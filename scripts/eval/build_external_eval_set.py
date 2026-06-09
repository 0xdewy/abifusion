#!/usr/bin/env python3
"""Build an external evaluation set from Sourcify.

Flow:
  1. Pull candidate address list from Sourcify API or local parquet.
  2. De-duplicate against existing data/contracts.parquet.
  3. Fetch metadata + bytecode (throttled, cached, resumable) until
     candidate-limit valid rows are cached OR source is exhausted.
  4. Post-fetch stratified sample down to --limit using composite buckets.
  5. Output data/external_eval.parquet + data/external_eval_meta.json.

Usage:
    python scripts/eval/build_external_eval_set.py \\
        --source auto \\
        --candidate-limit 1500 \\
        --limit 500 \\
        --rate-limit 1.0 \\
        --resume \\
        --output data/external_eval.parquet

Caching:
    Fetched contracts are cached under data/cache/external_eval/{chain_id}/{address}.json
    On --resume, skip any contract already cached.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import numpy as np
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")
CACHE_DIR = REPO_ROOT / "data" / "cache" / "external_eval"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "external_eval.parquet"
DEFAULT_META = REPO_ROOT / "data" / "external_eval_meta.json"
SOURCIFY_CONTRACTS_URL = "https://sourcify.dev/server/v2/contracts/1"
SOURCIFY_CONTRACT_URL = "https://sourcify.dev/server/v2/contract/{chain_id}/{address}"

MIN_FUNCTIONS = 2
MIN_COMPILER_VERSION = (0, 4, 0)


def parse_compiler_version(version: str) -> tuple:
    """Parse 'v0.8.24+commit.e11b9ed9' -> (0, 8, 24)."""
    v = version.lstrip("v").split("+")[0]
    parts = v.split(".")
    return tuple(int(p) for p in parts[:3])


def age_bucket(created_at: str | None) -> str:
    if not created_at:
        return "unknown"
    try:
        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return "unknown"
    now = datetime.now(timezone.utc)
    delta = (now - created).days
    if delta < 30:
        return "0-30d"
    elif delta < 180:
        return "30-180d"
    elif delta < 365:
        return "180d-1y"
    elif delta < 730:
        return "1y-2y"
    else:
        return "2y+"


def func_count_bucket(n: int) -> int:
    return min(n // 5, 20)


def bc_size_bucket(size_bytes: int) -> int:
    return min(size_bytes // 500, 20)


def compiler_bucket(version: str) -> str:
    try:
        major, minor, patch = parse_compiler_version(version)
        return f"{major}.{minor}"
    except (ValueError, TypeError):
        return "unknown"


def composite_bucket(row: dict) -> str:
    fb = func_count_bucket(row.get("abi_function_count", 0))
    bb = bc_size_bucket(row.get("bytecode_size", 0))
    cb = compiler_bucket(row.get("compiler_version", ""))
    ab = age_bucket(row.get("created_at"))
    return f"{fb}_{bb}_{cb}_{ab}"


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    df = df.copy()
    df["_bucket"] = df.apply(composite_bucket, axis=1)
    buckets = df.groupby("_bucket", group_keys=False)
    sampled = []
    rng = np.random.RandomState(seed)
    for _, group in buckets:
        k = max(1, round(n * len(group) / len(df)))
        idx = rng.choice(len(group), size=min(k, len(group)), replace=False)
        sampled.append(group.iloc[idx])
    result = pd.concat(sampled, ignore_index=True)
    if len(result) > n:
        idx = rng.choice(len(result), size=n, replace=False)
        result = result.iloc[idx]
    return result


def fetch_sourcify_addresses(source: str = "api", target_count: int = 1500) -> list[dict]:
    if source == "auto":
        source = "api"
    if source == "parquet":
        raise NotImplementedError("Local Sourcify parquet not yet implemented")
    logger.info("Fetching contract list from Sourcify API v2...")

    out: list[dict] = []
    after_match_id = None
    while len(out) < target_count:
        params = {"limit": 200, "sort": "desc"}
        if after_match_id is not None:
            params["afterMatchId"] = after_match_id
        resp = requests.get(SOURCIFY_CONTRACTS_URL, params=params, timeout=60)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            break
        out.extend(results)
        after_match_id = results[-1].get("matchId")
        if after_match_id is None:
            break
    return out


def fetch_contract_metadata(address: str, chain_id: int = 1) -> dict | None:
    url = SOURCIFY_CONTRACT_URL.format(chain_id=chain_id, address=address)
    resp = requests.get(url, params={"fields": "all"}, timeout=30)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def get_block_number(rpc_url: str) -> int:
    resp = requests.post(rpc_url, json={"jsonrpc": "2.0", "method": "eth_blockNumber", "params": [], "id": 1}, timeout=30)
    resp.raise_for_status()
    result = resp.json()["result"]
    return int(result, 16)


def get_code(rpc_url: str, address: str, block: int | str = "latest") -> str:
    resp = requests.post(
        rpc_url,
        json={"jsonrpc": "2.0", "method": "eth_getCode", "params": [address, hex(block) if isinstance(block, int) else block], "id": 1},
        timeout=30,
    )
    resp.raise_for_status()
    result = resp.json().get("result", "0x")
    return result[2:] if result.startswith("0x") else result


def cache_path(chain_id: int, address: str) -> Path:
    return CACHE_DIR / str(chain_id) / f"{address}.json"


def load_from_cache(chain_id: int, address: str) -> dict | None:
    p = cache_path(chain_id, address)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def save_to_cache(chain_id: int, address: str, data: dict) -> None:
    p = cache_path(chain_id, address)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(data, f)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build external evaluation set from Sourcify")
    parser.add_argument("--source", choices=["auto", "api", "parquet"], default="auto")
    parser.add_argument("--candidate-limit", type=int, default=1500)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--rate-limit", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output_path = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rpc_url = os.environ.get("ETH_RPC_URL")
    if not rpc_url:
        logger.error("ETH_RPC_URL environment variable is required")
        return 1

    # 1. Load existing addresses to exclude
    existing_df = pd.read_parquet(REPO_ROOT / "data" / "contracts.parquet")
    existing_addrs = set(existing_df["address"].str.lower())
    logger.info("Existing addresses to exclude: %d", len(existing_addrs))

    # 2. Fetch candidate address list
    try:
        candidates_raw = fetch_sourcify_addresses(args.source, args.candidate_limit * 3)
    except Exception as e:
        logger.error("Failed to fetch Sourcify contract list: %s", e)
        return 1

    candidates = [
        {
            "address": c["address"],
            "chain_id": int(c.get("chainId", 1)),
            "created_at": c.get("verifiedAt"),
        }
        for c in candidates_raw
        if c.get("address", "").startswith("0x") and c.get("address", "").lower() not in existing_addrs
    ]
    logger.info("Sourcify candidates (after dedup): %d", len(candidates))

    if not candidates:
        logger.error("No candidates found after de-duplication")
        return 1

    # 3. Fetch fixed block number for reproducibility
    try:
        fixed_block = get_block_number(rpc_url)
        logger.info("Using fixed block number: %d", fixed_block)
    except Exception as e:
        logger.error("Failed to get block number: %s", e)
        return 1

    # 4. Fetch metadata + bytecode for candidates
    fetched = []
    last_time = [0.0]

    def rate_limit():
        elapsed = time.time() - last_time[0]
        if elapsed < (1.0 / args.rate_limit):
            time.sleep((1.0 / args.rate_limit) - elapsed)
        last_time[0] = time.time()

    for i, cand in enumerate(candidates):
        addr = cand["address"].lower()
        chain = cand["chain_id"]

        if args.resume:
            cached = load_from_cache(chain, addr)
            if cached is not None:
                fetched.append({**cand, **cached})
                continue

        rate_limit()

        try:
            meta = fetch_contract_metadata(addr, chain)
            if meta is None:
                logger.debug("Skipping %s: not on Sourcify", addr)
                continue
        except Exception as e:
            logger.warning("Skipping %s: metadata fetch failed: %s", addr, e)
            continue

        try:
            bytecode = get_code(rpc_url, addr, fixed_block)
        except Exception as e:
            logger.warning("Skipping %s: RPC getCode failed: %s", addr, e)
            continue

        if bytecode in ("", "0x", "0x0"):
            logger.debug("Skipping %s: no runtime bytecode", addr)
            continue

        abi = meta.get("abi") or meta.get("metadata", {}).get("output", {}).get("abi", [])
        func_count = sum(1 for item in abi if isinstance(item, dict) and item.get("type") == "function")
        if func_count < MIN_FUNCTIONS:
            logger.debug("Skipping %s: only %d functions", addr, func_count)
            continue

        compiler_version = (
            meta.get("compilation", {}).get("compilerVersion")
            or meta.get("metadata", {}).get("compiler", {}).get("version", "")
        )
        try:
            major, minor, _ = parse_compiler_version(compiler_version)
            if (major, minor) < MIN_COMPILER_VERSION:
                logger.debug("Skipping %s: compiler %s too old", addr, compiler_version)
                continue
        except (ValueError, TypeError):
            pass

        bytecode_hash = hashlib.sha256(bytecode.encode()).hexdigest()
        fetched_at = datetime.now(timezone.utc).isoformat()

        cache_data = {
            "metadata": {"abi": abi, "compiler_version": compiler_version},
            "bytecode": bytecode,
            "bytecode_hash": bytecode_hash,
            "fetched_at": fetched_at,
            "fetched_block_number": fixed_block,
            "abi_function_count": func_count,
            "bytecode_size": len(bytecode) // 2,
        }
        save_to_cache(chain, addr, cache_data)

        fetched.append({**cand, **cache_data})

        if len(fetched) % 50 == 0:
            logger.info("Fetched %d/%d valid candidates", len(fetched), args.candidate_limit)

        if len(fetched) >= args.candidate_limit:
            logger.info("Reached candidate limit %d", args.candidate_limit)
            break

        if (i + 1) % 200 == 0:
            logger.info("Processed %d/%d raw candidates", i + 1, len(candidates))

    logger.info("Total valid fetched: %d", len(fetched))

    if len(fetched) < args.limit:
        logger.warning(
            "Only %d valid candidates fetched (target: %d). "
            "Consider increasing --candidate-limit or using a different source.",
            len(fetched), args.limit,
        )

    # 5. Stratified sample
    df = pd.DataFrame(fetched)
    df["abi"] = df["metadata"].apply(lambda m: json.dumps(m.get("abi", [])))
    df["compiler_version"] = df["metadata"].apply(lambda m: m.get("compiler_version", ""))

    sample_df = stratified_sample(df, min(args.limit, len(df)), seed=42)
    logger.info("Stratified sample: %d contracts", len(sample_df))

    # 6. Write output
    out_cols = [
        "address", "chain_id", "bytecode", "abi", "compiler_version",
        "bytecode_size", "abi_function_count", "created_at",
        "fetched_at", "fetched_block_number",
    ]
    sample_df[out_cols].to_parquet(output_path, index=False)

    meta = {
        "fetch_date": datetime.now(timezone.utc).date().isoformat(),
        "candidate_count": len(candidates),
        "valid_count": len(fetched),
        "final_count": len(sample_df),
        "selection_seed": 42,
        "source": "api",
        "fetched_block_number": fixed_block,
    }
    meta_path = DEFAULT_META if not args.output.is_absolute() else args.output.parent / "external_eval_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Wrote %s (%d contracts)", output_path, len(sample_df))
    logger.info("Wrote %s", meta_path)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
