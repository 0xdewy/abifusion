#!/usr/bin/env python3
"""Fetch verified contracts from Sourcify API and build a dataset.

Source: Sourcify REST API — no massive parquet downloads needed.
Fetches contract addresses, then bytecodes + ABIs, saves as parquet.

Usage:
    python scripts/fetch_more_contracts.py --limit 500 --output data/sourcify_fresh.parquet
"""

from __future__ import annotations

import json
import time
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

SOURCIFY_SERVER = "https://sourcify.dev/server"
USER_AGENT = "abifusion/0.1 (evaluation data collection)"
RATE_LIMIT = 0.3  # seconds between requests
BATCH_SIZE = 100  # contracts per pagination page

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})


def list_verified(chains: tuple = (1, 42161, 10, 8453), limit: int = 500) -> list[dict]:
    """Get verified contract addresses from Sourcify.

    Chain IDs: 1=mainnet, 42161=Arbitrum, 10=Optimism, 8453=Base
    """
    results = []
    seen = set()

    for chain in chains:
        page = 1
        while len(results) < limit:
            url = f"{SOURCIFY_SERVER}/contracts/{chain}"
            params = {"limit": min(BATCH_SIZE, limit - len(results)), "page": page, "match": "perfect"}
            print(f"  chain {chain} page {page} ...", end=" ", flush=True)
            try:
                resp = session.get(url, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                contracts = data.get("results", [])
                for c in contracts:
                    addr = (c.get("address", "")).lower()
                    if addr and addr not in seen:
                        seen.add(addr)
                        results.append({
                            "address": "0x" + addr,
                            "chain_id": chain,
                            "name": c.get("name", ""),
                        })
                total = data.get("pagination", {}).get("totalMatches", 0)
                print(f"{len(contracts)} results (total={total})")
                if len(contracts) < BATCH_SIZE or page * BATCH_SIZE >= total:
                    break
                page += 1
            except Exception as e:
                print(f"error: {e}")
                break
        if len(results) >= limit:
            break

    print(f"  got {len(results)} unique addresses")
    return results[:limit]


def fetch_contract(address: str, chain_id: int) -> Optional[dict]:
    """Fetch bytecode and ABI for one verified contract from Sourcify."""
    url = f"{SOURCIFY_SERVER}/files/any/{chain_id}/{address}"
    try:
        resp = session.get(url, timeout=30)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()

        bytecode = ""
        abi = []
        compiler = ""

        for f in data.get("files", []):
            name = f.get("name", "")
            content = f.get("content", "")
            if name.endswith(".json") and "metadata" in name.lower():
                try:
                    meta = json.loads(content) if isinstance(content, str) else content
                    compiler = meta.get("compiler", {}).get("version", "")
                except Exception:
                    pass

        # Sourcify returns the runtime bytecode in the metadata
        try:
            meta_file = next((f for f in data.get("files", []) if f["name"] == "metadata.json"), None)
            if meta_file:
                meta = json.loads(meta_file["content"])
                output = meta.get("output", {})
                abi = output.get("abi", [])
                deployed = output.get("deployedBytecode", {})
                if isinstance(deployed, dict):
                    bytecode = deployed.get("object", "")
                elif isinstance(deployed, str):
                    bytecode = deployed
                compiler = meta.get("compiler", {}).get("version", "")
                if not bytecode:
                    for name in ("deployedBytecode", "runtimeBytecode"):
                        extra = output.get(name, {})
                        if isinstance(extra, dict):
                            extra = extra.get("object", "")
                        if extra and extra != "0x":
                            bytecode = extra
                            break
        except Exception:
            pass

        if not bytecode or not abi:
            return None
        if not bytecode.startswith("0x"):
            bytecode = "0x" + bytecode
        if len(bytecode) < 100:
            return None

        return {
            "address": address,
            "chain_id": chain_id,
            "bytecode": bytecode,
            "abi": abi,
            "compiler_version": compiler or "unknown",
        }
    except Exception as e:
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=500, help="Target number of contracts")
    parser.add_argument("--output", default="data/sourcify_fresh.parquet", help="Output parquet path")
    parser.add_argument("--chains", default="1,42161,10,8453", help="Chain IDs to query")
    args = parser.parse_args()

    chains = tuple(int(c.strip()) for c in args.chains.split(",") if c.strip())

    print(f"Fetching up to {args.limit} contracts from chains {chains}...")
    addresses = list_verified(chains=chains, limit=args.limit)

    contracts = []
    fetched = 0
    failed = 0

    for i, item in enumerate(addresses):
        if i > 0 and i % 50 == 0:
            print(f"  progress: {len(contracts)} collected, {failed} failed ({i}/{len(addresses)})")

        time.sleep(RATE_LIMIT)
        result = fetch_contract(item["address"], item["chain_id"])
        if result:
            result["name"] = item.get("name", "")
            contracts.append(result)
            fetched += 1
        else:
            failed += 1

    print(f"\nDone: {len(contracts)} contracts fetched, {failed} failed")

    if contracts:
        df = pd.DataFrame(contracts)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.output, index=False)
        print(f"Saved {len(df)} contracts to {args.output}")
        print(f"Chains: {df.chain_id.value_counts().to_dict()}")
        print(f"With ABI: {(df['abi'].apply(len) > 0).sum()}")
    else:
        print("No contracts fetched", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
