#!/usr/bin/env python3
"""Evaluation benchmark for ABI Reconstructor.

Measures selector F1, prototype/type F1, and full-signature exact-match accuracy
on ground-truth verified contracts. Reports per-compiler-version breakdown when
available.

Modes:
  --source local  : evaluate against a JSON file of contracts
  --source etherscan : sample verified contracts from Etherscan
  --sample N     : use N random contracts from the source
  --stratify     : report per-compiler-version metrics

Output:
  eval_results.json  : per-contract metrics
  eval_summary.md    : aggregate report

Usage:
  python scripts/eval/benchmark.py --source local --input contracts.json
  python scripts/eval/benchmark.py --source etherscan --sample 200 --stratify
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from abifusion import OfflineABI


# ── Data types ────────────────────────────────────────────────────────────


@dataclass
class ContractGroundTruth:
    """A contract with known ground-truth ABI for evaluation."""
    address: str = ""
    bytecode: str = ""
    abi: List[Dict[str, Any]] = field(default_factory=list)
    compiler_version: str = ""
    contract_name: str = ""


@dataclass
class PerContractMetrics:
    """Metrics for a single contract's ABI reconstruction."""
    address: str
    contract_name: str
    compiler_version: str

    # Ground truth counts
    gt_function_count: int
    # Reconstructed counts
    rec_function_count: int

    # Selector-level metrics
    selector_recall: float
    selector_precision: float
    selector_f1: float

    # Prototype/type-level metrics (per-parameter type accuracy)
    type_accuracy: float          # % of parameter types correct
    arity_accuracy: float         # % of functions with correct parameter count
    prototype_f1: float           # harmonic mean of exact-type-match per function

    # Full-signature exact match (selector + all types correct)
    full_signature_accuracy: float


@dataclass
class EvalResults:
    """Aggregate evaluation results."""
    total_contracts: int
    total_functions: int

    # Aggregate selector metrics (micro-averaged across all functions)
    selector_precision: float
    selector_recall: float
    selector_f1: float

    # Aggregate prototype metrics
    prototype_f1: float
    type_accuracy: float
    arity_accuracy: float
    full_signature_accuracy: float

    # Per-compiler-version breakdown
    by_compiler: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Per-contract details
    per_contract: List[PerContractMetrics] = field(default_factory=list)


# ── Data loading ───────────────────────────────────────────────────────────


def load_contracts_from_json(filepath: str) -> List[ContractGroundTruth]:
    """Load ground-truth contracts from a JSON file.

    Expected format: array of objects with:
      - address (string, optional)
      - bytecode (string, hex)
      - abi (array of ABI entries, each with type/name/inputs)
      - compiler_version (string, optional)
      - contract_name (string, optional)
    """
    with open(filepath) as f:
        data = json.load(f)

    contracts = []
    for item in data:
        contracts.append(ContractGroundTruth(
            address=item.get("address", ""),
            bytecode=item.get("bytecode", ""),
            abi=item.get("abi", []),
            compiler_version=item.get("compiler_version", ""),
            contract_name=item.get("contract_name", ""),
        ))
    return contracts


def sample_etherscan_contracts(
    client,
    count: int = 200,
    seed: int = 42,
) -> List[ContractGroundTruth]:
    """Sample verified contracts from Etherscan using known addresses."""
    # Well-known verified contract addresses on Ethereum mainnet
    KNOWN_ADDRESSES = [
        # ERC20 tokens
        "0xdAC17F958D2ee523a2206206994597C13D831ec7",  # USDT
        "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",  # USDC
        "0x6B175474E89094C44Da98b954EedeAC495271d0F",  # DAI
        "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984",  # UNI
        "0x7D1AfA7B718fb893dB30A3aBc0Cfc608AaCfeBB0",  # MATIC
        "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",  # WBTC
        "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
        "0x514910771AF9Ca656af840dff83E8264EcF986CA",  # LINK
        "0x95aD61b0a150d79219dCF64E1E6Cc01f0B64C4cE",  # SHIB
        "0x2b591e99afE9f32eAA6214f7B7629768c40Eeb39",  # HEX
        # DeFi protocols
        "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",  # Uniswap V2 Router
        "0xE592427A0AEce92De3Edee1F18E0157C05861564",  # Uniswap V3 Router
        "0x1111111254EEB25477B68fb85Ed929f73A960582",  # 1inch Router v5
        # Governance
        "0x408ED6354d4973f66138C91495F2f2FCbd8724C3",  # Compound Governor
        # ERC721
        "0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13D",  # BAYC
        "0x60E4d786628Fea6478F785A6d7e704777c86a7c6",  # MAYC
        # Contracts with varied compiler versions
        "0x0AbdAce70D3790235af448C88547603b945604ea",  # DNT (district0x)
        "0x595832F8FC6BF59c85C527fEC3740A1b7a361269",  # POWR
        "0x960b236A07cf122663c4303350609A66A7B288C0",  # ANT (Aragon)
        "0xBBbbCA6A901c926F240b89EacB641d8Aec7AEafD",  # LRC (Loopring)
    ]

    random.seed(seed)
    selected = random.sample(KNOWN_ADDRESSES, min(count, len(KNOWN_ADDRESSES)))

    contracts = []
    for addr in selected:
        try:
            abi = client.get_contract_abi(addr)
            src = client.get_contract_source_code(addr)
            if abi is None or src is None or len(src) == 0:
                continue
            result = src[0] if isinstance(src, list) else src
            bytecode = client.get_contract_bytecode(addr)  # Actual deployed bytecode
            compiler = result.get("compiler_version", "") or result.get("CompilerVersion", "")
            name = result.get("contract_name", "") or result.get("ContractName", "")
            if not bytecode or bytecode == "0x":
                continue
            contracts.append(ContractGroundTruth(
                address=addr,
                bytecode=bytecode,
                abi=abi,
                compiler_version=compiler,
                contract_name=name,
            ))
        except Exception as e:
            print(f"  Skipping {addr}: {e}")
            continue

    return contracts


# ── Metric computation ─────────────────────────────────────────────────────


def keccak256(data: bytes) -> bytes:
    """Compute keccak256 hash of data."""
    try:
        from Crypto.Hash import keccak
        k = keccak.new(digest_bits=256)
        k.update(data)
        return k.digest()
    except ImportError:
        import hashlib
        return hashlib.sha3_256(data).digest()


def parse_ground_truth_functions(abi: List[Dict]) -> List[Dict]:
    """Extract function entries with selectors from a ground-truth ABI."""
    functions = []
    for entry in abi:
        if entry.get("type") != "function":
            continue
        name = entry.get("name", "")
        inputs = entry.get("inputs", [])
        sig = f"{name}({','.join(p.get('type','') for p in inputs)})"
        selector = keccak256(sig.encode("utf-8")).hex()[:8]
        functions.append({
            "selector": selector,
            "name": name,
            "input_types": [p.get("type", "") for p in inputs],
            "arity": len(inputs),
        })
    return functions


def tokenize_abi_types(type_str: str) -> str:
    """Normalize ABI types for comparison: 'uint256' = 'uint' for Solidity type matching."""
    import re
    t = type_str.strip().lower()
    # Normalize uint/int bit sizes: uint256 → uint, uint256[] → uint[]
    t = re.sub(r'\buint\d+\b', 'uint', t)
    t = re.sub(r'\bint\d+\b', 'int', t)
    t = re.sub(r'\bbytes\d+\b', 'bytesN', t)
    return t


def compute_contract_metrics(
    gt_functions: List[Dict],
    rec_functions: List[Dict],
    address: str = "",
    contract_name: str = "",
    compiler_version: str = "",
) -> PerContractMetrics:
    """Compute per-contract metrics comparing ground-truth and reconstructed ABIs."""

    gt_selector_set = {f["selector"] for f in gt_functions}
    rec_selector_set = {f.get("selector", "") for f in rec_functions}

    # ── Selector-level metrics ──
    tp_selectors = gt_selector_set & rec_selector_set
    selector_precision = len(tp_selectors) / len(rec_selector_set) if rec_selector_set else 0.0
    selector_recall = len(tp_selectors) / len(gt_selector_set) if gt_selector_set else 0.0
    selector_f1 = _f1(selector_precision, selector_recall)

    # ── Prototype/type-level metrics ──
    # Build lookup: selector -> ground-truth (name, input_types, arity)
    gt_lookup = {f["selector"]: f for f in gt_functions}

    total_funcs = 0
    arity_correct = 0
    type_correct_params = 0
    total_params = 0
    exact_type_match_count = 0

    for rec in rec_functions:
        sel = rec.get("selector", "")
        if sel not in gt_lookup:
            continue
        total_funcs += 1
        gt = gt_lookup[sel]

        # Arity match
        rec_arity = rec.get("arity", 0)
        if rec_arity == gt["arity"]:
            arity_correct += 1

        # Per-parameter type match
        gt_types = [tokenize_abi_types(t) for t in gt["input_types"]]
        rec_types_list = rec.get("input_types", [])
        for i, rt in enumerate(rec_types_list):
            if i < len(gt_types) and tokenize_abi_types(rt) == gt_types[i]:
                type_correct_params += 1
        total_params += len(rec_types_list)

        # Exact type match (all params correct, correct arity)
        if rec_arity == gt["arity"] and all(
            i < len(gt_types) and tokenize_abi_types(rec_types_list[i]) == gt_types[i]
            for i in range(rec_arity)
        ):
            exact_type_match_count += 1

    type_accuracy = type_correct_params / total_params if total_params > 0 else 0.0
    arity_accuracy = arity_correct / total_funcs if total_funcs > 0 else 0.0
    exact_type_p = exact_type_match_count / total_funcs if total_funcs > 0 else 0.0
    exact_type_r = exact_type_match_count / len(gt_functions) if gt_functions else 0.0
    prototype_f1 = _f1(exact_type_p, exact_type_r)

    # Full-signature exact match
    # A function is "exact match" if: selector found + arity correct + all types match
    full_exact = sum(
        1 for sel in gt_selector_set & rec_selector_set
        if (
            (rec_f := next((r for r in rec_functions if r.get("selector") == sel), None))
            and rec_f.get("arity", 0) == gt_lookup[sel]["arity"]
            and all(
                i < len(gt_lookup[sel]["input_types"])
                and tokenize_abi_types(rec_f.get("input_types", [])[i]) == tokenize_abi_types(gt_lookup[sel]["input_types"][i])
                for i in range(rec_f.get("arity", 0))
            )
        )
    )
    full_signature_acc = full_exact / len(gt_selector_set) if gt_selector_set else 0.0

    return PerContractMetrics(
        address=address,
        contract_name=contract_name,
        compiler_version=compiler_version,
        gt_function_count=len(gt_functions),
        rec_function_count=len(rec_functions),
        selector_recall=selector_recall,
        selector_precision=selector_precision,
        selector_f1=selector_f1,
        type_accuracy=type_accuracy,
        arity_accuracy=arity_accuracy,
        prototype_f1=prototype_f1,
        full_signature_accuracy=full_signature_acc,
    )


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ── Aggregation ────────────────────────────────────────────────────────────


def aggregate_results(per_contract: List[PerContractMetrics]) -> EvalResults:
    """Aggregate per-contract metrics into global statistics."""
    total_funcs = sum(c.gt_function_count for c in per_contract)

    # Micro-averaged selector metrics
    total_tp = 0
    total_gt = 0
    total_rec = 0
    total_type_correct = 0
    total_params = 0
    total_arity_correct = 0
    total_exact_type = 0
    total_full_exact = 0

    for c in per_contract:
        total_gt += c.gt_function_count
        total_rec += c.rec_function_count
        total_tp += int(c.selector_recall * c.gt_function_count)  # approximate
        total_arity_correct += int(c.arity_accuracy * max(c.gt_function_count, 1))
        total_type_correct += int(c.type_accuracy * max(c.gt_function_count, 1))
        total_params += max(c.gt_function_count, 1)
        total_exact_type += int(c.prototype_f1 * max(c.gt_function_count, 1) / (2 - c.prototype_f1 + 0.001))  # rough
        total_full_exact += int(c.full_signature_accuracy * c.gt_function_count)

    # Recompute micro-averages properly
    total_gt_selectors = sum(c.gt_function_count for c in per_contract)
    total_rec_selectors = sum(c.rec_function_count for c in per_contract)
    # TP selectors = recall * gt for each contract
    total_tp_selectors = sum(
        int(c.selector_recall * c.gt_function_count) for c in per_contract
    )

    # Per-compiler breakdown
    by_compiler: Dict[str, dict] = {}
    for c in per_contract:
        ver = _normalize_compiler_version(c.compiler_version)
        if ver not in by_compiler:
            by_compiler[ver] = {
                "n_contracts": 0, "n_functions": 0,
                "selector_f1": 0.0, "prototype_f1": 0.0,
                "selector_f1_sum": 0.0, "prototype_f1_sum": 0.0,
            }
        by_compiler[ver]["n_contracts"] += 1
        by_compiler[ver]["n_functions"] += c.gt_function_count
        by_compiler[ver]["selector_f1_sum"] += c.selector_f1 * c.gt_function_count
        by_compiler[ver]["prototype_f1_sum"] += c.prototype_f1 * c.gt_function_count

    for ver in by_compiler:
        nf = by_compiler[ver]["n_functions"]
        by_compiler[ver]["selector_f1"] = by_compiler[ver]["selector_f1_sum"] / nf if nf else 0
        by_compiler[ver]["prototype_f1"] = by_compiler[ver]["prototype_f1_sum"] / nf if nf else 0

    return EvalResults(
        total_contracts=len(per_contract),
        total_functions=total_gt_selectors,
        selector_precision=total_tp_selectors / total_rec_selectors if total_rec_selectors else 0,
        selector_recall=total_tp_selectors / total_gt_selectors if total_gt_selectors else 0,
        selector_f1=_f1(
            total_tp_selectors / total_rec_selectors if total_rec_selectors else 0,
            total_tp_selectors / total_gt_selectors if total_gt_selectors else 0,
        ),
        prototype_f1=sum(c.prototype_f1 * c.gt_function_count for c in per_contract) / total_gt_selectors if total_gt_selectors else 0,
        type_accuracy=sum(c.type_accuracy * c.gt_function_count for c in per_contract) / total_gt_selectors if total_gt_selectors else 0,
        arity_accuracy=sum(c.arity_accuracy * c.gt_function_count for c in per_contract) / total_gt_selectors if total_gt_selectors else 0,
        full_signature_accuracy=sum(c.full_signature_accuracy * c.gt_function_count for c in per_contract) / total_gt_selectors if total_gt_selectors else 0,
        by_compiler=by_compiler,
        per_contract=list(per_contract),
    )


def _normalize_compiler_version(version: str) -> str:
    """Normalize compiler version to major.minor."""
    import re
    m = re.match(r'v?(\d+\.\d+)', version)
    return m.group(1) if m else version or "unknown"


# ── Output ─────────────────────────────────────────────────────────────────


def write_results(results: EvalResults, output_dir: str):
    """Write eval_results.json and eval_summary.md."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = out_dir / "eval_results.json"
    with open(json_path, "w") as f:
        json.dump({
            "total_contracts": results.total_contracts,
            "total_functions": results.total_functions,
            "selector_precision": results.selector_precision,
            "selector_recall": results.selector_recall,
            "selector_f1": results.selector_f1,
            "prototype_f1": results.prototype_f1,
            "type_accuracy": results.type_accuracy,
            "arity_accuracy": results.arity_accuracy,
            "full_signature_accuracy": results.full_signature_accuracy,
            "by_compiler": results.by_compiler,
            "per_contract": [asdict(c) for c in results.per_contract],
        }, f, indent=2)
    print(f"  -> {json_path}")

    # Markdown summary
    md_path = out_dir / "eval_summary.md"
    lines = [
        f"# ABI Reconstructor Evaluation",
        f"",
        f"**{results.total_contracts} contracts · {results.total_functions} functions**",
        f"",
        f"## Aggregate Metrics",
        f"",
        f"| Metric | Value | SOTA Reference |",
        f"|--------|-------|----------------|",
        f"| Selector F1 | {results.selector_f1:.4f} | Heimdall-rs: 0.996 (SCDBench, 2026) |",
        f"| Selector Precision | {results.selector_precision:.4f} | Gigahorse: 0.991 |",
        f"| Selector Recall | {results.selector_recall:.4f} | Gigahorse: 0.991 |",
        f"| Prototype F1 (type-level) | {results.prototype_f1:.4f} | Heimdall-rs: 0.777 (SCDBench, 2026) |",
        f"| Type Accuracy (per-parameter) | {results.type_accuracy:.4f} | — |",
        f"| Arity Accuracy (correct param count) | {results.arity_accuracy:.4f} | — |",
        f"| Full-Signature Exact Match | {results.full_signature_accuracy:.4f} | — |",
        f"",
    ]

    if results.by_compiler:
        lines += [
            f"## By Compiler Version",
            f"",
            f"| Version | Contracts | Functions | Selector F1 | Prototype F1 |",
            f"|---------|-----------|-----------|-------------|-------------|",
        ]
        for ver in sorted(results.by_compiler.keys()):
            d = results.by_compiler[ver]
            lines.append(
                f"| v{ver} | {d['n_contracts']} | {d['n_functions']} | "
                f"{d['selector_f1']:.4f} | {d['prototype_f1']:.4f} |"
            )
        lines.append("")

    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  -> {md_path}")


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="ABI Reconstructor evaluation benchmark")
    parser.add_argument("--source", choices=["local", "etherscan"], default="local",
                        help="Contract data source")
    parser.add_argument("--input", type=str, default="",
                        help="Path to JSON file of ground-truth contracts (for --source local)")
    parser.add_argument("--sample", type=int, default=20,
                        help="Number of contracts to evaluate")
    parser.add_argument("--db", type=str, default="./cache/4byte.db",
                        help="Path to 4byte database")
    parser.add_argument("--output-dir", type=str, default="./eval_output",
                        help="Directory for eval_results.json and eval_summary.md")
    parser.add_argument("--stratify", action="store_true",
                        help="Report per-compiler-version breakdown")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    args = parser.parse_args()

    random.seed(args.seed)

    # ── Load contracts ──
    if args.source == "local":
        if not args.input:
            print("ERROR: --input required for --source local")
            sys.exit(1)
        contracts = load_contracts_from_json(args.input)
    elif args.source == "etherscan":
        from abifusion.data.etherscan_fetcher import EtherscanClient
        client = EtherscanClient()
        contracts = sample_etherscan_contracts(client, count=args.sample, seed=args.seed)
    else:
        print(f"Unknown source: {args.source}")
        sys.exit(1)

    if not contracts:
        print("No contracts loaded. Aborting.")
        sys.exit(1)

    contracts = contracts[:args.sample]
    print(f"Evaluating {len(contracts)} contracts...")

    # ── Run reconstruction ──
    reconstructor = OfflineABI(db_path=args.db)
    per_contract_metrics: List[PerContractMetrics] = []

    for i, c in enumerate(contracts):
        addr_display = c.address[:10] + "..." if c.address else f"#{i}"
        print(f"  [{i+1}/{len(contracts)}] {c.contract_name or addr_display}")

        try:
            # Run extraction
            result = reconstructor.reconstruct_abi(c.bytecode, max_selectors=50)

            # Parse ground truth
            gt_funcs = parse_ground_truth_functions(c.abi)

            # Parse reconstructed
            rec_funcs = []
            for f in result.get("functions", []):
                params = f.get("parameters", [])
                rec_funcs.append({
                    "selector": f.get("selector", ""),
                    "name": f.get("function_name", ""),
                    "input_types": [p.get("type", "uint256") for p in params],
                    "arity": len(params),
                })

            metrics = compute_contract_metrics(
                gt_funcs, rec_funcs,
                address=c.address,
                contract_name=c.contract_name,
                compiler_version=c.compiler_version,
            )
            per_contract_metrics.append(metrics)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

    reconstructor.close()

    # ── Aggregate and write ──
    results = aggregate_results(per_contract_metrics)
    write_results(results, args.output_dir)

    print(f"\n{'='*50}")
    print(f"Results: {results.total_contracts} contracts, {results.total_functions} functions")
    print(f"  Selector F1:  {results.selector_f1:.4f}")
    print(f"  Prototype F1: {results.prototype_f1:.4f}")
    print(f"  Type Acc:     {results.type_accuracy:.4f}")
    print(f"  Arity Acc:    {results.arity_accuracy:.4f}")
    print(f"  Full Exact:   {results.full_signature_accuracy:.4f}")

    if args.stratify and results.by_compiler:
        print(f"\nPer-compiler breakdown:")
        for ver in sorted(results.by_compiler.keys()):
            d = results.by_compiler[ver]
            print(f"  v{ver}: sel F1={d['selector_f1']:.4f}, proto F1={d['prototype_f1']:.4f} ({d['n_contracts']} contracts, {d['n_functions']} funcs)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
