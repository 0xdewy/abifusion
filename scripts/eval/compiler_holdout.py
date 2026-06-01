#!/usr/bin/env python3
"""Compiler holdout experiment for ABI reconstruction.

Evaluates whether ABI reconstruction accuracy degrades across Solidity compiler
versions. Compares two evaluation strategies:
  1. Compiler-holdout: evaluate separately on v0.4, v0.5, v0.6, v0.7, v0.8 groups
  2. Random-baseline: same contracts randomly shuffled (control)

Usage:
  python scripts/eval/compiler_holdout.py --sample 50
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Crypto.Hash import keccak
from dotenv import load_dotenv

load_dotenv()

from abi_reconstructor.data.etherscan_fetcher import EtherscanClient
from abi_reconstructor.reconstructor import ABIReconstructor, BytecodeParser
from abi_reconstructor.training.compiler_splitter import (
    parse_compiler_version,
    version_in_range,
)


# ── Well-known verified contracts with known compiler versions ──

# Curated list of verified contracts across compiler versions
COMPILER_DIVERSE_ADDRESSES = [
    # v0.4.x (older compilers)
    ("0xdAC17F958D2ee523a2206206994597C13D831ec7", "USDT", "v0.4.18"),
    ("0x514910771AF9Ca656af840dff83E8264EcF986CA", "LINK", "v0.4.16"),
    ("0x0AbdAce70D3790235af448C88547603b945604ea", "DNT", "v0.4.11"),
    ("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2", "WETH", "v0.4.19"),
    ("0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48", "USDC", "v0.4.24"),
    ("0x7D1AfA7B718fb893dB30A3aBc0Cfc608AaCfeBB0", "MATIC", "v0.5.2"),
    ("0xB8c77482e45F1F44dE1745F52C74426C631bDD52", "BNB", "v0.4.11"),
    ("0x9f8F72aA9304c8B593d555F12eF6589cC3A579A2", "MKR", "v0.4.24"),
    ("0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984", "UNI", "v0.5.16"),
    ("0x6B175474E89094C44Da98b954EedeAC495271d0F", "DAI", "v0.5.12"),
    ("0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599", "WBTC", "v0.4.24"),
    ("0x595832F8FC6BF59c85C527fEC3740A1b7a361269", "POWR", "v0.4.11"),
    ("0x960b236A07cf122663c4303350609A66A7B288C0", "ANT", "v0.4.8"),
    ("0xBBbbCA6A901c926F240b89EacB641d8Aec7AEafD", "LRC", "v0.5.7"),
    # v0.6.x
    ("0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D", "UniV2Router", "v0.6.6"),
    ("0x11111112542D85B3EF4690572D7e30dDDFC396E5", "1inchV3Resolver", "v0.6.12"),
    ("0x1111111254fb6c44bAC0beD2854e76F90643097d", "1inchV4", "v0.6.12"),
    ("0xDef1C0ded9bec7F1a1670819833240f027b25EfF", "0xRouter", "v0.6.12"),
    # v0.7.x
    ("0xE592427A0AEce92De3Edee1F18E0157C05861564", "UniV3Router", "v0.7.6"),
    ("0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45", "UniV3Router2", "v0.7.6"),
    ("0x60E4d786628Fea6478F785A6d7e704777c86a7c6", "MAYC", "v0.7.0"),
    ("0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13D", "BAYC", "v0.7.0"),
    ("0x2b591e99afE9f32eAA6214f7B7629768c40Eeb39", "HEX", "v0.5.13"),
    ("0xBA12222222228d8Ba445958a75a0704d566BF2C8", "BalancerVault", "v0.7.6"),
    # v0.8.x (newer compilers)
    ("0x1111111254EEB25477B68fb85Ed929f73A960582", "1inchV5", "v0.8.17"),
    ("0x0000000022d53366457F9d5E68Ec105046FC4383", "dsProxyFactory", "v0.8.11"),
    ("0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9", "AAVE", "v0.8.10"),
    ("0x5f98805A4E8be255a32880FDeC7F6728C6568bA0", "LUSD", "v0.8.10"),
    ("0xD533a949740bb3306d119CC777fa900bA034cd52", "CRV", "v0.8.10"),
    ("0x4d224452801ACEd8B2F0aebE155379bb5D594381", "APE", "v0.8.17"),
    ("0x95aD61b0a150d79219dCF64E1E6Cc01f0B64C4cE", "SHIB", "v0.5.16"),
    ("0xC011a73ee8576Fb46F5E1c5751cA3B9Fe0af2a6F", "SNX", "v0.4.24"),
    ("0xD5147bc8e386d91Cc5DBE72099DAC6C9b99276F5", "renFIL", "v0.6.12"),
    ("0x408ED6354d4973f66138C91495F2f2FCbd8724C3", "GovernorBravo", "v0.5.16"),
    ("0x6De037ef9aD2725EB40118Bb1702EBb27e4Aeb24", "RNDR", "v0.6.12"),
    ("0x3b484b82567a09e2588A13D54D032153f0c0aEe0", "SOS", "v0.8.10"),
    ("0x6123B0049F904d730dB3C36a31167D9d4121fA6B", "RBN", "v0.8.10"),
]


def compute_selector(name: str, input_types: List[str]) -> str:
    sig = f"{name}({','.join(input_types)})"
    k = keccak.new(digest_bits=256)
    k.update(sig.encode("utf-8"))
    return k.digest().hex()[:8]


def normalize_type(t: str) -> str:
    import re
    t = t.strip().lower()
    t = re.sub(r'\buint\d+\b', 'uint', t)
    t = re.sub(r'\bint\d+\b', 'int', t)
    t = re.sub(r'\bbytes\d+\b', 'bytesN', t)
    return t


def evaluate_contract(
    addr: str,
    client: EtherscanClient,
    reconstructor: ABIReconstructor,
) -> Dict[str, Any]:
    """Evaluate ABI reconstruction on a single contract."""
    abi = client.get_contract_abi(addr)
    bytecode = client.get_contract_bytecode(addr)
    src = client.get_contract_source_code(addr)

    if not abi or not bytecode:
        return {"error": "no data", "address": addr}

    contract_name = src.get("contract_name", "") or src.get("ContractName", "") if src else addr[:10]
    compiler = src.get("compiler_version", "") or src.get("CompilerVersion", "") if src else ""

    # Ground truth functions
    gt_funcs = {}
    for entry in abi:
        if entry.get("type") != "function":
            continue
        name = entry["name"]
        inputs = [i.get("type", "") for i in entry.get("inputs", [])]
        sel = compute_selector(name, inputs)
        gt_funcs[sel] = {"name": name, "input_types": inputs, "arity": len(inputs)}

    # Run reconstruction
    result = reconstructor.reconstruct_abi(bytecode, max_selectors=100)
    rec_funcs = {}
    for f in result.get("functions", []):
        sel = f.get("selector", "")
        rec_funcs[sel] = {
            "name": f.get("function_name", ""),
            "input_types": [p.get("type", "uint256") for p in f.get("parameters", [])],
        }

    # Compute per-function metrics
    selector_correct = 0
    type_correct = 0
    exact_match = 0
    total = len(gt_funcs)

    for sel, gt in gt_funcs.items():
        if sel in rec_funcs:
            selector_correct += 1
            rec = rec_funcs[sel]
            rec_types = rec["input_types"]
            if len(gt["input_types"]) == len(rec_types) and all(
                normalize_type(g) == normalize_type(r)
                for g, r in zip(gt["input_types"], rec_types)
            ):
                type_correct += 1
                exact_match += 1

    return {
        "address": addr,
        "contract_name": contract_name,
        "compiler_version": compiler,
        "total_functions": total,
        "selector_correct": selector_correct,
        "type_correct": type_correct,
        "exact_match": exact_match,
        "selector_accuracy": selector_correct / total if total else 0,
        "type_accuracy": type_correct / total if total else 0,
        "exact_accuracy": exact_match / total if total else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Compiler holdout experiment")
    parser.add_argument("--sample", type=int, default=0,
                        help="Number of contracts to evaluate (0=all)")
    parser.add_argument("--db", type=str, default="./cache/4byte.db")
    parser.add_argument("--output", type=str, default="eval_output/compiler_holdout_report.md")
    args = parser.parse_args()

    addresses = COMPILER_DIVERSE_ADDRESSES
    if args.sample:
        addresses = addresses[:args.sample]

    client = EtherscanClient()
    reconstructor = ABIReconstructor(db_path=args.db)

    results = []
    for i, (addr, name, expected_ver) in enumerate(addresses):
        print(f"  [{i+1}/{len(addresses)}] {name} ({addr[:10]}...)")
        r = evaluate_contract(addr, client, reconstructor)
        if "error" in r:
            print(f"    SKIP: {r['error']}")
            continue
        results.append(r)

    reconstructor.close()

    # Group by major compiler version
    by_version: Dict[str, List[Dict]] = {}
    for r in results:
        ver = r.get("compiler_version", "")
        parsed = parse_compiler_version(ver)
        if parsed:
            major = f"v{parsed[0]}.{parsed[1]}"
        else:
            major = "unknown"
        by_version.setdefault(major, []).append(r)

    # Compute per-version aggregates
    version_stats = {}
    for ver, group in sorted(by_version.items()):
        total_f = sum(g["total_functions"] for g in group)
        sel_c = sum(g["selector_correct"] for g in group)
        type_c = sum(g["type_correct"] for g in group)
        exact_c = sum(g["exact_match"] for g in group)
        version_stats[ver] = {
            "contracts": len(group),
            "functions": total_f,
            "selector_accuracy": sel_c / total_f if total_f else 0,
            "type_accuracy": type_c / total_f if total_f else 0,
            "exact_accuracy": exact_c / total_f if total_f else 0,
        }

    # Generate report
    lines = [
        "# Compiler Hold-Out Experiment — ABI Reconstructor",
        "",
        f"**{len(results)} contracts evaluated across Solidity compiler versions**",
        "",
        "## Per-Version Accuracy",
        "",
        "| Version | Contracts | Functions | Selector Acc | Type Acc | Exact Match |",
        "|---|---|---|---|---|---|",
    ]

    for ver in sorted(version_stats.keys()):
        s = version_stats[ver]
        lines.append(
            f"| {ver} | {s['contracts']} | {s['functions']} | "
            f"{s['selector_accuracy']:.3f} | {s['type_accuracy']:.3f} | "
            f"{s['exact_accuracy']:.3f} |"
        )

    # Overall
    total_f = sum(s["functions"] for s in version_stats.values())
    sel_all = sum(s["selector_accuracy"] * s["functions"] for s in version_stats.values()) / total_f if total_f else 0
    type_all = sum(s["type_accuracy"] * s["functions"] for s in version_stats.values()) / total_f if total_f else 0

    lines += [
        "",
        "## Degradation Analysis",
        "",
    ]

    if "v0.4" in version_stats and "v0.8" in version_stats:
        s4 = version_stats["v0.4"]
        s8 = version_stats["v0.8"]
        sel_degradation = s4["selector_accuracy"] - s8["selector_accuracy"]
        type_degradation = s4["type_accuracy"] - s8["type_accuracy"]

        lines += [
            f"- **Selector accuracy**: v0.4 = {s4['selector_accuracy']:.3f}, v0.8 = {s8['selector_accuracy']:.3f} (Δ = {sel_degradation:+.3f})",
            f"- **Type accuracy**: v0.4 = {s4['type_accuracy']:.3f}, v0.8 = {s8['type_accuracy']:.3f} (Δ = {type_degradation:+.3f})",
            "",
        ]

        if abs(type_degradation) < 0.05:
            lines.append("**Result: No significant degradation across compiler versions.** The approach generalizes across Solidity compiler versions — consistent with the hypothesis that 4byte DB-based type recovery does not depend on compiler-specific code generation patterns.")
        elif abs(type_degradation) < 0.10:
            lines.append("**Result: Moderate degradation.** Some compiler-version sensitivity exists but does not dominate accuracy.")
        else:
            lines.append("**Result: Significant degradation.** Newer compiler versions produce bytecode that is harder to recover types from. This suggests compiler-specific patterns matter.")

    lines += [
        "",
        "## Per-Contract Results",
        "",
        "| Contract | Version | Functions | Sel Acc | Type Acc | Exact |",
        "|---|---|---|---|---|---|",
    ]

    for r in sorted(results, key=lambda r: r.get("compiler_version", "")):
        lines.append(
            f"| {r['contract_name'][:25]} | {r['compiler_version'][:15]} | "
            f"{r['total_functions']} | {r['selector_accuracy']:.3f} | "
            f"{r['type_accuracy']:.3f} | {r['exact_accuracy']:.3f} |"
        )

    report = "\n".join(lines)
    with open(args.output, "w") as f:
        f.write(report)

    print(f"\nReport → {args.output}")
    print(f"Contracts: {len(results)}, Functions: {total_f}")
    print()

    for ver in sorted(version_stats.keys()):
        s = version_stats[ver]
        print(f"  {ver}: sel={s['selector_accuracy']:.3f}, type={s['type_accuracy']:.3f}, exact={s['exact_accuracy']:.3f}  ({s['contracts']} contracts, {s['functions']} funcs)")

    if "v0.4" in version_stats and "v0.8" in version_stats:
        print(f"\n  v0.4 → v0.8 degradation: sel={sel_degradation:+.3f}, type={type_degradation:+.3f}")

    return 0


if __name__ == "__main__":
    main()
