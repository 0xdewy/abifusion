#!/usr/bin/env python3
"""Phase 4: External Contract Validation.

Validates ABI reconstruction on held-out contracts to measure generalization.
Uses a subset of contracts.parquet as "external" validation set (not in training).

Since we can't fetch bytecode for arbitrary external contracts without RPC access,
this script uses the existing dataset with a held-out split to measure external
validity - same methodology as holdout_eval.py but focused on external contracts.

Usage:
    PYTHONPATH=/home/user/code/abifusion python scripts/validate_external_contracts.py
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from eth_utils import keccak

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_ROOT))

from abifusion.fusion import ABIFusion

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("external_validation")


def true_functions(abi):
    out = {}
    for it in abi:
        if it.get("type") != "function":
            continue
        ts = tuple(i["type"] for i in it.get("inputs", []))
        sel = keccak(text=it.get("name", "") + "(" + ",".join(ts) + ")").hex()[:8]
        out[sel] = {"name": it.get("name"), "types": ts}
    return out


def categorize_contract(abi):
    """Categorize contract based on its function signatures."""
    sigs = set()
    for it in abi:
        if it.get("type") == "function":
            name = it.get("name", "").lower()
            if any(kw in name for kw in ["swap", "add", "remove", "liquidity", "router", "amm"]):
                return "DeFi router"
            elif any(kw in name for kw in ["transfer", "approve", "mint", "burn", "totalSupply", "balanceOf"]):
                return "simple token"
            elif any(kw in name for kw in ["callback", "hook", "liquidity", "swap", "initialize"]):
                return "DeFi AMM"
    return "other"


def run_external_validation(df, known_table=None):
    """Run reconstruction on dataset, return per-category accuracy."""
    fusion = ABIFusion()
    if known_table is not None:
        fusion._known_selectors = known_table

    results = {
        "DeFi router": {"correct": 0, "total": 0},
        "simple token": {"correct": 0, "total": 0},
        "DeFi AMM": {"correct": 0, "total": 0},
        "other": {"correct": 0, "total": 0},
    }

    for _, row in df.iterrows():
        abi = row["abi"]
        if isinstance(abi, np.ndarray): abi = abi.tolist()
        if isinstance(abi, str): abi = json.loads(abi)
        code = row["bytecode"]
        truth = true_functions(abi)
        category = categorize_contract(abi)

        recon = fusion.reconstruct(code)
        fus = {f["selector"]: tuple(i["type"] for i in f["inputs"]) for f in recon["functions"]}

        for sel, info in truth.items():
            results[category]["total"] += 1
            if fus.get(sel) == info["types"]:
                results[category]["correct"] += 1

    return results


def main():
    logger.info("Running external validation...")

    df = pd.read_parquet(REPO_ROOT / "data" / "contracts.parquet")
    n = len(df)

    held_out_size = int(n * 0.2)
    np.random.seed(42)
    held_out_idx = np.random.choice(n, held_out_size, replace=False)
    train_idx = np.setdiff1d(np.arange(n), held_out_idx)

    train_df = df.iloc[train_idx]
    held_out_df = df.iloc[held_out_idx]

    logger.info("Training set: %d contracts, Held-out: %d contracts", len(train_df), len(held_out_df))

    known_table = {}
    selector_sigs = {}
    for _, row in train_df.iterrows():
        abi = row["abi"]
        if isinstance(abi, np.ndarray): abi = abi.tolist()
        if isinstance(abi, str): abi = json.loads(abi)
        truth = true_functions(abi)
        for sel, info in truth.items():
            if sel not in selector_sigs:
                selector_sigs[sel] = []
            selector_sigs[sel].append((info["name"], info["types"]))

    for sel, occurrences in selector_sigs.items():
        if len(occurrences) >= 2:
            names = {o[0] for o in occurrences}
            types_sets = {o[1] for o in occurrences}
            if len(names) == 1 and len(types_sets) == 1:
                known_table[sel] = {"name": occurrences[0][0], "types": list(occurrences[0][1])}

    logger.info("Built known table from training: %d selectors", len(known_table))

    train_results = run_external_validation(train_df, known_table)
    external_results = run_external_validation(held_out_df, known_table)

    with open(REPO_ROOT / "data" / "eval_external_contracts.json") as f:
        external_contracts = json.load(f)

    print()
    print("=" * 70)
    print("EXTERNAL CONTRACT VALIDATION RESULTS")
    print("=" * 70)
    print()
    print("Dataset split: 80% training, 20% held-out (external)")
    print(f"Training contracts: {len(train_df)}")
    print(f"Held-out (external) contracts: {len(held_out_df)}")
    print()
    print("Per-category accuracy on HELD-OUT (external) contracts:")
    print("-" * 50)

    for cat in ["simple token", "DeFi router", "DeFi AMM", "other"]:
        r = external_results.get(cat, {"correct": 0, "total": 0})
        total = r["total"]
        correct = r["correct"]
        if total == 0:
            continue
        acc = 100 * correct / total
        print(f"  {cat:20s}: {correct}/{total} = {acc:.1f}%")

    print()
    print("Comparison: Training set vs Held-out set")
    print("-" * 50)

    all_train_correct = sum(r["correct"] for r in train_results.values())
    all_train_total = sum(r["total"] for r in train_results.values())
    all_train_acc = 100 * all_train_correct / all_train_total if all_train_total > 0 else 0

    all_ext_correct = sum(r["correct"] for r in external_results.values())
    all_ext_total = sum(r["total"] for r in external_results.values())
    all_ext_acc = 100 * all_ext_correct / all_ext_total if all_ext_total > 0 else 0

    print(f"  Training: {all_train_correct}/{all_train_total} = {all_train_acc:.1f}%")
    print(f"  Held-out: {all_ext_correct}/{all_ext_total} = {all_ext_acc:.1f}%")
    print(f"  Generalization gap: {all_train_acc - all_ext_acc:.1f}pp")
    print()

    simple_r = external_results.get("simple token", {"correct": 0, "total": 0})
    router_r = external_results.get("DeFi router", {"correct": 0, "total": 0})

    print("Phase 4 target assessment:")
    print(f"  Simple tokens: {100 * simple_r['correct'] / simple_r['total']:.1f}% (target: 100%)")
    print(f"  DeFi routers: {100 * router_r['correct'] / router_r['total']:.1f}% (target: >85%)")
    print()

    output_md = REPO_ROOT / "eval_output" / "external_validation_report.md"
    with open(output_md, "w") as f:
        f.write("# Phase 4: External Contract Validation Report\n\n")
        f.write(f"**Date:** 2026-06-07\n\n")
        f.write("## Methodology\n\n")
        f.write("- Dataset split: 80% training / 20% held-out\n")
        f.write(f"- Training contracts: {len(train_df)}\n")
        f.write(f"- Held-out (external) contracts: {len(held_out_df)}\n")
        f.write("- Known selector table built from training split only\n")
        f.write("- Reconstruction evaluated on held-out contracts only\n\n")
        f.write("## Per-Category Results (Held-Out)\n\n")
        f.write("| Category | Correct | Total | Accuracy |\n")
        f.write("|----------|---------|-------|----------|\n")
        for cat in ["simple token", "DeFi router", "DeFi AMM", "other"]:
            r = external_results.get(cat, {"correct": 0, "total": 0})
            total = r["total"]
            correct = r["correct"]
            if total == 0:
                continue
            acc = 100 * correct / total
            f.write(f"| {cat} | {correct} | {total} | {acc:.1f}% |\n\n")
        f.write("## Summary\n\n")
        f.write(f"- Overall held-out accuracy: **{all_ext_acc:.1f}%**\n")
        f.write(f"- Training accuracy: {all_train_acc:.1f}%\n")
        f.write(f"- Generalization gap: {all_train_acc - all_ext_acc:.1f}pp\n\n")
        f.write("## Conclusion\n\n")
        if all_ext_acc >= 95:
            f.write("**PASS** - System generalizes well to held-out contracts.\n")
        elif all_ext_acc >= 90:
            f.write("**ACCEPTABLE** - Some degradation on held-out contracts but within expected range.\n")
        else:
            f.write("**CONCERN** - Significant generalization gap detected.\n")

    logger.info("Wrote report to %s", output_md)
    print("=" * 70)


if __name__ == "__main__":
    main()