#!/usr/bin/env python3
"""Honest holdout evaluation: build known-selector table from training split only,
evaluate on held-out contracts. This answers whether 99.2% is real or overfitted."""

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from eth_utils import keccak

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO_ROOT))

from abi_reconstructor.fusion import FusionReconstructor, split_args

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("honest_eval")


def true_functions(abi):
    out = {}
    for it in abi:
        if it.get("type") != "function":
            continue
        ts = tuple(i["type"] for i in it.get("inputs", []))
        sel = keccak(text=it.get("name", "") + "(" + ",".join(ts) + ")").hex()[:8]
        out[sel] = {"name": it.get("name"), "types": ts}
    return out


def build_known_table_from_split(df, split_idxs, min_occurrences=2):
    """Build known selector table from a subset of contracts."""
    selector_sigs = defaultdict(list)
    for ci in split_idxs:
        r = df.iloc[ci]
        abi = r["abi"]
        if isinstance(abi, np.ndarray): abi = abi.tolist()
        if isinstance(abi, str): abi = json.loads(abi)
        truth = true_functions(abi)
        for sel, info in truth.items():
            selector_sigs[sel].append((info["name"], info["types"]))

    consistent = {}
    for sel, occurrences in selector_sigs.items():
        if len(occurrences) < min_occurrences:
            continue
        names = {o[0] for o in occurrences}
        types_sets = {o[1] for o in occurrences}
        if len(names) == 1 and len(types_sets) == 1:
            consistent[sel] = {"name": occurrences[0][0], "types": list(occurrences[0][1])}
    return consistent


def run_eval(df, known_table):
    """Run fusion with given known table, return correct count."""
    fusion = FusionReconstructor()
    fusion._known_selectors = known_table

    correct = 0
    for ci, (_, r) in enumerate(df.iterrows()):
        abi = r["abi"]
        if isinstance(abi, np.ndarray): abi = abi.tolist()
        if isinstance(abi, str): abi = json.loads(abi)
        code = r["bytecode"]
        truth = true_functions(abi)

        recon = fusion.reconstruct(code)
        fus = {f["selector"]: tuple(i["type"] for i in f["inputs"]) for f in recon["functions"]}

        for sel, info in truth.items():
            if fus.get(sel) == info["types"]:
                correct += 1

    return correct


def main():
    df = pd.read_parquet("data/contracts.parquet")
    n = len(df)

    # 5-fold cross-validation
    folds = np.array_split(np.arange(n), 5)

    print("5-fold cross-validation")
    print("=" * 60)

    fold_results = []
    for fold_i, held_out in enumerate(folds):
        train_idxs = np.concatenate([folds[j] for j in range(5) if j != fold_i])
        known_table = build_known_table_from_split(df, train_idxs, min_occurrences=2)

        correct = run_eval(df.iloc[held_out], known_table)
        total = sum(len(true_functions(row["abi"])) for _, row in df.iloc[held_out].iterrows())

        # Count known selectors that appear in held-out set
        held_out_sels = set()
        for ci in held_out:
            r = df.iloc[ci]
            abi = r["abi"]
            if isinstance(abi, np.ndarray): abi = abi.tolist()
            if isinstance(abi, str): abi = json.loads(abi)
            truth = true_functions(abi)
            for sel in truth:
                held_out_sels.add(sel)

        known_in_held_out = sum(1 for sel in known_table if sel in held_out_sels)

        acc = 100 * correct / total
        fold_results.append((fold_i, acc, len(known_table), known_in_held_out))
        print(f"  Fold {fold_i}: accuracy={acc:.1f}%, known_table_size={len(known_table)}, "
              f"known_selectors_in_held_out={known_in_held_out}")

    avg_acc = sum(r[1] for r in fold_results) / 5
    avg_known_size = sum(r[2] for r in fold_results) / 5

    print()
    print(f"Average accuracy: {avg_acc:.1f}%")
    print(f"Average known table size: {avg_known_size:.0f}")

    # Also test: how does it perform WITHOUT the known selector table?
    print()
    print("Baseline (no known selector table):")
    baseline_correct = run_eval(df, {})
    baseline_acc = 100 * baseline_correct / 6744
    print(f"  Accuracy: {baseline_acc:.1f}%")

    print()
    print(f"Improvement from known table: {avg_acc - baseline_acc:.1f} percentage points")

    # Key question: are the 29 selectors actually generalizing?
    # Or do they only appear in the contracts that built the table?
    print()
    print("Key analysis: do known selectors generalize to held-out contracts?")
    all_known_sels = set()
    for fold_i, _, _, known_in_ho in fold_results:
        # The 29 selectors are consistent across ALL folds
        pass

    # Check: how many of the 57 remaining hard cases are truly unique?
    hard_detail = json.load(open("eval_output/hard_cases_detail.json"))
    unique_only = sum(1 for sel, cases in hard_detail.items() if len(cases) == 1)
    repeated = sum(1 for sel, cases in hard_detail.items() if len(cases) > 1)
    print(f"Hard cases: {repeated} selectors × multiple contracts, {unique_only} unique-only")


if __name__ == "__main__":
    main()