#!/usr/bin/env python3
"""Tier recall evaluation: measure per-tier accuracy and breakdown of failures.

This script runs reconstruction on the evaluation dataset and reports:
1. Per-tier recall (what fraction of selectors each tier resolves)
2. Tier 1 failure breakdown: no 4byte entry vs collision mis-pick
3. Phase 2 ROI validation: is family-guided disambiguation worth implementing?

Usage:
    PYTHONPATH=/home/user/code/abifusion python scripts/eval/tier_recall_eval.py
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from eth_utils import keccak

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO_ROOT))

from abifusion.fusion import ABIFusion

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("tier_recall")


def true_functions(abi):
    """Extract ground-truth functions from ABI."""
    out = {}
    for it in abi:
        if it.get("type") != "function":
            continue
        ts = tuple(i["type"] for i in it.get("inputs", []))
        sel = keccak(text=it.get("name", "") + "(" + ",".join(ts) + ")").hex()[:8]
        out[sel] = {"name": it.get("name"), "types": ts}
    return out


def get_tier(source: str) -> int:
    """Map source field to tier number."""
    return {
        "signature+evmole": 1,
        "signature": 1,
        "known-selector-table": 2,
        "ml": 3,
    }.get(source, 4)


def get_4byte_candidates(selector: str, sig_lookup) -> list:
    """Get 4byte candidates for a selector."""
    rows = sig_lookup.lookup_openchain(selector) or sig_lookup.lookup_4byte(selector)
    candidates = []
    for s in rows:
        text_sig = s.get("text_signature", "")
        i = text_sig.find("(")
        if i == -1:
            continue
        name = text_sig[:i]
        types_str = text_sig[i+1:text_sig.rfind(")")]
        types = tuple(t.strip() for t in types_str.split(",") if t.strip())
        candidates.append((name, types))
    return candidates


def run_tier_eval():
    """Run tier recall evaluation on full dataset."""
    df = pd.read_parquet("data/contracts.parquet")

    fusion = ABIFusion()
    sig_lookup = fusion.sig

    tier_counts = Counter()
    tier_correct = Counter()
    tier_total = Counter()

    tier1_no_4byte = 0
    tier1_single_candidate = 0
    tier1_collision = 0
    tier1_picked_correct = 0
    tier1_picked_wrong = 0

    total_functions = 0
    total_correct = 0

    for ci, (_, row) in enumerate(df.iterrows()):
        abi = row["abi"]
        if isinstance(abi, np.ndarray): abi = abi.tolist()
        if isinstance(abi, str): abi = json.loads(abi)
        code = row["bytecode"]
        truth = true_functions(abi)

        recon = fusion.reconstruct(code)
        fus = {f["selector"]: (tuple(i["type"] for i in f["inputs"]), f["source"]) for f in recon["functions"]}

        for sel, info in truth.items():
            true_types = info["types"]
            total_functions += 1

            if sel in fus:
                pred_types, source = fus[sel]
                tier = get_tier(source)
                tier_counts[tier] += 1
                tier_total[tier] += 1

                if pred_types == true_types:
                    tier_correct[tier] += 1
                    total_correct += 1
                    if tier == 1:
                        tier1_picked_correct += 1
                else:
                    if tier == 1:
                        tier1_picked_wrong += 1
            else:
                tier_total[4] += 1
                tier_counts[4] += 1

            # 4byte analysis for this selector
            candidates = get_4byte_candidates(sel, sig_lookup)
            if not candidates:
                tier1_no_4byte += 1
            elif len(candidates) == 1:
                tier1_single_candidate += 1
            else:
                tier1_collision += 1

    return tier_counts, tier_correct, tier_total, {
        "no_4byte": tier1_no_4byte,
        "single_candidate": tier1_single_candidate,
        "collision": tier1_collision,
        "picked_correct": tier1_picked_correct,
        "picked_wrong": tier1_picked_wrong,
        "total": total_functions,
        "total_correct": total_correct,
    }


def main():
    logger.info("Running tier recall evaluation...")

    tier_counts, tier_correct, tier_total, data = run_tier_eval()

    print()
    print("=" * 60)
    print("TIER RECALL BREAKDOWN")
    print("=" * 60)

    total_functions = data["total"]
    total_correct = data["total_correct"]
    overall_acc = 100 * total_correct / total_functions if total_functions > 0 else 0

    print()
    print("Per-tier resolution:")
    for tier in sorted(tier_counts.keys()):
        total = tier_total[tier]
        correct = tier_correct.get(tier, 0)
        resolved = tier_counts[tier]
        acc = 100 * correct / total if total > 0 else 0
        pct = 100 * resolved / total_functions
        print(f"  Tier {tier}: {resolved}/{total_functions} resolved ({pct:.1f}%), accuracy={acc:.1f}%")

    print()
    print(f"Overall: {total_correct}/{total_functions} correct = {overall_acc:.1f}%")

    no_4byte = data["no_4byte"]
    single = data["single_candidate"]
    collision = data["collision"]
    picked_correct = data["picked_correct"]
    picked_wrong = data["picked_wrong"]

    print()
    print("Tier 1 (4byte) failure breakdown:")
    print(f"  Total selectors: {total_functions}")
    print(f"  No 4byte entry: {no_4byte} ({100*no_4byte/total_functions:.1f}%)")
    print(f"  Single candidate (no collision): {single} ({100*single/total_functions:.1f}%)")
    print(f"  Multiple candidates (collision): {collision} ({100*collision/total_functions:.1f}%)")

    tier1_resolved = picked_correct + picked_wrong
    if tier1_resolved > 0:
        print()
        print(f"  Among resolved by Tier 1 ({tier1_resolved}):")
        print(f"    Picked correct: {picked_correct} ({100*picked_correct/tier1_resolved:.1f}% of resolved)")
        print(f"    Picked wrong: {picked_wrong} ({100*picked_wrong/tier1_resolved:.1f}% of resolved)")

    addressable_by_ml = picked_wrong
    print()
    print("Phase 2 ROI assessment:")
    print(f"  Collision mis-picks (addressable by ML disambiguation): {addressable_by_ml}")
    print(f"  As % of all selectors: {100*addressable_by_ml/total_functions:.2f}%")
    if tier1_resolved > 0:
        print(f"  As % of Tier 1 resolved: {100*addressable_by_ml/tier1_resolved:.1f}%")

    if addressable_by_ml > 50:
        print(f"  -> Phase 2.1 (family-guided disambiguation) is HIGH VALUE")
    elif addressable_by_ml > 10:
        print(f"  -> Phase 2.1 (family-guided disambiguation) is MODERATE VALUE")
    else:
        print(f"  -> Phase 2.1 (family-guided disambiguation) is LOW VALUE — consider skipping to Phase 3")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
