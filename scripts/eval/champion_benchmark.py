#!/usr/bin/env python3
"""Champion benchmark: regression gate against the current best reconstructor.

Usage:
 # Bootstrap: create baseline from current state
  python scripts/eval/champion_benchmark.py --init

  # Compare current branch against baseline
  python scripts/eval/champion_benchmark.py

Exit codes:
  0 current results meet or exceed baseline
  1  regression detected beyond tolerance
  2  no baseline exists (run --init first)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from abifusion.fusion import ABIFusion
from scripts.eval._shared import (
    accuracy,
    build_known_selector_table,
    extract_bytecode_metadata,
    resolve_data_path,
    split_dataframe,
    true_functions,
)
from scripts.eval.mine_failures import classify_failure

logger = logging.getLogger("champion_benchmark")

DEFAULT_BASELINE = REPO_ROOT / "eval_output" / "champion_baseline.json"
DEFAULT_FAILURES = REPO_ROOT / "eval_output" / "failures.json"
DEFAULT_REPORT = REPO_ROOT / "eval_output" / "champion_benchmark.md"
REGRESSION_TOLERANCE = 0.5


def evaluate_contracts(
    df: pd.DataFrame, known_table: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    fusion = ABIFusion()
    if known_table is not None:
        fusion._known_selectors = known_table

    results: Dict[str, Any] = {
        "total": 0,
        "correct": 0,
        "by_source": defaultdict(lambda: {"correct": 0, "total": 0}),
        "by_confidence": defaultdict(lambda: {"correct": 0, "total": 0}),
        "by_category": defaultdict(lambda: {"correct": 0, "total": 0}),
        "selector_coverage": {"found": 0, "total": 0},
        "failures": [],
    }

    for index, (_, row) in enumerate(df.iterrows(), start=1):
        if index == 1 or index % 50 == 0:
            logger.info("evaluating contract %d/%d", index, len(df))

        bytecode = row["bytecode"]
        truth = true_functions(row["abi"])
        reconstructed = fusion.reconstruct(bytecode)
        predicted = {
            fn["selector"]: tuple(inp["type"] for inp in fn["inputs"])
            for fn in reconstructed["functions"]
        }

        bytecode_selectors, evmole_types_map, sig_db_candidates_map = extract_bytecode_metadata(
            bytecode, fusion, truth
        )

        for selector, info in truth.items():
            results["total"] += 1
            results["selector_coverage"]["total"] += 1

            is_correct = predicted.get(selector) == info["types"]
            if is_correct:
                results["correct"] += 1
                results["selector_coverage"]["found"] += 1

            reconstructed_func = next(
                (fn for fn in reconstructed["functions"] if fn["selector"] == selector),
                {},
            )
            source = reconstructed_func.get("source", "unknown")
            confidence = reconstructed_func.get("confidence", "unknown")

            results["by_source"][source]["total"] += 1
            if is_correct:
                results["by_source"][source]["correct"] += 1

            results["by_confidence"][confidence]["total"] += 1
            if is_correct:
                results["by_confidence"][confidence]["correct"] += 1

            evmole_t = evmole_types_map.get(selector)
            sig_candidates = sig_db_candidates_map.get(selector, [])
            known_hit = selector in fusion._known_selectors

            failure_cat, _ = classify_failure(
                selector,
                info["types"],
                predicted.get(selector) or (),
                reconstructed_func,
                bytecode_selectors,
                sig_candidates,
                evmole_t,
                known_hit,
            )

            results["by_category"][failure_cat]["total"] += 1
            if is_correct:
                results["by_category"][failure_cat]["correct"] += 1
            else:
                results["failures"].append({
                    "selector": selector,
                    "ground_truth_types": list(info["types"]),
                    "predicted_types": list(predicted.get(selector, [])),
                    "source": source,
                    "confidence": confidence,
                    "failure_category": failure_cat,
                })

    results["by_source"] = dict(results["by_source"])
    results["by_confidence"] = dict(results["by_confidence"])
    results["by_category"] = dict(results["by_category"])
    return results


def run_benchmark(
    df: pd.DataFrame,
    known_table: Dict[str, Dict[str, Any]],
    seed: int,
    heldout_fraction: float,
) -> Dict[str, Any]:
    results = evaluate_contracts(df, known_table)

    total = results["total"]
    correct = results["correct"]

    benchmark = {
        "date": date.today().isoformat(),
        "seed": seed,
        "heldout_fraction": heldout_fraction,
        "total_functions": total,
        "correct": correct,
        "accuracy": accuracy(correct, total),
        "selector_coverage": results["selector_coverage"],
        "by_source": {},
        "by_confidence": {},
        "by_category": {},
        "failures": results["failures"],
    }

    for source, vals in results["by_source"].items():
        benchmark["by_source"][source] = {
            "correct": vals["correct"],
            "total": vals["total"],
            "accuracy": accuracy(vals["correct"], vals["total"]),
        }

    for conf, vals in results["by_confidence"].items():
        benchmark["by_confidence"][conf] = {
            "correct": vals["correct"],
            "total": vals["total"],
            "accuracy": accuracy(vals["correct"], vals["total"]),
        }

    for cat, vals in results["by_category"].items():
        benchmark["by_category"][cat] = {
            "correct": vals["correct"],
            "total": vals["total"],
            "accuracy": accuracy(vals["correct"], vals["total"]),
        }

    return benchmark


def compare(baseline: Dict[str, Any], current: Dict[str, Any]) -> Tuple[bool, List[str]]:
    messages: List[str] = []
    regressions = False

    base_acc = baseline.get("accuracy", 0.0)
    curr_acc = current.get("accuracy", 0.0)
    diff = base_acc - curr_acc

    messages.append(
        f"Accuracy: {curr_acc:.1f}% (baseline {base_acc:.1f}%, diff {diff:+.1f}pp)"
    )

    if diff > REGRESSION_TOLERANCE:
        messages.append(f"REGRESSION: accuracy dropped by {diff:.1f}pp (tolerance {REGRESSION_TOLERANCE}pp)")
        regressions = True

    for source in baseline.get("by_source", {}):
        base_src = baseline["by_source"].get(source, {"total": 0, "correct": 0})
        curr_src = current["by_source"].get(source, {"total": 0, "correct": 0})
        base_acc_s = accuracy(base_src["correct"], base_src["total"])
        curr_acc_s = accuracy(curr_src["correct"], curr_src["total"])
        diff_s = base_acc_s - curr_acc_s
        if diff_s > REGRESSION_TOLERANCE:
            messages.append(
                f"REGRESSION [{source}]: {curr_acc_s:.1f}% vs baseline {base_acc_s:.1f}% ({diff_s:+.1f}pp)"
            )
            regressions = True

    for conf in baseline.get("by_confidence", {}):
        base_c = baseline["by_confidence"].get(conf, {"total": 0, "correct": 0})
        curr_c = current["by_confidence"].get(conf, {"total": 0, "correct": 0})
        base_acc_c = accuracy(base_c["correct"], base_c["total"])
        curr_acc_c = accuracy(curr_c["correct"], curr_c["total"])
        diff_c = base_acc_c - curr_acc_c
        if diff_c > REGRESSION_TOLERANCE:
            messages.append(
                f"REGRESSION [{conf} confidence]: {curr_acc_c:.1f}% vs baseline {base_acc_c:.1f}% ({diff_c:+.1f}pp)"
            )
            regressions = True

    for cat in baseline.get("by_category", {}):
        base_cat = baseline["by_category"].get(cat, {"total": 0, "correct": 0})
        curr_cat = current["by_category"].get(cat, {"total": 0, "correct": 0})
        base_acc_cat = accuracy(base_cat["correct"], base_cat["total"])
        curr_acc_cat = accuracy(curr_cat["correct"], curr_cat["total"])
        diff_cat = base_acc_cat - curr_acc_cat
        if diff_cat > REGRESSION_TOLERANCE:
            messages.append(
                f"REGRESSION [{cat}]: {curr_acc_cat:.1f}% vs baseline {base_acc_cat:.1f}% ({diff_cat:+.1f}pp)"
            )
            regressions = True

    base_high = baseline.get("by_confidence", {}).get("high", {})
    high_conf = current.get("by_confidence", {}).get("high", {})
    if high_conf and high_conf.get("total", 0) > 0:
        high_conf_acc = accuracy(high_conf["correct"], high_conf["total"])
        base_high_acc = accuracy(base_high.get("correct", 0), base_high.get("total", 0))
        diff_hc = base_high_acc - high_conf_acc
        if diff_hc > REGRESSION_TOLERANCE:
            messages.append(
                f"REGRESSION [high-conf]: {high_conf_acc:.1f}% vs baseline {base_high_acc:.1f}% ({diff_hc:+.1f}pp)"
            )
            regressions = True

    return regressions, messages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/contracts.parquet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="Create baseline from current state (overwrites --baseline)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=REGRESSION_TOLERANCE,
        help=f"Regression tolerance in pp (default: {REGRESSION_TOLERANCE})",
    )
    parser.add_argument(
        "--external",
        action="store_true",
        help="Run on external eval set. Report-only mode: compares against the "
        "champion baseline but never fails the build. Uses shipped runtime "
        "tables only; no table building from external data.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    data_path = resolve_data_path(args.data, REPO_ROOT)

    df = pd.read_parquet(data_path)

    if args.external:
        # External mode: no split, use shipped runtime tables only
        from abifusion.fusion import ABIFusion
        ABIFusion._known_selectors = None
        ABIFusion._known_interfaces = None
        eval_df = df
        logger.info("external eval contracts: %d", len(eval_df))
        current = run_benchmark(eval_df, None, args.seed, args.heldout_fraction)
    else:
        train_df, heldout_df = split_dataframe(df, args.heldout_fraction, args.seed)
        logger.info("training contracts: %d", len(train_df))
        logger.info("held-out contracts: %d", len(heldout_df))

        known_table = build_known_selector_table(train_df)
        logger.info("known selector table entries: %d", len(known_table))

        current = run_benchmark(heldout_df, known_table, args.seed, args.heldout_fraction)

    if args.init:
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        with open(args.baseline, "w") as f:
            json.dump(current, f, indent=2)
        logger.info("wrote baseline to %s", args.baseline)
        print(f"Baseline created: {args.baseline}")
        print(f"Accuracy: {current['accuracy']:.1f}%")
        return 0

    if not args.baseline.exists():
        logger.error("no baseline found at %s (run --init first)", args.baseline)
        print(f"ERROR: no baseline found at {args.baseline}")
        print("Run: python scripts/eval/champion_benchmark.py --init")
        return 2

    with open(args.baseline) as f:
        baseline = json.load(f)

    if args.external:
        # External mode: report-only, never fails
        regressions, messages = compare(baseline, current)

        report_path = REPO_ROOT / "eval_output" / "external_generalization.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        heldout_acc = baseline.get("accuracy", 0.0)
        external_acc = current.get("accuracy", 0.0)
        gap = heldout_acc - external_acc

        all_categories = set(baseline.get("by_category", {}).keys()) | set(
            current.get("by_category", {}).keys()
        )

        report_lines = [
            "# External Generalization Report",
            "",
            f"**Date:** {date.today().isoformat()}",
            f"**Baseline:** {args.baseline} (held-out: {heldout_acc:.1f}%)",
            "",
            "## Accuracy",
            "",
            "| Metric | Held-out (canonical) | External |",
            "|---|---|---|",
            f"| Accuracy | {heldout_acc:.1f}% | {external_acc:.1f}% |",
            f"| Generalization gap | — | {gap:+.1f}pp |",
            "",
            "## By Failure Category",
            "",
            "| Category | Held-out Acc | External Acc | Diff |",
            "|---|---|---|---|",
        ]

        for cat in sorted(all_categories):
            base_cat = baseline.get("by_category", {}).get(cat, {"total": 0, "correct": 0})
            curr_cat = current.get("by_category", {}).get(cat, {"total": 0, "correct": 0})
            base_acc = accuracy(base_cat["correct"], base_cat["total"])
            curr_acc = accuracy(curr_cat["correct"], curr_cat["total"])
            diff = base_acc - curr_acc
            report_lines.append(
                f"| {cat} | {base_acc:.1f}% ({base_cat['total']}) | {curr_acc:.1f}% ({curr_cat['total']}) | {diff:+.1f}pp |"
            )

        report_lines.extend(["", "## Messages", ""])
        for msg in messages:
            report_lines.append(f"- {msg}")

        report_lines.extend(["", "## Decision", ""])
        report_lines.append("")
        report_lines.append("[See plan for decision criteria]")

        report_path.write_text("\n".join(report_lines) + "\n")
        logger.info("wrote report to %s", report_path)

        print("\n" + "=" * 60)
        print("EXTERNAL GENERALIZATION REPORT")
        print("=" * 60)
        print(f"Date: {date.today().isoformat()}")
        print(f"Baseline: {args.baseline} ({heldout_acc:.1f}%)")
        print(f"External accuracy: {external_acc:.1f}%")
        print(f"Generalization gap: {gap:+.1f}pp")
        print()
        for msg in messages:
            print(f"  {msg}")
        print()
        print("RESULT: EXTERNAL REPORT (not a regression gate)")
        print("=" * 60)
        return 0

    regressions, messages = compare(baseline, current)

    report_path = DEFAULT_REPORT
    report_path.parent.mkdir(parents=True, exist_ok=True)

    all_categories = set(baseline.get("by_category", {}).keys()) | set(
        current.get("by_category", {}).keys()
    )

    report_lines = [
        "# Champion Benchmark Report",
        "",
        f"**Date:** {date.today().isoformat()}",
        f"**Baseline:** {args.baseline}",
        "",
        "## Overall",
        "",
        f"| Metric | Baseline | Current | Diff |",
        f"|--------|----------|---------|------|",
        f"| Accuracy | {baseline.get('accuracy', 0.0):.1f}% | {current.get('accuracy', 0.0):.1f}% | {baseline.get('accuracy', 0.0) - current.get('accuracy', 0.0):+.1f}pp |",
        f"| Total functions | {baseline.get('total_functions', 0)} | {current.get('total_functions', 0)} | |",
        "",
        "## By Failure Category",
        "",
        f"| Category | Baseline Acc | Current Acc | Diff |",
        f"|----------|--------------|-------------|------|",
    ]

    for cat in sorted(all_categories):
        base_cat = baseline.get("by_category", {}).get(cat, {"total": 0, "correct": 0})
        curr_cat = current.get("by_category", {}).get(cat, {"total": 0, "correct": 0})
        base_acc = accuracy(base_cat["correct"], base_cat["total"])
        curr_acc = accuracy(curr_cat["correct"], curr_cat["total"])
        diff = base_acc - curr_acc
        report_lines.append(
            f"| {cat} | {base_acc:.1f}% ({base_cat['total']}) | {curr_acc:.1f}% ({curr_cat['total']}) | {diff:+.1f}pp |"
        )

    report_lines.extend(["", "## Messages", ""])
    for msg in messages:
        report_lines.append(f"- {msg}")

    report_path.write_text("\n".join(report_lines) + "\n")
    logger.info("wrote report to %s", report_path)

    print("\n" + "=" * 60)
    print("CHAMPION BENCHMARK REPORT")
    print("=" * 60)
    print(f"Date: {date.today().isoformat()}")
    print(f"Baseline: {args.baseline}")
    print()
    for msg in messages:
        print(f"  {msg}")
    print()

    if regressions:
        print("RESULT: REGRESSION DETECTED")
        print("=" * 60)
        return 1
    else:
        print("RESULT: NO REGRESSION (within tolerance)")
        print("=" * 60)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
