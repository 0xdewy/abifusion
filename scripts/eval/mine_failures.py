#!/usr/bin/env python3
"""Mine current evaluation failures from the held-out split.

Runs on the same deterministic split (seed=42, 20% held-out) as run_evaluation.py,
compares fusion reconstructor against ground truth, and emits:
  - eval_output/failures.json   one structured entry per failed function
  - eval_output/failure_summary.md  counts by failure category and addressability
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from abi_reconstructor.fusion import FusionReconstructor
from scripts.eval._shared import (
    accuracy,
    build_known_selector_table,
    extract_bytecode_metadata,
    normalize_abi,
    resolve_data_path,
    split_dataframe,
    true_functions,
)

logger = logging.getLogger("mine_failures")


FAILURE_CATEGORIES = [
    "selector_missing_from_bytecode",
    "signature_db_miss",
    "evmole_no_signal",
    "evmole_arity_miss",
    "evmole_type_miss",
    "signature_collision_mispick",
    "known_selector_table_miss",
    "tuple_or_dynamic_type_mismatch",
    "output_not_recovered",
    "other",
]

ADDRESSABILITY_CLASSES = [
    "not_addressable_from_bytecode",
    "deterministically_addressable",
    "model_or_scoring_addressable",
]


def classify_failure(
    selector: str,
    truth_types: Tuple[str, ...],
    pred_types: Tuple[str, ...],
    reconstructed_func: Dict[str, Any],
    bytecode_selectors: set,
    sig_db_candidates: List[Tuple[str, Tuple[str, ...]]],
    evmole_types: Optional[Tuple[str, ...]],
    known_table_hit: bool,
) -> Tuple[str, str]:
    """Classify a failed function into a failure category and addressability class."""

    if selector not in bytecode_selectors:
        if known_table_hit and pred_types != truth_types:
            return "known_selector_table_miss", "deterministically_addressable"
        if not sig_db_candidates and evmole_types is None:
            return "selector_missing_from_bytecode", "not_addressable_from_bytecode"
        return "selector_missing_from_bytecode", "model_or_scoring_addressable"

    if not sig_db_candidates:
        if evmole_types is None:
            return "evmole_no_signal", "not_addressable_from_bytecode"
        return "evmole_no_signal", "model_or_scoring_addressable"

    if len(sig_db_candidates) > 1:
        return "signature_collision_mispick", "model_or_scoring_addressable"

    if evmole_types is None:
        return "signature_db_miss", "model_or_scoring_addressable"

    if len(evmole_types) != len(truth_types):
        return "evmole_arity_miss", "model_or_scoring_addressable"

    if evmole_types != pred_types:
        return "evmole_type_miss", "model_or_scoring_addressable"

    if any(t in ("tuple", "bytes", "string") or t.endswith("[]") for t in truth_types):
        return "tuple_or_dynamic_type_mismatch", "model_or_scoring_addressable"

    if known_table_hit and pred_types != truth_types:
        return "known_selector_table_miss", "deterministically_addressable"

    return "output_not_recovered", "model_or_scoring_addressable"


def mine_failures(
    df: pd.DataFrame, known_table: Dict[str, Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run failure mining on a dataframe and return failure entries + summary stats."""

    fusion = FusionReconstructor()
    if known_table is not None:
        fusion._known_selectors = known_table

    failures: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        "total_functions": 0,
        "correct": 0,
        "failed": 0,
        "by_category": {c: 0 for c in FAILURE_CATEGORIES},
        "by_addressability": {a: 0 for a in ADDRESSABILITY_CLASSES},
        "by_source": defaultdict(int),
        "by_confidence": defaultdict(int),
    }

    for index, (_, row) in enumerate(df.iterrows(), start=1):
        if index == 1 or index % 50 == 0:
            logger.info("mining contract %d/%d", index, len(df))

        bytecode = row["bytecode"]
        truth = true_functions(row["abi"])
        reconstructed = fusion.reconstruct(bytecode)

        predicted: Dict[str, Tuple[str, ...]] = {
            fn["selector"]: tuple(inp["type"] for inp in fn["inputs"])
            for fn in reconstructed["functions"]
        }

        bytecode_selectors, evmole_types_map, sig_db_candidates_map = extract_bytecode_metadata(
            bytecode, fusion, truth
        )

        for selector, info in truth.items():
            summary["total_functions"] += 1
            pred_types = predicted.get(selector)
            truth_types = info["types"]

            if pred_types == truth_types:
                summary["correct"] += 1
                continue

            summary["failed"] += 1

            reconstructed_func = next(
                (fn for fn in reconstructed["functions"] if fn["selector"] == selector),
                {},
            )

            evmole_t = evmole_types_map.get(selector)
            sig_candidates = sig_db_candidates_map.get(selector, [])
            known_hit = selector in fusion._known_selectors if fusion._known_selectors else False

            failure_cat, addressability = classify_failure(
                selector,
                truth_types,
                pred_types or (),
                reconstructed_func,
                bytecode_selectors,
                sig_candidates,
                evmole_t,
                known_hit,
            )

            summary["by_category"][failure_cat] += 1
            summary["by_addressability"][addressability] += 1

            source = reconstructed_func.get("source", "unknown")
            confidence = reconstructed_func.get("confidence", "unknown")
            summary["by_source"][source] += 1
            summary["by_confidence"][confidence] += 1

            failure_entry = {
                "contract_address": str(row.get("address", "")),
                "contract_name": str(row.get("name", "")),
                "compiler_version": str(row.get("compiler_version", "")),
                "selector": selector,
                "selector_in_bytecode": selector in bytecode_selectors,
                "ground_truth": {
                    "name": info["name"],
                    "types": list(truth_types),
                },
                "predicted": {
                    "name": reconstructed_func.get("name", ""),
                    "types": list(pred_types) if pred_types else [],
                },
                "source": source,
                "confidence": confidence,
                "candidates": reconstructed_func.get("candidates"),
                "evmole_recovered_types": list(evmole_t) if evmole_t is not None else None,
                "sig_db_candidates": [
                    {"name": c[0], "types": list(c[1])} for c in sig_candidates
                ],
                "known_selector_table_hit": known_hit,
                "failure_category": failure_cat,
                "addressability_class": addressability,
            }
            failures.append(failure_entry)

    summary["by_source"] = dict(summary["by_source"])
    summary["by_confidence"] = dict(summary["by_confidence"])
    return failures, summary


def write_failures_json(failures: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(failures, f, indent=2)
    logger.info("wrote %s (%d entries)", out_path, len(failures))


def write_failure_summary(
    summary: Dict[str, Any],
    failures: List[Dict[str, Any]],
    seed: int,
    heldout_fraction: float,
    out_path: Path,
) -> None:
    total = summary["total_functions"]
    correct = summary["correct"]
    failed = summary["failed"]
    acc = accuracy(correct, total)

    lines = [
        "# ABI Reconstructor Failure Summary",
        "",
        f"**Date:** {date.today().isoformat()}",
        "",
        "## Overview",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total functions | {total} |",
        f"| Correct | {correct} |",
        f"| Failed | {failed} |",
        f"| Accuracy | {acc:.1f}% |",
        f"| Split seed | {seed} |",
        f"| Held-out fraction | {heldout_fraction} |",
        "",
        "## Failure Categories",
        "",
 f"| Category | Count | % |",
        f"|----------|-------|-------|",
    ]

    for cat in FAILURE_CATEGORIES:
        count = summary["by_category"].get(cat, 0)
        if count == 0:
            continue
        pct = 100.0 * count / total if total else 0.0
        lines.append(f"| {cat} | {count} | {pct:.1f}% |")

    lines.extend(["", "## Addressability", "", f"| Class | Count | % |", f"|----------|-------|-------|"])

    for addr_class in ADDRESSABILITY_CLASSES:
        count = summary["by_addressability"].get(addr_class, 0)
        if count == 0:
            continue
        pct = 100.0 * count / total if total else 0.0
        lines.append(f"| {addr_class} | {count} | {pct:.1f}% |")

    lines.extend(["", "## Source Tier Breakdown", "", f"| Source | Count | % |", f"|----------|-------|-------|"])

    for source, count in sorted(summary["by_source"].items(), key=lambda x: -x[1]):
        pct = 100.0 * count / total if total else 0.0
        lines.append(f"| {source} | {count} | {pct:.1f}% |")

    lines.extend(["", "## Confidence Breakdown", "", f"| Confidence | Count | % |", f"|----------|-------|-------|"])

    for conf, count in sorted(summary["by_confidence"].items(), key=lambda x: -x[1]):
        pct = 100.0 * count / total if total else 0.0
        lines.append(f"| {conf} | {count} | {pct:.1f}% |")

    model_scoring_failures = [
        f for f in failures if f["addressability_class"] == "model_or_scoring_addressable"
    ]
    if model_scoring_failures:
        lines.extend(
            [
                "",
                "## Top Addressable Failures (model_or_scoring_addressable)",
                "",
                f"| Selector | Ground Truth | Predicted | Category |",
                f"|----------|-------------|-----------|----------|",
            ]
        )
        for f in model_scoring_failures[:20]:
            gt = f"{f['ground_truth']['name']}({','.join(f['ground_truth']['types'])})"
            pred = f"{f['predicted']['name']}({','.join(f['predicted']['types'])})"
            lines.append(f"| `{f['selector']}` | {gt} | {pred} | {f['failure_category']} |")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    logger.info("wrote %s", out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/contracts.parquet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    parser.add_argument(
        "--failures-json", default="eval_output/failures.json"
    )
    parser.add_argument(
        "--failure-summary", default="eval_output/failure_summary.md"
    )
    parser.add_argument(
        "--out-json",
        dest="out_json",
        help="Output path for failures JSON (default: eval_output/failures.json or eval_output/external_failures.json in external mode)",
    )
    parser.add_argument(
        "--out-md",
        dest="out_md",
        help="Output path for failure summary Markdown (default: eval_output/failure_summary.md or eval_output/external_failure_summary.md in external mode)",
    )
    parser.add_argument(
        "--external",
        action="store_true",
        help="Run on external eval set. Uses shipped runtime tables only; "
        "does not build tables from the external data.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    data_path = resolve_data_path(args.data, REPO_ROOT)

    df = pd.read_parquet(data_path)

    if args.external:
        FusionReconstructor._known_selectors = None
        FusionReconstructor._known_interfaces = None
        eval_df = df
        logger.info("external eval contracts: %d", len(eval_df))
        known_table = None
        failures, summary = mine_failures(eval_df, known_table)
        failures_json_path = resolve_data_path(
            args.out_json or "eval_output/external_failures.json", REPO_ROOT
        )
        write_failures_json(failures, failures_json_path)

        summary_path = resolve_data_path(
            args.out_md or "eval_output/external_failure_summary.md", REPO_ROOT
        )
        write_failure_summary(
            summary, failures, args.seed, args.heldout_fraction, summary_path
        )
    else:
        train_df, heldout_df = split_dataframe(df, args.heldout_fraction, args.seed)
        logger.info("training contracts: %d", len(train_df))
        logger.info("held-out contracts: %d", len(heldout_df))

        known_table = build_known_selector_table(train_df)
        logger.info("known selector table entries: %d", len(known_table))

        failures, summary = mine_failures(heldout_df, known_table)

        failures_json_path = resolve_data_path(
            args.out_json or args.failures_json, REPO_ROOT
        )
        write_failures_json(failures, failures_json_path)

        summary_path = resolve_data_path(
            args.out_md or args.failure_summary, REPO_ROOT
        )
        write_failure_summary(
            summary, failures, args.seed, args.heldout_fraction, summary_path
        )

    logger.info(
        "done: %d/%d correct (%.1f%%), %d failures",
        summary["correct"],
        summary["total_functions"],
        100.0 * summary["correct"] / summary["total_functions"]
        if summary["total_functions"]
        else 0.0,
        summary["failed"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
