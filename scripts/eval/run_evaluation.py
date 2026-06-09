#!/usr/bin/env python3
"""Canonical ABI reconstruction evaluation.

This is the headline evaluation path for the project:

1. Load verified contracts with bytecode and ground-truth ABIs.
2. Build the known-selector table from the training split only.
3. Evaluate reconstruction on the held-out split only.
4. Write a markdown report with the held-out exact parameter-type accuracy.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from eth_utils import keccak

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from abi_reconstructor.fusion import FusionReconstructor, split_args  # noqa: E402

logger = logging.getLogger("run_evaluation")


def evaluate_evmole_baseline(df: pd.DataFrame) -> Tuple[int, int]:
    """Compute evmole-only accuracy on a dataset.

    Uses evmole.contract_info() to recover argument types directly,
    without 4byte disambiguation. This is the 'evmole baseline' —
    evmole's static analysis alone, before fusion with signature DBs.
    """
    try:
        import evmole
    except Exception:
        logger.warning("evmole not available, skipping baseline")
        return 0, 0

    from abi_reconstructor.reconstructor import BytecodeParser

    parser = BytecodeParser()
    correct = 0
    total = 0

    for _, row in df.iterrows():
        code = row["bytecode"][2:] if row["bytecode"].startswith("0x") else row["bytecode"]
        truth = true_functions(row["abi"])
        pred = {}

        try:
            info = evmole.contract_info(code, selectors=True, arguments=True)
            if info and info.functions:
                for f in info.functions:
                    types = tuple(split_args(f.arguments or ""))
                    pred[f.selector.lower()] = types
        except Exception:
            pass

        try:
            for sel in parser.extract_selectors(row["bytecode"]):
                sel_hex = sel.selector.lower()
                if sel_hex not in pred:
                    pred[sel_hex] = ()
        except Exception:
            pass

        for sel, info in truth.items():
            total += 1
            if pred.get(sel) == info["types"]:
                correct += 1

    return correct, total


FunctionTruth = Dict[str, Dict[str, Any]]
KnownSelectorTable = Dict[str, Dict[str, Any]]
CategoryResults = Dict[str, Dict[str, int]]


def normalize_abi(abi: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(abi, np.ndarray):
        return abi.tolist()
    if isinstance(abi, str):
        return json.loads(abi)
    return abi


def true_functions(abi: Any) -> FunctionTruth:
    out: FunctionTruth = {}
    for item in normalize_abi(abi):
        if item.get("type") != "function":
            continue
        types = tuple(inp["type"] for inp in item.get("inputs", []))
        signature = f"{item.get('name', '')}({','.join(types)})"
        selector = keccak(text=signature).hex()[:8]
        out[selector] = {"name": item.get("name", ""), "types": types}
    return out


def categorize_contract(abi: Any) -> str:
    for item in normalize_abi(abi):
        if item.get("type") != "function":
            continue
        name = item.get("name", "").lower()
        if any(
            term in name
            for term in ("swap", "add", "remove", "liquidity", "router", "amm")
        ):
            return "DeFi router"
        if any(
            term in name
            for term in ("transfer", "approve", "mint", "burn", "totalsupply", "balanceof")
        ):
            return "simple token"
        if any(
            term in name
            for term in ("callback", "hook", "liquidity", "swap", "initialize")
        ):
            return "DeFi AMM"
    return "other"


def split_dataframe(
    df: pd.DataFrame, heldout_fraction: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < heldout_fraction < 1:
        raise ValueError("--heldout-fraction must be between 0 and 1")

    rng = np.random.RandomState(seed)
    heldout_size = int(len(df) * heldout_fraction)
    heldout_idx = rng.choice(len(df), heldout_size, replace=False)
    train_idx = np.setdiff1d(np.arange(len(df)), heldout_idx)
    return df.iloc[train_idx], df.iloc[heldout_idx]


def build_known_selector_table(train_df: pd.DataFrame) -> KnownSelectorTable:
    selector_sigs: Dict[str, list] = defaultdict(list)
    for _, row in train_df.iterrows():
        for selector, info in true_functions(row["abi"]).items():
            selector_sigs[selector].append((info["name"], info["types"]))

    known_table: KnownSelectorTable = {}
    for selector, occurrences in selector_sigs.items():
        if len(occurrences) < 2:
            continue
        names = {name for name, _ in occurrences}
        type_sets = {types for _, types in occurrences}
        if len(names) == 1 and len(type_sets) == 1:
            name, types = occurrences[0]
            known_table[selector] = {"name": name, "types": list(types)}
    return known_table


def evaluate_contracts(
    df: pd.DataFrame, known_table: KnownSelectorTable | None = None
) -> Tuple[CategoryResults, int, int]:
    fusion = FusionReconstructor()
    if known_table is not None:
        fusion._known_selectors = known_table

    by_category: CategoryResults = defaultdict(lambda: {"correct": 0, "total": 0})
    correct = 0
    total = 0

    for index, (_, row) in enumerate(df.iterrows(), start=1):
        if index == 1 or index % 50 == 0:
            logger.info("evaluating contract %d/%d", index, len(df))

        truth = true_functions(row["abi"])
        reconstructed = fusion.reconstruct(row["bytecode"])
        predicted = {
            fn["selector"]: tuple(inp["type"] for inp in fn["inputs"])
            for fn in reconstructed["functions"]
        }

        category = categorize_contract(row["abi"])
        for selector, info in truth.items():
            total += 1
            by_category[category]["total"] += 1
            if predicted.get(selector) == info["types"]:
                correct += 1
                by_category[category]["correct"] += 1

    return dict(by_category), correct, total


def pct(correct: int, total: int) -> float:
    return 100.0 * correct / total if total else 0.0


def render_report(
    *,
    data_path: Path,
    seed: int,
    heldout_fraction: float,
    train_contracts: int,
    heldout_contracts: int,
    known_table_size: int,
    train_correct: int,
    train_total: int,
    heldout_correct: int,
    heldout_total: int,
    heldout_by_category: CategoryResults,
    evmole_correct: int = 0,
    evmole_total: int = 0,
) -> str:
    train_acc = pct(train_correct, train_total)
    heldout_acc = pct(heldout_correct, heldout_total)
    evmole_acc = pct(evmole_correct, evmole_total)
    lines = [
        "# ABI Reconstructor Canonical Evaluation",
        "",
        f"**Date:** {date.today().isoformat()}",
        "",
        "## Headline",
        "",
        "| Reconstructor | Held-out exact parameter-type accuracy |",
        "|---|---|",
        f"| **Fusion** (recommended) | **{heldout_acc:.1f}%** |",
        f"| evmole baseline | {evmole_acc:.1f}% |",
        "",
        f"- Fusion fixes {heldout_correct - evmole_correct} functions evmole gets wrong",
        f"- Held-out functions: {heldout_correct}/{heldout_total}",
        f"- Training accuracy: {train_acc:.1f}%",
        f"- Generalization gap: {train_acc - heldout_acc:.1f}pp",
        "",
        "## Methodology",
        "",
        f"- Dataset: `{data_path}`",
        f"- Split: {100 * (1 - heldout_fraction):.0f}% training / {100 * heldout_fraction:.0f}% held-out",
        f"- Split seed: {seed}",
        f"- Training contracts: {train_contracts}",
        f"- Held-out contracts: {heldout_contracts}",
        f"- Known selector table entries: {known_table_size}",
        "- Known selector table built from training contracts only",
        "- Both fusion and evmole baseline evaluated on the same held-out split",
        "",
        "## Held-Out By Category",
        "",
        "| Category | Correct | Total | Accuracy |",
        "|----------|---------|-------|----------|",
    ]
    for category in sorted(heldout_by_category):
        row = heldout_by_category[category]
        lines.append(
            f"| {category} | {row['correct']} | {row['total']} | "
            f"{pct(row['correct'], row['total']):.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            "This report is the canonical project headline. README accuracy claims should",
            "reference this held-out result instead of full-dataset or exploratory reports.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_external_report(
    *,
    data_path: Path,
    eval_contracts: int,
    eval_correct: int,
    eval_total: int,
    eval_by_category: CategoryResults,
    evmole_correct: int = 0,
    evmole_total: int = 0,
) -> str:
    eval_acc = pct(eval_correct, eval_total)
    evmole_acc = pct(evmole_correct, evmole_total)
    lines = [
        "# ABI Reconstructor External Evaluation",
        "",
        f"**Date:** {date.today().isoformat()}",
        "",
        "## Headline",
        "",
        "| Reconstructor | External exact parameter-type accuracy |",
        "|---|---|",
        f"| **Fusion** (recommended) | **{eval_acc:.1f}%** |",
        f"| evmole baseline | {evmole_acc:.1f}% |",
        "",
        f"- Fusion fixes {eval_correct - evmole_correct} functions evmole gets wrong",
        f"- External functions: {eval_correct}/{eval_total}",
        "",
        "## Methodology",
        "",
        f"- Dataset: `{data_path}` (external, not split)",
        "- Uses shipped runtime tables only (no table building from external data)",
        f"- External contracts: {eval_contracts}",
        "",
        "## External By Category",
        "",
        "| Category | Correct | Total | Accuracy |",
        "|----------|---------|-------|----------|",
    ]
    for category in sorted(eval_by_category):
        row = eval_by_category[category]
        lines.append(
            f"| {category} | {row['correct']} | {row['total']} | "
            f"{pct(row['correct'], row['total']):.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            "This report is the external generalization evaluation. ",
            "It uses shipped runtime tables only — no table building from external data.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/contracts.parquet")
    parser.add_argument("--out", default="eval_output/canonical_evaluation.md")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    parser.add_argument(
        "--external",
        action="store_true",
        help="Run on external eval set. Uses shipped runtime tables only; "
        "does not build tables from the external data.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = REPO_ROOT / data_path
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path

    df = pd.read_parquet(data_path)

    if args.external:
        # External mode: no split, use shipped runtime tables only
        if not args.out or args.out == "eval_output/canonical_evaluation.md":
            out_path = REPO_ROOT / "eval_output" / "external_evaluation.md"
        FusionReconstructor._known_selectors = None
        FusionReconstructor._known_interfaces = None
        eval_df = df
        logger.info("external eval contracts: %d", len(eval_df))
        eval_by_category, eval_correct, eval_total = evaluate_contracts(eval_df, known_table=None)
        evmole_correct, evmole_total = evaluate_evmole_baseline(eval_df)
        logger.info("evmole baseline: %d/%d = %.1f%%", evmole_correct, evmole_total,
                    pct(evmole_correct, evmole_total))
        report = render_external_report(
            data_path=data_path.relative_to(REPO_ROOT),
            eval_contracts=len(eval_df),
            eval_correct=eval_correct,
            eval_total=eval_total,
            eval_by_category=eval_by_category,
            evmole_correct=evmole_correct,
            evmole_total=evmole_total,
        )
    else:
        train_df, heldout_df = split_dataframe(df, args.heldout_fraction, args.seed)
        logger.info("training contracts: %d", len(train_df))
        logger.info("held-out contracts: %d", len(heldout_df))

        known_table = build_known_selector_table(train_df)
        logger.info("known selector table entries: %d", len(known_table))

        train_by_category, train_correct, train_total = evaluate_contracts(
            train_df, known_table
        )
        heldout_by_category, heldout_correct, heldout_total = evaluate_contracts(
            heldout_df, known_table
        )
        _ = train_by_category

        evmole_correct, evmole_total = evaluate_evmole_baseline(heldout_df)
        logger.info("evmole baseline: %d/%d = %.1f%%", evmole_correct, evmole_total,
                    pct(evmole_correct, evmole_total))

        report = render_report(
            data_path=data_path.relative_to(REPO_ROOT),
            seed=args.seed,
            heldout_fraction=args.heldout_fraction,
            train_contracts=len(train_df),
            heldout_contracts=len(heldout_df),
            known_table_size=len(known_table),
            train_correct=train_correct,
            train_total=train_total,
            heldout_correct=heldout_correct,
            heldout_total=heldout_total,
            heldout_by_category=heldout_by_category,
            evmole_correct=evmole_correct,
            evmole_total=evmole_total,
        )

    print(report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    logger.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
