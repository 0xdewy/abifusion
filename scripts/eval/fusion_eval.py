#!/usr/bin/env python3
"""Phase 3 — evaluate the FusionReconstructor end-to-end vs evmole.

Runs the actual FusionReconstructor.reconstruct() on each contract and compares
its per-selector parameter types against the ground-truth ABI, alongside evmole
as the baseline. Reports exact-type accuracy and which evmole errors were fixed.
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from eth_utils import keccak

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import evmole  # noqa: E402

from abi_reconstructor.fusion import FusionReconstructor, split_args  # noqa: E402

logger = logging.getLogger("fusion_eval")


def true_functions(abi):
    out = {}
    for it in abi:
        if it.get("type") != "function":
            continue
        ts = tuple(i["type"] for i in it.get("inputs", []))
        sel = keccak(text=it.get("name", "") + "(" + ",".join(ts) + ")").hex()[:8]
        out[sel] = ts
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/contracts.parquet")
    ap.add_argument("--holdout", type=float, default=0.0,
                    help="evaluate only the last fraction of contracts (e.g. 0.3)")
    ap.add_argument("--out", default="eval_output/fusion_eval.md")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    df = pd.read_parquet(args.data)
    if args.holdout:
        df = df.iloc[int(len(df) * (1 - args.holdout)):]
    logger.info("contracts: %d", len(df))

    fusion = FusionReconstructor()
    n = evmole_ok = fusion_ok = 0
    fixed = broke = 0
    fixed_buckets = Counter()

    for ci, (_, r) in enumerate(df.iterrows()):
        if ci % 50 == 0:
            logger.info("  contract %d/%d", ci, len(df))
        abi = r["abi"]
        if isinstance(abi, np.ndarray):
            abi = abi.tolist()
        if isinstance(abi, str):
            abi = json.loads(abi)
        truth = true_functions(abi)
        code = r["bytecode"]
        code = code[2:] if str(code).startswith("0x") else code

        try:
            info = evmole.contract_info(code, arguments=True)
            ev = {f.selector.lower(): split_args(f.arguments or "") for f in info.functions}
        except Exception:
            ev = {}
        recon = fusion.reconstruct(r["bytecode"])
        fus = {
            f["selector"]: tuple(i["type"] for i in f["inputs"])
            for f in recon["functions"]
        }

        for sel, true_t in truth.items():
            n += 1
            e_ok = ev.get(sel) == true_t
            f_ok = fus.get(sel) == true_t
            evmole_ok += e_ok
            fusion_ok += f_ok
            if f_ok and not e_ok:
                fixed += 1
                # classify what evmole had gotten wrong
                et = ev.get(sel)
                if et is None:
                    fixed_buckets["evmole_missing"] += 1
                elif len(et) != len(true_t):
                    fixed_buckets["arity"] += 1
                else:
                    fixed_buckets["type_swap"] += 1
            if e_ok and not f_ok:
                broke += 1

    def pct(x):
        return 100.0 * x / max(n, 1)

    lines = [
        "# FusionReconstructor vs evmole (Phase 3)",
        "",
        f"Functions evaluated: **{n}**" + (f" (held-out {args.holdout:.0%})" if args.holdout else " (full set)"),
        "",
        "| Method | Exact type-tuple accuracy |",
        "|---|---|",
        f"| evmole | **{pct(evmole_ok):.1f}%** |",
        f"| FusionReconstructor | **{pct(fusion_ok):.1f}%** |",
        "",
        f"Net vs evmole: **+{fixed} fixed**, **-{broke} broke** "
        f"({pct(fusion_ok) - pct(evmole_ok):+.1f} pp)",
        "",
        "Fixed-error breakdown:",
    ]
    for k, v in fixed_buckets.most_common():
        lines.append(f"  - {k}: {v}")
    report = "\n".join(lines)
    print("\n" + report + "\n")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report + "\n")
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
