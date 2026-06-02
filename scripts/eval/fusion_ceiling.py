#!/usr/bin/env python3
"""Phase 1 — measure the ceiling for a fusion reconstructor vs evmole.

For every function in the ground-truth set (data/contracts.parquet), compare the
true parameter-type tuple against:

  * evmole              — static analysis (baseline)
  * 4byte recall        — is the true signature among the 4byte candidates?
  * oracle fusion       — true types if 4byte recall hits, else evmole (perfect
                          disambiguation upper bound)
  * evmole-guided fusion— pick the 4byte candidate whose types best match
                          evmole's; else evmole (realistic, no ML)

4byte lookups are cached (cache/signatures), so re-runs are fast.
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

from abi_reconstructor.utils.signature_lookup import SignatureLookup  # noqa: E402

logger = logging.getLogger("fusion_ceiling")


def split_args(arg_str: str):
    """Split a Solidity arg list, respecting nested tuple parens."""
    arg_str = arg_str.strip()
    if not arg_str:
        return ()
    out, depth, cur = [], 0, ""
    for ch in arg_str:
        if ch == "(":
            depth += 1
            cur += ch
        elif ch == ")":
            depth -= 1
            cur += ch
        elif ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return tuple(out)


def types_from_signature(text_sig: str):
    """('transfer(address,uint256)') -> ('address','uint256')."""
    i = text_sig.find("(")
    if i == -1:
        return None
    return split_args(text_sig[i + 1 : text_sig.rfind(")")])


def ground_truth(df):
    """List of (selector, true_types) per function instance."""
    rows = []
    for _, r in df.iterrows():
        abi = r["abi"]
        if isinstance(abi, np.ndarray):
            abi = abi.tolist()
        if isinstance(abi, str):
            abi = json.loads(abi)
        code = r["bytecode"]
        code = code[2:] if str(code).startswith("0x") else code
        try:
            info = evmole.contract_info(code, arguments=True)
        except Exception:
            info = None
        ev = (
            {f.selector.lower(): split_args(f.arguments or "") for f in info.functions}
            if info
            else {}
        )
        for it in abi:
            if it.get("type") != "function":
                continue
            ts = tuple(i["type"] for i in it.get("inputs", []))
            sel = keccak(text=it.get("name", "") + "(" + ",".join(ts) + ")").hex()[:8]
            rows.append((sel, ts, ev.get(sel)))
    return rows


def pick_fusion(true_types, evmole_types, candidate_type_lists):
    """Realistic, ML-free fusion choice of a parameter-type tuple."""
    cands = candidate_type_lists
    if cands:
        # 1) candidate that exactly equals evmole's structure
        if evmole_types is not None:
            for c in cands:
                if c == evmole_types:
                    return c
            # 2) same arity, maximal per-position agreement with evmole
            same_arity = [c for c in cands if len(c) == len(evmole_types)]
            if same_arity:
                return max(
                    same_arity,
                    key=lambda c: sum(a == b for a, b in zip(c, evmole_types)),
                )
        # 3) no evmole help: most common candidate arity, first such
        return cands[0]
    return evmole_types if evmole_types is not None else ()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/contracts.parquet")
    ap.add_argument("--limit", type=int, default=0, help="cap function instances (0=all)")
    ap.add_argument("--out", default="eval_output/fusion_ceiling.md")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    df = pd.read_parquet(args.data)
    rows = ground_truth(df)
    if args.limit:
        rows = rows[: args.limit]
    logger.info("function instances: %d", len(rows))

    sl = SignatureLookup()
    sig_cache = {}  # selector -> list[type-tuple]

    def candidates(sel):
        if sel not in sig_cache:
            cands = []
            for s in sl.lookup_4byte(sel):
                t = types_from_signature(s.get("text_signature", ""))
                if t is not None:
                    cands.append(t)
            sig_cache[sel] = cands
        return sig_cache[sel]

    n = len(rows)
    evmole_ok = recall_hit = oracle_ok = fusion_ok = fourbyte_present = 0
    fusion_fixes = fusion_breaks = 0  # vs evmole
    coll = Counter()
    for idx, (sel, true_t, ev_t) in enumerate(rows):
        if idx % 500 == 0:
            logger.info("  %d/%d (unique selectors looked up: %d)", idx, n, len(sig_cache))
        cands = candidates(sel)
        if cands:
            fourbyte_present += 1
            coll[min(len(cands), 10)] += 1
        recall = true_t in cands
        e_ok = ev_t == true_t
        f_ok = pick_fusion(true_t, ev_t, cands) == true_t
        evmole_ok += e_ok
        recall_hit += recall
        oracle_ok += recall or e_ok
        fusion_ok += f_ok
        fusion_fixes += f_ok and not e_ok
        fusion_breaks += e_ok and not f_ok

    def pct(x):
        return 100.0 * x / n

    lines = [
        "# Fusion ceiling vs evmole (Phase 1)",
        "",
        f"Ground-truth function instances: **{n}**",
        "",
        "| Method | Exact type-tuple accuracy |",
        "|---|---|",
        f"| evmole (baseline) | **{pct(evmole_ok):.1f}%** |",
        f"| 4byte recall (true sig present) | {pct(recall_hit):.1f}% |",
        f"| Oracle fusion (4byte-true else evmole) | **{pct(oracle_ok):.1f}%** |",
        f"| evmole-guided fusion (no ML) | **{pct(fusion_ok):.1f}%** |",
        "",
        f"Fusion vs evmole: **+{fusion_fixes} fixed**, **-{fusion_breaks} broke** "
        f"(net {fusion_fixes - fusion_breaks:+d} = {pct(fusion_ok) - pct(evmole_ok):+.1f} pp)",
        "",
        f"4byte selector coverage: {pct(fourbyte_present):.1f}%",
        "",
        "4byte candidate-count distribution (collision degree):",
    ]
    for k in sorted(coll):
        label = f"{k}+" if k == 10 else str(k)
        lines.append(f"  - {label} candidates: {coll[k]} ({100.0*coll[k]/max(fourbyte_present,1):.1f}% of covered)")
    report = "\n".join(lines)
    print("\n" + report + "\n")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report + "\n")
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
