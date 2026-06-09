#!/usr/bin/env python3
"""Characterize where the fusion reconstructor still errs, to target coverage work.

For every function the fusion gets wrong, classify the cause so we know whether
*more signature coverage* can help (vs. it being a disambiguation problem or a
truly-unknown selector):

  * no_sig_anywhere       — neither openchain nor 4byte returns any candidate
                            (truly unknown selector; only a decompiler can guess)
  * db_lacks_true_sig     — DB(s) have candidates but not the true signature
                            (the DB is incomplete → a more complete source helps)
  * true_sig_mispicked    — true sig IS among candidates but disambiguation chose
                            wrong (NOT a coverage problem)

Buckets no_sig_anywhere + db_lacks_true_sig bound what signature-coverage
expansion could recover.
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

from abifusion.fusion import (  # noqa: E402
    choose_candidate,
    parse_signature,
    split_args,
)
from abifusion.utils.signature_lookup import SignatureLookup  # noqa: E402

logger = logging.getLogger("coverage_gap")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/contracts.parquet")
    ap.add_argument("--out", default="eval_output/coverage_gap.md")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    df = pd.read_parquet(args.data)
    sl = SignatureLookup()
    oc_cache, fb_cache = {}, {}

    def cands(lookup, cache, sel):
        if sel not in cache:
            cache[sel] = [
                parse_signature(s.get("text_signature", ""))
                for s in lookup(sel)
                if parse_signature(s.get("text_signature", ""))
            ]
        return cache[sel]

    n = wrong = 0
    buckets = Counter()
    for ci, (_, r) in enumerate(df.iterrows()):
        if ci % 50 == 0:
            logger.info("  contract %d/%d", ci, len(df))
        abi = r["abi"]
        if isinstance(abi, np.ndarray):
            abi = abi.tolist()
        if isinstance(abi, str):
            abi = json.loads(abi)
        code = r["bytecode"]
        code = code[2:] if str(code).startswith("0x") else code
        try:
            info = evmole.contract_info(code, arguments=True)
            ev = {f.selector.lower(): split_args(f.arguments or "") for f in info.functions}
        except Exception:
            ev = {}

        for it in abi:
            if it.get("type") != "function":
                continue
            ts = tuple(i["type"] for i in it.get("inputs", []))
            sel = keccak(text=it.get("name", "") + "(" + ",".join(ts) + ")").hex()[:8]
            n += 1

            oc = cands(sl.lookup_openchain, oc_cache, sel)
            fb = cands(sl.lookup_4byte, fb_cache, sel)
            candidates = oc or fb  # the source the fusion actually uses
            ev_t = ev.get(sel)
            pick = choose_candidate(candidates, ev_t)
            ptypes = pick[1] if pick else (ev_t if ev_t is not None else ())
            if ptypes == ts:
                continue
            wrong += 1

            all_cand_types = {t for _, t in oc} | {t for _, t in fb}
            if not oc and not fb:
                buckets["no_sig_anywhere"] += 1
            elif ts not in all_cand_types:
                buckets["db_lacks_true_sig"] += 1
            else:
                buckets["true_sig_mispicked"] += 1

    lines = [
        "# Coverage-gap analysis of the fusion reconstructor",
        "",
        f"Functions: **{n}**, wrong: **{wrong}** ({100*wrong/n:.1f}%)",
        "",
        "Residual error causes:",
    ]
    for k, v in buckets.most_common():
        lines.append(f"  - **{k}**: {v} ({100*v/n:.1f}% of all, {100*v/wrong:.1f}% of errors)")
    lines += [
        "",
        "Coverage-addressable (no_sig_anywhere + db_lacks_true_sig): "
        f"**{100*(buckets['no_sig_anywhere']+buckets['db_lacks_true_sig'])/n:.1f}%** of all functions.",
    ]
    report = "\n".join(lines)
    print("\n" + report + "\n")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report + "\n")


if __name__ == "__main__":
    main()
