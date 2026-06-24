#!/usr/bin/env python3
"""Pre-seed the signature cache for all selectors in a dataset.

Walks every contract, extracts function selectors, queries openchain + 4byte
for each, and writes results to the standard cache dir. After this,
``ABIFusion`` runs at full speed — no network latency during evaluation.
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path
from typing import List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from abifusion.utils.signature_lookup import SignatureLookup
from abifusion.utils.bytecode import extract_push4_selectors


def extract_selectors(bytecode: str) -> List[str]:
    return extract_push4_selectors(bytecode)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/contracts_1k.parquet")
    args = parser.parse_args()

    data_path = REPO_ROOT / args.data
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} contracts from {data_path}")

    all_selectors = OrderedDict()
    for i, (_, row) in enumerate(df.iterrows()):
        sels = extract_selectors(row["bytecode"])
        for s in sels:
            all_selectors[s] = True

    all_selectors_list = list(all_selectors.keys())
    print(f"Found {len(all_selectors_list)} unique selectors across {len(df)} contracts")

    sl = SignatureLookup()
    for i, sel in enumerate(all_selectors_list):
        if i % 50 == 0:
            print(f"  caching {i + 1}/{len(all_selectors_list)}: 0x{sel} ...")

        # Trigger cache writes for both sources
        openchain = sl.lookup_openchain(sel)
        fourbyte = sl.lookup_4byte(sel)
        count = len(openchain) + len(fourbyte)

    print(f"Cached {len(all_selectors_list)} selectors. Cache dir: {sl.cache_dir}")
    print("Ready for offline evaluation: python scripts/eval/run_evaluation.py --data data/contracts_1k.parquet")


if __name__ == "__main__":
    main()
