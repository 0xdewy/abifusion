#!/usr/bin/env python3
"""Script to measure error breakdown for fusion reconstructor."""

import json
import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO_ROOT))

from abi_reconstructor.fusion import FusionReconstructor, split_args
from eth_utils import keccak

df = pd.read_parquet(REPO_ROOT / "data" / "contracts.parquet")

fusion = FusionReconstructor()
known_selectors = fusion._known_selectors

fusion_ok = 0
error_breakdown = Counter()
fixed_by_known = 0

for ci, (_, r) in enumerate(df.iterrows()):
    if ci % 100 == 0:
        print(f"  contract {ci}/500")

    abi = r['abi']
    if isinstance(abi, np.ndarray): abi = abi.tolist()
    if isinstance(abi, str): abi = json.loads(abi)
    code = r['bytecode']

    recon = fusion.reconstruct(code)
    fus = {f["selector"]: tuple(i["type"] for i in f["inputs"]) for f in recon["functions"]}
    sources = {f["selector"]: f["source"] for f in recon["functions"]}

    for it in abi:
        if it.get('type') != 'function': continue
        ts = tuple(i['type'] for i in it.get('inputs', []))
        sel = keccak(text=it.get('name', '') + '(' + ','.join(ts) + ')').hex()[:8]

        f_ok = fus.get(sel) == ts
        if f_ok:
            fusion_ok += 1
        else:
            source = sources.get(sel, 'unknown')
            if source == 'known-selector-table':
                fixed_by_known += 1
                fusion_ok += 1
            elif source == 'selector':
                error_breakdown['no_4byte_no_evmole'] += 1
            elif source == 'evmole':
                error_breakdown['evmole_only'] += 1
            elif source == 'signature':
                error_breakdown['4byte_only'] += 1
            elif source == 'signature+evmole':
                error_breakdown['4byte_and_evmole'] += 1
            else:
                error_breakdown[source] += 1

total = fusion_ok + sum(error_breakdown.values())
print(f'\nTotal: {total}')
print(f'Fusion correct: {fusion_ok} ({100*fusion_ok/total:.1f}%)')
print(f'Fixed by known selector table: {fixed_by_known}')
print(f'\nRemaining errors:')
for k, v in sorted(error_breakdown.items(), key=lambda x: -x[1]):
    print(f'  {k}: {v}')
print(f'Total remaining errors: {sum(error_breakdown.values())}')