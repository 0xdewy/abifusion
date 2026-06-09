#!/usr/bin/env python3
"""Phase 0 verification: CALLDATALOAD offset analysis for per-parameter localization.

This script validates that CALLDATALOAD offset analysis can identify per-parameter
bytecode regions. It scans function bodies for PUSHn <offset> CALLDATALOAD patterns
and compares the number of unique offsets found to the arity reported by evmole.

If >90% of functions have matching offset count vs arity, Phase 0 PASSED.
If <90%, Phase 0 FAILED — fallback to global features.

Usage:
    python scripts/eval/verify_calldataload_offset.py
"""

import logging
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"

if not (DATA_DIR / "contracts.parquet").exists():
    REPO_ROOT = Path("/home/user/code/abi_reconstructor")
    DATA_DIR = REPO_ROOT / "data"

import sys

sys.path.insert(0, str(REPO_ROOT))

from abi_reconstructor.fusion import split_args
from abi_reconstructor.utils.bytecode import extract_calldata_offsets

logger = logging.getLogger("verify_calldataload_offset")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    df = pd.read_parquet(DATA_DIR / "contracts.parquet")
    logger.info("Loaded %d contracts", len(df))

    num_contracts = min(10, len(df))
    total_functions = 0
    matching_functions = 0
    mismatches = []

    for ci in range(num_contracts):
        row = df.iloc[ci]
        bytecode = row["bytecode"]
        if isinstance(bytecode, str) and bytecode.startswith("0x"):
            code = bytecode[2:]
        else:
            code = bytecode

        try:
            import evmole
            info = evmole.contract_info(code, selectors=True, arguments=True)
        except Exception as e:
            logger.warning("evmole failed for contract %d: %s", ci, e)
            continue

        if info is None or info.functions is None:
            continue

        for f in info.functions:
            if f.arguments is None:
                continue

            arity = len(split_args(f.arguments))
            if arity == 0:
                continue

            bytecode_offset = f.bytecode_offset
            if bytecode_offset is None or bytecode_offset < 0:
                continue

            raw = bytes.fromhex(code)
            if bytecode_offset >= len(raw):
                continue

            offsets_found = extract_calldata_offsets(raw, bytecode_offset)
            num_offsets = len(offsets_found)

            total_functions += 1

            if num_offsets == arity:
                matching_functions += 1
            else:
                mismatches.append({
                    "contract": ci,
                    "selector": f.selector,
                    "arity": arity,
                    "offsets_found": num_offsets,
                    "offset_values": offsets_found[:arity + 2] if offsets_found else [],
                })

    if total_functions == 0:
        print("\nPhase 0 RESULT: FAILED (no functions with arguments found)")
        return

    match_rate = matching_functions / total_functions * 100
    print(f"\n{'=' * 60}")
    print("Phase 0 Verification: CALLDATALOAD Offset Analysis")
    print(f"{'=' * 60}")
    print(f"  Contracts analyzed: {num_contracts}")
    print(f"  Total functions with arguments: {total_functions}")
    print(f"  Matching (offsets == arity): {matching_functions}")
    print(f"  Mismatching: {len(mismatches)}")
    print(f"  Match rate: {match_rate:.1f}%")
    print()

    if mismatches and len(mismatches) <= 20:
        print("Sample mismatches:")
        for m in mismatches[:10]:
            print(f"  sel={m['selector']} arity={m['arity']} offsets={m['offsets_found']} values={m['offset_values']}")

    print()
    if match_rate >= 90:
        print(f"Phase 0: PASSED (≥90% threshold met: {match_rate:.1f}%)")
        print("Per-position feature extraction is VIABLE.")
    else:
        print(f"Phase 0: FAILED (<90% threshold: {match_rate:.1f}%)")
        print("Falling back to global features with arity augmentation.")


if __name__ == "__main__":
    main()
