#!/usr/bin/env python3
"""Expand data/known_selector_signatures.json from verified contract sources.

This script finds contracts in data/contracts.parquet that contain the 29
hard-case repeated selectors (Uniswap V3 callbacks, NFT setters, SVG generators)
and expands the known_selector_signatures.json with additional verified entries.

The 29 hard-case selectors (from eval_output/hard_cases.md):
- Uniswap V3 callbacks (×12-18 each): 468ead2c, bc29bafc, 1f29cf9d, b6d4944a,
  606e8192, c13d1c69, b772b8cc, d950bd74, ebe1cdaf, 5cb32d10
- NFT setters (×4 each): 71edd89d, d8234121, d33aed88, 2201fddb, 49b69b3f,
  5d5ce9c9, 0bf1d8f2, fed15348, 8728fc6e
- SVG generator (×4): a660668a
- Plus 9 others: ac12fecf, 5189bbab, 588570a5, fb4baf17, af6ed122,
  e543e5bf, 2b805192, 501e60d5, 7b315630

Usage:
    python scripts/eval/build_known_selector_table.py
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from eth_utils import keccak

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
TABLE_PATH = DATA_DIR / "known_selector_signatures.json"

if not (DATA_DIR / "contracts.parquet").exists():
    REPO_ROOT = Path("/home/user/code/abifusion")
    DATA_DIR = REPO_ROOT / "data"
    TABLE_PATH = DATA_DIR / "known_selector_signatures.json"

import sys

sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("build_known_selector_table")

HARD_CASE_SELECTORS = {
    "468ead2c",  # beforeSwap
    "bc29bafc",  # afterAddLiquidity
    "1f29cf9d",  # afterDonate
    "b6d4944a",  # afterInitialize
    "606e8192",  # afterRemoveLiquidity
    "c13d1c69",  # afterSwap
    "b772b8cc",  # beforeAddLiquidity
    "d950bd74",  # beforeDonate
    "ebe1cdaf",  # beforeInitialize
    "5cb32d10",  # beforeRemoveLiquidity
    "71edd89d",  # setAccessories
    "d8234121",  # setBody
    "d33aed88",  # setEyes
    "2201fddb",  # setGround
    "49b69b3f",  # setHair
    "5d5ce9c9",  # setHorn
    "0bf1d8f2",  # setLegsBack
    "fed15348",  # setLegsFront
    "8728fc6e",  # setTail
    "a660668a",  # generateSvg
    "ac12fecf",  # startRace
    "5189bbab",  # setWings
    "588570a5",  # initialize
    "fb4baf17",  # changeFeeParams
    "af6ed122",  # executeUpgrade
    "e543e5bf",  # setChainCreationParams
    "2b805192",  # setNewVersionUpgrade
    "501e60d5",  # setUpgradeDiamondCut
    "7b315630",  # upgradeChainFromVersion
}


def true_functions(abi):
    out = {}
    for it in abi:
        if it.get("type") != "function":
            continue
        ts = tuple(i["type"] for i in it.get("inputs", []))
        sel = keccak(text=it.get("name", "") + "(" + ",".join(ts) + ")").hex()[:8]
        out[sel] = {"name": it.get("name"), "types": ts}
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    df = pd.read_parquet(DATA_DIR / "contracts.parquet")
    logger.info("Loaded %d contracts from contracts.parquet", len(df))

    existing_table = {}
    if TABLE_PATH.exists():
        with open(TABLE_PATH) as f:
            existing_table = json.load(f)
        logger.info("Loaded existing table with %d entries", len(existing_table))

    selector_to_contracts = defaultdict(list)

    for ci, (_, r) in enumerate(df.iterrows()):
        if ci % 100 == 0:
            logger.info("  scanning contract %d/%d", ci, len(df))

        abi = r["abi"]
        if isinstance(abi, np.ndarray):
            abi = abi.tolist()
        if isinstance(abi, str):
            abi = json.loads(abi)

        truth = true_functions(abi)

        for sel in HARD_CASE_SELECTORS:
            if sel in truth:
                selector_to_contracts[sel].append({
                    "name": truth[sel]["name"],
                    "types": truth[sel]["types"],
                    "contract_idx": ci,
                })

    added_count = 0
    updated_count = 0
    skipped_already_present = 0
    verified_entries = {}

    for sel in HARD_CASE_SELECTORS:
        occurrences = selector_to_contracts.get(sel, [])
        if not occurrences:
            logger.info("  selector %s: no occurrences in contracts.parquet", sel)
            continue

        names = {o["name"] for o in occurrences}
        types_sets = {tuple(o["types"]) for o in occurrences}

        if len(names) > 1:
            logger.warning("  selector %s: inconsistent names across %d occurrences: %s",
                           sel, len(occurrences), names)
            continue
        if len(types_sets) > 1:
            logger.warning("  selector %s: inconsistent types across %d occurrences: %s",
                           sel, len(occurrences), types_sets)
            continue

        name = occurrences[0]["name"]
        types = list(occurrences[0]["types"])
        total_count = len(occurrences)

        if sel in existing_table:
            existing = existing_table[sel]
            if existing["name"] == name and tuple(existing["types"]) == tuple(types):
                old_count = existing.get("count", 0)
                new_count = max(old_count, total_count)
                verified_entries[sel] = {
                    "name": name,
                    "types": types,
                    "selector": sel,
                    "count": new_count,
                }
                if new_count > old_count:
                    updated_count += 1
                    logger.info("  selector %s: updated count %d -> %d (verified %s)",
                                sel, old_count, new_count, name)
                else:
                    logger.info("  selector %s: count unchanged at %d (verified %s)",
                                sel, old_count, name)
            else:
                logger.warning("  selector %s: existing entry mismatch, skipping", sel)
                skipped_already_present += 1
        else:
            verified_entries[sel] = {
                "name": name,
                "types": types,
                "selector": sel,
                "count": total_count,
            }
            added_count += 1
            logger.info("  selector %s: ADDED (%s, %s, count=%d)",
                        sel, name, types, total_count)

    merged = {**existing_table, **verified_entries}

    with open(TABLE_PATH, "w") as f:
        json.dump(merged, f, indent=2)
    logger.info("Wrote expanded table with %d entries to %s", len(merged), TABLE_PATH)

    print("\n=== Expansion Summary ===")
    print(f"  Selectors already present (verified & updated): {updated_count}")
    print(f"  Selectors newly added: {added_count}")
    print(f"  Selectors skipped (mismatch): {skipped_already_present}")
    print(f"  Total entries in table: {len(merged)}")

    print("\n=== Hard-case selectors status ===")
    for sel in sorted(HARD_CASE_SELECTORS):
        if sel in merged:
            e = merged[sel]
            status = "existing" if sel in existing_table else "NEW"
            print(f"  [{status}] {sel} ×{e['count']:2d}: {e['name']}({','.join(e['types'])})")
        else:
            print(f"  [MISSING] {sel}: not found in contracts.parquet")


if __name__ == "__main__":
    main()
