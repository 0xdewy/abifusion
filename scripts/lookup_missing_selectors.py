#!/usr/bin/env python3
"""Phase 3: Query 4byte.directory API for 57 unique selectors.

These selectors have no 4byte entry AND no evmole detection.
We query the 4byte API to find their signatures.

Usage:
    PYTHONPATH=/home/user/code/abifusion python scripts/lookup_missing_selectors.py

Output:
    - Updated data/known_selector_signatures.json with found selectors
    - eval_output/external_lookup_results.md with detailed results
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("lookup_missing")


FOURBYTE_API = "https://www.4byte.directory/api/v1/signatures/"
REQUEST_INTERVAL = 1.1


def get_unique_selectors():
    """Load the 57 unique selectors from hard_cases_detail.json."""
    with open(REPO_ROOT / "eval_output" / "hard_cases_detail.json") as f:
        data = json.load(f)

    unique = []
    for sel, entries in data.items():
        total_occ = sum(e["occurrences"] for e in entries)
        if len(entries) == 1 and total_occ == 1:
            entry = entries[0]
            unique.append({
                "selector": sel,
                "name": entry["name"],
                "types": entry["types"],
                "contract_idx": entry["ci"],
            })
    return unique


def lookup_4byte(selector: str) -> dict | None:
    """Query 4byte API for a selector. Returns signature dict or None."""
    url = f"{FOURBYTE_API}?hex_signature=0x{selector}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if results:
            return results[0]
        return None
    except Exception as e:
        logger.warning("4byte lookup failed for %s: %s", selector, e)
        return None


def parse_text_signature(text_sig: str) -> tuple[str, list] | None:
    """Parse 'funcName(type1,type2,...)' into (name, [types])."""
    try:
        i = text_sig.find("(")
        j = text_sig.rfind(")")
        if i == -1 or j == -1:
            return None
        name = text_sig[:i].strip()
        types_str = text_sig[i+1:j]
        types = [t.strip() for t in types_str.split(",") if t.strip()]
        return name, types
    except Exception:
        return None


def main():
    unique_selectors = get_unique_selectors()
    logger.info("Loaded %d unique selectors to lookup", len(unique_selectors))

    found = []
    not_found = []
    errors = []

    existing_table_path = REPO_ROOT / "data" / "known_selector_signatures.json"
    with open(existing_table_path) as f:
        known_table = json.load(f)

    already_in_table = 0
    for sel_info in unique_selectors:
        sel = sel_info["selector"]
        if sel in known_table:
            already_in_table += 1
            logger.debug("%s already in known table as %s", sel, known_table[sel]["name"])

    logger.info("Already in known table: %d/%d", already_in_table, len(unique_selectors))

    to_lookup = [s for s in unique_selectors if s["selector"] not in known_table]
    logger.info("Need to lookup: %d selectors", len(to_lookup))

    for i, sel_info in enumerate(to_lookup):
        sel = sel_info["selector"]
        logger.info("[%d/%d] Looking up %s (%s)...", i + 1, len(to_lookup), sel, sel_info["name"])

        result = lookup_4byte(sel)
        time.sleep(REQUEST_INTERVAL)

        if result is None:
            not_found.append(sel_info)
            logger.info("  -> NOT FOUND")
        else:
            text_sig = result.get("text_signature", "")
            parsed = parse_text_signature(text_sig)
            if parsed:
                name, types = parsed
                entry = {
                    "name": name,
                    "types": types,
                    "source": "4byte",
                    "from_selector": sel_info["name"],
                }
                known_table[sel] = entry
                found.append({**sel_info, "4byte_name": name, "4byte_types": types})
                logger.info("  -> FOUND: %s(%s)", name, types)
            else:
                errors.append({**sel_info, "text_sig": text_sig})
                logger.warning("  -> Could not parse: %s", text_sig)

    with open(existing_table_path, "w") as f:
        json.dump(known_table, f, indent=2)
    logger.info("Updated known_selector_signatures.json")

    output_dir = REPO_ROOT / "eval_output"
    results_md = output_dir / "external_lookup_results.md"

    with open(results_md, "w") as f:
        f.write("# Phase 3: External 4byte Lookup Results\n\n")
        f.write(f"**Date:** 2026-06-07\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"- Unique selectors targeted: {len(unique_selectors)}\n")
        f.write(f"- Already in known table: {already_in_table}\n")
        f.write(f"- Queried 4byte API: {len(to_lookup)}\n")
        f.write(f"- **Found via 4byte: {len(found)}**\n")
        f.write(f"- Not found: {len(not_found)}\n")
        f.write(f"- Parse errors: {len(errors)}\n\n")

        f.write(f"## Found Selectors ({len(found)})\n\n")
        if found:
            f.write("| Selector | Ground Truth Name | 4byte Name | Types |\n")
            f.write("|----------|-------------------|------------|-------|\n")
            for item in found:
                sel = item["selector"]
                gt_name = item["name"]
                name = item["4byte_name"]
                types = item["4byte_types"]
                f.write(f"| `{sel}` | `{gt_name}` | `{name}` | `{types}` |\n")
        else:
            f.write("None.\n\n")

        f.write(f"\n## Not Found ({len(not_found)})\n\n")
        if not_found:
            f.write("| Selector | Name | Contract Idx |\n")
            f.write("|----------|------|--------------|\n")
            for item in not_found:
                f.write(f"| `{item['selector']}` | `{item['name']}` | {item['contract_idx']} |\n")
        else:
            f.write("None.\n\n")

        f.write(f"\n## Parse Errors ({len(errors)})\n\n")
        if errors:
            for item in errors:
                f.write(f"- `{item['selector']}`: could not parse '{item.get('text_sig', '')}'\n")
        else:
            f.write("None.\n\n")

        remaining = len(not_found) + len(errors)
        f.write(f"\n## Final Gap\n\n")
        f.write(f"- Remaining unfound selectors: {remaining}\n")
        f.write(f"- As % of all 6744 selectors: {100 * remaining / 6744:.2f}%\n")
        f.write(f"- These are genuinely unfixable without source code or contract address lookup.\n")

    logger.info("Wrote results to %s", results_md)

    print()
    print("=" * 60)
    print("4BYTE LOOKUP RESULTS")
    print("=" * 60)
    print(f"  Unique selectors targeted: {len(unique_selectors)}")
    print(f"  Already in known table: {already_in_table}")
    print(f"  Queried 4byte API: {len(to_lookup)}")
    print(f"  Found: {len(found)}")
    print(f"  Not found: {len(not_found)}")
    print(f"  Parse errors: {len(errors)}")
    print()
    if found:
        print("Found selectors:")
        for item in found:
            print(f"  {item['selector']}: {item['4byte_name']}({item['4byte_types']})")
    print()
    print(f"Results written to: {results_md}")
    print("=" * 60)


if __name__ == "__main__":
    main()