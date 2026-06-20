#!/usr/bin/env python3
"""Step 1.1: Characterize the 244 hard cases in detail.

For each of the 244 functions where fusion has no 4byte candidate AND evmole
finds nothing:
- Compute the true selector from ground truth
- Check if the same selector appears elsewhere in the dataset (repeated)
- Check if the function name appears with known 4byte coverage elsewhere
- Categorize by function family (Uniswap V3 callbacks, NFT setters, etc.)
"""

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from eth_utils import keccak

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO_ROOT))

from abifusion.fusion import split_args
from abifusion.utils.signature_lookup import SignatureLookup

logger = logging.getLogger("analyze_hard_cases")


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/contracts.parquet")
    ap.add_argument("--out", default="eval_output/hard_cases.md")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    df = pd.read_parquet(args.data)
    sl = SignatureLookup()

    # Collect all errors
    all_true_sigs = {}  # selector -> list of (contract_idx, name, types)
    all_func_names = Counter()  # func_name -> count across contracts

    for ci, (_, r) in enumerate(df.iterrows()):
        if ci % 50 == 0:
            logger.info("  contract %d/%d", ci, len(df))

        abi = r["abi"]
        if isinstance(abi, np.ndarray):
            abi = abi.tolist()
        if isinstance(abi, str):
            abi = json.loads(abi)

        truth = true_functions(abi)

        for sel, info in truth.items():
            all_true_sigs.setdefault(sel, []).append({"ci": ci, "name": info["name"], "types": info["types"]})
            all_func_names[info["name"]] += 1

    # Now find hard cases
    hard_selector_map = defaultdict(list)  # selector -> list of hard case info
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
            import evmole
            info = evmole.contract_info(code, arguments=True)
            evmole_types = {f.selector.lower(): split_args(f.arguments or "") for f in info.functions}
        except Exception:
            evmole_types = {}

        try:
            from tooling.abifusion_legacy.reconstructor import BytecodeParser
            extra_sels = {s.selector.lower() for s in BytecodeParser().extract_selectors(r["bytecode"])}
        except Exception:
            extra_sels = set()

        truth = true_functions(abi)

        for sel, info in truth.items():
            ev = evmole_types.get(sel)
            oc = [s for s in sl.lookup_openchain(sel) if s.get("text_signature")]
            fb = [s for s in sl.lookup_4byte(sel) if s.get("text_signature")]
            has_4byte = bool(oc or fb)

            # Hard case: no 4byte AND evmole finds nothing AND not in extra selectors
            if not has_4byte and ev is None and sel not in extra_sels:
                # Check how many times this selector appears in the dataset
                occurrences = all_true_sigs.get(sel, [])
                hard_selector_map[sel].append({
                    "ci": ci,
                    "name": info["name"],
                    "types": info["types"],
                    "occurrences": len(occurrences),
                })

    # Build the report
    total_hard = sum(len(v) for v in hard_selector_map.values())
    unique_selectors = len(hard_selector_map)

    lines = [
        "# Hard Cases Analysis: Selectors with No 4byte AND No evmole",
        "",
        f"Total hard cases: **{total_hard}**",
        f"Unique selectors: **{unique_selectors}**",
        "",
    ]

    # Categorize by function family
    func_family = Counter()
    for sel, cases in hard_selector_map.items():
        for c in cases:
            name = c["name"]
            if name.startswith(("after", "before")):
                func_family["Uniswap V3 callbacks"] += 1
            elif name.startswith("set") and len(name) > 3:
                func_family["NFT setters"] += 1
            elif name in ("execute", "executeAndSendBalance", "executeFinalCall"):
                func_family["execute helpers"] += 1
            elif name.startswith("generate"):
                func_family["SVG generators"] += 1
            else:
                func_family["other"] += 1

    lines.append("## Function family breakdown:")
    for k, v in func_family.most_common():
        lines.append(f"  - **{k}**: {v}")
    lines.append("")

    # Repeated selectors
    repeated = {s: c for s, c in hard_selector_map.items() if len(c) > 1}
    unique_only = {s: c for s, c in hard_selector_map.items() if len(c) == 1}

    lines.append(f"## Selector frequency:")
    lines.append(f"  - Repeated selectors: **{len(repeated)}** ({sum(len(v) for v in repeated.values())} occurrences)")
    lines.append(f"  - Unique-only selectors: **{len(unique_only)}**")
    lines.append("")

    # Top repeated selectors
    lines.append("## Top repeated selectors:")
    for sel, cases in sorted(hard_selector_map.items(), key=lambda x: -len(x[1]))[:20]:
        names = [c["name"] for c in cases]
        types = [c["types"] for c in cases]
        lines.append(f"  - `{sel}` ×{len(cases)}: {names[0]}({','.join(types[0])})")
    lines.append("")

    # Sample of unique-only selectors
    lines.append("## Sample unique selectors (first 15):")
    for sel, cases in list(unique_only.items())[:15]:
        c = cases[0]
        lines.append(f"  - `{sel}`: {c['name']}({','.join(c['types'])})")

    # Overall selector frequency distribution
    lines.append("")
    lines.append("## Selector frequency distribution:")
    freq_counter = Counter()
    for sel, cases in hard_selector_map.items():
        freq_counter[len(cases)] += 1
    for freq in sorted(freq_counter.keys(), reverse=True):
        lines.append(f"  - ×{freq}: {freq_counter[freq]} selectors")

    report = "\n".join(lines)
    print("\n" + report + "\n")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report + "\n")
    logger.info("wrote %s", args.out)

    # Also write the detailed mapping as JSON for downstream use
    detail_path = Path(args.out).parent / "hard_cases_detail.json"
    detail = {
        sel: [{"ci": c["ci"], "name": c["name"], "types": list(c["types"]), "occurrences": c["occurrences"]} for c in cases]
        for sel, cases in hard_selector_map.items()
    }
    with open(detail_path, "w") as f:
        json.dump(detail, f, indent=2)
    logger.info("wrote detail JSON to %s", detail_path)


if __name__ == "__main__":
    main()
