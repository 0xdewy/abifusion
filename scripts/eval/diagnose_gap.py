#!/usr/bin/env python3
"""Gap analysis for ABI reconstruction benchmark results.

Independently fetches ground-truth ABIs and runs reconstruction on each
contract, then categorizes every function-level failure.

Usage:
    python scripts/eval/diagnose_gap.py
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Crypto.Hash import keccak
from dotenv import load_dotenv

load_dotenv()

from abifusion.database import FourByteDatabase
from abifusion.data.etherscan_fetcher import EtherscanClient
from abifusion.reconstructor import OfflineABI


# ── Category labels ──

CAT_SELECTOR_MISSING = "A. Selector not found"
CAT_DB_MISS = "B. 4byte DB has no entry"
CAT_DB_COLLISION = "C. 4byte DB collision (multiple signatures)"
CAT_PROXY = "D. Proxy contract (structural)"
CAT_TYPE_BUG = "E. Type mismatch despite DB hit"
CAT_CORRECT = "Correct"


def selector_of(name: str, input_types: List[str]) -> str:
    """Compute the 4-byte keccak selector for a function signature."""
    sig = f"{name}({','.join(input_types)})"
    k = keccak.new(digest_bits=256)
    k.update(sig.encode("utf-8"))
    return k.digest().hex()[:8]


def normalize_type(t: str) -> str:
    """Normalize Solidity type for comparison."""
    import re
    t = t.strip().lower()
    t = re.sub(r'\buint\d+\b', 'uint', t)
    t = re.sub(r'\bint\d+\b', 'int', t)
    t = re.sub(r'\bbytes\d+\b', 'bytesN', t)
    return t


def types_match(gt: List[str], rec: List[str]) -> bool:
    if len(gt) != len(rec):
        return False
    return all(normalize_type(g) == normalize_type(r) for g, r in zip(gt, rec))


def is_proxy_contract(abi: List[Dict], client: EtherscanClient, addr: str) -> Tuple[bool, str]:
    """Heuristic check for proxy contracts."""
    proxy_sigs = {"implementation", "_implementation", "admin", "_admin",
                  "upgradeTo", "upgradeToAndCall", "changeAdmin"}
    for entry in abi:
        if entry.get("name", "") in proxy_sigs:
            return True, f"proxy function '{entry['name']}' in ABI"
    try:
        bc = client.get_contract_bytecode(addr) or ""
        if bc and bc.lower().count("f4") >= 3:
            return True, "delegatecall in bytecode"
    except Exception:
        pass
    return False, ""


def analyze_contract(
    addr: str,
    client: EtherscanClient,
    reconstructor: OfflineABI,
    db: FourByteDatabase,
) -> Dict[str, List[Dict]]:
    """Analyze one contract and categorize all function-level results."""

    out: Dict[str, List[Dict]] = {
        CAT_SELECTOR_MISSING: [],
        CAT_DB_MISS: [],
        CAT_DB_COLLISION: [],
        CAT_PROXY: [],
        CAT_TYPE_BUG: [],
        CAT_CORRECT: [],
    }

    # Fetch ground truth
    abi = client.get_contract_abi(addr)
    bytecode = client.get_contract_bytecode(addr)
    src = client.get_contract_source_code(addr)

    if not abi or not bytecode:
        return out

    contract_name = ""
    if src:
        contract_name = src.get("contract_name", "") or src.get("ContractName", "")

    proxy, proxy_reason = is_proxy_contract(abi, client, addr)

    # Build ground-truth function map
    gt_funcs: Dict[str, Dict] = {}
    for entry in abi:
        if entry.get("type") != "function":
            continue
        name = entry.get("name", "")
        inputs = entry.get("inputs", [])
        input_types = [i.get("type", "") for i in inputs]
        sel = selector_of(name, input_types)
        gt_funcs[sel] = {
            "name": name,
            "input_types": input_types,
            "arity": len(inputs),
        }

    # Run reconstruction
    result = reconstructor.reconstruct_abi(bytecode, max_selectors=100)
    rec_funcs = {}
    for f in result.get("functions", []):
        sel = f.get("selector", "")
        rec_funcs[sel] = [p.get("type", "uint256") for p in f.get("parameters", [])]

    # Categorize each ground-truth function
    for sel, gt in gt_funcs.items():
        base = {
            "contract": contract_name or addr[:10],
            "address": addr,
            "selector": sel,
            "function": gt["name"],
            "gt_types": gt["input_types"],
        }

        if sel not in rec_funcs:
            out[CAT_SELECTOR_MISSING].append(base)
            continue

        rec_types = rec_funcs[sel]

        if types_match(gt["input_types"], rec_types):
            out[CAT_CORRECT].append(base)
            continue

        # Failure — categorize why
        entry = dict(base, rec_types=rec_types)

        if proxy:
            entry["proxy_reason"] = proxy_reason
            out[CAT_PROXY].append(entry)
            continue

        db_sigs = db.lookup(sel, fetch_api=False)

        if not db_sigs:
            entry["rec_types"] = rec_types
            out[CAT_DB_MISS].append(entry)
        elif len(db_sigs) > 1:
            entry["db_signatures"] = [s.text_signature for s in db_sigs]
            entry["rec_types"] = rec_types
            out[CAT_DB_COLLISION].append(entry)
        else:
            entry["db_signature"] = db_sigs[0].text_signature
            entry["rec_types"] = rec_types
            out[CAT_TYPE_BUG].append(entry)

    return out


def main():
    parser = argparse.ArgumentParser(description="Gap analysis for ABI reconstruction")
    parser.add_argument("--results", default="eval_output/eval_results.json")
    parser.add_argument("--db", default="./cache/4byte.db")
    parser.add_argument("--output", default="eval_output/gap_analysis.md")
    parser.add_argument("--sample", type=int, default=0,
                        help="Analyze first N contracts (0=all)")
    args = parser.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    addresses = [c["address"] for c in results.get("per_contract", []) if c.get("address")]
    if args.sample:
        addresses = addresses[:args.sample]

    client = EtherscanClient()
    reconstructor = OfflineABI(db_path=args.db)
    db = FourByteDatabase(db_path=args.db)

    all_categories: Dict[str, List[Dict]] = {
        CAT_SELECTOR_MISSING: [],
        CAT_DB_MISS: [],
        CAT_DB_COLLISION: [],
        CAT_PROXY: [],
        CAT_TYPE_BUG: [],
        CAT_CORRECT: [],
    }

    for i, addr in enumerate(addresses):
        print(f"  [{i+1}/{len(addresses)}] {addr[:10]}...")
        contract_cats = analyze_contract(addr, client, reconstructor, db)
        for cat in all_categories:
            all_categories[cat].extend(contract_cats[cat])

    reconstructor.close()
    db.close()

    total = sum(len(v) for v in all_categories.values())
    correct = len(all_categories[CAT_CORRECT])
    failures = total - correct

    # Generate report
    lines = [
        "# Gap Analysis — ABI Reconstructor",
        "",
        f"**{len(addresses)} contracts · {total} ground-truth functions**",
        f"**{correct} correct ({correct/total*100:.1f}%) · {failures} failures ({failures/total*100:.1f}%)**",
        "",
        "## Failure Breakdown",
        "",
        "| Category | Count | % of Gap | Fixable? |",
        "|---|---|---|---|",
    ]

    for cat, fix in [
        (CAT_SELECTOR_MISSING, "Maybe (parser improvement)"),
        (CAT_DB_MISS, "Yes (ML or DB expansion)"),
        (CAT_DB_COLLISION, "Yes (signature disambiguation)"),
        (CAT_PROXY, "No (structural)"),
        (CAT_TYPE_BUG, "Bug — investigate"),
    ]:
        n = len(all_categories[cat])
        pct = n / failures * 100 if failures else 0
        lines.append(f"| {cat} | {n} | {pct:.1f}% | {fix} |")

    fixable_c = len(all_categories[CAT_DB_MISS]) + len(all_categories[CAT_DB_COLLISION])
    maybe_c = len(all_categories[CAT_SELECTOR_MISSING])
    structural_c = len(all_categories[CAT_PROXY])
    bug_c = len(all_categories[CAT_TYPE_BUG])

    lines += [
        "",
        "## Fixability Summary",
        "",
        f"- **Fixable** (DB miss or collision): {fixable_c} ({fixable_c/failures*100:.1f}% of gap) — these can be resolved by ML type prediction or signature disambiguation",
        f"- **Maybe fixable** (selector extraction): {maybe_c} ({maybe_c/failures*100:.1f}% of gap) — these selectors exist in bytecode but our parser misses them",
        f"- **Structural** (proxy contracts): {structural_c} ({structural_c/failures*100:.1f}% of gap) — proxy bytecode selectors ≠ logical ABI, not fixable without implementation resolution",
        f"- **Bugs**: {bug_c} ({bug_c/failures*100:.1f}% of gap) — type mismatch despite single DB entry, needs investigation",
        "",
        "## Projected improvement",
        "",
    ]

    if failures > 0:
        current_proto = correct / total
        if fixable_c > 0:
            after_fix = (correct + fixable_c) / total
            lines.append(f"- After fixing DB misses + collisions: prototype accuracy {correct/total:.3f} → {after_fix:.3f}")
        if maybe_c > 0:
            after_all = (correct + fixable_c + maybe_c) / total
            lines.append(f"- After selector extraction improvement: → {after_all:.3f}")

    # Examples
    for cat in [CAT_SELECTOR_MISSING, CAT_DB_MISS, CAT_DB_COLLISION, CAT_PROXY, CAT_TYPE_BUG]:
        entries = all_categories[cat]
        if not entries:
            continue
        lines.append(f"")
        lines.append(f"## {cat} ({len(entries)} examples)")
        lines.append("")
        for e in entries[:8]:
            lines.append(f"- `{e['contract']}.{e['function']}({','.join(e['gt_types'])})`")
            lines.append(f"  - selector `{e['selector']}`")
            rec = e.get("rec_types")
            if rec:
                lines.append(f"  - reconstructed as `({','.join(rec)})`")
            db_sig = e.get("db_signature")
            if db_sig:
                lines.append(f"  - DB signature: `{db_sig}`")
            db_sigs = e.get("db_signatures")
            if db_sigs:
                lines.append(f"  - DB returns {len(db_sigs)} signatures: {db_sigs[:3]}")
            proxy_r = e.get("proxy_reason")
            if proxy_r:
                lines.append(f"  - proxy reason: {proxy_r}")
            lines.append("")
        if len(entries) > 8:
            lines.append(f"  ... and {len(entries) - 8} more like this")
            lines.append("")

    report = "\n".join(lines)
    with open(args.output, "w") as f:
        f.write(report)

    print(f"\nGap analysis → {args.output}")
    print(f"Total: {total} · Correct: {correct} ({correct/total*100:.1f}%) · Failures: {failures} ({failures/total*100:.1f}%)")
    print(f"  A-Selector: {len(all_categories[CAT_SELECTOR_MISSING])}")
    print(f"  B-DB miss:  {len(all_categories[CAT_DB_MISS])}")
    print(f"  C-Collision:{len(all_categories[CAT_DB_COLLISION])}")
    print(f"  D-Proxy:    {len(all_categories[CAT_PROXY])}")
    print(f"  E-Type bug: {len(all_categories[CAT_TYPE_BUG])}")


if __name__ == "__main__":
    main()
