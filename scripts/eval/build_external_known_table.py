"""Build known interface sets from training-only data.

Discovers groups of functions that appear together in verified ABIs but lack
bytecode evidence (PUSH4+EQ extraction misses them). For each group, finds
trigger selectors — bytecode selectors that strongly co-occur — to enable
conservative interface completion at reconstruction time.

Usage:
    python scripts/eval/build_external_known_table.py

Output: data/known_interface_sets.json
"""

from __future__ import annotations

import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from abifusion.fusion import parse_signature, split_args
from scripts.eval._shared import (
    split_dataframe,
    true_functions,
    resolve_data_path,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MIN_TRIGGER_PRECISION = 0.95
MIN_TRIGGER_RECALL = 0.95
MIN_CLUSTER_SIMILARITY = 0.6
MIN_CONTRACT_COUNT = 5


def extract_bytecode_selectors(bytecode: str) -> set[str]:
    """Extract PUSH4+EQ selectors from raw bytecode."""
    code = bytecode[2:] if bytecode.startswith("0x") else bytecode
    return set(re.findall(r"63([0-9a-f]{8})14", code))


def jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def cluster_by_similarity(
    contract_data: list[dict],
    min_similarity: float = 0.7,
    min_cluster_size: int = 2,
) -> list[list[int]]:
    """Cluster contracts by Jaccard similarity of their abi_only sets."""
    n = len(contract_data)
    sets = [set(cd["abi_only"].keys()) for cd in contract_data]

    visited = [False] * n
    clusters = []

    for i in range(n):
        if visited[i] or len(sets[i]) < 2:
            continue
        visited[i] = True
        cluster = [i]

        for j in range(i + 1, n):
            if visited[j] or len(sets[j]) < 2:
                continue
            sim = jaccard(sets[i], sets[j])
            if sim >= min_similarity:
                visited[j] = True
                cluster.append(j)

        if len(cluster) >= min_cluster_size:
            clusters.append(cluster)

    return clusters


def build_interface_sets(
    train_df: pd.DataFrame,
) -> list[dict]:
    """Discover interface sets and their trigger selectors from training data.

    An "interface set" is a group of ABI functions that:
    - Appear together in multiple training contracts
    - Are never found in bytecode via PUSH4+EQ extraction

    Returns list of interface sets with:
    - name: human-readable label
    - functions: list of {selector, name, types}
    - triggers: {selector_set, min_matches}
    - precision, recall: measured on training data
    """

    all_bc_sel_freq: dict[str, int] = defaultdict(int)
    for _, row in train_df.iterrows():
        bc_sels = extract_bytecode_selectors(row["bytecode"])
        for s in bc_sels:
            all_bc_sel_freq[s] += 1

    total_contracts = len(train_df)

    contract_data: list[dict] = []
    for _, row in train_df.iterrows():
        abi_funcs = true_functions(row["abi"])
        bc_sels = extract_bytecode_selectors(row["bytecode"])

        abi_only = {
            sel: info
            for sel, info in abi_funcs.items()
            if sel not in bc_sels
        }

        contract_data.append({
            "address": row["address"],
            "bc_selectors": bc_sels,
            "abi_only": abi_only,
        })

    clusters = cluster_by_similarity(contract_data, min_similarity=0.6)

    candidate_interfaces = []
    for cluster_indices in clusters:
        all_sels = set()
        for ci in cluster_indices:
            all_sels |= set(contract_data[ci]["abi_only"].keys())

        min_occurrence = max(2, int(len(cluster_indices) * 0.7))
        consensus_sels = [
            sel for sel in all_sels
            if sum(1 for ci in cluster_indices if sel in contract_data[ci]["abi_only"])
               >= min_occurrence
        ]

        if len(consensus_sels) < 2:
            continue

        first = contract_data[cluster_indices[0]]
        functions = []
        for sel in sorted(consensus_sels):
            if sel in first["abi_only"]:
                fname = first["abi_only"][sel]["name"]
                ftypes = list(first["abi_only"][sel]["types"])
            else:
                for ci in cluster_indices:
                    if sel in contract_data[ci]["abi_only"]:
                        fname = contract_data[ci]["abi_only"][sel]["name"]
                        ftypes = list(contract_data[ci]["abi_only"][sel]["types"])
                        break
            functions.append({"selector": sel, "name": fname, "types": ftypes})

        candidate_interfaces.append({
            "functions": functions,
            "contract_indices": cluster_indices,
            "total_contracts_matching": len(cluster_indices),
        })

    interface_sets = []
    for candidate in candidate_interfaces:
        functions = candidate["functions"]
        cluster_indices = candidate["contract_indices"]
        n_interface = len(cluster_indices)

        function_selectors = {f["selector"] for f in functions}

        expanded_indices = set(cluster_indices)
        for ci, cd in enumerate(contract_data):
            if ci in expanded_indices:
                continue
            if function_selectors & set(cd["abi_only"].keys()):
                expanded_indices.add(ci)

        all_indices = sorted(expanded_indices)
        n_expanded = len(all_indices)

        bc_sels_in_expanded = set()
        for ci in all_indices:
            bc_sels_in_expanded |= contract_data[ci]["bc_selectors"]

        trigger_candidates = []
        for sel in sorted(bc_sels_in_expanded):
            recall = sum(
                1 for ci in all_indices if sel in contract_data[ci]["bc_selectors"]
            ) / n_expanded

            non_interface_with = sum(
                1
                for ci, cd in enumerate(contract_data)
                if ci not in all_indices and sel in cd["bc_selectors"]
            )
            if n_expanded + non_interface_with == 0:
                continue
            precision = n_expanded / (n_expanded + non_interface_with)

            if recall >= MIN_TRIGGER_RECALL and precision >= MIN_TRIGGER_PRECISION:
                trigger_candidates.append({
                    "selector": sel,
                    "recall": recall,
                    "precision": precision,
                })

        if not trigger_candidates or n_expanded < MIN_CONTRACT_COUNT:
            logger.info(
                "no triggers found for interface set with %d functions, %d contracts",
                len(functions), n_expanded,
            )
            continue

        trigger_candidates.sort(key=lambda t: (t["precision"], t["recall"]), reverse=True)
        trigger_sel_set = [t["selector"] for t in trigger_candidates]

        max_recall = max(t["recall"] for t in trigger_candidates)
        min_precision = min(t["precision"] for t in trigger_candidates)

        min_matches = len(trigger_sel_set)
        for candidate_min in range(2, len(trigger_sel_set) + 1):
            n_triggered = sum(
                1 for ci in all_indices
                if len(set(trigger_sel_set[:candidate_min]) & contract_data[ci]["bc_selectors"]) >= candidate_min
            )
            if n_triggered / n_expanded >= MIN_TRIGGER_RECALL:
                min_matches = candidate_min
                break

        # Build interface set name from function names
        func_names = sorted({f["name"] for f in functions})
        if len(func_names) == 1:
            iface_name = func_names[0]
        elif len(func_names) <= 3:
            iface_name = "_".join(func_names)
        else:
            iface_name = func_names[0] + "_group"

        interface_sets.append({
            "name": iface_name,
            "functions": functions,
            "triggers": {
                "selector_set": trigger_sel_set,
                "min_matches": min_matches,
            },
            "precision": min_precision,
            "recall": max_recall,
            "contract_count": n_expanded,
        })

    return interface_sets


def main() -> None:
    data_path = resolve_data_path("data/contracts.parquet", REPO_ROOT)
    output_path = REPO_ROOT / "data" / "known_interface_sets.json"

    df = pd.read_parquet(data_path)
    train_df, heldout_df = split_dataframe(df, heldout_fraction=0.2, seed=42)

    logger.info("training contracts: %d, heldout contracts: %d", len(train_df), len(heldout_df))

    interface_sets = build_interface_sets(train_df)

    logger.info("discovered %d interface sets", len(interface_sets))
    for iface in interface_sets:
        func_count = len(iface["functions"])
        trigger_count = len(iface["triggers"]["selector_set"])
        func_names = ", ".join(f["name"] for f in iface["functions"][:5])
        if func_count > 5:
            func_names += f" ... (+{func_count - 5})"
        logger.info(
            "  %s: %d functions [%s], %d triggers, P=%.2f R=%.2f, %d contracts",
            iface["name"], func_count, func_names, trigger_count,
            iface["precision"], iface["recall"], iface["contract_count"],
        )

    with open(output_path, "w") as f:
        json.dump(interface_sets, f, indent=2)
    logger.info("wrote %s (%d interface sets)", output_path, len(interface_sets))


if __name__ == "__main__":
    main()
