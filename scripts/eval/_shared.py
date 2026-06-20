from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from eth_utils import keccak


def resolve_data_path(arg: str, default: Path) -> Path:
    p = Path(arg)
    return p if p.is_absolute() else default / p


def normalize_abi(abi: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(abi, np.ndarray):
        return abi.tolist()
    if isinstance(abi, str):
        return json.loads(abi)
    return abi


def true_functions(abi: Any) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for item in normalize_abi(abi):
        if item.get("type") != "function":
            continue
        types = tuple(inp["type"] for inp in item.get("inputs", []))
        signature = f"{item.get('name', '')}({','.join(types)})"
        selector = keccak(text=signature).hex()[:8]
        out[selector] = {"name": item.get("name", ""), "types": types}
    return out


def split_dataframe(
    df: pd.DataFrame, heldout_fraction: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.RandomState(seed)
    heldout_size = int(len(df) * heldout_fraction)
    heldout_idx = rng.choice(len(df), heldout_size, replace=False)
    train_idx = np.setdiff1d(np.arange(len(df)), heldout_idx)
    return df.iloc[train_idx], df.iloc[heldout_idx]


def build_known_selector_table(train_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    selector_sigs: Dict[str, list] = defaultdict(list)
    for _, row in train_df.iterrows():
        for selector, info in true_functions(row["abi"]).items():
            selector_sigs[selector].append((info["name"], info["types"]))

    known_table: Dict[str, Dict[str, Any]] = {}
    for selector, occurrences in selector_sigs.items():
        if len(occurrences) < 2:
            continue
        names = {name for name, _ in occurrences}
        type_sets = {types for _, types in occurrences}
        if len(names) == 1 and len(type_sets) == 1:
            name, types = occurrences[0]
            known_table[selector] = {"name": name, "types": list(types)}
    return known_table


def accuracy(correct: int, total: int) -> float:
    return 100.0 * correct / total if total else 0.0


def extract_bytecode_metadata(
    bytecode: str,
    fusion: "ABIFusion",
    truth: Dict[str, Dict[str, Any]],
) -> tuple[
    set[str],
    Dict[str, Optional[Tuple[str, ...]]],
    Dict[str, list],
]:
    code = bytecode[2:] if bytecode.startswith("0x") else bytecode

    bytecode_selectors: set[str] = set()
    try:
        from tooling.abifusion_legacy.reconstructor import BytecodeParser

        for s in BytecodeParser().extract_selectors(bytecode):
            bytecode_selectors.add(s.selector.lower())
    except Exception:
        pass

    evmole_types_map: Dict[str, Optional[Tuple[str, ...]]] = {}
    try:
        import evmole
        from abifusion.fusion import split_args

        info = evmole.contract_info(code, selectors=True, arguments=True)
        if info and info.functions:
            for f in info.functions:
                evmole_types_map[f.selector.lower()] = split_args(f.arguments or "")
    except Exception:
        pass

    sig_db_candidates_map: Dict[str, list] = {}
    for selector in truth:
        from abifusion.fusion import parse_signature

        rows = fusion.sig.lookup_openchain(selector) or fusion.sig.lookup_4byte(selector)
        parsed_candidates = []
        for s in rows:
            p = parse_signature(s.get("text_signature", ""))
            if p is not None:
                parsed_candidates.append(p)
        sig_db_candidates_map[selector] = parsed_candidates

    return bytecode_selectors, evmole_types_map, sig_db_candidates_map
