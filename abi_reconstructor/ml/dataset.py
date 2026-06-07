"""ML dataset for zero-shot ABI reconstruction.

Prepares training data from contracts.parquet:
- Train: first 400 contracts (~5,667 functions)
- Val: next 50 contracts
- Test: last 50 contracts +57 unique-selector functions

Each sample: (feature_matrix, bytecode_tokens, function_name_label, type_labels)
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from abi_reconstructor.features.global_bytecode_features import build_feature_matrix
from abi_reconstructor.utils.bytecode import parse_bytecode_hex


def compute_selector(func_sig: str) -> str:
    from Crypto.Hash import keccak
    k = keccak.new(digest_bits=256)
    k.update(func_sig.encode())
    return k.hexdigest()[:8]


def sig_from_abi_func(func: Dict[str, Any]) -> str:
    inputs = ",".join([i["type"] for i in func.get("inputs", [])])
    return f"{func['name']}({inputs})"


def coarse_type(abi_type: str) -> int:
    if abi_type in ("address", "uint160"):
        return 0
    if abi_type in ("uint256", "uint128", "uint8", "uint16", "uint32", "uint64", "uint112", "uint224", "int256", "int128", "int8", "int16", "int32", "int64", "int112", "int224"):
        return 1
    if abi_type.startswith("bytes") or abi_type.startswith("bytes32"):
        return 2
    return 3


class FunctionSample:
    __slots__ = ("selector", "name", "types", "bytecode_offset", "feature_matrix", "bytecode_tokens", "coarse_types")

    def __init__(
        self,
        selector: str,
        name: str,
        types: List[str],
        bytecode_offset: int,
        feature_matrix: List[List[float]],
        bytecode_tokens: List[int],
        coarse_types: List[int],
    ):
        self.selector = selector
        self.name = name
        self.types = types
        self.bytecode_offset = bytecode_offset
        self.feature_matrix = feature_matrix
        self.bytecode_tokens = bytecode_tokens
        self.coarse_types = coarse_types


class ContractDataset(Dataset):
    def __init__(
        self,
        contracts_df: pd.DataFrame,
        name_vocab: Dict[str, int],
        max_params: int = 16,
        window_bytes: int = 100,
        max_len: int = 5000,
        use_families: bool = False,
    ):
        self.contracts_df = contracts_df.reset_index(drop=True)
        self.name_vocab = name_vocab
        self.max_params = max_params
        self.window_bytes = window_bytes
        self.max_len = max_len
        self.use_families = use_families
        self.samples: List[FunctionSample] = []
        self._build()

    def _build(self) -> None:
        for idx in range(len(self.contracts_df)):
            row = self.contracts_df.iloc[idx]
            self._process_contract(idx, row)

    def _process_contract(self, idx: int, row: pd.Series) -> None:
        bytecode_hex = row["bytecode"]
        if bytecode_hex.startswith("0x"):
            bytecode_hex = bytecode_hex[2:]
        code = parse_bytecode_hex(bytecode_hex)
        if code is None:
            return

        abi = row["abi"]
        funcs = [f for f in abi if f.get("type") == "function"]

        evmole_types: Dict[str, Tuple[str, ...]] = {}
        evmole_offsets: Dict[str, int] = {}
        try:
            import evmole

            from abi_reconstructor.fusion import split_args

            info = evmole.contract_info(bytecode_hex, selectors=True, arguments=True)
            if info is not None and info.functions is not None:
                for f in info.functions:
                    evmole_types[f.selector.lower()] = split_args(f.arguments or "")
                    evmole_offsets[f.selector.lower()] = f.bytecode_offset
        except Exception:
            pass

        for func in funcs:
            sig = sig_from_abi_func(func)
            sel = compute_selector(sig)
            name = func["name"]
            abi_types = [i["type"] for i in func.get("inputs", [])]
            num_params = len(abi_types)

            offset = evmole_offsets.get(sel.lower(),0)
            feature_mat = build_feature_matrix(bytecode_hex, offset, num_params, self.window_bytes)

            tokens = self._tokenize_bytecode(code, offset)
            coarse = [coarse_type(t) for t in abi_types]

            self.samples.append(
                FunctionSample(
                    selector=sel,
                    name=name,
                    types=abi_types,
                    bytecode_offset=offset,
                    feature_matrix=feature_mat,
                    bytecode_tokens=tokens,
                    coarse_types=coarse,
                )
            )

    def _tokenize_bytecode(self, code: bytes, offset: int) -> List[int]:
        start = max(0, offset)
        end = min(len(code), offset + self.max_len)
        window = code[start:end]
        return list(window)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        feature_mat = sample.feature_matrix[: self.max_params]
        while len(feature_mat) < self.max_params:
            feature_mat.append([0.0] * 9)
        feature_tensor = torch.tensor(feature_mat, dtype=torch.float32)

        tokens = sample.bytecode_tokens[: self.max_len]
        while len(tokens) < self.max_len:
            tokens.append(0)
        token_tensor = torch.tensor(tokens, dtype=torch.long)

        if self.use_families:
            from abi_reconstructor.ml.families import get_family
            label = get_family(sample.name)
        else:
            label = self.name_vocab.get(sample.name, 0)
        name_tensor = torch.tensor(label, dtype=torch.long)

        coarse = sample.coarse_types[: self.max_params]
        while len(coarse) < self.max_params:
            coarse.append(0)
        type_tensor = torch.tensor(coarse, dtype=torch.long)

        return feature_tensor, token_tensor, name_tensor, type_tensor


def build_vocabs(samples: List[FunctionSample]) -> Tuple[Dict[str, int], int]:
    name_counts = Counter(s.name for s in samples)
    names = ["<unk>"] + [n for n, _ in name_counts.most_common()]
    name_vocab = {n: i for i, n in enumerate(names)}
    return name_vocab, len(names)


def prepare_test_set(
    contracts_df: pd.DataFrame,
    name_vocab: Dict[str, int],
    test_selectors: List[str],
    max_params: int = 16,
    window_bytes: int = 100,
    max_len: int = 5000,
) -> List[FunctionSample]:
    samples: List[FunctionSample] = []
    for idx in range(len(contracts_df)):
        row = contracts_df.iloc[idx]
        bytecode_hex = row["bytecode"]
        if bytecode_hex.startswith("0x"):
            bytecode_hex = bytecode_hex[2:]
        code = parse_bytecode_hex(bytecode_hex)
        if code is None:
            continue

        abi = row["abi"]
        funcs = [f for f in abi if f.get("type") == "function"]

        evmole_offsets: Dict[str, int] = {}
        try:
            import evmole

            info = evmole.contract_info(bytecode_hex, selectors=True, arguments=True)
            if info is not None and info.functions is not None:
                for f in info.functions:
                    evmole_offsets[f.selector.lower()] = f.bytecode_offset
        except Exception:
            pass

        for func in funcs:
            sig = sig_from_abi_func(func)
            sel = compute_selector(sig)
            if sel.lower() not in test_selectors:
                continue
            name = func["name"]
            abi_types = [i["type"] for i in func.get("inputs", [])]
            num_params = len(abi_types)
            offset = evmole_offsets.get(sel.lower(), 0)
            feature_mat = build_feature_matrix(bytecode_hex, offset, num_params, window_bytes)
            tokens = list(code[max(0, offset) : min(len(code), offset + max_len)])
            while len(tokens) < max_len:
                tokens.append(0)
            coarse = [coarse_type(t) for t in abi_types]
            samples.append(
                FunctionSample(
                    selector=sel,
                    name=name,
                    types=abi_types,
                    bytecode_offset=offset,
                    feature_matrix=feature_mat,
                    bytecode_tokens=tokens[:max_len],
                    coarse_types=coarse,
                )
            )
    return samples


def load_contracts(path: str = "data/contracts.parquet") -> pd.DataFrame:
    return pd.read_parquet(path)
