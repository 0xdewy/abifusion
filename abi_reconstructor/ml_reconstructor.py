"""ML-based ABI reconstructor using function family classification.

This module provides an MLReconstructor class that predicts:
- Function family (Top-3) from bytecode (e.g., "ERC20_BASIC", "AMM_SWAP")
- Per-position parameter types (4-class: address, uint-like, bytes-like, other)

The model uses a 1D CNN bytecode encoder + per-position feature fusion + two
classification heads. It was trained on 5,667 functions from 400 contracts and
achieves 95.4% Top-3 family accuracy on the zero-shot test set.

The 4-class type vocabulary:
- address: address, uint160
- uint-like: uint256, uint128, uint8, int256, etc.
- bytes-like: bytes, bytes32, bytes<M>
- other: bool, string, arrays, tuple
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from abi_reconstructor.ml.families import FAMILY_NAMES, get_family_name
from abi_reconstructor.ml.model import MLAbiModel


class MLReconstructor:
    _instance: Optional["MLReconstructor"] = None

    def __init__(self):
        self.model: Optional[MLAbiModel] = None
        self.model_path: Optional[str] = None
        self.device: str = "cpu"
        self._loaded = False

    @classmethod
    def get_instance(cls) -> "MLReconstructor":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_model(self, model_path: str) -> None:
        self.model_path = model_path
        try:
            import pathlib

            if not pathlib.Path(model_path).exists():
                raise FileNotFoundError(f"Model not found at {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model = MLAbiModel(
                num_name_classes=len(FAMILY_NAMES),
                num_type_classes=4,
                embed_dim=64,
                num_filters=128,
                max_params=16,
            )
            self.model.load_state_dict(checkpoint, strict=False)
            self.model.to(self.device)
            self.model.eval()
            self._loaded = True
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}") from e

    def _build_feature_matrix(self, bytecode: str, max_params: int = 16) -> Tuple[List[List[float]], List[int]]:
        from abi_reconstructor.features.global_bytecode_features import (
            build_feature_matrix,
        )
        from abi_reconstructor.utils.bytecode import parse_bytecode_hex

        code = parse_bytecode_hex(bytecode)
        if code is None:
            return [[0.0] * 8 for _ in range(max_params)], [0] * 5000

        bytecode_hex = bytecode[2:] if bytecode.startswith("0x") else bytecode

        evmole_offset = 0
        try:
            import evmole

            info = evmole.contract_info(bytecode_hex, selectors=True, arguments=True)
            if info is not None and info.functions is not None and info.functions:
                evmole_offset = info.functions[0].bytecode_offset
        except Exception:
            pass

        feature_mat = build_feature_matrix(bytecode_hex, evmole_offset, max_params, window_bytes=100)
        tokens = list(code[max(0, evmole_offset) : min(len(code), evmole_offset + 5000)])
        while len(tokens) < 5000:
            tokens.append(0)
        return feature_mat, tokens

    def predict(self, bytecode: str) -> Dict[str, Any]:
        if self.model is None or not self._loaded:
            return self._stub_response(False)

        feature_mat, tokens = self._build_feature_matrix(bytecode)
        max_params = 16
        max_len = 5000

        fm = feature_mat[:max_params]
        while len(fm) < max_params:
            fm.append([0.0] * 8)
        while len(tokens) < max_len:
            tokens.append(0)

        feature_tensor = torch.tensor([fm], dtype=torch.float32, device=self.device)
        token_tensor = torch.tensor([tokens[:max_len]], dtype=torch.long, device=self.device)

        with torch.no_grad():
            name_logits, type_logits = self.model(feature_tensor, token_tensor)

        name_probs = F.softmax(name_logits, dim=-1)
        top3_idx = name_logits[0].topk(3).indices.tolist()
        top3_families = [get_family_name(idx) for idx in top3_idx]
        top3_probs = [name_probs[0, idx].item() for idx in top3_idx]

        num_params = min(max_params, len(feature_mat))
        type_preds = []
        type_probs_list = []
        for p in range(num_params):
            type_logits_pos = type_logits[0, p]
            type_probs = F.softmax(type_logits_pos, dim=-1)
            type_pred = type_logits_pos.argmax().item()
            type_conf = type_probs[type_pred].item()
            type_name = self._type_idx_to_name(type_pred)
            type_preds.append(type_name)
            type_probs_list.append(type_conf)

        return {
            "source": "ml",
            "family": top3_families[0],
            "family_top3": top3_families,
            "family_probs": top3_probs,
            "types": type_preds,
            "type_confidences": type_probs_list,
            "loaded": self._loaded,
            "model_path": self.model_path,
        }

    def _stub_response(self, loaded: bool) -> Dict[str, Any]:
        return {
            "source": "ml_stub",
            "family": None,
            "family_top3": [],
            "family_probs": [],
            "types": [],
            "type_confidences": [],
            "loaded": loaded,
            "model_path": self.model_path,
        }

    def _type_idx_to_name(self, idx: int) -> str:
        names = ["address", "uint-like", "bytes-like", "other"]
        return names[idx] if idx < len(names) else "other"

    def predict_from_features(
        self,
        feature_matrix: List[List[float]],
        bytecode_tokens: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        if self.model is None or not self._loaded:
            return self._stub_response(False)
        raise NotImplementedError("Use predict(bytecode) instead")
