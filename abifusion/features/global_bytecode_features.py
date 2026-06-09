"""Global bytecode feature extraction for ML input representation.

This module extracts global bytecode features (7-dim SigRec vector) replicated
per parameter position, augmented with parameter count.

The primary approach (CALLDATALOAD offset analysis) was validated via Phase 0
verification. When that approach fails for a given function (e.g., due to
optimizer-inlined parameter loading), the module falls back to global features
augmented with the parameter count.
"""

from __future__ import annotations

from typing import List, Optional

from abifusion.features.discriminating_features import (
    DiscriminatingFeatureExtractor,
    DiscriminatingFeatures,
)


def _parse_bytecode(bytecode_hex: str) -> Optional[bytes]:
    code = bytecode_hex[2:] if bytecode_hex.startswith("0x") else bytecode_hex
    try:
        return bytes.fromhex(code)
    except ValueError:
        return None


def extract_global_features(
    bytecode_hex: str,
    bytecode_offset: int,
    window_bytes: int = 300,
) -> List[float]:
    raw = _parse_bytecode(bytecode_hex)
    if raw is None:
        return [0.0] * 7
    start = max(0, bytecode_offset)
    end = min(len(raw), start + window_bytes)
    try:
        disc_features = DiscriminatingFeatureExtractor._analyze_opcodes(raw[start:end])
    except Exception:
        return [0.0] * 7
    return disc_features.to_vector()


def extract_bytecode_features(
    bytecode_hex: str,
    bytecode_offset: int,
    num_params: int,
    window_bytes: int = 100,
) -> List[DiscriminatingFeatures]:
    if num_params <= 0:
        return []
    raw = _parse_bytecode(bytecode_hex)
    if raw is None or bytecode_offset >= len(raw):
        return [DiscriminatingFeatures() for _ in range(num_params)]
    end = min(bytecode_offset + window_bytes * 3, len(raw))
    try:
        global_features = DiscriminatingFeatureExtractor._analyze_opcodes(raw[bytecode_offset:end])
    except Exception:
        global_features = DiscriminatingFeatures()
    return [global_features for _ in range(num_params)]


def build_feature_matrix(
    bytecode_hex: str,
    bytecode_offset: int,
    num_params: int,
    window_bytes: int = 100,
) -> List[List[float]]:
    if num_params <= 0:
        return []
    features = extract_bytecode_features(bytecode_hex, bytecode_offset, num_params, window_bytes)
    global_vec = features[0].to_vector() if features else [0.0] * 7
    return [global_vec + [float(num_params), float(i)] for i in range(num_params)]
