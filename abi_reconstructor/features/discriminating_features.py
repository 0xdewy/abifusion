"""Discriminating-instruction feature extractor for EVM bytecode.

Implements SigRec-inspired analysis: identifies EVM instructions near function
selectors that reveal parameter type information (AND masking, SIGNEXTEND,
signed math, BYTE access). These features connect to SigRec rules R11-R18
(Chen et al., 2021, IEEE TSE).

References:
  - Chen et al. "SigRec: Automatic Recovery of Function Signatures in Smart
    Contracts." IEEE TSE, 2021. DOI: 10.1109/tse.2021.3078342
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# EVM opcodes relevant to type discrimination
AND = 0x16
SIGNEXTEND = 0x0B
BYTE_OP = 0x1A
SDIV = 0x05
SMOD = 0x07
SLT = 0x12
SGT = 0x13
ISZERO = 0x15
CALLDATALOAD = 0x35
CALLDATACOPY = 0x37
MLOAD = 0x51
MSTORE = 0x52
ADD = 0x01
MUL = 0x02
SUB = 0x03
DIV = 0x04
EQ = 0x14
JUMPI = 0x57
JUMP = 0x56
PUSH1 = 0x60
PUSH32 = 0x7F


@dataclass
class DiscriminatingFeatures:
    """Structured features extracted from bytecode around a function entry point.

    Each boolean flag indicates whether the corresponding type-revealing EVM
    instruction was observed in the parameter-handling bytecode near the
    function selector.
    """
    has_and_mask_leading_zeros: bool = False    # uint<M> pattern (R11)
    has_and_mask_trailing_zeros: bool = False   # bytes<M> pattern (R12)
    has_signextend: bool = False                # int<M> pattern (R13)
    has_signed_math: bool = False               # int256 pattern (R15): SDIV/SMOD/SLT/SGT
    has_byte_instruction: bool = False          # bytes32 pattern (R18)
    has_double_iszero: bool = False             # bool pattern (R14)
    has_address_pattern: bool = False           # address pattern (R16): 20-byte value, no math
    and_mask_value: int = 0                     # the AND mask constant observed
    and_mask_bytes: int = 0                     # number of non-zero bytes in the mask

    def to_vector(self) -> List[float]:
        """Convert to a feature vector for ML models."""
        return [
            float(self.has_and_mask_leading_zeros),
            float(self.has_and_mask_trailing_zeros),
            float(self.has_signextend),
            float(self.has_signed_math),
            float(self.has_byte_instruction),
            float(self.has_double_iszero),
            float(self.has_address_pattern),
        ]

    def to_dict(self) -> Dict[str, object]:
        return {
            "has_and_mask_leading_zeros": self.has_and_mask_leading_zeros,
            "has_and_mask_trailing_zeros": self.has_and_mask_trailing_zeros,
            "has_signextend": self.has_signextend,
            "has_signed_math": self.has_signed_math,
            "has_byte_instruction": self.has_byte_instruction,
            "has_double_iszero": self.has_double_iszero,
            "has_address_pattern": self.has_address_pattern,
        }


class DiscriminatingFeatureExtractor:
    """Analyze EVM bytecode to extract type-discriminating instruction patterns.

    Operates on a window of bytecode around a function selector. Detects the
    instruction patterns that SigRec's rules R11-R18 describe for distinguishing
    uint, int, bytes, address, and bool parameter types.
    """

    @staticmethod
    def extract_from_bytecode(
        bytecode_hex: str,
        selector: str,
        window_bytes: int = 300,
    ) -> DiscriminatingFeatures:
        """Extract discriminating features from bytecode near a selector.

        Args:
            bytecode_hex: Clean hex bytecode (no 0x prefix).
            selector: 8-char hex selector string.
            window_bytes: Number of bytes of context around the selector.

        Returns:
            DiscriminatingFeatures with observed type-revealing patterns.
        """
        pos = bytecode_hex.lower().find(selector.lower())
        if pos == -1:
            return DiscriminatingFeatures()

        # Get window around the selector
        start = max(0, pos - window_bytes // 2)
        end = min(len(bytecode_hex), pos + window_bytes // 2)

        # Parse raw bytes
        raw = bytes.fromhex(bytecode_hex[start:end])
        return DiscriminatingFeatureExtractor._analyze_opcodes(raw)

    @staticmethod
    def extract_from_context(
        bytecode_hex: str,
        context: str,
    ) -> DiscriminatingFeatures:
        """Extract discriminating features from a bytecode context string.

        Args:
            bytecode_hex: Full clean bytecode (for broader analysis).
            context: Hex context string around a function selector.

        Returns:
            DiscriminatingFeatures with observed type-revealing patterns.
        """
        raw = bytes.fromhex(context)
        return DiscriminatingFeatureExtractor._analyze_opcodes(raw)

    @staticmethod
    def _analyze_opcodes(raw: bytes) -> DiscriminatingFeatures:
        """Scan a byte sequence for discriminating EVM instructions.

        Walks the byte sequence instruction-by-instruction, tracking:
        - PUSH instructions and their pushed data (for AND mask detection)
        - Opcodes that reveal types (SIGNEXTEND, BYTE, SDIV, SMOD, SLT, SGT)
        - ISZERO repetition (bool pattern)
        """
        features = DiscriminatingFeatures()
        i = 0
        n = len(raw)
        last_was_iszero = False
        last_push_data: Optional[bytes] = None
        last_push_size: int = 0

        while i < n:
            op = raw[i]

            # Track PUSH data for AND mask detection
            if PUSH1 <= op <= PUSH32:
                push_size = op - PUSH1 + 1
                if i + 1 + push_size <= n:
                    last_push_data = raw[i + 1 : i + 1 + push_size]
                    last_push_size = push_size
                else:
                    last_push_data = None
                i += 1 + push_size
                continue

            # AND after a PUSH → check the mask
            if op == AND and last_push_data is not None:
                mask_int = int.from_bytes(last_push_data, "big")
                if mask_int != 0 and last_push_size <= 32:
                    features.and_mask_value = mask_int
                    # Count non-zero bytes in the mask
                    features.and_mask_bytes = sum(1 for b in last_push_data if b != 0)
                    # Leading zeros → uint (value is right-aligned in the word)
                    # Trailing zeros → bytes (value is left-aligned in the word)
                    mask_hex = last_push_data.hex()
                    if mask_hex.startswith("00" * (last_push_size - features.and_mask_bytes)):
                        features.has_and_mask_leading_zeros = True
                    if mask_hex.endswith("00" * (last_push_size - features.and_mask_bytes)):
                        features.has_and_mask_trailing_zeros = True

            # SIGNEXTEND → signed integer
            elif op == SIGNEXTEND:
                features.has_signextend = True

            # BYTE → bytes type access
            elif op == BYTE_OP:
                features.has_byte_instruction = True

            # Signed math → int256
            elif op in (SDIV, SMOD, SLT, SGT):
                features.has_signed_math = True

            # ISZERO → track for double-ISZERO (bool pattern)
            elif op == ISZERO:
                if last_was_iszero:
                    features.has_double_iszero = True
                last_was_iszero = True
            else:
                last_was_iszero = False

            # Reset push data after non-PUSH instruction
            last_push_data = None
            i += 1

        return features

    @staticmethod
    def infer_types_from_features(
        features: DiscriminatingFeatures,
        default: str = "uint256",
    ) -> List[str]:
        """Infer parameter types from discriminating features.

        Uses the same logic as SigRec's rules R4, R11-R18 to determine the
        most likely Solidity type from observed instruction patterns.

        Args:
            features: Extracted discriminating features.
            default: Default type when no discriminating signal is present.

        Returns:
            List of inferred type strings (may be empty if nothing found).
        """
        types = []

        # Order of specificity (most specific first):
        if features.has_double_iszero:
            types.append("bool")
        if features.has_signextend:
            types.append("int256")  # Could be int<N> but default to int256
        if features.has_signed_math:
            types.append("int256")
        if features.has_byte_instruction:
            types.append("bytes32")
        if features.has_and_mask_trailing_zeros and not features.has_and_mask_leading_zeros:
            mask_bytes = features.and_mask_bytes
            types.append(f"bytes{mask_bytes}" if mask_bytes <= 32 else "bytes")
        if features.has_and_mask_leading_zeros:
            mask_bytes = features.and_mask_bytes
            if mask_bytes == 20 and not features.has_signed_math:
                types.append("address")
            else:
                types.append(f"uint{mask_bytes * 8}" if mask_bytes <= 32 else "uint256")
        if features.has_address_pattern:
            types.append("address")

        if not types:
            types.append(default)

        return types
