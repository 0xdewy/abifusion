"""Tests for discriminating-instruction feature extraction."""

import pytest
from abifusion.features.discriminating_features import (
    DiscriminatingFeatures,
    DiscriminatingFeatureExtractor,
)


class TestDiscriminatingFeatureExtractor:
    """Test the EvM bytecode opcode analysis for type-discriminating features."""

    def test_uint_mask_leading_zeros(self):
        """AND mask with leading zeros → uint type (SigRec R11)."""
        ext = DiscriminatingFeatureExtractor()
        # PUSH32 <32 bytes of mask> AND
        # uint32 mask: 0x00000000...00000000FFFFFFFF (leading zeros)
        mask_hex = "00000000000000000000000000000000000000000000000000000000ffffffff"
        raw = bytes.fromhex("7f" + mask_hex + "16")
        features = ext._analyze_opcodes(raw)

        assert features.has_and_mask_leading_zeros is True
        assert features.has_and_mask_trailing_zeros is False
        assert features.and_mask_bytes == 4

        types = ext.infer_types_from_features(features)
        assert "uint32" in types

    def test_bytes_mask_trailing_zeros(self):
        """AND mask with trailing zeros → bytes type (SigRec R12)."""
        ext = DiscriminatingFeatureExtractor()
        # bytes4 mask: 0xFFFFFFFF00000000... (trailing zeros)
        mask_hex = "ffffffff00000000000000000000000000000000000000000000000000000000"
        raw = bytes.fromhex("7f" + mask_hex + "16")
        features = ext._analyze_opcodes(raw)

        assert features.has_and_mask_trailing_zeros is True
        assert features.has_and_mask_leading_zeros is False
        assert features.and_mask_bytes == 4

        types = ext.infer_types_from_features(features)
        assert "bytes4" in types

    def test_signextend_signed_integer(self):
        """SIGNEXTEND instruction → signed integer (SigRec R13)."""
        ext = DiscriminatingFeatureExtractor()
        raw = bytes.fromhex("0b")
        features = ext._analyze_opcodes(raw)

        assert features.has_signextend is True
        types = ext.infer_types_from_features(features)
        assert "int256" in types

    def test_signed_math_int256(self):
        """SDIV/SMOD/SLT/SGT → int256 (SigRec R15)."""
        ext = DiscriminatingFeatureExtractor()
        for op_hex in ["05", "07", "12", "13"]:
            features = ext._analyze_opcodes(bytes.fromhex(op_hex))
            assert features.has_signed_math is True, f"opcode {op_hex}"
            types = ext.infer_types_from_features(features)
            assert "int256" in types

    def test_byte_instruction_bytes32(self):
        """BYTE instruction → bytes32 (SigRec R18)."""
        ext = DiscriminatingFeatureExtractor()
        raw = bytes.fromhex("1a")
        features = ext._analyze_opcodes(raw)

        assert features.has_byte_instruction is True
        types = ext.infer_types_from_features(features)
        assert "bytes32" in types

    def test_double_iszero_bool(self):
        """Two consecutive ISZERO → bool (SigRec R14)."""
        ext = DiscriminatingFeatureExtractor()
        raw = bytes.fromhex("1515")
        features = ext._analyze_opcodes(raw)

        assert features.has_double_iszero is True
        types = ext.infer_types_from_features(features)
        assert "bool" in types

    def test_single_iszero_not_bool(self):
        """Single ISZERO should NOT trigger bool pattern."""
        ext = DiscriminatingFeatureExtractor()
        raw = bytes.fromhex("1550")  # ISZERO followed by POP
        features = ext._analyze_opcodes(raw)

        assert features.has_double_iszero is False

    def test_address_mask_20_bytes(self):
        """AND mask with 20 non-zero bytes → address (SigRec R16)."""
        ext = DiscriminatingFeatureExtractor()
        mask_hex = "000000000000000000000000ffffffffffffffffffffffffffffffffffffffff"
        raw = bytes.fromhex("7f" + mask_hex + "16")
        features = ext._analyze_opcodes(raw)

        assert features.has_and_mask_leading_zeros is True
        assert features.and_mask_bytes == 20

        types = ext.infer_types_from_features(features)
        assert "address" in types

    def test_default_to_uint256(self):
        """No discriminating instructions → uint256 (SigRec R4 fallback)."""
        ext = DiscriminatingFeatureExtractor()
        # Just some ADD and MUL — nothing type-revealing
        raw = bytes.fromhex("0102")
        features = ext._analyze_opcodes(raw)

        types = ext.infer_types_from_features(features)
        assert types == ["uint256"]

    def test_multiple_discriminating_signals(self):
        """Multiple discriminating instructions → multiple types inferred."""
        ext = DiscriminatingFeatureExtractor()
        # Double ISZERO + SIGNEXTEND → bool + int256
        raw = bytes.fromhex("1515500b")
        features = ext._analyze_opcodes(raw)

        types = ext.infer_types_from_features(features)
        assert "bool" in types
        assert "int256" in types

    def test_to_vector(self):
        """to_vector returns correct length and values."""
        features = DiscriminatingFeatures(
            has_and_mask_leading_zeros=True,
            has_signextend=True,
            has_signed_math=True,
        )
        vec = features.to_vector()
        assert len(vec) == 7
        assert vec[0] == 1.0  # has_and_mask_leading_zeros
        assert vec[2] == 1.0  # has_signextend
        assert vec[3] == 1.0  # has_signed_math

    def test_to_dict(self):
        """to_dict returns correct keys."""
        features = DiscriminatingFeatures()
        d = features.to_dict()
        assert "has_and_mask_leading_zeros" in d
        assert "has_signextend" in d
        assert "has_double_iszero" in d

    def test_extract_from_context_handles_selector(self):
        """extract_from_context works with a hex context string."""
        ext = DiscriminatingFeatureExtractor()
        # PUSH32 <mask> AND in hex context
        mask_hex = "00000000000000000000000000000000000000000000000000000000ffffffff"
        context = "7f" + mask_hex + "16" + "50" * 10
        features = ext.extract_from_context(context, context)

        assert features.has_and_mask_leading_zeros is True

    def test_empty_bytecode(self):
        """Empty bytecode returns default features."""
        ext = DiscriminatingFeatureExtractor()
        features = ext._analyze_opcodes(b"")
        assert features.has_and_mask_leading_zeros is False
        assert features.has_signextend is False
        types = ext.infer_types_from_features(features)
        assert types == ["uint256"]


class TestParameterReconstructorWithDiscriminatingFeatures:
    """Integration tests for discriminating features in ParameterReconstructor."""

    def test_unknown_function_uses_features(self):
        """Unknown function names use discriminating instruction analysis."""
        from abifusion.reconstructor import ParameterReconstructor

        pr = ParameterReconstructor()

        # Context with SIGNEXTEND → should infer int256
        context = "0b" * 10  # SIGNEXTEND instructions
        types = pr._infer_types_from_context(
            bytecode="0b" * 20,
            selector="deadbeef",
            context=context,
        )
        assert "int256" in types

    def test_unknown_function_default_uint256(self):
        """Unknown function with no type signal defaults to uint256."""
        from abifusion.reconstructor import ParameterReconstructor

        pr = ParameterReconstructor()

        # Context with only generic opcodes (ADD, MUL, POP)
        context = "01500250"
        types = pr._infer_types_from_context(
            bytecode="01500250" * 5,
            selector="deadbeef",
            context=context,
        )
        assert types == ["uint256"]
