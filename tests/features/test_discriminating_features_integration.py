"""Integration tests for discriminating features in the full pipeline.

Tests that discriminating features (SigRec R11-R18 opcode analysis) flow through
the pipeline — from bytecode analysis → feature vector → model input → output change.
"""

import pytest
import torch
import numpy as np
from abi_reconstructor.features.discriminating_features import (
    DiscriminatingFeatureExtractor,
    DiscriminatingFeatures,
)
from abi_reconstructor.reconstructor import ParameterReconstructor, ABIReconstructor
from abi_reconstructor.models.parameter_prediction_model import ParameterPredictionModel


# ── Synthetic but realistic bytecode snippets ──

# PUSH4 a9059cbb (transfer selector) JUMPDEST ... PUSH32 uint_mask AND
# The address AND mask: 0x0000...0000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
BYTECODE_TRANSFER_WITH_UINT_MASK = (
    "63a9059cbb"  # PUSH4 selector
    + "5b"         # JUMPDEST
    + "7f"         # PUSH32
    + "000000000000000000000000ffffffffffffffffffffffffffffffffffffffff"  # uint160 mask
    + "16"         # AND
    + "50" * 10    # POP padding
)

# PUSH4 40c10f19 (mint selector) JUMPDEST ... PUSH32 address_mask AND ... SIGNEXTEND absent
# Address mask + no SIGNEXTEND → address + uint
BYTECODE_MINT_WITH_ADDRESS_MASK = (
    "6340c10f19"  # PUSH4 mint(address,uint256)
    + "5b"
    + "7f"
    + "000000000000000000000000ffffffffffffffffffffffffffffffffffffffff"  # address mask (20 bytes)
    + "16"         # AND
    + "50" * 10
)

# PUSH4 deadbeef (unknown) JUMPDEST ... SIGNEXTEND → int pattern
BYTECODE_SIGNEXTEND_PATTERN = (
    "63deadbeef"
    + "5b"
    + "0b"         # SIGNEXTEND
    + "50" * 10
)

# PUSH4 cafebabe (unknown) JUMPDEST ... ISZERO ISZERO → bool pattern
BYTECODE_BOOL_PATTERN = (
    "63cafebabe"
    + "5b"
    + "1515"       # ISZERO ISZERO
    + "50" * 10
)

# PUSH4 0badf00d (unknown) JUMPDEST ... no discriminating instructions
BYTECODE_NO_FEATURES = (
    "630badf00d"
    + "5b"
    + "50" * 20   # Just POPs — no type-revealing opcodes
)


class TestDiscriminatingFeaturesOnRealPatterns:
    """Test that discriminating features extract correctly from realistic bytecode."""

    def test_erc20_transfer_detects_uint_mask(self):
        """AND mask with leading zeros → uint (SigRec R11)."""
        features = DiscriminatingFeatureExtractor._analyze_opcodes(
            bytes.fromhex(BYTECODE_TRANSFER_WITH_UINT_MASK)
        )
        assert features.has_and_mask_leading_zeros is True
        assert features.and_mask_bytes == 20  # 160-bit mask
        types = DiscriminatingFeatureExtractor.infer_types_from_features(features)
        assert "address" in types

    def test_mint_detects_address_mask(self):
        """20-byte AND mask without signed math → address (SigRec R16)."""
        features = DiscriminatingFeatureExtractor._analyze_opcodes(
            bytes.fromhex(BYTECODE_MINT_WITH_ADDRESS_MASK)
        )
        assert features.has_and_mask_leading_zeros is True
        assert features.and_mask_bytes == 20
        types = DiscriminatingFeatureExtractor.infer_types_from_features(features)
        assert "address" in types

    def test_signextend_detects_int(self):
        """SIGNEXTEND → int256 (SigRec R13)."""
        features = DiscriminatingFeatureExtractor._analyze_opcodes(
            bytes.fromhex(BYTECODE_SIGNEXTEND_PATTERN)
        )
        assert features.has_signextend is True
        types = DiscriminatingFeatureExtractor.infer_types_from_features(features)
        assert "int256" in types

    def test_double_iszero_detects_bool(self):
        """Double ISZERO → bool (SigRec R14)."""
        features = DiscriminatingFeatureExtractor._analyze_opcodes(
            bytes.fromhex(BYTECODE_BOOL_PATTERN)
        )
        assert features.has_double_iszero is True
        types = DiscriminatingFeatureExtractor.infer_types_from_features(features)
        assert "bool" in types

    def test_no_features_defaults_to_uint256(self):
        """No discriminating instructions → uint256 (SigRec R4 fallback)."""
        features = DiscriminatingFeatureExtractor._analyze_opcodes(
            bytes.fromhex(BYTECODE_NO_FEATURES)
        )
        types = DiscriminatingFeatureExtractor.infer_types_from_features(features)
        assert types == ["uint256"]

    def test_signed_math_detects_int256(self):
        """SDIV/SMOD/SLT/SGT → int256 (SigRec R15)."""
        for op_hex in ["05", "07", "12", "13"]:
            features = DiscriminatingFeatureExtractor._analyze_opcodes(
                bytes.fromhex(op_hex + "50" * 5)
            )
            assert features.has_signed_math is True, f"opcode 0x{op_hex}"
            types = DiscriminatingFeatureExtractor.infer_types_from_features(features)
            assert "int256" in types, f"opcode 0x{op_hex}"

    def test_byte_instruction_detects_bytes(self):
        """BYTE instruction → bytes32 (SigRec R18)."""
        features = DiscriminatingFeatureExtractor._analyze_opcodes(
            bytes.fromhex("1a" + "50" * 5)
        )
        assert features.has_byte_instruction is True
        types = DiscriminatingFeatureExtractor.infer_types_from_features(features)
        assert "bytes32" in types


class TestDiscriminatingFeaturesViaParameterReconstructor:
    """Test that ParameterReconstructor uses discriminating features for unknowns."""

    def test_unknown_function_with_signextend_gets_int(self):
        """Unknown selector + SIGNEXTEND bytecode → infer int256."""
        pr = ParameterReconstructor()
        types = pr._infer_types_from_context(
            bytecode=BYTECODE_SIGNEXTEND_PATTERN,
            selector="deadbeef",
            context=BYTECODE_SIGNEXTEND_PATTERN,
        )
        assert "int256" in types

    def test_unknown_function_with_bool_pattern(self):
        """Unknown selector + double ISZERO → infer bool."""
        pr = ParameterReconstructor()
        types = pr._infer_types_from_context(
            bytecode=BYTECODE_BOOL_PATTERN,
            selector="cafebabe",
            context=BYTECODE_BOOL_PATTERN,
        )
        assert "bool" in types

    def test_unknown_function_no_features_defaults(self):
        """Unknown selector + no type signal → default uint256."""
        pr = ParameterReconstructor()
        types = pr._infer_types_from_context(
            bytecode=BYTECODE_NO_FEATURES,
            selector="0badf00d",
            context=BYTECODE_NO_FEATURES,
        )
        assert types == ["uint256"]

    def test_selector_not_found_falls_back(self):
        """Missing selector → still returns default."""
        pr = ParameterReconstructor()
        types = pr._infer_types_from_context(
            bytecode=BYTECODE_NO_FEATURES,
            selector="ffffffff",  # not in bytecode
        )
        assert types == ["uint256"]


class TestDiscriminatingFeaturesChangeModelOutput:
    """Test that discriminating features actually change model predictions."""

    @pytest.fixture
    def model(self):
        return ParameterPredictionModel(
            num_type_classes=22,
            max_parameters=12,
            encoder_type="cnn",
            hidden_dim=256,
            use_function_conditioning=False,
            use_cuda=False,
        )

    @pytest.fixture
    def sample_batch(self):
        """Create a batch of 4 samples with fixed bytecode tokens."""
        torch.manual_seed(42)
        tokens = torch.randint(0, 256, (4, 512)).float()
        return {
            "bytecode_tokens": tokens.unsqueeze(-1).expand(-1, -1, 256),
            "parameter_count": torch.tensor([2, 1, 0, 2]),
            "parameter_types": torch.randint(0, 22, (4, 12)),
            "parameter_mask": torch.zeros(4, 12),
        }

    def test_discriminating_features_produce_different_output(self, model, sample_batch):
        """Passing non-zero features changes model output vs. zeros."""
        sample_batch["parameter_mask"][:, 0] = 1.0

        # Run with zeros (backward compat default)
        zeros = torch.zeros(4, 7)
        out_zeros = model(
            sample_batch["bytecode_tokens"],
            discriminating_features=zeros,
        )

        # Run with real-looking features (AND mask + SIGNEXTEND)
        non_zero = torch.zeros(4, 7)
        non_zero[:, 0] = 1.0  # has_and_mask_leading_zeros
        non_zero[:, 2] = 1.0  # has_signextend
        non_zero[:, 3] = 1.0  # has_signed_math

        out_real = model(
            sample_batch["bytecode_tokens"],
            discriminating_features=non_zero,
        )

        # Type logits should differ
        assert not torch.allclose(
            out_zeros["type_logits"], out_real["type_logits"], atol=1e-6
        ), "Discriminating features had no effect on model output"

    def test_zero_features_produces_deterministic_output(self, model, sample_batch):
        """Zero discriminating features produce deterministic output.

        The discriminating projection includes learned parameters that transform
        even zero inputs, but the output is deterministic (same input → same output).
        With dropout in eval mode, results are deterministic.
        """
        sample_batch["parameter_mask"][:, 0] = 1.0
        model.eval()  # Disable dropout for determinism

        out_zeros_1 = model(
            sample_batch["bytecode_tokens"],
            discriminating_features=torch.zeros(4, 7),
        )
        out_zeros_2 = model(
            sample_batch["bytecode_tokens"],
            discriminating_features=torch.zeros(4, 7),
        )

        # Two calls with zeros produce identical output (deterministic)
        assert torch.allclose(
            out_zeros_1["type_logits"], out_zeros_2["type_logits"], atol=1e-6
        ), "Zero features should be deterministic"
        assert torch.allclose(
            out_zeros_1["count_logits"], out_zeros_2["count_logits"], atol=1e-6
        )

    def test_batch_size_one_works(self, model):
        """Batch size 1 with discriminating features works."""
        tokens = torch.randint(0, 256, (1, 512)).float().unsqueeze(-1).expand(-1, -1, 256)
        disc = torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
        out = model(tokens, discriminating_features=disc)
        assert out["type_logits"].shape == (1, 12, 22)
        assert out["count_logits"].shape == (1, 13)

    def test_compute_loss_with_discriminating_features(self, model, sample_batch):
        """compute_loss with discriminating features works."""
        sample_batch["parameter_mask"][:, 0] = 1.0
        disc = torch.randn(4, 7)
        loss = model.compute_loss(
            sample_batch["bytecode_tokens"],
            sample_batch["parameter_count"],
            sample_batch["parameter_types"],
            sample_batch["parameter_mask"],
            discriminating_features=disc,
        )
        assert "total_loss" in loss
        assert loss["total_loss"].item() > 0

    def test_compute_loss_defaults_to_none(self, model, sample_batch):
        """compute_loss without discriminating_features still works."""
        sample_batch["parameter_mask"][:, 0] = 1.0
        loss = model.compute_loss(
            sample_batch["bytecode_tokens"],
            sample_batch["parameter_count"],
            sample_batch["parameter_types"],
            sample_batch["parameter_mask"],
        )
        assert "total_loss" in loss


class TestDiscriminatingFeaturesCheckpointCompatibility:
    """Test that models with discriminating layers save/load correctly."""

    def test_state_dict_contains_discriminating_layers(self):
        model = ParameterPredictionModel(
            num_type_classes=22,
            max_parameters=12,
            encoder_type="cnn",
            hidden_dim=256,
            use_function_conditioning=False,
            use_cuda=False,
        )
        state = model.state_dict()
        disc_keys = [k for k in state if "discriminating" in k]
        assert len(disc_keys) >= 2, f"Expected >=2 discriminating layers, got {disc_keys}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_discriminating_features_on_gpu(self):
        """Discriminating features work when model is on GPU."""
        model = ParameterPredictionModel(
            num_type_classes=22,
            max_parameters=12,
            encoder_type="cnn",
            hidden_dim=256,
            use_cuda=True,
        )
        tokens = torch.randint(0, 256, (2, 512)).float().unsqueeze(-1).expand(-1, -1, 256)
        tokens = tokens.cuda()
        disc = torch.randn(2, 7).cuda()
        out = model(tokens, discriminating_features=disc)
        assert out["type_logits"].device.type == "cuda"
