"""Tests for bytecode_features.py."""

import json
import torch
import pytest

from abi_reconstructor.features.bytecode_features import BytecodeFeatureExtractor


class TestBytecodeFeatureExtractor:
    """Test suite for BytecodeFeatureExtractor class."""

    def test_init(self):
        """Test initialization."""
        extractor = BytecodeFeatureExtractor()
        assert extractor is not None
        assert extractor.vocab_size == 256
        assert extractor.feature_dim == 256

    def test_forward_empty_input(self):
        """Test forward pass with empty input."""
        extractor = BytecodeFeatureExtractor()
        batch_size = 2
        seq_len = 10
        input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        attention_mask = torch.ones(batch_size, seq_len)

        outputs = extractor.forward(input_ids, attention_mask)

        assert "sequence_features" in outputs
        assert "pooled_features" in outputs
        assert "attention_weights" in outputs
        assert outputs["pooled_features"].shape == (batch_size, extractor.feature_dim)

    def test_extract_features_pooled(self):
        """Test extracting pooled features from bytecode."""
        extractor = BytecodeFeatureExtractor()
        batch_size = 2
        seq_len = 20

        input_ids = torch.randint(0, 256, (batch_size, seq_len), dtype=torch.long)
        attention_mask = torch.ones(batch_size, seq_len)

        features = extractor.extract_features(input_ids, attention_mask, return_sequence=False)

        assert isinstance(features, torch.Tensor)
        assert features.shape == (batch_size, extractor.feature_dim)

    def test_extract_features_sequence(self):
        """Test extracting sequence features from bytecode."""
        extractor = BytecodeFeatureExtractor()
        batch_size = 1
        seq_len = 30

        input_ids = torch.randint(0, 256, (batch_size, seq_len), dtype=torch.long)
        attention_mask = torch.ones(batch_size, seq_len)

        features = extractor.extract_features(input_ids, attention_mask, return_sequence=True)

        assert isinstance(features, torch.Tensor)
        assert features.shape == (batch_size, seq_len, extractor.feature_dim)

    def test_get_attention_patterns(self):
        """Test getting attention patterns for interpretability."""
        extractor = BytecodeFeatureExtractor()
        extractor.eval()

        batch_size = 1
        seq_len = 40
        input_ids = torch.randint(0, 256, (batch_size, seq_len), dtype=torch.long)
        attention_mask = torch.ones(batch_size, seq_len)

        result = extractor.get_attention_patterns(input_ids, attention_mask)

        assert "attention_patterns" in result
        assert "attention_weights" in result
        assert len(result["attention_patterns"]) == batch_size

    def test_forward_with_bytecode_tokens(self):
        """Test forward pass with realistic bytecode tokens."""
        extractor = BytecodeFeatureExtractor()

        bytecode_tokens = [
            0x60, 0x80, 0x60, 0x40, 0x52, 0x60, 0x17, 0x80, 0xfd,
            0x63, 0xa9, 0x05, 0x9c, 0xbb, 0x14, 0x60, 0x2d, 0x57,
        ][:20]
        while len(bytecode_tokens) < 20:
            bytecode_tokens.append(0)

        input_ids = torch.tensor([bytecode_tokens], dtype=torch.long)
        attention_mask = torch.ones(1, 20)

        outputs = extractor.forward(input_ids, attention_mask)

        assert "pooled_features" in outputs
        assert "sequence_features" in outputs
        assert outputs["pooled_features"].shape[0] == 1

    def test_model_trainable(self):
        """Test that model can be set to train mode."""
        extractor = BytecodeFeatureExtractor()
        extractor.train()

        assert extractor.training is True

        extractor.eval()
        assert extractor.training is False

    def test_attention_mask_applied(self):
        """Test that attention mask is properly applied in pooling."""
        extractor = BytecodeFeatureExtractor()
        batch_size = 1
        seq_len = 20

        input_ids = torch.randint(0, 256, (batch_size, seq_len), dtype=torch.long)
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[0, 10:] = 0

        outputs = extractor.forward(input_ids, attention_mask)
        pooled = outputs["pooled_features"]

        assert pooled.shape == (batch_size, extractor.feature_dim)

    def test_conv_layers_output_shape(self):
        """Test that convolutional layers produce correct output shape."""
        extractor = BytecodeFeatureExtractor()
        batch_size = 2
        seq_len = 50

        input_ids = torch.randint(0, 256, (batch_size, seq_len), dtype=torch.long)
        attention_mask = torch.ones(batch_size, seq_len)

        outputs = extractor.forward(input_ids, attention_mask)
        seq_features = outputs["sequence_features"]

        assert seq_features.shape == (batch_size, seq_len, extractor.feature_dim)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])