"""Tests for selector_extractor.py."""

import torch
import pytest

from abi_reconstructor.features.selector_extractor import NeuralSelectorExtractor


class TestNeuralSelectorExtractor:
    """Test suite for NeuralSelectorExtractor class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        extractor = NeuralSelectorExtractor()
        assert extractor.vocab_size == 256
        assert extractor.hidden_dim == 128
        assert extractor.dropout == 0.2

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        extractor = NeuralSelectorExtractor(vocab_size=512, hidden_dim=256, dropout=0.3)
        assert extractor.vocab_size == 512
        assert extractor.hidden_dim == 256
        assert extractor.dropout == 0.3

    def test_forward_empty_input(self):
        """Test forward pass with empty input."""
        extractor = NeuralSelectorExtractor()
        batch_size = 2
        seq_len = 10
        input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        attention_mask = torch.ones(batch_size, seq_len)

        outputs = extractor.forward(input_ids, attention_mask)

        assert "selector_logits" in outputs
        assert "attention_weights" in outputs
        assert outputs["selector_logits"].shape == (batch_size, seq_len)

    def test_forward_with_real_bytecode(self):
        """Test forward pass with realistic bytecode tokens."""
        extractor = NeuralSelectorExtractor()

        bytecode_tokens = [
            0x60, 0x80, 0x60, 0x40, 0x52, 0x60, 0x17, 0x80, 0xfd,
            0x63, 0xa9, 0x05, 0x9c, 0xbb, 0x14, 0x60, 0x2d, 0x57,
        ][:10]
        while len(bytecode_tokens) < 10:
            bytecode_tokens.append(0)

        input_ids = torch.tensor([bytecode_tokens], dtype=torch.long)
        attention_mask = torch.ones(1, 10)

        outputs = extractor.forward(input_ids, attention_mask)

        assert "selector_logits" in outputs
        assert "attention_weights" in outputs
        assert outputs["selector_logits"].shape[0] == 1

    def test_extract_selectors_returns_positions(self):
        """Test that extract_selectors returns positions above threshold."""
        extractor = NeuralSelectorExtractor()
        extractor.eval()

        batch_size = 1
        seq_len = 20
        input_ids = torch.randint(0, 256, (batch_size, seq_len), dtype=torch.long)
        attention_mask = torch.ones(batch_size, seq_len)

        result = extractor.extract_selectors(input_ids, attention_mask, threshold=0.5)

        assert "selector_positions" in result
        assert "selector_confidences" in result
        assert "selector_probs" in result

    def test_selector_logits_shape(self):
        """Test that selector_logits has correct shape."""
        extractor = NeuralSelectorExtractor()
        batch_size = 4
        seq_len = 100

        input_ids = torch.randint(0, 256, (batch_size, seq_len), dtype=torch.long)
        attention_mask = torch.ones(batch_size, seq_len)

        outputs = extractor.forward(input_ids, attention_mask)
        selector_logits = outputs["selector_logits"]

        assert selector_logits.shape == (batch_size, seq_len)

    def test_attention_weights_shape(self):
        """Test that attention weights have correct shape."""
        extractor = NeuralSelectorExtractor()
        batch_size = 2
        seq_len = 50

        input_ids = torch.randint(0, 256, (batch_size, seq_len), dtype=torch.long)
        attention_mask = torch.ones(batch_size, seq_len)

        outputs = extractor.forward(input_ids, attention_mask)
        attn_weights = outputs["attention_weights"]

        assert attn_weights is not None

    def test_model_trainable(self):
        """Test that model can be set to train mode."""
        extractor = NeuralSelectorExtractor()
        extractor.train()

        assert extractor.training is True

        extractor.eval()
        assert extractor.training is False

    def test_extract_selectors_threshold(self):
        """Test that threshold controls selector detection."""
        extractor = NeuralSelectorExtractor()
        extractor.eval()

        input_ids = torch.randint(0, 256, (1, 50), dtype=torch.long)
        attention_mask = torch.ones(1, 50)

        result_low = extractor.extract_selectors(input_ids, attention_mask, threshold=0.1)
        result_high = extractor.extract_selectors(input_ids, attention_mask, threshold=0.9)

        assert "selector_probs" in result_low
        assert "selector_probs" in result_high


if __name__ == "__main__":
    pytest.main([__file__, "-v"])