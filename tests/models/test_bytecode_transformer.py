"""Unit tests for BytecodeTransformer and BytecodeCNNEncoder."""

import pytest
import torch
import numpy as np

from abi_reconstructor.models.bytecode_transformer import (
    BytecodeTransformer,
    BytecodeCNNEncoder,
)


class TestBytecodeTransformer:
    """Test suite for BytecodeTransformer."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        model = BytecodeTransformer(
            vocab_size=256,
            d_model=256,
            nhead=8,
            num_layers=4,
            dim_feedforward=1024,
            dropout=0.1,
            max_seq_len=512,
            use_positional_encoding=True,
            use_cuda=False,
        )

        assert model.vocab_size == 256
        assert model.d_model == 256
        assert model.nhead == 8
        assert model.num_layers == 4
        assert model.dim_feedforward == 1024
        assert model.dropout == 0.1
        assert model.max_seq_len == 512
        assert model.use_positional_encoding == True
        assert model.use_cuda == False

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        model = BytecodeTransformer(
            vocab_size=128,
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=256,
            dropout=0.2,
            max_seq_len=256,
            use_positional_encoding=False,
            use_cuda=False,
        )

        assert model.vocab_size == 128
        assert model.d_model == 64
        assert model.nhead == 4
        assert model.num_layers == 2
        assert model.dim_feedforward == 256
        assert model.dropout == 0.2
        assert model.max_seq_len == 256
        assert model.use_positional_encoding == False

    def test_forward_basic(self):
        """Test forward pass with basic input."""
        model = BytecodeTransformer(
            vocab_size=256,
            d_model=64,
            nhead=2,
            num_layers=1,
            dim_feedforward=128,
            dropout=0.1,
            max_seq_len=512,
            use_positional_encoding=True,
            use_cuda=False,
        )

        batch_size = 2
        seq_len = 100
        tokens = torch.randint(0, 256, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        outputs = model(tokens, attention_mask)

        assert isinstance(outputs, dict)
        assert "last_hidden_state" in outputs
        assert "pooled_output" in outputs

        # Check shapes
        assert outputs["last_hidden_state"].shape == (batch_size, seq_len, 64)
        assert outputs["pooled_output"].shape == (batch_size, 64)

        # Pooled output uses mean pooling (weighted by attention mask)
        # Since attention_mask is all ones, it's just mean pooling
        expected_pooled = outputs["last_hidden_state"].mean(dim=1)
        assert torch.allclose(outputs["pooled_output"], expected_pooled, rtol=1e-5)

    def test_forward_no_attention_mask(self):
        """Test forward pass without attention mask."""
        model = BytecodeTransformer(
            vocab_size=256,
            d_model=64,
            nhead=2,
            num_layers=1,
            dim_feedforward=128,
            dropout=0.1,
            max_seq_len=512,
            use_positional_encoding=True,
            use_cuda=False,
        )

        batch_size = 2
        seq_len = 100
        tokens = torch.randint(0, 256, (batch_size, seq_len))

        outputs = model(tokens)

        assert outputs["last_hidden_state"].shape == (batch_size, seq_len, 64)
        assert outputs["pooled_output"].shape == (batch_size, 64)

    def test_forward_with_attention_mask(self):
        """Test forward pass with attention mask (some positions masked)."""
        model = BytecodeTransformer(
            vocab_size=256,
            d_model=64,
            nhead=2,
            num_layers=1,
            dim_feedforward=128,
            dropout=0.1,
            max_seq_len=512,
            use_positional_encoding=True,
            use_cuda=False,
        )

        batch_size = 2
        seq_len = 100
        tokens = torch.randint(0, 256, (batch_size, seq_len))

        # Create mask where first 50 positions are 1, rest are 0
        attention_mask = torch.zeros(batch_size, seq_len)
        attention_mask[:, :50] = 1

        outputs = model(tokens, attention_mask)

        # Should still produce correct shapes
        assert outputs["last_hidden_state"].shape == (batch_size, seq_len, 64)
        assert outputs["pooled_output"].shape == (batch_size, 64)

    def test_forward_empty_batch(self):
        """Test forward pass with empty batch."""
        model = BytecodeTransformer(
            vocab_size=256,
            d_model=64,
            nhead=2,
            num_layers=1,
            dim_feedforward=128,
            dropout=0.1,
            max_seq_len=512,
            use_positional_encoding=True,
            use_cuda=False,
        )

        batch_size = 0
        seq_len = 100
        tokens = torch.randint(0, 256, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        outputs = model(tokens, attention_mask)

        assert outputs["last_hidden_state"].shape == (0, seq_len, 64)
        assert outputs["pooled_output"].shape == (0, 64)

    def test_forward_max_seq_len(self):
        """Test forward pass with sequence at max length."""
        max_seq_len = 128
        model = BytecodeTransformer(
            vocab_size=256,
            d_model=64,
            nhead=2,
            num_layers=1,
            dim_feedforward=128,
            dropout=0.1,
            max_seq_len=max_seq_len,
            use_positional_encoding=True,
            use_cuda=False,
        )

        batch_size = 2
        seq_len = max_seq_len  # Exactly max length
        tokens = torch.randint(0, 256, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        outputs = model(tokens, attention_mask)

        assert outputs["last_hidden_state"].shape == (batch_size, seq_len, 64)

    def test_forward_seq_len_exceeds_max(self):
        """Test forward pass with sequence longer than max length (should truncate)."""
        max_seq_len = 64
        model = BytecodeTransformer(
            vocab_size=256,
            d_model=64,
            nhead=2,
            num_layers=1,
            dim_feedforward=128,
            dropout=0.1,
            max_seq_len=max_seq_len,
            use_positional_encoding=True,
            use_cuda=False,
        )

        batch_size = 2
        seq_len = 100  # Longer than max_seq_len
        tokens = torch.randint(0, 256, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        outputs = model(tokens, attention_mask)

        # Should be truncated to max_seq_len
        assert outputs["last_hidden_state"].shape == (batch_size, max_seq_len, 64)

    def test_positional_encoding_disabled(self):
        """Test forward pass without positional encoding."""
        model = BytecodeTransformer(
            vocab_size=256,
            d_model=64,
            nhead=2,
            num_layers=1,
            dim_feedforward=128,
            dropout=0.1,
            max_seq_len=512,
            use_positional_encoding=False,  # Disabled
            use_cuda=False,
        )

        batch_size = 2
        seq_len = 100
        tokens = torch.randint(0, 256, (batch_size, seq_len))

        outputs = model(tokens)

        # Should still work without positional encoding
        assert outputs["last_hidden_state"].shape == (batch_size, seq_len, 64)

    def test_model_on_gpu_if_available(self):
        """Test model moves to GPU when available and requested."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        model = BytecodeTransformer(
            vocab_size=256,
            d_model=64,
            nhead=2,
            num_layers=1,
            dim_feedforward=128,
            dropout=0.1,
            max_seq_len=512,
            use_positional_encoding=True,
            use_cuda=True,
        )

        # Check model is on GPU
        assert next(model.parameters()).is_cuda

        # Test forward pass on GPU
        batch_size = 2
        seq_len = 100
        tokens = torch.randint(0, 256, (batch_size, seq_len)).cuda()
        attention_mask = torch.ones(batch_size, seq_len).cuda()

        outputs = model(tokens, attention_mask)

        # Outputs should also be on GPU
        assert outputs["last_hidden_state"].is_cuda


class TestBytecodeCNNEncoder:
    """Test suite for BytecodeCNNEncoder."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        model = BytecodeCNNEncoder(
            vocab_size=256,
            embedding_dim=128,
            num_filters=128,
            filter_sizes=(3, 4, 5),
            dropout=0.1,
            use_cuda=False,
        )

        assert model.vocab_size == 256
        assert model.embedding_dim == 128
        assert model.num_filters == 128
        assert model.filter_sizes == (3, 4, 5)
        assert model.dropout == 0.1
        assert model.use_cuda == False

        # Check output dimension calculation
        expected_output_dim = 128 * len((3, 4, 5))  # num_filters * num_filter_sizes
        assert model.output_dim == expected_output_dim

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        model = BytecodeCNNEncoder(
            vocab_size=128,
            embedding_dim=64,
            num_filters=64,
            filter_sizes=(2, 3),
            dropout=0.2,
            use_cuda=False,
        )

        assert model.vocab_size == 128
        assert model.embedding_dim == 64
        assert model.num_filters == 64
        assert model.filter_sizes == (2, 3)
        assert model.dropout == 0.2

        expected_output_dim = 64 * 2  # num_filters * num_filter_sizes
        assert model.output_dim == expected_output_dim

    def test_forward_basic(self):
        """Test forward pass with basic input."""
        model = BytecodeCNNEncoder(
            vocab_size=256,
            embedding_dim=64,
            num_filters=64,
            filter_sizes=(3, 4),
            dropout=0.1,
            use_cuda=False,
        )

        batch_size = 2
        seq_len = 100
        tokens = torch.randint(0, 256, (batch_size, seq_len))

        outputs = model(tokens)

        assert isinstance(outputs, dict)
        assert "last_hidden_state" in outputs
        assert "pooled_output" in outputs

        # Check shapes
        expected_output_dim = 64 * 2  # num_filters * num_filter_sizes
        assert outputs["last_hidden_state"].shape == (batch_size, expected_output_dim)
        assert outputs["pooled_output"].shape == (batch_size, expected_output_dim)

    def test_forward_with_attention_mask(self):
        """Test forward pass with attention mask (CNN ignores it but accepts it)."""
        model = BytecodeCNNEncoder(
            vocab_size=256,
            embedding_dim=64,
            num_filters=64,
            filter_sizes=(3, 4),
            dropout=0.1,
            use_cuda=False,
        )

        batch_size = 2
        seq_len = 100
        tokens = torch.randint(0, 256, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        # Should accept attention_mask parameter even if not used
        outputs = model(tokens, attention_mask)

        expected_output_dim = 64 * 2
        assert outputs["last_hidden_state"].shape == (batch_size, expected_output_dim)

    def test_forward_empty_batch(self):
        """Test forward pass with empty batch."""
        model = BytecodeCNNEncoder(
            vocab_size=256,
            embedding_dim=64,
            num_filters=64,
            filter_sizes=(3, 4),
            dropout=0.1,
            use_cuda=False,
        )

        batch_size = 0
        seq_len = 100
        tokens = torch.randint(0, 256, (batch_size, seq_len))

        outputs = model(tokens)

        expected_output_dim = 64 * 2
        assert outputs["last_hidden_state"].shape == (0, expected_output_dim)
        assert outputs["pooled_output"].shape == (0, expected_output_dim)

    def test_forward_short_sequence(self):
        """Test forward pass with sequence shorter than largest filter size."""
        largest_filter = 5
        model = BytecodeCNNEncoder(
            vocab_size=256,
            embedding_dim=64,
            num_filters=64,
            filter_sizes=(3, 4, largest_filter),
            dropout=0.1,
            use_cuda=False,
        )

        batch_size = 2
        seq_len = 3  # Shorter than largest filter
        tokens = torch.randint(0, 256, (batch_size, seq_len))

        # Should handle short sequences (padding internally)
        outputs = model(tokens)

        expected_output_dim = 64 * 3
        assert outputs["last_hidden_state"].shape == (batch_size, expected_output_dim)

    def test_model_on_gpu_if_available(self):
        """Test model moves to GPU when available and requested."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        model = BytecodeCNNEncoder(
            vocab_size=256,
            embedding_dim=64,
            num_filters=64,
            filter_sizes=(3, 4),
            dropout=0.1,
            use_cuda=True,
        )

        # Check model is on GPU
        assert next(model.parameters()).is_cuda

        # Test forward pass on GPU
        batch_size = 2
        seq_len = 100
        tokens = torch.randint(0, 256, (batch_size, seq_len)).cuda()

        outputs = model(tokens)

        # Outputs should also be on GPU
        assert outputs["last_hidden_state"].is_cuda

    def test_different_filter_sizes(self):
        """Test with different filter size configurations."""
        test_cases = [
            ((2,), 64 * 1),  # Single filter size
            ((2, 3, 4), 64 * 3),  # Three filter sizes
            ((1, 2, 3, 4, 5), 64 * 5),  # Many filter sizes
        ]

        for filter_sizes, expected_output_dim in test_cases:
            model = BytecodeCNNEncoder(
                vocab_size=256,
                embedding_dim=64,
                num_filters=64,
                filter_sizes=filter_sizes,
                dropout=0.1,
                use_cuda=False,
            )

            assert model.output_dim == expected_output_dim

            # Test forward pass
            batch_size = 2
            seq_len = 50
            tokens = torch.randint(0, 256, (batch_size, seq_len))

            outputs = model(tokens)

            assert outputs["last_hidden_state"].shape == (
                batch_size,
                expected_output_dim,
            )
            assert outputs["pooled_output"].shape == (batch_size, expected_output_dim)
