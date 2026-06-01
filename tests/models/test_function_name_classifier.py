"""Unit tests for FunctionNameClassifier model."""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from abi_reconstructor.models.function_name_classifier import FunctionNameClassifier


class TestFunctionNameClassifier:
    """Test suite for FunctionNameClassifier."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        model = FunctionNameClassifier(
            num_classes=50,
            encoder_type="transformer",
            use_cuda=False,
        )

        assert model.num_classes == 50
        assert model.encoder_type == "transformer"
        assert model.hidden_dim == 256
        assert model.dropout == 0.2
        assert model.use_cuda == False
        assert isinstance(model.encoder, torch.nn.Module)
        assert isinstance(model.classifier, torch.nn.Sequential)
        assert isinstance(model.temperature, torch.nn.Parameter)
        assert isinstance(model.criterion, torch.nn.CrossEntropyLoss)

    def test_init_cnn_encoder(self):
        """Test initialization with CNN encoder."""
        model = FunctionNameClassifier(
            num_classes=50,
            encoder_type="cnn",
            use_cuda=False,
        )

        assert model.encoder_type == "cnn"
        # Check that encoder was created (exact type check might be internal)
        assert hasattr(model.encoder, "forward")

    def test_init_custom_config(self):
        """Test initialization with custom encoder configuration."""
        encoder_config = {
            "vocab_size": 256,
            "d_model": 128,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 512,
            "dropout": 0.1,
            "max_seq_len": 256,
            "use_positional_encoding": True,
            "use_cuda": False,
        }

        model = FunctionNameClassifier(
            num_classes=100,
            encoder_type="transformer",
            encoder_config=encoder_config,
            hidden_dim=128,
            dropout=0.3,
            use_cuda=False,
        )

        assert model.num_classes == 100
        assert model.hidden_dim == 128
        assert model.dropout == 0.3
        assert model.encoder_config["d_model"] == 128
        assert model.encoder_config["nhead"] == 4

    def test_forward_basic(self):
        """Test forward pass with basic input."""
        model = FunctionNameClassifier(
            num_classes=50,
            encoder_type="transformer",
            encoder_config={
                "vocab_size": 256,
                "d_model": 64,
                "nhead": 2,
                "num_layers": 1,
                "dim_feedforward": 128,
                "dropout": 0.1,
                "max_seq_len": 512,
                "use_positional_encoding": True,
                "use_cuda": False,
            },
            hidden_dim=64,
            dropout=0.1,
            use_cuda=False,
        )

        batch_size = 2
        seq_len = 100
        bytecode_tokens = torch.randint(0, 256, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        outputs = model(bytecode_tokens, attention_mask)

        # Check output structure
        assert isinstance(outputs, dict)
        assert "logits" in outputs
        assert "calibrated_logits" in outputs
        assert "probs" in outputs
        assert "confidence" in outputs
        assert "predicted_class" in outputs
        assert "features" in outputs

        # Check shapes
        assert outputs["logits"].shape == (batch_size, 50)
        assert outputs["calibrated_logits"].shape == (batch_size, 50)
        assert outputs["probs"].shape == (batch_size, 50)
        assert outputs["confidence"].shape == (batch_size,)
        assert outputs["predicted_class"].shape == (batch_size,)

        # Check probability properties
        probs_sum = outputs["probs"].sum(dim=1)
        assert torch.allclose(probs_sum, torch.ones(batch_size), rtol=1e-5)
        assert torch.all(outputs["probs"] >= 0)
        assert torch.all(outputs["probs"] <= 1)

        # Check confidence properties
        assert torch.all(outputs["confidence"] >= 0)
        assert torch.all(outputs["confidence"] <= 1)

    def test_forward_no_attention_mask(self):
        """Test forward pass without attention mask."""
        model = FunctionNameClassifier(
            num_classes=50,
            encoder_type="transformer",
            encoder_config={
                "vocab_size": 256,
                "d_model": 64,
                "nhead": 2,
                "num_layers": 1,
                "dim_feedforward": 128,
                "dropout": 0.1,
                "max_seq_len": 512,
                "use_positional_encoding": True,
                "use_cuda": False,
            },
            use_cuda=False,
        )

        batch_size = 2
        seq_len = 100
        bytecode_tokens = torch.randint(0, 256, (batch_size, seq_len))

        outputs = model(bytecode_tokens)

        assert isinstance(outputs, dict)
        assert "logits" in outputs
        assert outputs["logits"].shape == (batch_size, 50)

    def test_forward_empty_batch(self):
        """Test forward pass with empty batch (edge case)."""
        model = FunctionNameClassifier(
            num_classes=50,
            encoder_type="transformer",
            encoder_config={
                "vocab_size": 256,
                "d_model": 64,
                "nhead": 2,
                "num_layers": 1,
                "dim_feedforward": 128,
                "dropout": 0.1,
                "max_seq_len": 512,
                "use_positional_encoding": True,
                "use_cuda": False,
            },
            use_cuda=False,
        )

        batch_size = 0
        seq_len = 100
        bytecode_tokens = torch.randint(0, 256, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        outputs = model(bytecode_tokens, attention_mask)

        assert outputs["logits"].shape == (0, 50)
        assert outputs["probs"].shape == (0, 50)
        assert outputs["confidence"].shape == (0,)
        assert outputs["predicted_class"].shape == (0,)

    def test_compute_loss_basic(self):
        """Test compute_loss with basic input."""
        model = FunctionNameClassifier(
            num_classes=50,
            encoder_type="transformer",
            encoder_config={
                "vocab_size": 256,
                "d_model": 64,
                "nhead": 2,
                "num_layers": 1,
                "dim_feedforward": 128,
                "dropout": 0.1,
                "max_seq_len": 512,
                "use_positional_encoding": True,
                "use_cuda": False,
            },
            use_cuda=False,
        )

        batch_size = 4
        seq_len = 100
        bytecode_tokens = torch.randint(0, 256, (batch_size, seq_len))
        labels = torch.randint(0, 50, (batch_size,))
        attention_mask = torch.ones(batch_size, seq_len)

        loss_dict = model.compute_loss(bytecode_tokens, labels, attention_mask)

        assert isinstance(loss_dict, dict)
        assert "loss" in loss_dict
        assert isinstance(loss_dict["loss"], torch.Tensor)
        assert loss_dict["loss"].dim() == 0  # Scalar

        # Loss should be positive
        assert loss_dict["loss"].item() > 0

    def test_compute_loss_no_attention_mask(self):
        """Test compute_loss without attention mask."""
        model = FunctionNameClassifier(
            num_classes=50,
            encoder_type="transformer",
            encoder_config={
                "vocab_size": 256,
                "d_model": 64,
                "nhead": 2,
                "num_layers": 1,
                "dim_feedforward": 128,
                "dropout": 0.1,
                "max_seq_len": 512,
                "use_positional_encoding": True,
                "use_cuda": False,
            },
            use_cuda=False,
        )

        batch_size = 4
        seq_len = 100
        bytecode_tokens = torch.randint(0, 256, (batch_size, seq_len))
        labels = torch.randint(0, 50, (batch_size,))

        loss_dict = model.compute_loss(bytecode_tokens, labels)

        assert isinstance(loss_dict, dict)
        assert "loss" in loss_dict
        assert loss_dict["loss"].item() > 0

    def test_predict_basic(self):
        """Test predict method with basic input."""
        model = FunctionNameClassifier(
            num_classes=50,
            encoder_type="transformer",
            encoder_config={
                "vocab_size": 256,
                "d_model": 64,
                "nhead": 2,
                "num_layers": 1,
                "dim_feedforward": 128,
                "dropout": 0.1,
                "max_seq_len": 512,
                "use_positional_encoding": True,
                "use_cuda": False,
            },
            use_cuda=False,
        )

        batch_size = 3
        seq_len = 100
        bytecode_tokens = torch.randint(0, 256, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        predictions = model.predict(bytecode_tokens, attention_mask)

        assert isinstance(predictions, dict)
        assert "predictions" in predictions
        assert "raw_outputs" in predictions

        pred_list = predictions["predictions"]
        assert len(pred_list) == batch_size

        # Check each prediction has required fields
        for pred in pred_list:
            assert "top_indices" in pred
            assert "top_probs" in pred
            assert "best_index" in pred
            assert "best_prob" in pred
            assert "confidence" in pred

            # Top probabilities should be filtered by threshold
            if pred["top_probs"]:
                assert all(0 <= prob <= 1 for prob in pred["top_probs"])

            # Best index should be in top indices
            if pred["top_indices"]:
                assert pred["best_index"] in pred["top_indices"]
                assert 0 <= pred["best_class_prob"] <= 1

    def test_predict_with_custom_threshold(self):
        """Test predict method with custom confidence threshold."""
        model = FunctionNameClassifier(
            num_classes=50,
            encoder_type="transformer",
            encoder_config={
                "vocab_size": 256,
                "d_model": 64,
                "nhead": 2,
                "num_layers": 1,
                "dim_feedforward": 128,
                "dropout": 0.1,
                "max_seq_len": 512,
                "use_positional_encoding": True,
                "use_cuda": False,
            },
            use_cuda=False,
        )

        batch_size = 2
        seq_len = 100
        bytecode_tokens = torch.randint(0, 256, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        # Test with higher threshold
        predictions = model.predict(bytecode_tokens, attention_mask, threshold=0.5)

        for pred in predictions["predictions"]:
            if pred["top_probs"]:
                assert all(0 <= prob <= 1 for prob in pred["top_probs"])

    def test_temperature_calibration(self):
        """Test temperature calibration affects predictions."""
        model = FunctionNameClassifier(
            num_classes=50,
            encoder_type="transformer",
            encoder_config={
                "vocab_size": 256,
                "d_model": 64,
                "nhead": 2,
                "num_layers": 1,
                "dim_feedforward": 128,
                "dropout": 0.1,
                "max_seq_len": 512,
                "use_positional_encoding": True,
                "use_cuda": False,
            },
            use_cuda=False,
        )

        batch_size = 2
        seq_len = 100
        bytecode_tokens = torch.randint(0, 256, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        # Get predictions with default temperature
        outputs1 = model(bytecode_tokens, attention_mask)

        # Change temperature
        model.temperature.data.fill_(2.0)

        # Get predictions with different temperature
        outputs2 = model(bytecode_tokens, attention_mask)

        # Logits should be different after temperature scaling
        assert not torch.allclose(outputs1["logits"], outputs2["logits"], rtol=1e-5)

        # Calibrated logits should be logits / temperature
        expected_calibrated = outputs2["logits"] / 2.0
        assert torch.allclose(
            outputs2["calibrated_logits"], expected_calibrated, rtol=1e-5
        )

    def test_save_and_load_model(self, tmp_path):
        """Test model save and load functionality."""
        # Create and save model
        model1 = FunctionNameClassifier(
            num_classes=50,
            encoder_type="transformer",
            encoder_config={
                "vocab_size": 256,
                "d_model": 64,
                "nhead": 2,
                "num_layers": 1,
                "dim_feedforward": 128,
                "dropout": 0.1,
                "max_seq_len": 512,
                "use_positional_encoding": True,
                "use_cuda": False,
            },
            hidden_dim=128,
            dropout=0.3,
            use_cuda=False,
        )

        # Set temperature to non-default value
        model1.temperature.data.fill_(1.5)

        # Save model
        save_path = tmp_path / "test_model.pth"
        model1.save_model(str(save_path))

        # Load model
        model2 = FunctionNameClassifier.load_model(str(save_path), use_cuda=False)

        # Check architecture matches
        assert model2.num_classes == model1.num_classes
        assert model2.encoder_type == model1.encoder_type
        assert model2.hidden_dim == model1.hidden_dim
        assert model2.dropout >= 0.1 and model2.dropout <= 0.3
        assert model2.use_cuda == model1.use_cuda

        # Check temperature was restored
        assert torch.allclose(model2.temperature.data, torch.tensor([1.5]), rtol=1e-5)

        # Test forward pass gives same results with same input
        batch_size = 2
        seq_len = 50
        bytecode_tokens = torch.randint(0, 256, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        model1.eval()
        model2.eval()

        with torch.no_grad():
            outputs1 = model1(bytecode_tokens, attention_mask)
            outputs2 = model2(bytecode_tokens, attention_mask)

        # Predictions should be the same
        assert torch.allclose(outputs1["predicted_class"], outputs2["predicted_class"])

    def test_model_on_gpu_if_available(self):
        """Test model moves to GPU when available and requested."""
        # This test will be skipped if CUDA is not available
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        model = FunctionNameClassifier(
            num_classes=50,
            encoder_type="transformer",
            encoder_config={
                "vocab_size": 256,
                "d_model": 64,
                "nhead": 2,
                "num_layers": 1,
                "dim_feedforward": 128,
                "dropout": 0.1,
                "max_seq_len": 512,
                "use_positional_encoding": True,
                "use_cuda": True,
            },
            use_cuda=True,
        )

        # Check model is on GPU
        assert next(model.parameters()).is_cuda

        # Test forward pass on GPU
        batch_size = 2
        seq_len = 100
        bytecode_tokens = torch.randint(0, 256, (batch_size, seq_len)).cuda()
        attention_mask = torch.ones(batch_size, seq_len).cuda()

        outputs = model(bytecode_tokens, attention_mask)

        # Outputs should also be on GPU
        assert outputs["logits"].is_cuda

    def test_model_stays_on_cpu_when_requested(self):
        """Test model stays on CPU when use_cuda=False even if GPU is available."""
        model = FunctionNameClassifier(
            num_classes=50,
            encoder_type="transformer",
            encoder_config={
                "vocab_size": 256,
                "d_model": 64,
                "nhead": 2,
                "num_layers": 1,
                "dim_feedforward": 128,
                "dropout": 0.1,
                "max_seq_len": 512,
                "use_positional_encoding": True,
                "use_cuda": False,  # Explicitly false
            },
            use_cuda=False,
        )

        # Check model is on CPU
        assert not next(model.parameters()).is_cuda

        # Test forward pass
        batch_size = 2
        seq_len = 100
        bytecode_tokens = torch.randint(0, 256, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        outputs = model(bytecode_tokens, attention_mask)

        # Outputs should also be on CPU
        assert not outputs["logits"].is_cuda
