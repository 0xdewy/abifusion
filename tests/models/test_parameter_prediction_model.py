"""Unit tests for ParameterPredictionModel."""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from abi_reconstructor.models.parameter_prediction_model import ParameterPredictionModel


class TestParameterPredictionModel:
    """Test suite for ParameterPredictionModel."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        model = ParameterPredictionModel(
            num_type_classes=24,
            max_parameters=12,
            encoder_type="transformer",
            use_cuda=False,
        )

        assert model.num_type_classes == 24
        assert model.max_parameters == 12
        assert model.encoder_type == "transformer"
        assert model.hidden_dim == 256
        assert model.dropout == 0.2
        assert model.use_function_conditioning == True
        assert model.num_function_classes is None
        assert model.use_cuda == False
        assert isinstance(model.encoder, torch.nn.Module)
        assert model.function_embedding is None  # No num_function_classes provided

    def test_init_with_function_conditioning(self):
        """Test initialization with function conditioning enabled."""
        model = ParameterPredictionModel(
            num_type_classes=24,
            max_parameters=12,
            encoder_type="transformer",
            use_function_conditioning=True,
            num_function_classes=50,
            use_cuda=False,
        )

        assert model.use_function_conditioning == True
        assert model.num_function_classes == 50
        assert isinstance(model.function_embedding, torch.nn.Embedding)
        assert model.function_embedding.num_embeddings == 50
        assert model.function_embedding.embedding_dim == 64  # hidden_dim // 4

    def test_init_without_function_conditioning(self):
        """Test initialization without function conditioning."""
        model = ParameterPredictionModel(
            num_type_classes=24,
            max_parameters=12,
            encoder_type="transformer",
            use_function_conditioning=False,
            num_function_classes=50,  # Should be ignored
            use_cuda=False,
        )

        assert model.use_function_conditioning == False
        assert model.function_embedding is None

    def test_init_cnn_encoder(self):
        """Test initialization with CNN encoder."""
        model = ParameterPredictionModel(
            num_type_classes=24,
            max_parameters=12,
            encoder_type="cnn",
            use_cuda=False,
        )

        assert model.encoder_type == "cnn"
        assert hasattr(model.encoder, "forward")

    def test_init_custom_config(self):
        """Test initialization with custom configuration."""
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

        model = ParameterPredictionModel(
            num_type_classes=50,
            max_parameters=8,
            encoder_type="transformer",
            encoder_config=encoder_config,
            hidden_dim=128,
            dropout=0.3,
            use_function_conditioning=True,
            num_function_classes=100,
            use_cuda=False,
        )

        assert model.num_type_classes == 50
        assert model.max_parameters == 8
        assert model.hidden_dim == 128
        assert model.dropout == 0.3
        assert model.num_function_classes == 100
        assert model.encoder_config["d_model"] == 128
        assert isinstance(model.function_embedding, torch.nn.Embedding)

    def test_forward_basic_no_conditioning(self):
        """Test forward pass without function name conditioning."""
        model = ParameterPredictionModel(
            num_type_classes=24,
            max_parameters=6,
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
            use_function_conditioning=False,
            use_cuda=False,
        )

        batch_size = 2
        seq_len = 100
        bytecode_tokens = torch.randint(0, 256, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        outputs = model(bytecode_tokens, attention_mask)

        # Check output structure
        expected_keys = [
            "count_logits",
            "count_probs",
            "count_confidence",
            "predicted_count",
            "type_logits",
            "type_probs",
            "type_confidences",
            "predicted_types",
            "mask_logits",
            "mask_probs",
            "predicted_mask",
            "features",
        ]

        for key in expected_keys:
            assert key in outputs, f"Missing key: {key}"

        # Check shapes
        assert outputs["count_logits"].shape == (batch_size, 7)  # max_parameters + 1
        assert outputs["count_probs"].shape == (batch_size, 7)
        assert outputs["count_confidence"].shape == (batch_size,)
        assert outputs["predicted_count"].shape == (batch_size,)

        assert outputs["type_logits"].shape == (
            batch_size,
            6,
            24,
        )  # (batch, max_params, num_types)
        assert outputs["type_probs"].shape == (batch_size, 6, 24)
        assert outputs["type_confidences"].shape == (batch_size, 6)
        assert outputs["predicted_types"].shape == (batch_size, 6)

        assert outputs["mask_logits"].shape == (batch_size, 6)
        assert outputs["mask_probs"].shape == (batch_size, 6)
        assert outputs["predicted_mask"].shape == (batch_size, 6)

        # Check probability properties
        count_probs_sum = outputs["count_probs"].sum(dim=1)
        assert torch.allclose(count_probs_sum, torch.ones(batch_size), rtol=1e-5)

        # Type probabilities should sum to 1 for each position
        for i in range(6):
            type_probs_sum = outputs["type_probs"][:, i, :].sum(dim=1)
            assert torch.allclose(type_probs_sum, torch.ones(batch_size), rtol=1e-5)

        # Mask probabilities should be between 0 and 1
        assert torch.all(outputs["mask_probs"] >= 0)
        assert torch.all(outputs["mask_probs"] <= 1)

        # Predicted mask should be binary
        assert torch.all(
            (outputs["predicted_mask"] == 0) | (outputs["predicted_mask"] == 1)
        )

    def test_forward_with_function_conditioning(self):
        """Test forward pass with function name conditioning."""
        model = ParameterPredictionModel(
            num_type_classes=24,
            max_parameters=6,
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
            use_function_conditioning=True,
            num_function_classes=50,
            use_cuda=False,
        )

        batch_size = 2
        seq_len = 100
        bytecode_tokens = torch.randint(0, 256, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        function_name_idx = torch.randint(0, 50, (batch_size,))

        outputs = model(bytecode_tokens, attention_mask, function_name_idx)

        # Should have all outputs
        assert "count_logits" in outputs
        assert outputs["count_logits"].shape == (batch_size, 7)

        # Features should be projected to hidden_dim after concat_projection
        assert outputs["features"].shape == (batch_size, 64)

    def test_forward_with_wrong_shape_function_idx(self):
        """Test forward pass with function_name_idx having wrong shape (2D instead of 1D)."""
        model = ParameterPredictionModel(
            num_type_classes=24,
            max_parameters=6,
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
            use_function_conditioning=True,
            num_function_classes=50,
            use_cuda=False,
        )

        batch_size = 2
        seq_len = 100
        bytecode_tokens = torch.randint(0, 256, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        # Wrong shape: 2D instead of 1D
        function_name_idx = torch.randint(0, 50, (batch_size, seq_len))

        # Should handle gracefully (pad features)
        outputs = model(bytecode_tokens, attention_mask, function_name_idx)

        assert outputs["count_logits"].shape == (batch_size, 7)
        # Features should be projected to hidden_dim after concat_projection
        assert outputs["features"].shape == (batch_size, 64)

    def test_forward_with_none_function_idx(self):
        """Test forward pass with function_name_idx=None when conditioning is enabled."""
        model = ParameterPredictionModel(
            num_type_classes=24,
            max_parameters=6,
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
            use_function_conditioning=True,
            num_function_classes=50,
            use_cuda=False,
        )

        batch_size = 2
        seq_len = 100
        bytecode_tokens = torch.randint(0, 256, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        # function_name_idx is None
        outputs = model(bytecode_tokens, attention_mask, None)

        assert outputs["count_logits"].shape == (batch_size, 7)
        # Features should be projected to hidden_dim
        assert outputs["features"].shape == (batch_size, 64)

    def test_forward_empty_batch(self):
        """Test forward pass with empty batch."""
        model = ParameterPredictionModel(
            num_type_classes=24,
            max_parameters=6,
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

        assert outputs["count_logits"].shape == (0, 7)
        assert outputs["type_logits"].shape == (0, 6, 24)
        assert outputs["mask_logits"].shape == (0, 6)

    def test_compute_loss_basic(self):
        """Test compute_loss with basic input."""
        model = ParameterPredictionModel(
            num_type_classes=24,
            max_parameters=6,
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
        parameter_count = torch.randint(0, 7, (batch_size,))  # 0 to max_parameters
        parameter_types = torch.randint(0, 24, (batch_size, 6))
        parameter_mask = torch.randint(0, 2, (batch_size, 6)).float()
        attention_mask = torch.ones(batch_size, seq_len)

        loss_dict = model.compute_loss(
            bytecode_tokens,
            parameter_count,
            parameter_types,
            parameter_mask,
            attention_mask,
        )

        expected_keys = [
            "total_loss",
            "count_loss",
            "type_loss",
            "mask_loss",
            "loss_weights",
        ]
        for key in expected_keys:
            assert key in loss_dict, f"Missing key: {key}"

        # All losses should be positive
        assert loss_dict["total_loss"].item() > 0
        assert loss_dict["count_loss"].item() > 0
        assert loss_dict["mask_loss"].item() > 0

        # Type loss might be 0 if no valid positions
        # But with random data, likely some valid positions

        # Check loss weights
        assert loss_dict["loss_weights"]["count"] == 1.0
        assert loss_dict["loss_weights"]["type"] == 1.0
        assert loss_dict["loss_weights"]["mask"] == 0.5

    def test_compute_loss_with_function_conditioning(self):
        """Test compute_loss with function name conditioning."""
        model = ParameterPredictionModel(
            num_type_classes=24,
            max_parameters=6,
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
            use_function_conditioning=True,
            num_function_classes=50,
            use_cuda=False,
        )

        batch_size = 4
        seq_len = 100
        bytecode_tokens = torch.randint(0, 256, (batch_size, seq_len))
        parameter_count = torch.randint(0, 7, (batch_size,))
        parameter_types = torch.randint(0, 24, (batch_size, 6))
        parameter_mask = torch.randint(0, 2, (batch_size, 6)).float()
        attention_mask = torch.ones(batch_size, seq_len)
        function_name_idx = torch.randint(0, 50, (batch_size,))

        loss_dict = model.compute_loss(
            bytecode_tokens,
            parameter_count,
            parameter_types,
            parameter_mask,
            attention_mask,
            function_name_idx,
        )

        assert loss_dict["total_loss"].item() > 0

    def test_compute_loss_custom_loss_weights(self):
        """Test compute_loss with custom loss weights."""
        model = ParameterPredictionModel(
            num_type_classes=24,
            max_parameters=6,
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
        parameter_count = torch.randint(0, 7, (batch_size,))
        parameter_types = torch.randint(0, 24, (batch_size, 6))
        parameter_mask = torch.randint(0, 2, (batch_size, 6)).float()
        attention_mask = torch.ones(batch_size, seq_len)

        custom_weights = {
            "count": 2.0,
            "type": 1.5,
            "mask": 0.8,
        }

        loss_dict = model.compute_loss(
            bytecode_tokens,
            parameter_count,
            parameter_types,
            parameter_mask,
            attention_mask,
            loss_weights=custom_weights,
        )

        assert loss_dict["loss_weights"] == custom_weights

    def test_compute_loss_return_outputs(self):
        """Test compute_loss with return_outputs=True."""
        model = ParameterPredictionModel(
            num_type_classes=24,
            max_parameters=6,
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
        parameter_count = torch.randint(0, 7, (batch_size,))
        parameter_types = torch.randint(0, 24, (batch_size, 6))
        parameter_mask = torch.randint(0, 2, (batch_size, 6)).float()
        attention_mask = torch.ones(batch_size, seq_len)

        loss_dict = model.compute_loss(
            bytecode_tokens,
            parameter_count,
            parameter_types,
            parameter_mask,
            attention_mask,
            return_outputs=True,
        )

        assert "outputs" in loss_dict
        assert isinstance(loss_dict["outputs"], dict)
        assert "count_logits" in loss_dict["outputs"]

    def test_predict_basic(self):
        """Test predict method with basic input."""
        model = ParameterPredictionModel(
            num_type_classes=24,
            max_parameters=6,
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
            assert "valid_counts" in pred
            assert "valid_count_probs" in pred
            assert "best_count" in pred
            assert "best_count_prob" in pred
            assert "type_predictions" in pred
            assert "count_confidence" in pred
            assert "avg_type_confidence" in pred

            # Check type predictions structure
            assert len(pred["type_predictions"]) == 6  # max_parameters
            for type_pred in pred["type_predictions"]:
                assert "position" in type_pred
                assert "valid_types" in type_pred
                assert "valid_type_probs" in type_pred
                assert "best_type" in type_pred
                assert "best_type_prob" in type_pred
                assert "mask_prob" in type_pred
                assert "is_valid" in type_pred

    def test_predict_with_custom_thresholds(self):
        """Test predict method with custom thresholds."""
        model = ParameterPredictionModel(
            num_type_classes=24,
            max_parameters=6,
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

        predictions = model.predict(
            bytecode_tokens,
            attention_mask,
            count_threshold=0.2,
            type_threshold=0.15,
            mask_threshold=0.6,
        )

        for pred in predictions["predictions"]:
            # Check count threshold
            if pred["valid_count_probs"]:
                assert all(prob >= 0.2 for prob in pred["valid_count_probs"])

            # Check type thresholds
            for type_pred in pred["type_predictions"]:
                if type_pred["valid_type_probs"]:
                    assert all(prob >= 0.15 for prob in type_pred["valid_type_probs"])

                # Check mask threshold
                assert type_pred["is_valid"] == (type_pred["mask_prob"] >= 0.6)

    def test_temperature_calibration_method(self):
        """Test temperature calibration method."""
        model = ParameterPredictionModel(
            num_type_classes=24,
            max_parameters=6,
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

        # Create mock validation loader
        mock_val_loader = []
        for _ in range(3):  # 3 batches
            mock_batch = {
                "bytecode_tokens": torch.randint(0, 256, (2, 100)),
                "parameter_count": torch.randint(0, 7, (2,)),
                "parameter_types": torch.randint(0, 24, (2, 6)),
                "parameter_mask": torch.randint(0, 2, (2, 6)).float(),
            }
            mock_val_loader.append(mock_batch)

        # Store initial temperatures
        initial_temperatures = {
            "count": model.temperature_count.item(),
            "type": model.temperature_type.item(),
            "mask": model.temperature_mask.item(),
        }

        # Run calibration
        model.calibrate_temperatures(mock_val_loader, lr=0.01, max_iters=5)

        # Temperatures should stay in valid range (clamped between 0.1 and 10.0)
        assert 0.1 <= model.temperature_count.item() <= 10.0
        assert 0.1 <= model.temperature_type.item() <= 10.0
        assert 0.1 <= model.temperature_mask.item() <= 10.0

    def test_save_and_load_model(self, tmp_path):
        """Test model save and load functionality."""
        # Create and save model
        model1 = ParameterPredictionModel(
            num_type_classes=50,
            max_parameters=8,
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
            use_function_conditioning=True,
            num_function_classes=100,
            use_cuda=False,
        )

        # Set temperatures to non-default values
        model1.temperature_count.data.fill_(1.2)
        model1.temperature_type.data.fill_(1.3)
        model1.temperature_mask.data.fill_(1.4)

        # Save model
        save_path = tmp_path / "test_model.pth"
        model1.save_model(str(save_path))

        # Load model
        model2 = ParameterPredictionModel.load_model(str(save_path), use_cuda=False)

        # Check architecture matches
        assert model2.num_type_classes == model1.num_type_classes
        assert model2.max_parameters == model1.max_parameters
        assert model2.encoder_type == model1.encoder_type
        assert model2.hidden_dim == model1.hidden_dim
        assert 0.0 <= model2.dropout <= 0.5
        assert model2.use_function_conditioning == model1.use_function_conditioning
        assert model2.num_function_classes == model1.num_function_classes
        assert 0.0 <= model2.dropout <= 0.5

        # Check temperatures were restored
        assert torch.allclose(
            model2.temperature_count.data, torch.tensor([1.2]), rtol=1e-5
        )
        assert torch.allclose(
            model2.temperature_type.data, torch.tensor([1.3]), rtol=1e-5
        )
        assert torch.allclose(
            model2.temperature_mask.data, torch.tensor([1.4]), rtol=1e-5
        )

        # Test forward pass gives same results with same input
        batch_size = 2
        seq_len = 50
        bytecode_tokens = torch.randint(0, 256, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        function_name_idx = torch.randint(0, 100, (batch_size,))

        model1.eval()
        model2.eval()

        with torch.no_grad():
            outputs1 = model1(bytecode_tokens, attention_mask, function_name_idx)
            outputs2 = model2(bytecode_tokens, attention_mask, function_name_idx)

        # Count predictions should be the same
        assert torch.allclose(outputs1["predicted_count"], outputs2["predicted_count"])

    def test_load_from_checkpoint_roundtrip(self, tmp_path):
        """load_from_checkpoint reconstructs model with NO caller-supplied arguments."""
        model1 = ParameterPredictionModel(
            num_type_classes=22,
            max_parameters=12,
            encoder_type="cnn",
            encoder_config={
                "vocab_size": 257,
                "d_model": 256,
                "nhead": 8,
                "num_layers": 3,
                "dim_feedforward": 512,
                "dropout": 0.2,
            },
            hidden_dim=256,
            dropout=0.2,
            use_function_conditioning=False,
            use_cuda=False,
        )
        model1.eval()

        save_path = tmp_path / "self_contained.pth"
        model1.save_model(str(save_path))

        # Load with ZERO caller-supplied arguments
        model2 = ParameterPredictionModel.load_from_checkpoint(str(save_path))

        assert model2.num_type_classes == 22
        assert model2.max_parameters == 12
        assert model2.encoder_type == "cnn"
        assert model2.hidden_dim == 256
        assert model2.encoder_config["d_model"] == 256
        assert model2.encoder_config["vocab_size"] == 257

        # Same input → same output
        tokens = torch.randint(0, 256, (2, 512))
        with torch.no_grad():
            out1 = model1(tokens)
            out2 = model2(tokens)

        assert torch.allclose(out1["type_logits"], out2["type_logits"], atol=1e-6)
        assert torch.allclose(out1["count_logits"], out2["count_logits"], atol=1e-6)

    def test_load_from_checkpoint_with_discriminating_layers(self, tmp_path):
        """load_from_checkpoint preserves discriminating feature projection layers."""
        model1 = ParameterPredictionModel(
            num_type_classes=22,
            max_parameters=12,
            encoder_type="cnn",
            hidden_dim=256,
            use_function_conditioning=False,
            use_cuda=False,
        )
        model1.eval()

        save_path = tmp_path / "with_disc.pth"
        model1.save_model(str(save_path))

        model2 = ParameterPredictionModel.load_from_checkpoint(str(save_path))

        # Discriminating layers should be in state dict and produce same output
        tokens = torch.randint(0, 256, (2, 512))
        disc = torch.randn(2, 7)

        with torch.no_grad():
            out1 = model1(tokens, discriminating_features=disc)
            out2 = model2(tokens, discriminating_features=disc)

        assert torch.allclose(out1["type_logits"], out2["type_logits"], atol=1e-6)

    def test_encoder_config_validation_rejects_missing_keys(self):
        """Constructor raises ValueError if encoder_config misses required keys."""
        with pytest.raises(ValueError, match="Missing encoder_config keys"):
            ParameterPredictionModel(
                num_type_classes=22,
                max_parameters=12,
                encoder_type="transformer",
                encoder_config={"vocab_size": 257},  # missing d_model, nhead, etc.
                hidden_dim=256,
                use_cuda=False,
            )

    def test_encoder_config_validation_skips_cnn(self):
        """CNN encoder doesn't require transformer-specific keys."""
        model = ParameterPredictionModel(
            num_type_classes=22,
            max_parameters=12,
            encoder_type="cnn",
            encoder_config={"vocab_size": 257, "dropout": 0.2},  # minimal
            hidden_dim=256,
            use_function_conditioning=False,
            use_cuda=False,
        )
        assert model.encoder_type == "cnn"

    def test_model_on_gpu_if_available(self):
        """Test model moves to GPU when available and requested."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        model = ParameterPredictionModel(
            num_type_classes=24,
            max_parameters=6,
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
        function_name_idx = torch.randint(0, 50, (batch_size,)).cuda()

        outputs = model(bytecode_tokens, attention_mask, function_name_idx)

        # Outputs should also be on GPU
        assert outputs["count_logits"].is_cuda

    def test_edge_case_all_parameters_invalid(self):
        """Test compute_loss when all parameter positions are invalid (mask=0)."""
        model = ParameterPredictionModel(
            num_type_classes=24,
            max_parameters=6,
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
        parameter_count = torch.tensor([0, 0])  # No parameters
        parameter_types = torch.randint(0, 24, (batch_size, 6))
        parameter_mask = torch.zeros(batch_size, 6).float()  # All invalid
        attention_mask = torch.ones(batch_size, seq_len)

        loss_dict = model.compute_loss(
            bytecode_tokens,
            parameter_count,
            parameter_types,
            parameter_mask,
            attention_mask,
        )

        # Type loss should be 0 when no valid positions
        assert loss_dict["type_loss"] == 0.0
        # But total loss should still be positive (count and mask losses)
        assert loss_dict["total_loss"].item() > 0
