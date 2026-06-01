"""Simple integration tests for training pipeline components."""

import pytest
import torch
import tempfile
from pathlib import Path


def test_model_imports():
    """Test that all model classes can be imported."""
    from abi_reconstructor.models.function_name_classifier import FunctionNameClassifier
    from abi_reconstructor.models.parameter_prediction_model import (
        ParameterPredictionModel,
    )
    from abi_reconstructor.models.bytecode_transformer import (
        BytecodeTransformer,
        BytecodeCNNEncoder,
    )

    # Just verify imports work
    assert FunctionNameClassifier is not None
    assert ParameterPredictionModel is not None
    assert BytecodeTransformer is not None
    assert BytecodeCNNEncoder is not None


def test_training_imports():
    """Test that training modules can be imported."""
    from abi_reconstructor.training import train_ml_models
    from abi_reconstructor.training import parameter_dataset

    # Just verify imports work
    assert train_ml_models is not None
    assert parameter_dataset is not None


def test_simple_model_creation():
    """Test creating simple instances of models."""
    from abi_reconstructor.models.function_name_classifier import FunctionNameClassifier
    from abi_reconstructor.models.parameter_prediction_model import (
        ParameterPredictionModel,
    )

    # Create simple function name classifier
    fn_model = FunctionNameClassifier(
        num_classes=10,
        encoder_type="transformer",
        encoder_config={
            "vocab_size": 256,
            "d_model": 32,
            "nhead": 2,
            "num_layers": 1,
            "dim_feedforward": 64,
            "dropout": 0.1,
            "max_seq_len": 128,
            "use_positional_encoding": True,
            "use_cuda": False,
        },
        use_cuda=False,
    )

    # Create simple parameter prediction model
    param_model = ParameterPredictionModel(
        num_type_classes=10,
        max_parameters=4,
        encoder_type="transformer",
        encoder_config={
            "vocab_size": 256,
            "d_model": 32,
            "nhead": 2,
            "num_layers": 1,
            "dim_feedforward": 64,
            "dropout": 0.1,
            "max_seq_len": 128,
            "use_positional_encoding": True,
            "use_cuda": False,
        },
        use_cuda=False,
    )

    # Verify models were created
    assert fn_model is not None
    assert param_model is not None

    # Verify they have the expected attributes
    assert fn_model.num_classes == 10
    assert param_model.num_type_classes == 10
    assert param_model.max_parameters == 4


def test_model_forward_pass():
    """Test simple forward pass through models."""
    from abi_reconstructor.models.function_name_classifier import FunctionNameClassifier
    from abi_reconstructor.models.parameter_prediction_model import (
        ParameterPredictionModel,
    )

    # Create simple models
    fn_model = FunctionNameClassifier(
        num_classes=10,
        encoder_type="transformer",
        encoder_config={
            "vocab_size": 256,
            "d_model": 32,
            "nhead": 2,
            "num_layers": 1,
            "dim_feedforward": 64,
            "dropout": 0.1,
            "max_seq_len": 128,
            "use_positional_encoding": True,
            "use_cuda": False,
        },
        use_cuda=False,
    )

    param_model = ParameterPredictionModel(
        num_type_classes=10,
        max_parameters=4,
        encoder_type="transformer",
        encoder_config={
            "vocab_size": 256,
            "d_model": 32,
            "nhead": 2,
            "num_layers": 1,
            "dim_feedforward": 64,
            "dropout": 0.1,
            "max_seq_len": 128,
            "use_positional_encoding": True,
            "use_cuda": False,
        },
        use_cuda=False,
    )

    # Create test input
    batch_size = 2
    seq_len = 50
    bytecode_tokens = torch.randint(0, 256, (batch_size, seq_len))

    # Test forward pass
    with torch.no_grad():
        fn_outputs = fn_model(bytecode_tokens)
        param_outputs = param_model(bytecode_tokens)

    # Check outputs have expected keys
    assert "predicted_class" in fn_outputs
    assert "predicted_count" in param_outputs
    assert "predicted_types" in param_outputs

    # Check output shapes
    assert fn_outputs["predicted_class"].shape == (batch_size,)
    assert param_outputs["predicted_count"].shape == (batch_size,)
    assert param_outputs["predicted_types"].shape == (batch_size, 4)  # max_parameters=4


def test_model_save_load():
    """Test model save and load functionality."""
    from abi_reconstructor.models.function_name_classifier import FunctionNameClassifier
    from abi_reconstructor.models.parameter_prediction_model import (
        ParameterPredictionModel,
    )

    # Create simple models
    fn_model = FunctionNameClassifier(
        num_classes=10,
        encoder_type="transformer",
        encoder_config={
            "vocab_size": 256,
            "d_model": 32,
            "nhead": 2,
            "num_layers": 1,
            "dim_feedforward": 64,
            "dropout": 0.1,
            "max_seq_len": 128,
            "use_positional_encoding": True,
            "use_cuda": False,
        },
        use_cuda=False,
    )

    param_model = ParameterPredictionModel(
        num_type_classes=10,
        max_parameters=4,
        encoder_type="transformer",
        encoder_config={
            "vocab_size": 256,
            "d_model": 32,
            "nhead": 2,
            "num_layers": 1,
            "dim_feedforward": 64,
            "dropout": 0.1,
            "max_seq_len": 128,
            "use_positional_encoding": True,
            "use_cuda": False,
        },
        use_cuda=False,
    )

    # Save models
    with tempfile.TemporaryDirectory() as tmpdir:
        fn_path = Path(tmpdir) / "fn_model.pth"
        param_path = Path(tmpdir) / "param_model.pth"

        fn_model.save_model(str(fn_path))
        param_model.save_model(str(param_path))

        # Check files exist
        assert fn_path.exists()
        assert param_path.exists()

        # Load models back
        loaded_fn_model = FunctionNameClassifier.load_model(
            str(fn_path), use_cuda=False
        )
        loaded_param_model = ParameterPredictionModel.load_model(
            str(param_path), use_cuda=False
        )

        # Check architecture matches
        assert loaded_fn_model.num_classes == fn_model.num_classes
        assert loaded_param_model.num_type_classes == param_model.num_type_classes
        assert loaded_param_model.max_parameters == param_model.max_parameters


def test_training_components_exist():
    """Test that training components exist and can be imported."""
    from abi_reconstructor.training.train_ml_models import (
        ModelTrainer,
        FunctionNameTrainer,
        ParameterPredictionTrainer,
        prepare_datasets,
    )

    # Just verify imports work
    assert ModelTrainer is not None
    assert FunctionNameTrainer is not None
    assert ParameterPredictionTrainer is not None
    assert prepare_datasets is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
