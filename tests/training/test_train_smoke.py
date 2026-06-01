"""Smoke tests for the training pipeline in train_ml_models.py.

Covers one parameter-prediction training step end to end (compute_batch_loss →
backward → optimizer step) and the self-contained checkpoint round-trip
introduced in plan #3 (the saved checkpoint must carry encoder_config and the
architecture needed to rebuild the model). Uses a tiny CNN model so the test is
fast and needs no real dataset.
"""

import torch

from abi_reconstructor.models.parameter_prediction_model import ParameterPredictionModel
from abi_reconstructor.training.train_ml_models import ParameterPredictionTrainer

NUM_TYPES = 4
MAX_PARAMS = 4
HIDDEN = 32
SEQ = 64


def _tiny_model():
    return ParameterPredictionModel(
        num_type_classes=NUM_TYPES,
        max_parameters=MAX_PARAMS,
        encoder_type="cnn",
        hidden_dim=HIDDEN,
        use_function_conditioning=False,
        use_cuda=False,
    )


def _tiny_trainer(model, tmp_path):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return ParameterPredictionTrainer(
        model=model,
        train_loader=None,
        val_loader=None,
        optimizer=optimizer,
        scheduler=None,
        use_cuda=False,
        checkpoint_dir=str(tmp_path),
        model_name="param_predictor_test",
    )


def _synthetic_batch(batch_size=2):
    # Deliberately leave tensors on CPU: the model/trainer may place weights on
    # GPU (via DeviceManager), and compute_loss/forward must align inputs to the
    # model's device. This guards the device-handling fix.
    tokens = (
        torch.randint(0, 256, (batch_size, SEQ))
    )
    return {
        "bytecode_tokens": tokens,
        "parameter_count": torch.tensor([1, 2][:batch_size]),
        "parameter_types": torch.randint(0, NUM_TYPES, (batch_size, MAX_PARAMS)),
        "parameter_mask": torch.zeros(batch_size, MAX_PARAMS),
    }


class TestTrainingStep:
    def test_compute_batch_loss_is_differentiable(self, tmp_path):
        model = _tiny_model()
        trainer = _tiny_trainer(model, tmp_path)

        loss_dict = trainer.compute_batch_loss(_synthetic_batch())
        loss = loss_dict["total_loss"]

        assert torch.is_tensor(loss)
        assert torch.isfinite(loss)
        assert loss.requires_grad

    def test_one_optimizer_step_changes_weights(self, tmp_path):
        model = _tiny_model()
        trainer = _tiny_trainer(model, tmp_path)
        # A parameter from a trainable head.
        ref = next(model.count_predictor.parameters())
        before = ref.detach().clone()

        loss = trainer.compute_batch_loss(_synthetic_batch())["total_loss"]
        trainer.optimizer.zero_grad()
        loss.backward()
        trainer.optimizer.step()

        assert not torch.equal(before, ref.detach())


class TestCheckpointRoundTrip:
    def test_checkpoint_is_self_contained(self, tmp_path):
        model = _tiny_model()
        trainer = _tiny_trainer(model, tmp_path)

        trainer.save_checkpoint("ckpt.pth")
        ckpt = torch.load(tmp_path / "ckpt.pth", map_location="cpu", weights_only=False)

        # Plan #3: architecture is reconstructable without guessing.
        assert ckpt["encoder_config"] == model.encoder_config
        assert ckpt["num_type_classes"] == NUM_TYPES
        assert ckpt["max_parameters"] == MAX_PARAMS
        assert ckpt["encoder_type"] == "cnn"
        assert ckpt["hidden_dim"] == HIDDEN
        assert ckpt["format_version"] == 2
        assert "model_state_dict" in ckpt

    def test_checkpoint_reloads_into_fresh_model(self, tmp_path):
        model = _tiny_model()
        trainer = _tiny_trainer(model, tmp_path)
        trainer.save_checkpoint("ckpt.pth")

        fresh = _tiny_model()
        fresh_trainer = _tiny_trainer(fresh, tmp_path)
        fresh_trainer.load_checkpoint("ckpt.pth")

        # Weights match the saved model after load.
        for k, v in model.state_dict().items():
            assert torch.equal(v, fresh.state_dict()[k])

    def test_load_model_with_class_weights_roundtrips(self, tmp_path):
        # Models trained with class weights carry a type_criterion_weighted
        # buffer; load_model must reconstruct it instead of failing on an
        # unexpected key.
        model = _tiny_model()
        model.set_type_class_weights(torch.rand(NUM_TYPES))
        path = tmp_path / "weighted.pth"
        model.save_model(str(path))

        loaded = ParameterPredictionModel.load_model(str(path), use_cuda=False)
        assert loaded.num_type_classes == NUM_TYPES
        assert hasattr(loaded, "type_criterion_weighted")
