# 03 — Self-Contained Checkpoint

**Priority:** Medium | **Status:** ✅ Complete | **Depends on:** —

## Goal

The model checkpoint must contain everything needed to reconstruct the model — `state_dict()` plus `encoder_config`. Currently only `state_dict()` is saved. The load path reconstructs `encoder_config` from scratch, guessing `d_model` from weight shapes. This sometimes crashes.

## Current behavior

**Save path** (`train_ml_models.py` or model save method):
```python
torch.save(model.state_dict(), path)
```

**Load path** (`train_ml_models.py`):
```python
# Reconstructs encoder_config from scratch — may miss d_model
encoder_config = {
    "vocab_size": data_dict.get("vocab_size", 257),
    "d_model": hidden_dim,  # GUESSED, not from checkpoint
    ...
}
model = ParameterPredictionModel(encoder_config=encoder_config, ...)
model.load_state_dict(torch.load(path))
```

The bug: `d_model` is inferred from the caller context, not from the checkpoint. If `hidden_dim` doesn't match, the model silently produces wrong output.

## Fix

Save `encoder_config` as JSON alongside the `.pth` file:

```python
# Save
checkpoint = {
    "state_dict": model.state_dict(),
    "encoder_config": model.encoder_config,
    "num_type_classes": model.num_type_classes,
    "max_parameters": model.max_parameters,
}
torch.save(checkpoint, path)

# Load
checkpoint = torch.load(path, weights_only=False)
model = ParameterPredictionModel(
    num_type_classes=checkpoint["num_type_classes"],
    encoder_config=checkpoint["encoder_config"],
    ...
)
model.load_state_dict(checkpoint["state_dict"])
```

At save time, assert that `encoder_config` contains all required keys: `vocab_size`, `d_model`, `nhead`, `num_layers`, `dim_feedforward`, `dropout`.

## Files changed

- `abi_reconstructor/models/parameter_prediction_model.py` — add `save_checkpoint()` and `load_checkpoint()` static methods
- `abi_reconstructor/training/train_ml_models.py` — use new save/load methods
- `tests/models/test_parameter_prediction_model.py` — add roundtrip test

## Test

```python
def test_checkpoint_roundtrip_with_discriminating_features():
    model = ParameterPredictionModel(...)
    with tempfile.NamedTemporaryFile(suffix=".pth") as f:
        model.save_checkpoint(f.name)
        loaded = ParameterPredictionModel.load_checkpoint(f.name)
    # Same input → same output
    x = torch.randn(2, 512, 256)
    assert torch.allclose(model(x)["type_logits"], loaded(x)["type_logits"])
```

## Risk

- Existing checkpoints in `checkpoints/` may not load after this change. Add a migration path: if loading old format (bare state_dict), reconstruct config from legacy defaults.
