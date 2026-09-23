# 04 — Device Management Cleanup

**Priority:** Medium | **Status:** ✅ Complete | **Depends on:** —

## Goal

Eliminate ad-hoc `to(device)` calls scattered through the codebase. Route everything through `DeviceManager`. Fix the test that triggers CUDA code paths when `use_cuda=False`.

## Current state

`DeviceManager` exists at `abifusion/device.py` but is used inconsistently. Models construct their own device logic in `__init__`:

```python
# parameter_prediction_model.py
if use_cuda and torch.cuda.is_available():
    self.cuda()
```

```python
# bytecode_transformer.py / function_name_classifier.py
self._device = get_device_for_cuda_flag(use_cuda)
```

Tests that set `use_cuda=False` sometimes trigger CUDA initialization because ad-hoc device logic checks `torch.cuda.is_available()` instead of using the explicit flag.

## Fix

1. **Make `DeviceManager` the single entry point** for all device decisions.
2. **Remove all direct `to(device)` calls** from model `__init__` methods. Models stay on CPU by default; callers move them via `DeviceManager`.
3. **Add a `device` parameter** to all model constructors. Default to `None` (CPU). `DeviceManager.get_device()` returns the configured device.
4. **Update tests** to use the explicit `device` parameter instead of relying on `use_cuda` flags being respected.

## API change

```python
# Before
model = ParameterPredictionModel(use_cuda=False, ...)  # ad-hoc

# After
device = DeviceManager.get_device()  # "cuda:0" or "cpu"
model = ParameterPredictionModel(device=device, ...)    # explicit
model = DeviceManager.to_device(model)                  # or move after init
```

## Files changed

- `abifusion/device.py` — audit and simplify, make `get_device()` the canonical entry point
- `abifusion/models/parameter_prediction_model.py` — remove `use_cuda`, add `device` param
- `abifusion/models/function_name_classifier.py` — same
- `abifusion/models/bytecode_transformer.py` — same
- `abifusion/features/bytecode_features.py` — same
- `tests/` — update model construction in all tests
- `abifusion/training/train_ml_models.py` — use DeviceManager

## Risk

- Model tests currently rely on `use_cuda=False` implicitly. After this change, they must explicitly pass `device=torch.device("cpu")`. This is a test-only impact — production code already has a clear device selection point via CLI flags.
