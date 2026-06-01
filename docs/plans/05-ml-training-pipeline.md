# 05 — ML Training Pipeline

**Priority:** Medium | **Status:** Spec ready | **Depends on:** #3 (self-contained checkpoint)

## Goal

Train the `ParameterPredictionModel` with discriminating features on real contract data. Currently the pipeline is fully wired but untrained. A trained model would close the remaining selector extraction gap (especially for complex routers like Uniswap V3 and 1inch), and would produce genuinely learned type predictions rather than relying solely on the 4byte DB.

## Current state

The full pipeline is wired:
- ✓ Discriminating features extracted in `ParameterDataset._prepare_data_dict()`
- ✓ Features loaded in `ParameterDatasetTorch`
- ✓ Collate function stacks features into batches
- ✓ `ParameterPredictionModel.forward()` accepts `discriminating_features`
- ✓ `ParameterPredictionModel.compute_loss()` propagates features
- ✓ `train_ml_models.py` passes features from batch to model

What's missing: a training run with enough contracts to produce meaningful weights.

## Plan

### 1. Build training dataset (Phase 1 of training script)

Fetch 500+ verified contracts from Etherscan. For each contract:
- Extract runtime bytecode via `eth_getCode`
- Extract ground-truth ABI (function names + parameter types)
- Extract compiler version for hold-out splitting
- Build `ParameterDataset` with discriminating features

### 2. Train the model

- Split: random 70/15/15 (train/val/test) OR compiler-holdout if #2 is done first
- Model: CNN encoder, `hidden_dim=256`, `max_parameters=12`, with discriminating projection
- Train for N epochs until validation loss plateaus
- Save checkpoint with full `encoder_config` (depends on #3)

### 3. Evaluate

- Run `benchmark.py` on the held-out test set
- Compare type recovery metrics against the current rule-based pipeline
- Target: prototype F1 ≥ current 0.972 on the test set (proves ML doesn't regress)
- Stretch: prototype F1 ≥ 0.98 (proves discriminating features help beyond 4byte DB)

### 4. Analyze

- Which functions does the ML model get right that the rule-based pipeline gets wrong?
- Are the discriminating features actually used? (Ablation: zero them out and measure degradation)
- Does the ML model generalize to unseen compiler versions? (If #2 also done)

## Deliverable

Script `scripts/training/train_with_discriminating_features.py` that:
1. Fetches contracts from Etherscan
2. Builds annotated ParameterDataset
3. Trains ParameterPredictionModel
4. Saves checkpoint
5. Evaluates on held-out test set via benchmark

## Metrics to track

| Metric | Current (rule-based) | Target (ML) |
|---|---|---|
| Prototype F1 | 0.972 | ≥ 0.972 |
| Type Accuracy | 0.995 | ≥ 0.995 |
| Full Exact Match | 0.956 | ≥ 0.960 |
| Per-compiler degradation | TBD | < 5 pp |

## Dependencies

- #3 (checkpoint save/load) — required for proper training workflow
- #2 (compiler holdout) — optional, would make the evaluation more interesting
- Etherscan API access — available
- GPU or patience — training on CPU is feasible for 500 contracts
