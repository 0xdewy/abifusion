# 05 — ML Training Pipeline

**Priority:** Medium | **Status:** Executed at scale; ML type head below targets | **Depends on:** #3 (self-contained checkpoint)

## Full-scale result (2026-06-01)

Ran at 500 contracts / 15 epochs — see `eval_output/plan05_fullscale.md`.
Type accuracy improved 1.9% → **32.1%** and function Top-3 → **85.1%**, but the
Plan targets (type ≥ 0.995) are **not met**: the ML type head is far below the
rule-based ~0.97. The **discriminating-features ablation is negative** (zeroing
them slightly *improves* type accuracy, −0.0095), so they add no value as
currently wired. Recommendation: keep the rule-based pipeline as the primary
type source, investigate/redesign the discriminating features, and scale data
further before relying on the ML head. Inference now passes discriminating
features (train/serve skew closed) regardless.

## Run log (2026-06-01)

Deliverable script `scripts/training/train_with_discriminating_features.py` is
implemented and was run end-to-end at small scale (20 verified mainnet
contracts → 385 parameter samples → transformer encoder, 5 epochs):

- Pipeline works: fetch (Etherscan V2) → dataset with discriminating features →
  train → held-out validation. The model learns (val type-accuracy 0.47 → 0.53,
  val loss 4.45 → 4.07 over 5 epochs); checkpoint is self-contained
  (`encoder_config` + `format_version: 2`, per #3).
- **Findings surfaced and fixed** while executing:
  - `ParameterPredictionModel.forward`/`compute_loss`/`calibrate_temperatures`
    did not align inputs/targets with the model's device → broke GPU training.
    Fixed by aligning to the actual parameter device.
  - The **CNN encoder was incompatible** with the token-id dataset (it expected
    pre-channelised `(batch, seq, 256)` input, with no embedding step).
    **Fixed:** the CNN now embeds token ids (`CNNWithEmbedding`) in both models,
    trains on real data, and round-trips through `load_model`.
  - The `--evaluate` flag runs the *full* pipeline and needs a function
    classifier; its existing checkpoint had drifted (missing
    `encoder_projection` keys). **Fixed:** the script now trains the function
    classifier too, and three pre-existing `evaluate_models`/`load_model` bugs
    (predict() call signature, return-key mismatch, `type_criterion_weighted`
    key) were fixed. `--evaluate` now runs end-to-end from both freshly trained
    and disk-loaded checkpoints, and `ABIReconstructorPipeline.load_pipeline`
    loads from canonical `checkpoints/`.
- **Caveat:** an earlier demo run wrote to `checkpoints/` and overwrote the
  interim `*_parameter_predictor.pth` (gitignored, unrecoverable). The script
  defaults to `checkpoints/plan05/` to avoid clobbering, but the function-
  classifier retrain was deliberately run into `checkpoints/` to replace the
  stale (unloadable) canonical function checkpoint.

**Remaining for full completion:** run at Plan-scale (500+ contracts, ~15
epochs) via `--addresses-file` to pursue the accuracy targets below (the small
functional retrain has low type accuracy, as expected).

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
