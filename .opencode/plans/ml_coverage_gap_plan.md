# Plan: ML for Coverage Gaps (4byte Disambiguation)

## Context

- **Problem**: 244 selectors have NO 4byte coverage and evmole fails to find them (returns nothing)
- **Goal**: Generalize to bytecode patterns, not memorize specific selectors
- **Approach**: Extend existing ParameterPredictionModel with type category prediction + combinatorial fallback

---

## Phase 1: Understand the Baseline

### 1.1 Run existing fusion on test set
- Baseline: 96.4% accuracy (6,500/6,744)
- 244 error cases = selectors with NO 4byte coverage

### 1.2 Analyze 244 hard cases
- evmole fails completely for these (doesn't find the function at all)
- Same function name can have DIFFERENT types across contracts
- Function families: Uniswap V2/V3 callbacks, NFT setters, DeFi helpers
- Type distribution: 33% tuple, 19% address, 14% bytes, 11% uint256, 8% tuple[]

### 1.3 Test current model on hard cases (zero-shot)
- Run ParameterPredictionModel on 244 hard cases
- See if it predicts anything useful without 4byte

---

## Phase 2: Extend ParameterPredictionModel

### 2.1 Add type category prediction head
- **New output**: `type_category_logits` (batch, max_params, 4)
- Categories: uint-like, bytes-like, address-like, tuple-like
- Why: Direct type prediction too granular (24+ types), categories easier to learn

### 2.2 Modify compute_loss
```python
category_targets = convert_types_to_categories(parameter_types)
category_loss = CrossEntropyLoss(category_logits, category_targets)
total_loss = count_loss + type_loss + 0.5*mask_loss + 0.3*category_loss
```

### 2.3 Add inference method `predict_with_categories`

---

## Phase 3: Train Model

### 3.1 Training data
- Existing: 5,662 samples from 500 contracts
- Augment with 244 hard cases (ground truth)
- Use function name + bytecode features as input

### 3.2 Train
```bash
python -m abifusion.training.train_ml_models --model parameter_prediction
```

---

## Phase 4: Combinatorial Signature Generator

### 4.1 When 4byte + ML fail
- ML predicts: count + categories
- Enumerate concrete types from categories
- Cross product generates candidate signatures

### 4.2 Integrate into fusion.py
- If no 4byte candidate and ML predicts:
  - Generate combinatorial candidates
  - Rank by bytecode pattern match

---

## Phase 5: Evaluation

| Metric | Target |
|--------|--------|
| Coverage | >96.4% |
| Category accuracy | >80% |
| Count accuracy | >90% |

---

## Files to Modify

| File | Change |
|------|--------|
| `models/parameter_prediction_model.py` | Add category head, modify compute_loss |
| `data/parameter_dataset.py` | Add category labels |
| `training/train_ml_models.py` | Train with new head |
| `fusion.py` | Add ML fallback |
| `features/signature_generator.py` | New: combinatorial candidates |

---

## Expected Outcome

- **Best**: Coverage >96.4%
- **Realistic**: Reduce 244 errors to ~100-150
- **Minimum**: Combinatorial generator provides ranked candidates