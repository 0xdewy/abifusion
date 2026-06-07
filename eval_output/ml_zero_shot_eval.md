# ML Zero-Shot Evaluation Results

## Training Run Summary

**Date:** 2026-06-07
**Model:** 1D CNN bytecode encoder + per-position feature fusion + two classification heads
**Training:** 20 epochs, 5,667 functions from 400 contracts
**Approach:** Function family classification (33 classes) instead of exact name classification (1,656 classes)

## Results

| Metric | Value | Notes |
|--------|-------|-------|
| **Family Top-3 accuracy** | 95.4% (166/174) | Far exceeds 60% threshold |
| **Family Top-1 accuracy** | 94.8% (165/174) | |
| **Type accuracy (4-class)** | 38.9% (91/234) | vs 25% random baseline |
| **Val Family Top-3** | 89.3% | Best epoch 18 |
| **Val Family Top-1** | 63.4% | |

## Key Finding: Family Classification Works

The exact-name prediction approach FAILED (0% accuracy) because 160/174 test names are `<unk>` (not in vocab).
The function family approach SUCCEEDS (95.4% accuracy) because:

1. **33 family classes vs1,656 name classes** — dramatically easier classification
2. **Bytecode patterns map to semantic families** — ERC20 bytecode patterns are distinguishable from AMM patterns
3. **Family names are reusable** — "transfer" appears in both training and test as ERC20_BASIC

## Type Prediction Analysis

The type prediction head shows modest signal:

- **True distribution:** 95 uint-like, 70 address, 51 bytes-like, 18 other
- **Accuracy:** 38.9% (vs 25% random for 4-class)
- **Note:** Type accuracy is lower than in the exact-name experiment (40.6%) but still above random

## What the Model Learned

The model correctly identifies function families from bytecode:

- **ERC20_BASIC** (transfer, approve, balanceOf) — bytecode patterns like AND masking for address handling
- **AMM_SWAP** (swap, beforeSwap) — branch-heavy code with multiple CALLDATALOAD patterns
- **ERC721_METADATA** (tokenURI) — different memory access patterns
- **ACCESS_OWNABLE** (owner, pendingOwner) — simple getter patterns

## Integration Decision

**PASS:95.4% >= 60% threshold — integrate ML model as Tier 3 fallback.**

The model is integrated into `FusionReconstructor` as a third-tier fallback:
- Tier 1: 4byte/openchain signature
- Tier 2: Known-selector table
- Tier 3: ML family + type prediction
- Tier 4: `function_<selector>` with no types

## Recommendations for Type Prediction

Type accuracy (38.9%) is above random but below useful. Improvements:
1. Better per-position features (CALLDATALOAD offset analysis if optimizer patterns can be detected)
2. Larger training set with more diverse contracts
3. Type-specific loss weighting (bytes-like and other are underrepresented)
