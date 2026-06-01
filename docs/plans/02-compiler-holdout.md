# 02 — Compiler Hold-Out Experiment

**Priority:** High | **Status:** ✅ Complete | **Depends on:** benchmark data with compiler metadata

## Goal

Determine whether the ML models learn genuine EVM semantic invariants or merely fingerprint known Solidity compiler code-generation patterns. This is the novel insight from RESEARCH.md — no paper has published a compiler-holdout experiment for ABI recovery.

## Research context

SigRec's 98.7% accuracy is achieved by encoding 31 compiler-specific rules. When a new compiler version or obfuscation changes code-generation patterns, accuracy could silently degrade. Our current type recovery (96.2% prototype F1) relies on the 4byte DB. The question: does ML-based parameter prediction generalize across compiler versions?

## Experiment design

| Split | Contracts from | Test |
|---|---|---|
| Train | v0.4.x, v0.5.x, v0.6.x, v0.7.x | — |
| Test | v0.8.x | Evaluate type recovery |
| Baseline | Random 80/20 split (same data) | Compare degradation |

**Key metric:** prototype F1 degradation between random-split and compiler-holdout. If degradation < 5 pp → ML model generalizes (genuine novelty). If degradation > 10 pp → the model is fingerprinting compiler output (justifies investment in compiler-agnostic features).

## Implementation

1. **Fetch training data**: 100+ verified contracts from Etherscan, annotated with `compiler_version` from `getsourcecode`. Store as parquet or JSON.
2. **Build dataset**: `ParameterDataset.prepare_training_data()` — already extracts discriminating features + handles `compiler_version` metadata.
3. **Split**: `split_by_compiler_version(samples, train="0.4-0.7", test="0.8")` from `compiler_splitter.py`.
4. **Train**: `ParameterPredictionModel` on v0.4–v0.7 split.
5. **Evaluate**: `benchmark.py` on v0.8 test contracts. Compare against random-baseline split.
6. **Report**: selector F1 + prototype F1 for both splits, with per-compiler-version breakdown.

## Deliverable

Script `scripts/eval/compiler_holdout.py` that:
1. Fetches contracts, builds annotated dataset
2. Creates both splits (compiler-holdout + random baseline)
3. Trains on each split
4. Evaluates on test set
5. Outputs `eval_output/compiler_holdout_report.md` comparing both splits

## Dependencies

- `compiler_splitter.py` ✓ (built)
- `benchmark.py` ✓ (built)
- `parameter_dataset.py` + discriminating features ✓ (built)
- `ParameterPredictionModel` ✓ (accepts discriminating features)
- Contract data with compiler metadata (needs Etherscan API key — available)
