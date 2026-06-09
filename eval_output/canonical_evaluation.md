# abifusion Canonical Evaluation

**Date:** 2026-06-08

## Headline

| ABIFusion | Held-out exact parameter-type accuracy |
|---|---|
| **Fusion** (recommended) | **98.5%** |
| evmole baseline | 90.3% |

- Fusion fixes 102 functions evmole gets wrong
- Held-out functions: 1223/1241
- Training accuracy: 98.2%
- Generalization gap: -0.4pp

## Methodology

- Dataset: `data/contracts.parquet`
- Split: 80% training / 20% held-out
- Split seed: 42
- Training contracts: 400
- Held-out contracts: 100
- Known selector table entries: 538
- Known selector table built from training contracts only
- Both fusion and evmole baseline evaluated on the same held-out split

## Held-Out By Category

| Category | Correct | Total | Accuracy |
|----------|---------|-------|----------|
| DeFi AMM | 92 | 92 | 100.0% |
| DeFi router | 532 | 546 | 97.4% |
| other | 23 | 23 | 100.0% |
| simple token | 576 | 580 | 99.3% |

## Summary

This report is the canonical project headline. README accuracy claims should
reference this held-out result instead of full-dataset or exploratory reports.
