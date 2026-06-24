# ABI Reconstructor Canonical Evaluation

**Date:** 2026-06-24

## Headline

| Reconstructor | Held-out exact parameter-type accuracy |
|---|---|
| **Fusion** (recommended) | **98.2%** |
| evmole baseline | 91.7% |

- Fusion fixes 124 functions evmole gets wrong
- Held-out functions: 1890/1925
- Training accuracy: 98.6%
- Generalization gap: 0.4pp

## Methodology

- Dataset: `data/contracts_1k.parquet`
- Split: 80% training / 20% held-out
- Split seed: 42
- Training contracts: 800
- Held-out contracts: 200
- Known selector table entries: 681
- Known selector table built from training contracts only
- Both fusion and evmole baseline evaluated on the same held-out split

## Held-Out By Category

| Category | Correct | Total | Accuracy |
|----------|---------|-------|----------|
| DeFi AMM | 93 | 95 | 97.9% |
| DeFi router | 404 | 425 | 95.1% |
| other | 127 | 134 | 94.8% |
| simple token | 1266 | 1271 | 99.6% |

## Summary

This report is the canonical project headline. README accuracy claims should
reference this held-out result instead of full-dataset or exploratory reports.
