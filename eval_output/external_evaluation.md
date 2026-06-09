# abifusion External Evaluation

**Date:** 2026-06-09

## Headline

| ABIFusion | External exact parameter-type accuracy |
|---|---|
| **Fusion** (recommended) | **96.8%** |
| evmole baseline | 92.5% |

- Fusion fixes 395 functions evmole gets wrong
- External functions: 8950/9249

## Methodology

- Dataset: `data/external_eval.parquet` (external, not split)
- Uses shipped runtime tables only (no table building from external data)
- External contracts: 500

## External By Category

| Category | Correct | Total | Accuracy |
|----------|---------|-------|----------|
| DeFi AMM | 416 | 450 | 92.4% |
| DeFi router | 2018 | 2139 | 94.3% |
| other | 492 | 531 | 92.7% |
| simple token | 6024 | 6129 | 98.3% |

## Summary

This report is the external generalization evaluation. 
It uses shipped runtime tables only — no table building from external data.
