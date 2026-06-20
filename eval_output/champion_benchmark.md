# Champion Benchmark Report

**Date:** 2026-06-09
**Baseline:** /home/user/code/abi_reconstructor/eval_output/champion_baseline.json

## Overall

| Metric | Baseline | Current | Diff |
|--------|----------|---------|------|
| Accuracy | 98.5% | 98.5% | +0.0pp |
| Total functions | 1241 | 1241 | |

## By Failure Category

| Category | Baseline Acc | Current Acc | Diff |
|----------|--------------|-------------|------|
| evmole_arity_miss | 100.0% (5) | 100.0% (5) | +0.0pp |
| evmole_type_miss | 100.0% (60) | 100.0% (60) | +0.0pp |
| output_not_recovered | 100.0% (1035) | 100.0% (1035) | +0.0pp |
| selector_missing_from_bytecode | 65.4% (52) | 65.4% (52) | +0.0pp |
| signature_collision_mispick | 100.0% (58) | 100.0% (58) | +0.0pp |
| tuple_or_dynamic_type_mismatch | 100.0% (31) | 100.0% (31) | +0.0pp |

## Messages

- Accuracy: 98.5% (baseline 98.5%, diff +0.0pp)
