# External Generalization Report

**Date:** 2026-06-09
**Baseline:** /home/user/code/abi_reconstructor/eval_output/champion_baseline.json (held-out: 98.5%)

## Accuracy

| Metric | Held-out (canonical) | External |
|---|---|---|
| Accuracy | 98.5% | 98.5% |
| Generalization gap | — | +0.0pp |

## By Failure Category

| Category | Held-out Acc | External Acc | Diff |
|---|---|---|---|
| evmole_arity_miss | 100.0% (5) | 100.0% (29) | +0.0pp |
| evmole_type_miss | 100.0% (60) | 100.0% (438) | +0.0pp |
| known_selector_table_miss | 0.0% (0) | 0.0% (2) | +0.0pp |
| output_not_recovered | 100.0% (1035) | 100.0% (7743) | +0.0pp |
| selector_missing_from_bytecode | 65.4% (52) | 46.9% (256) | +18.5pp |
| signature_collision_mispick | 100.0% (58) | 100.0% (599) | +0.0pp |
| tuple_or_dynamic_type_mismatch | 100.0% (31) | 100.0% (182) | +0.0pp |

## Messages

- Accuracy: 98.5% (baseline 98.5%, diff +0.0pp)
- REGRESSION [selector_missing_from_bytecode]: 46.9% vs baseline 65.4% (+18.5pp)

## Decision


[See plan for decision criteria]
