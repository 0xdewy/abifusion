# Plans — ABI Reconstructor

Status index for remaining work. Each plan traces to a research finding or IDEAS.md item.

| # | Plan | Priority | Status | Depends on |
|---|---|---|---|---|
| [01](./01-gap-analysis.md) | Gap Analysis | high | ✅ complete | — |
| [02](./02-compiler-holdout.md) | Compiler Hold-Out Experiment | high | ✅ complete | — |
| [03](./03-self-contained-checkpoint.md) | Self-Contained Checkpoint | medium | ✅ complete | — |
| [04](./04-device-management.md) | Device Management Cleanup | medium | ✅ complete | — |
| [05](./05-ml-training-pipeline.md) | ML Training Pipeline | medium | spec ready | #3 (checkpoint fix) |

## Done

| # | Plan | Key result |
|---|---|---|
| — | Measurement Apparatus | `scripts/eval/benchmark.py` — SCDBench-style metrics per compiler version |
| — | Discriminating Features | SigRec R11-R18 opcode analysis integrated into rule-based + ML pipeline |
| — | Type Recovery from 4byte DB | Prototype F1 0.292 → 0.972 by parsing types from DB signatures |
| — | Selector Extraction Fix | MIN_SELECTOR_VALUE lowered from 0x10M to 0x100, recovered 52 selectors |
| — | Etherscan v2 Migration | Client migrated to v2 API with chainid support |
| 01 | Gap Analysis | 96.2% correct, remaining 3.8% = 17 complex router selectors + 1 proxy |

## Baseline (after all fixes)

| Metric | Ours | SOTA |
|---|---|---|
| Selector F1 | 0.9636 | Heimdall-rs: 0.996 |
| Prototype F1 | 0.9718 | Heimdall-rs: 0.777 |
| Full Exact Match | 0.9563 | — |
