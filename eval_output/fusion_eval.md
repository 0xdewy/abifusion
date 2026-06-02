# FusionReconstructor vs evmole (Phase 3)

**The fusion reconstructor beats evmole on exact parameter types, strictly (zero
regressions).** With openchain.xyz as the primary signature source it fixes 429
of evmole's errors and breaks none — 374 type-swaps (`bytes32`/`uint256`,
`address`/`uint160`, resolved by the exact signature), 54 arity corrections, and
1 selector evmole missed entirely. The method is deterministic (no trained
params), so this full-set number is also the generalization estimate.

Functions evaluated: **6744** (full set)

| Method | Exact type-tuple accuracy |
|---|---|
| evmole | **90.0%** |
| FusionReconstructor | **96.4%** |

Net vs evmole: **+429 fixed**, **-0 broke** (+6.4 pp)

Fixed-error breakdown:
  - type_swap: 374
  - arity: 54
  - evmole_missing: 1
