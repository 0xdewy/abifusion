# FusionReconstructor vs evmole (Phase 3)

**The fusion reconstructor beats evmole on exact parameter-type accuracy,
strictly (zero regressions).** The method is deterministic — 4byte lookups +
evmole + a fixed disambiguation heuristic, no trained parameters — so this
full-set number is also the generalization estimate (no train/test split to
overfit). The 291 fixes are 245 type-swaps (4byte exact types resolve
`bytes32`/`uint256`, `address`/`uint160`) and 46 arity corrections.

Functions evaluated: **6744** (full set)

| Method | Exact type-tuple accuracy |
|---|---|
| evmole | **90.0%** |
| FusionReconstructor | **94.3%** |

Net vs evmole: **+291 fixed**, **-0 broke** (+4.3 pp)

Fixed-error breakdown:
  - type_swap: 245
  - arity: 46
