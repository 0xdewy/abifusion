# Fusion ceiling vs evmole (Phase 1)

**Conclusion: the fusion thesis is validated.** Combining 4byte candidate
signatures with evmole's type structure beats evmole on exact parameter-type
accuracy **90.0% → 94.3%** (+4.3pp), fixing 291 functions and **breaking zero** —
strictly better, with no ML. The realistic heuristic (94.3%) nearly reaches the
oracle ceiling (94.5%), so simple evmole-guided disambiguation captures almost
all the headroom; ML is optional refinement, not load-bearing.

How it works: 4byte returns candidate signatures for a selector (often including
spam, e.g. `a9059cbb` → `workMyDirefulOwner(uint256,uint256)` ahead of the real
`transfer(address,uint256)`). evmole's recovered type structure
(`address,uint256`) selects the right candidate and rejects the spam, and the
4byte signature supplies the exact type evmole can't infer (e.g. `bytes32` vs
`uint256`). The ceiling is capped at 94.5% by 4byte coverage (78%): the remaining
errors are mostly selectors absent from 4byte where evmole is the only signal.

Ground-truth function instances: **6744**

| Method | Exact type-tuple accuracy |
|---|---|
| evmole (baseline) | **90.0%** |
| 4byte recall (true sig present) | 78.0% |
| Oracle fusion (4byte-true else evmole) | **94.5%** |
| evmole-guided fusion (no ML) | **94.3%** |

Fusion vs evmole: **+291 fixed**, **-0 broke** (net +291 = +4.3 pp)

4byte selector coverage: 78.0%

4byte candidate-count distribution (collision degree):
  - 1 candidates: 3193 (60.7% of covered)
  - 2 candidates: 496 (9.4% of covered)
  - 3 candidates: 519 (9.9% of covered)
  - 4 candidates: 556 (10.6% of covered)
  - 5 candidates: 336 (6.4% of covered)
  - 6 candidates: 160 (3.0% of covered)
