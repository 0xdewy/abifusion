# Fusion ceiling vs evmole (Phase 1)

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
