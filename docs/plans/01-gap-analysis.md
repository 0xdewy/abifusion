# 01 — Gap Analysis

**Priority:** High | **Status:** In progress | **Depends on:** benchmark baseline data

## Goal

Diagnose every function where full-signature accuracy < 1.0 in the benchmark results. Categorize failures to determine which gaps are fixable vs. structural.

## Baseline

| Metric | Ours | SOTA |
|---|---|---|
| Selector F1 | 0.9043 | 0.996 (Heimdall-rs) |
| Prototype F1 | 0.9190 | 0.777 (Heimdall-rs) |
| Full Exact Match | 0.8542 | — |

85.4% of functions have completely correct selector + types. The remaining 14.6% gap breaks down into two sub-gaps:

- **9.6% selector recall gap** (0.904 selector F1): selectors in ground truth that our parser doesn't extract
- **5.0% type mismatch gap**: selectors found but types parsed incorrectly

## Failure Categories

For each contract × function where types are wrong or selector is missing:

| Category | Detection | Fixable? |
|---|---|---|
| **A. Selector not extracted** | Ground-truth selector missing from BytecodeParser output | Maybe — parser improvement |
| **B. DB miss** | Selector found but no entry in 4byte DB | Yes — ML fallback or DB expansion |
| **C. DB collision** | 2+ signatures for one selector, picked wrong one | Yes — disambiguation |
| **D. Proxy contract** | Proxy delegate — bytecode selectors ≠ logical ABI | No — structural |
| **E. Type mismatch despite DB hit** | DB had signature but types don't match ground truth | Bug — investigate |

## Deliverable

Script `scripts/eval/diagnose_gap.py` that reads `eval_output/eval_results.json` and the 4byte DB, then outputs:

1. `eval_output/gap_analysis.md` — human-readable report with counts per category + specific examples
2. Category breakdown showing which % of the 14.6% gap is fixable

## Implementation

1. For each contract in eval_results, load its ground-truth ABI and reconstructed output
2. For each ground-truth function, check if selector is in reconstructed output → if not, Category A
3. For each reconstructed function with wrong types, check if selector has DB entry → if not, Category B
4. If DB has multiple entries for the same selector → Category C
5. Flag proxy contracts (detect by `implementation()` in ABI or delegatecall in bytecode)
6. If DB has a single entry but types don't match → Category E
