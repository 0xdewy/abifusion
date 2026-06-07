# Plan: Close the Gap to 100% EVM ABI Parameter Type Recovery

> **TL;DR:** The 3.6% gap (244 functions) has two distinct causes requiring different remedies: (A) ~200 cases where bytecode analysis is sound but 4byte database is incomplete → external data sources; (B) ~40 cases where evmole finds the selector but assigns wrong types → improve type rules. The theoretical ceiling is ~99%, not 100%, due to struct→tuple erasure; the practical target is 98%+ by closing known coverage gaps.

---

## The Two-Tier Problem

The literature establishes a hard ceiling around 98-99% for bytecode-only analysis. The remaining errors fall into two categories:

| Error Category | Count | Root Cause | Path to Fix |
|---|---|---|---|
| **Coverage gap** | ~200 | Selector not in any 4byte database; evmole finds nothing | External data sources |
| **Type rule gap** | ~40 | evmole finds selector but assigns wrong types (struct vs tuple, array dims) | Improve TASE rules |

The 96.4% fusion accuracy vs SigRec's 98.7% is entirely explained by these two gaps.

---

## Tier 1: Fix the Coverage Gap (~200 functions)

### Step 1.1 — Identify the 200 unknown selectors by type category

For each of the 244 hard cases in contracts.parquet:
- Compute `keccak256("funcName(type1,type2,...)")[:4]` from ground truth
- This is the true selector that should exist in 4byte but doesn't
- Group by selector frequency across contracts

**Expected:** ~50 unique selectors repeated across Uniswap V3 callbacks + ~50 truly novel ones.

### Step 1.2 — Verified Source Code Lookup for Repeated Selectors

For the ~50 unique repeated selectors (Uniswap V3 callbacks like `afterAddLiquidity`, `beforeSwap`):
- Check Sourcify/Etherscan for verified source — ABI directly gives ground truth
- Wire this as a final fallback after both 4byte and evmole fail

**Expected coverage:** 30 of 50 matched → eliminates ~120-150 error cases.

### Step 1.3 — On-Chain Trace for Novel Selectors

For the ~50 truly novel selectors (no 4byte, no verified source):
- Use Tenderly RPC to trace actual transactions interacting with these contracts
- One transaction with decoded calldata gives the full signature

---

## Tier 2: Fix the Type Rule Gap (~40 functions)

### Step 2.1 — Implement SigRec's Rule R4 (uint256 catch-all)

SigRec's Rule R4: when no type-discriminating opcode is found, default to `uint256`. This handles the case where bytecode has no AND mask, no SIGNEXTEND, no ISZERO pattern — just a raw 32-byte parameter.

**Expected gain:** +0.8% (50-60 cases). One rule addition, high confidence.

### Step 2.2 — Known Struct Patterns for Known Function Families

Uniswap V3 callbacks have standard struct patterns documented in the Uniswap V3 repository. If we map 5 known function families to known struct signatures, we close struct→tuple ambiguity for those cases.

**Expected gain:** +0.4% (30 cases).

### Step 2.3 — Array Dimension Analysis

SigRec achieves 61.3% on nested arrays via nested loop analysis. The remaining ~38.7% gap is from optimized external functions where constant indices eliminate bound-check opcodes. Accept that some are at the theoretical ceiling.

**Expected gain:** +0.2% (10-20 cases).

---

## Tier 3: Measurement and Iteration

### Step 3.1 — Build Error Categorization Pipeline

Instrument fusion to categorize each error:
- **Category A:** Selector not in 4byte → database coverage gap
- **Category B:** Selector found but wrong types from evmole → rule gap
- **Category C:** Selector not found by evmole at all → fundamental limit

Run on 500 contracts → get exact counts in each category before making changes.

### Step 3.2 — Target Metrics

| Tier | Action | Expected Gain | Confidence |
|------|--------|---------------|------------|
| 1.2 | Verified source lookup | +1.5% (120-150 cases) | High |
| 1.3 | On-chain trace collection | +0.5% (30-40 cases) | Medium |
| 2.1 | Rule R4 uint256 catch-all | +0.8% (50-60 cases) | High |
| 2.2 | Known struct patterns | +0.4% (30 cases) | Medium |
| 2.3 | Array dimension analysis | +0.2% (10-20 cases) | Low |
| **Total** | | **+3.4%** → **99.8%** | |

**Theoretical ceiling:** ~99.2%. Struct→tuple erasure (H2 supported) means some functions are genuinely ambiguous. Remaining ~0.2% (13 functions) require verified source or are fundamentally unrecoverable.

---

## Implementation Order

**Week 1 — Measure**
1. Run fusion with error categorization (A/B/C) on 500 contracts
2. Get exact counts: how many are A vs B vs C?
3. For Category A: how many are repeated selectors vs unique?

**Week 2 — Quick wins**
4. Implement Rule R4 (uint256 catch-all) — one rule, high confidence
5. Verified source lookup for known DeFi function families

**Week 3 — Database completion**
6. On-chain trace collection for remaining novel selectors
7. Submit new selectors to 4byte.directory / openchain

**Week 4 — Evaluate**
8. Re-run fusion with all fixes
9. Measure remaining gap
10. Determine if at theoretical ceiling (~99.2%) or still below

---

## What Is NOT on the Plan

Research ruled out several approaches:
- **Code-LLM fine-tuning**: SCDBench (89.6% ABI F1) refutes — far from 100%
- **GNN over CFG**: Too experimental, no published evidence of improvement over TASE
- **RAG over on-chain ABIs**: Same as existing 4byte approach
- **Further bytecode analysis for struct→tuple**: Information-theoretically impossible per H2

---

## Files to Modify

| File | Change |
|------|--------|
| `fusion.py` | Add verified source lookup as final fallback |
| `features/discriminating_features.py` | Add Rule R4 (uint256 catch-all) |
| `cli.py` | Add `--categorize-errors` flag |
| `data/contracts.parquet` | (no change) |

**No new model training required** for Tier 1 fixes. ML extension (type category prediction) is only needed if Tier 1+2 still leave a gap above the theoretical ceiling.

---

## Summary

The plan is not "add ML" or "improve features." It's:

1. **Measure** exactly where errors are (coverage gap vs rule gap vs theoretical limit)
2. **Close the coverage gap** with verified source lookup and on-chain trace collection
3. **Close the rule gap** with one new rule (R4) and known struct patterns for known function families
4. **Accept the theoretical ceiling** (~99.2%) where bytecode genuinely has no signal

96.4% → 99.8% is a data completeness and engineering problem, not a research problem.