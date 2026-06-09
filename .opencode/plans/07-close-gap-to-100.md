# Plan 07: Close the Gap to 100%

## Context

After reviewer corrections:

| State | Gap | What's left |
|-------|-----|-------------|
| Pre-Tier-2 (fusion_eval.md) | 3.6% (244 functions) | All failures |
| Post-Tier-2 (fusion_eval_v2.md) | **0.8%** (~54 functions) | **57 unique-only selectors** |
| Post-Tier-3 (current) | ~0.8% | Same — ML type prediction only reached when Tiers 1+2 fail |

**The real problem:** 57 selectors appear exactly once in the dataset — no training signal, no 4byte coverage, no evmole detection. The 187 "hard cases" were fixed by Tier 2 (known-selector table), not by ML.

---

## Phase 1: Honest Holdout + Tier Breakdown

**Two scripts, two deliverables before Phase 2 coding.**

```bash
PYTHONPATH=/home/user/code/abifusion python scripts/eval/holdout_eval.py
PYTHONPATH=/home/user/code/abifusion python scripts/eval/tier_recall_eval.py
```

### Phase 1.1: Holdout Evaluation

**Deliverable:** `eval_output/holdout_eval_results.md`

- 5-fold cross-validation
- Know if known-selector table generalizes or leaks contract identity
- If holdout accuracy ≈ 99.2% → gap is truly ~0.8%
- If holdout accuracy << 99.2% → gap is larger, Phases 2-3 become more relevant

### Phase 1.2: Tier Recall Breakdown

**Deliverable:** `eval_output/tier_breakdown.md`

Instrument `reconstruct()` to track tier resolution per selector. Write `scripts/eval/tier_recall_eval.py`:

1. Run `reconstruct()` on evaluation dataset with per-tier tracking
2. Group results by tier
3. For Tier 1: measure recall as selectors resolved by 4byte / total selectors
4. Break down Tier 1 failures into:
   - **No 4byte entry** — selector not in 4byte at all (can't be fixed by disambiguation)
   - **Collision mis-pick** — 4byte has candidates but wrong one selected (fixable by ML)
5. Report: overall Tier 1 recall, collision rate, no-coverage rate

**Tier assignment in `fusion.py`:**
```python
tier = 1 if source in ("signature+evmole", "signature") else \
       2 if source == "known-selector-table" else \
       3 if source == "ml" else 4
```

**Phase 2 ROI validation:** Phase 1 tells us whether collision mis-pick rate is significant. If yes → Phase 2.1 is high-value. If no → skip to Phase 3.

---

## Phase 2: ML-Powered 4byte Disambiguation

**Only proceed if Phase 1 validates ROI** (collision mis-pick rate is significant).

### Phase 2.1: Implement family-guided tiebreaker (~20 lines + import)

**Lazy call — only invoke ml.predict() in Branch 3 (tiebreaker).**

`fusion.py` changes:
1. Import: `from abifusion.ml.families import get_family, get_family_name`
2. Add `ml_family: Optional[str] = None` parameter to `choose_candidate()`
3. Implement Branch 3:
```python
if candidates and ml_family:
    scored = [(name, types, get_family_name(get_family(name)) == ml_family)
              for name, types in candidates]
    return max(scored, key=lambda x: x[2])[:2]
```

**Current `choose_candidate()` flow:**
```
Branch 1: exact evmole match → return immediately (fast path, no ML)
Branch 2: same-arity max-overlap → return immediately (fast path, no ML)
Branch 3: tiebreaker → call ml.predict() for family (new, rarely hit)
```

**ML call site:** In `reconstruct()`, before `choose_candidate()`: call `ml.predict(bytecode, selector)` only when Branch 3 is reached. Cost: ~17 calls worst case (10% of 170 selectors) = under 1 second. No caching needed.

**Note:** `ml.predict()` internally re-extracts evmole offsets (duplicate work). Don't fix now — negligible cost.

**Bug fixed:** `get_family(name)` returns an int (index), `ml_family` is a str (e.g., `"AMM_CALLBACK"`). Use `get_family_name(get_family(name)) == ml_family` for comparison.

**Deliverable:** `eval_output/4byte_disambiguation_results.md` — Tier 1 recall before/after, collision resolution rate.

### Phase 2.2: Evaluate Tier 1 Improvement

Target: 78% → 85%+ (validated by Phase 1 ROI assessment first).

### Phase 2.3: Class-Weighted Loss for Type Prediction

**Low-effort.** Address is overestimated by 2.2x. Class-weighted cross-entropy loss is standard and won't hurt type accuracy for the ~0.8% that reach Tier 3.

**Deliverable:** `eval_output/type_prediction_results.md`

---

## Phase 3: External DB Lookups for 57 Unique Selectors

**4byte.directory API only** (selector-based, free, no key needed).
**Sourcify and Etherscan are address-based — skip entirely.**

`scripts/lookup_missing_selectors.py`:
1. Load 57 selectors from `eval_output/hard_cases.md`
2. Query 4byte API: `GET https://www.4byte.directory/api/v1/signatures/?hex_signature=0x<selector>`
3. If found → add to `data/known_selector_signatures.json`
4. If not found → genuinely unfixable without source code access

**Deliverable:** `eval_output/external_lookup_results.md` — selectors found, final gap.

---

## Phase 4: External Contract Validation (30-50 contracts)

**Hardcoded addresses in `data/eval_external_contracts.json`** (~20 lines).

Three reasons for hardcoding:
1. **Reproducibility** — same contracts today and next month, no moving target
2. **Stability** — top DeFi router contracts deployed at fixed addresses
3. **Simplicity** — 20 lines vs a DefiLlama+Etherscan pipeline that can fail in subtle ways

| Category | Count | Target |
|----------|-------|--------|
| DeFi routers | 10 | >85% |
| Simple tokens | 10-20 | 100% |
| Diverse/novel | 10 | >95% overall |

**Deliverable:** `eval_output/external_validation_report.md`

---

## Implementation Order

```
Phase 1: Holdout + Tier Breakdown (1-2 days)
  ├─ holdout_eval.py → know if known-selector table generalizes
  └─ tier_recall_eval.py → validate Phase 2 ROI before coding

Phase 2: 4byte Disambiguation (only if ROI validated) (2-3 days)
  ├─ 2.1 Implement family-guided ranking (~20 lines + import)
  ├─ 2.2 Evaluate Tier 1 improvement
  └─ 2.3 Class-weighted loss

Phase 3: External DB Lookups (1 day)
  ├─ 3.1 Query 4byte for 57 selectors
  ├─ 3.2 Automate lookup script
  └─ 3.3 Accept rest as inherent ceiling

Phase 4: External Validation (ongoing)
  ├─ 4.1 Hardcode 30-50 contracts in eval_external_contracts.json
  ├─ 4.2 Evaluate against verified ABIs
  └─ 4.3 Document final results
```

---

## Success Metrics

| Phase | Metric | Baseline | Target |
|-------|--------|----------|--------|
| 1 | Holdout accuracy (5-fold) | ~99.2% (assumed) | >97% |
| 1 | Tier 1 collision mis-pick rate | TBD (Phase 1) | Validates Phase 2 ROI |
| 2 | Tier 1 recall | 78% | >85% |
| 2 | Type prediction accuracy | 40.6% | >45% |
| 3 | Unique selectors found via 4byte | 0/57 | Maximize |
| 4 | Simple tokens | N/A | 100% |
| 4 | Complex routers | 73-85% | >85% |
| 4 | Overall external | N/A | >95% |

---

## What This Plan Does NOT Include (And Why)

| Rejected | Reason |
|----------|--------|
| TypeHead architecture "fix" | Cosmetic — identical output, zero accuracy gain |
| Per-position CALLDATALOAD analysis | Already tested, 0% match rate, known-failed |
| Bytecode heuristics for Tier 4 | Same as above |
| Sourcify/Etherscan for 57 selectors | Address-based APIs — useless without contract address |
| Manual curation of unique selectors | One-off selectors not worth manual effort |
| Caching layer for ML calls | Cost negligible, lazy approach simpler |
| Dynamic DefiLlama sourcing for eval | Moving target — hardcode for reproducibility |

---

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `scripts/eval/tier_recall_eval.py` | Create | Measure per-tier recall + collision breakdown |
| `abifusion/fusion.py` | Modify | Add `ml_family` arg to `choose_candidate()`, import families |
| `scripts/lookup_missing_selectors.py` | Create | Query 4byte for 57 selectors |
| `scripts/train/train_family_model.py` | Modify | Add class-weighted loss option |
| `data/eval_external_contracts.json` | Create | Hardcoded 30-50 external contracts |
| `eval_output/holdout_eval_results.md` | Create | Phase 1 deliverable 1 |
| `eval_output/tier_breakdown.md` | Create | Phase 1 deliverable 2 + Phase 2 ROI validation |
| `eval_output/4byte_disambiguation_results.md` | Create | Phase 2 results |
| `eval_output/type_prediction_results.md` | Create | Phase 2 results |
| `eval_output/external_lookup_results.md` | Create | Phase 3 results |
| `eval_output/external_validation_report.md` | Create | Phase 4 results |