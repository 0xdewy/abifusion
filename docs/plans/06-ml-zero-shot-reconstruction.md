# Plan: Reintroducing ML for Zero-Shot ABI Reconstruction

## The Actual Problem (Honest Framing)

The 3.6% residual error (244 functions, eval_output/hard_cases.md) is not one problem — it is two:

| Problem | Count | Nature | Right Tool |
|---|---|---|---|
| Repeated unknown selectors | 187 occ | Same selector across multiple contracts; in the eval dataset but absent from 4byte/openchain | Known-selector table |
| Unique unknown selectors | 57 occ | Truly novel — one occurrence, one contract, no prior human label anywhere | ML (potentially) |

The previous ML system (Plan 05, now removed) was evaluated on the full dataset including known-selector functions. It was never evaluated on the 57 unique selectors. Its architecture — a global 7-dimensional feature vector ("some address mask exists somewhere") applied to the full function — was wrong for the problem it was solving (known-selector disambiguation, where fusion already wins at 96.4%). It was never brought to bear on the problem it could actually solve (zero-shot type inference for unknown selectors).

---

## Step 0 — Before Any ML: Expand the Known-Selector Table

**Why first:** The 187 repeated-selector cases (Uniswap V3 callbacks ×12-18 each, NFT setters ×4 each) have a deterministic solution that requires no ML and no model training. The `data/known_selector_signatures.json` has only 29 entries, all sourced from the eval set. These 29 selectors account for 187 of the 244 hard cases. Expanding the table from verified contract sources closes 76% of the residual gap.

**How:**
- Write `scripts/eval/build_known_selector_table.py`
- Source contracts: contracts in `data/contracts.parquet` whose selectors appear in the 29 hard-case repeated selectors, plus any additional verified contracts from Sourcify that contain these selectors
- For each candidate selector, verify the entry against ground-truth ABI before adding
- Update `FusionReconstructor._ensure_known_selectors()` to load the expanded table

**Success metric:** After expansion, `eval_output/hard_cases.md` shows 0 occurrences for the 29 repeated selectors.

**Risk:** Low. This is deterministic lookup — no model, no generalization.

---

## Step 1 — Define the Honest ML Problem

After the known-selector table expansion, the residual is **57 unique-selector functions** (0.85% of 6,744). This is the ML problem. Everything below targets this residual.

**The problem statement:** Given bytecode for a function whose selector has no entry in any 4byte database and whose argument structure evmole cannot recover, predict:
1. The function name (or a plausible Top-3 ranked list)
2. The parameter types (as a ranked list of candidates)

**What ML does NOT do:** Disambiguate known-selector collisions, improve type accuracy for selectors already in 4byte, or compete with fusion on any function where fusion already works. ML exists only for the silence — functions where fusion has no signal.

**The critical empirical unknown:** Whether a model trained on known-selector functions (where ground truth exists) can generalize to the 57 unique selectors never seen during training. This is tested in Step 5, not assumed.

---

## Step 2 — Per-Position Feature Engineering

**Verified fact about evmole:** `evmole.contract_info(..., arguments=True)` returns `Function` objects with:
- `bytecode_offset`: byte offset of the function's JUMPDEST entry point
- `arguments`: a flattened type string (e.g., `"address,uint256"`), not per-parameter offsets
- `selector`: the 4-byte selector
- `state_mutability`: function mutability

**evmole does NOT provide per-parameter bytecode offsets.** There is no API call that tells you "parameter 0 is handled at bytecode offset X, parameter 1 at offset Y." That information must be derived from the bytecode itself.

**The correct primary approach: CALLDATALOAD offset analysis.**

Solidity always loads function parameters from calldata at fixed 32-byte intervals: offset 4 (first param), offset 36 (second), offset 68 (third), and so on. The pattern is:

```
PUSH1 0x04  ; push offset
CALLDATALOAD ; load first parameter (32 bytes)
... use parameter ...
PUSH1 0x24  ; 36 in decimal — 0x04 + 0x20
CALLDATALOAD ; load second parameter
```

By scanning the function body for `PUSHn <offset> CALLDATALOAD` sequences, we can identify how many parameters the function takes and, critically, which bytecode region handles each parameter. This is the per-position localization.

**Phase 0 verification (required before proceeding with per-position features):**
Write a short script that:
1. Calls `evmole.contract_info(code, selectors=True, arguments=True)` on 10 contracts
2. For each function, locates its `bytecode_offset` (function body entry)
3. Scans the bytecode window at that offset for `PUSHn <N> CALLDATALOAD` patterns
4. Collects the set of unique offsets found
5. Compares the count of offsets to `len(split_args(f.arguments))` (the arity evmole reports)

If the offset count matches arity for >90% of functions, the method works. If not, fall back to global features with arity augmentation.

**The extraction (once Phase 0 is verified):**
1. For each function, get `bytecode_offset` from evmole
2. Scan the bytecode window at that offset for `PUSHn CALLDATALOAD` patterns to identify per-parameter regions
3. For each parameter position, extract a local bytecode window (e.g., 100 bytes surrounding that parameter's handling code)
4. Run `DiscriminatingFeatureExtractor._analyze_opcodes()` on that local window to produce a per-position feature vector
5. Concatenate all per-position vectors in order

**The fallback path (if Phase 0 verification fails):**
Use the 7 SigRec features extracted from the function body as a single [7-dim] vector, with the parameter count from evmole appended as an [8th] dimension — this differs from Plan 05's pure global vector by including the arity signal. Concatenate this 8-dim vector with the bytecode encoder output as the model input. This is what the previous system effectively did — it is weaker, but it exists and can be baseline.

**Result:** A feature matrix of shape `[num_params × 8]` (7 SigRec features + 1 arity feature per position) rather than the previous `[1 × 7]` global vector.

---

## Step 3 — Training Pipeline

**Data:** 500 contracts from `data/contracts.parquet`. Train/val on the first 400 contracts (5,667 functions). Test only on the 57 unique-selector functions. This simulates the actual deployment condition: training on functions with known selectors, inference on functions with unknown selectors.

**Train on:** All 5,667 training functions. Tag which have 4byte coverage and which do not. The model learns bytecode patterns from both — the goal is to extract patterns that generalize to selectors where no prior human label exists.

**Evaluate on:** Only the 57 unique-selector functions. This is the honest test. The previous system evaluated on all 6,744 functions, which included the cases it was trained to handle — a fundamental eval design flaw.

**Note on test set selection bias:** The 57 unique-selector test functions are unique within the eval dataset; their zero-shot status is scoped to this dataset and may not generalize to truly novel selectors encountered in production.

**Model architecture:**
- **Bytecode encoder:** 1D CNN over tokenized bytecode bytes (0-255), producing a per-position embedding
- **Per-position features:** `[num_params × 8]` feature matrix from Step 2, concatenated with bytecode encoder output at each parameter position
- **Two heads:**
  - **Function name head:** Top-3 classification over the ground-truth function name vocabulary in training data (size: ~2,000 unique names)
  - **Parameter type head:** Per-position 4-class classification: `bytes-like` (bytes, bytes32, bytes<M>), `uint-like` (uint256, uint128, int256), `address`, `other` (bool, string, arrays, tuple)

**Why 4-class:** Most zero-shot failures are `bytes32`/`uint256` or `address`/`uint160` ambiguity. A coarser vocabulary is more learnable from limited data. Refinement to exact types can follow from the 4-class prediction using rule-based inference (e.g., `bytes-like` + evmole's arity → `bytes32` for single-parameter functions).

**Training:** Re-implement training using PyTorch. Use the existing checkpoint references (`cache/checkpoints/`) as architecture guides but train fresh with the per-position features from Step 2. The 1D CNN bytecode encoder from the removed Plan 05 was sound — only the input representation (global vs per-position) was wrong.

**Metric:** Top-3 function name accuracy on the 57 unique-selector test set. Partial recovery (correct function family) has value for human review.

---

## Step 4 — Integration Decision

**Decision threshold: >60% Top-3 function name accuracy on the 57 unique-selector test set.**

If the model achieves >60%:
1. Create `abi_reconstructor/ml_reconstructor.py` with the trained model
2. Integrate as a third-tier fallback in `FusionReconstructor.reconstruct()`:
   - Tier 1: 4byte/openchain signature (existing fusion path)
   - Tier 2: known-selector table (Step 0 expansion)
   - Tier 3: ML model prediction
   - Tier 4: `function_{selector}` with no types (existing fallback)
3. Use the Top-3 function name prediction to name the ABI function; use the 4-class type prediction to populate parameter types

If the model does NOT achieve >60%:
- Publish the results in `eval_output/ml_zero_shot_eval.md`
- Document the generalization failure honestly
- Accept 3.6% residual as the ceiling without additional training data

**Why 60%:** At 60% Top-3, the model is providing useful signal more often than not. Below that, the false positive rate is too high to be useful in an automated pipeline. The threshold is not arbitrary — below it, a human reviewing the output would spend more time correcting the model than simply using the `function_{selector}` fallback.

---

## Step 5 — Evaluation

Add `scripts/eval/ml_zero_shot_eval.py`:

1. Load the expanded known-selector table (Step 0)
2. Run `FusionReconstructor` on all 500 contracts
3. Identify functions where fusion returns `source="selector"` (no signal from Tier 1 or Tier 2)
4. Run the ML model on those functions
5. Report:
   - How many of the original 244 hard cases remain after Tier 2 lookup (target: 57)
   - Top-3 function name accuracy on those 57
   - Per-position coarse type accuracy (4-class)
   - Breakdown of errors: which function families the model confused
6. Write results to `eval_output/ml_zero_shot_eval.md`

**The eval is the honesty mechanism.** If the model achieves 40% accuracy on the 57, that is the result — not 96.4% on the full dataset. The number that matters is the recovery rate on the no-signal subset, and the eval must report it without smoothing it into aggregate metrics.

---

## Assumptions Named

1. Additional verified contracts with the 29 repeated selectors exist in Sourcify or the eval dataset and can be sourced for the known-selector table
2. CALLDATALOAD offset analysis can identify per-parameter bytecode regions with >90% accuracy (verified in Step 2 Phase 0)
3. 57 unique-selector functions provides enough training signal for a model to generalize — this is the critical empirical unknown
4. The 4-class coarse type vocabulary is learnable from bytecode patterns at the per-position granularity
5. The function name head generalizes better than the parameter type head (85% Top-3 was achievable in Plan 05; this plan uses per-position not global features)

---

## Proportion and Risk

| Step | Effort | Impact | Risk |
|---|---|---|---|
| Step 0: Known-selector table expansion | Low | Closes 76% of residual (187/244 functions) | Low — deterministic |
| Step 1: Define problem | Trivial | Focuses all subsequent work | None |
| Step 2: Per-position features (Phase 0 verify + implement) | Medium | Enables the correct ML input representation | Medium — Phase 0 may reveal the method doesn't work |
| Step 3: Training pipeline | High | Core of the ML work | High — 57 examples may not be enough |
| Step 4: Integration | Medium | Makes ML useful in production | Low |
| Step 5: Evaluation | Low | Honesty mechanism | None |

**The plan's proportion reflects its risk:** low-effort/high-impact steps first (Step 0 closes the most with the least work), high-risk steps deferred until their prerequisites are confirmed (Step 2 Phase 0 must pass before Step 3 begins).