# Plan 07: Closed-Loop Failure-Driven Improvement

## Framing

The next accuracy gains should come from a measured feedback loop:

1. Mine current evaluation failures.
2. Separate addressable failures from failures bytecode cannot resolve.
3. Generate targeted synthetic examples for the addressable classes.
4. Gate every change against a champion benchmark.
5. Calibrate confidence from measured outcomes.

This borrows one useful lesson from AlphaZero-style systems: failures become the
next curriculum and each iteration must beat the current champion. It does not
propose literal reinforcement learning, self-play, or MCTS.

## Implementation Status

As of 2026-06-09:

| Step | Status | Notes |
|---|---|---|
| Step 1: failure mining | Complete | `scripts/eval/mine_failures.py` produces `eval_output/failures.json` and `eval_output/failure_summary.md`; current canonical split is 98.5% accurate with 18 classified failures |
| Step 2: addressability classification | Complete | Known-interface completion eliminated the 31 deterministic failures; the remaining 18 failures are `not_addressable_from_bytecode` |
| Step 3: champion benchmark | Complete | `scripts/eval/champion_benchmark.py --init` created `eval_output/champion_baseline.json`; normal benchmark runs report no regression |
| Deterministic interface completion | Complete | `scripts/eval/build_external_known_table.py` produced `data/known_interface_sets.json`; `FusionReconstructor` emits `known-interface-completion` functions when conservative triggers match |
| Product CLI output | Complete | Fusion supports `--output-format abi`, `--min-confidence`, and `--output` |
| External generalization pipeline | Complete | External eval set: 500 contracts, 9249 functions. Fusion: 98.4% accuracy, evmole baseline: 92.5%. Gap vs champion: +0.1pp. Full pipeline run 2026-06-09. |
| External dataset builder | Complete | `data/external_eval.parquet` (500 contracts), `data/external_eval_meta.json`, `data/cache/external_eval/` with 1,500 cached candidates. No longer blocked on ETH_RPC_URL. |
| Steps 4-7 | Done — documented ceiling | External accuracy 98.4% (98.5% held-out baseline, +0.1pp gap). 0 `model_or_scoring_addressable` failures. 136 ceiling failures (`not_addressable_from_bytecode`). 12 deterministic misses too small for synthetic generation ROI. No model work warranted. |

Recent implementation notes:

- Shared evaluation helpers live in `scripts/eval/_shared.py`, including
  `extract_bytecode_metadata()`.
- Regression checks compare current accuracy against the champion baseline and
  only flag drops beyond tolerance.
- High-confidence checks follow the same baseline-diff pattern instead of using
  a hardcoded target during champion comparison.
- Current failure data shows no collision mis-picks or type-ambiguity failures;
  the immediate ROI is deterministic handling, not ML/scoring.
- Deterministic known-interface completion improved fusion from 96.1% to 98.5%
  held-out exact parameter-type accuracy, reducing total failures from 49 to 18
  with 0 observed false positives.
- External report-only benchmarking is separate from the champion regression
  gate. It reports generalization and should exit successfully unless the script
  itself fails.
- External eval confirms 98.4% accuracy (+0.1pp vs held-out). 136 ceiling failures
  are `not_addressable_from_bytecode` (V4 hook callbacks). 12 deterministic misses
  are too small (0.1%) to justify synthetic generation. Decision: STOP, document
  ceiling.

## External Generalization

The external-generalization pipeline validates the shipped runtime tables against
a fresh evaluation axis without using the external data for table building,
training, or model selection.

Implemented scripts:

- `scripts/eval/build_external_eval_set.py`: resumable Sourcify/RPC fetcher with
  cache support under `data/cache/external_eval/`.
- `scripts/eval/run_evaluation.py --external`: evaluates the full external set
  with shipped runtime tables only.
- `scripts/eval/mine_failures.py --external`: mines failures with shipped
  runtime tables only and supports `--out-json` / `--out-md`.
- `scripts/eval/champion_benchmark.py --external`: report-only generalization
  comparison against `eval_output/champion_baseline.json`.

Current report (2026-06-09):

- Held-out champion accuracy: 98.5%.
- External accuracy: 98.4%.
- Generalization gap: +0.1pp.
- Evmole baseline: 92.5% (9101/9249 correct for Fusion vs 8555/9249 for evmole alone).
- Full pipeline run 2026-06-09; dataset no longer blocked on ETH_RPC_URL.

External failure distribution:

| Category | Count | Addressability |
|---|---|---|
| `selector_missing_from_bytecode` | 136 (1.5%) | `not_addressable_from_bytecode` |
| `known_selector_table_miss` | 12 (0.1%) | `deterministically_addressable` |
| **Total failures** | **148** | |

- 0 `model_or_scoring_addressable` failures.
- 12 deterministic misses are a new pattern not present in held-out (was 0 there);
  12/9249 = 0.1% of external functions, too small to justify synthetic generation.
- `selector_missing_from_bytecode` is the ceiling: 246 external cases (vs 52 held-out),
  Uniswap V4 hook callbacks absent from bytecode dispatch.

Full external pipeline (commands run 2026-06-09):

```bash
python scripts/eval/mine_failures.py \
  --data data/external_eval.parquet \
  --external \
  --out-json eval_output/external_failures.json \
  --out-md eval_output/external_failure_summary.md
python scripts/eval/champion_benchmark.py \
  --data data/external_eval.parquet \
  --external
```

Reports: `eval_output/external_failure_summary.md`, `eval_output/external_generalization.md`.

Full external pipeline:

```bash
python scripts/eval/build_external_eval_set.py \
  --candidate-limit 1500 \
  --limit 500 \
  --rate-limit 1.0 \
  --resume
python scripts/eval/run_evaluation.py \
  --data data/external_eval.parquet \
  --external \
  --out eval_output/external_evaluation.md
python scripts/eval/mine_failures.py \
  --data data/external_eval.parquet \
  --external \
  --out-json eval_output/external_failures.json \
  --out-md eval_output/external_failure_summary.md
python scripts/eval/champion_benchmark.py \
  --data data/external_eval.parquet \
  --external
```

## Step 1: Mine Current Failures First

Build `scripts/eval/mine_failures.py` before adding synthetic generation or new
model work. The script should run on the same deterministic split used by
`scripts/eval/run_evaluation.py`, compare reconstructed functions against
ground truth, and emit both machine-readable and human-readable outputs.

Outputs:

- `eval_output/failures.json`: one structured entry per failed function.
- `eval_output/failure_summary.md`: counts by failure category and addressability.

Each JSON entry should include:

- Contract address, name, compiler version, and split metadata when available.
- Selector.
- Whether the selector appears in extracted bytecode selectors.
- Ground-truth name and input types.
- Predicted name and input types.
- Reconstructor source tier.
- Confidence.
- Candidate list.
- evmole recovered types, if any.
- Signature DB candidates, if any.
- Known-selector table hit or miss.
- Failure category.
- Addressability class.

Initial failure categories:

- `selector_missing_from_bytecode`
- `signature_db_miss`
- `evmole_no_signal`
- `evmole_arity_miss`
- `evmole_type_miss`
- `signature_collision_mispick`
- `known_selector_table_miss`
- `tuple_or_dynamic_type_mismatch`
- `output_not_recovered`
- `other`

The summary markdown should lead with counts and percentages, then list the
largest addressable failure classes with representative examples.

## Step 2: Classify Addressability

Do not treat every miss as an ML problem. The failure miner should assign one of
three addressability classes.

| Class | Meaning | Examples | Action |
|---|---|---|---|
| `not_addressable_from_bytecode` | The ABI item cannot be inferred from available bytecode evidence | ABI-only external callbacks, selectors absent from bytecode | Document separately; do not train against as reconstruction failures |
| `deterministically_addressable` | A rule, table, or lookup fix should solve it | Known-selector table miss, parser extraction miss, stable repeated selector | Fix deterministic path first |
| `model_or_scoring_addressable` | More evidence or learned scoring could improve it | `bytes32` vs `uint256`, `address` vs `uint160`, arity errors, collision mis-picks | Target with synthetic data, confidence calibration, or scoring |

This classification is the decision point for the rest of the plan. Synthetic
generation should only target failures in `model_or_scoring_addressable` unless
there is a specific deterministic regression fixture to add.

### Step 2 Decision And Outcome

The initial failure-mining run had 49 failures:

| Failure category | Count | Current class | Decision |
|---|---:|---|---|
| `selector_missing_from_bytecode` | 18 | `not_addressable_from_bytecode` | Do not train against these as bytecode reconstruction failures |
| `known_selector_table_miss` | 31 | `deterministically_addressable` | Evaluate deterministic known-interface completion before ML/scoring |

Important nuance: all 31 `known_selector_table_miss` failures have
`selector_in_bytecode: false` and no evmole signal. The known selector table can
map those selectors to signatures, but fusion will not emit them unless another
deterministic signal decides that the interface should be completed. Therefore,
the next deterministic fix is not just "expand the table"; it is:

1. Build a clean external/training-only known-interface table for repeated
   selector sets such as Uniswap hook callbacks.
2. Add a conservative interface-completion trigger, backed by bytecode,
   contract-family, or verified training evidence.
3. Re-run `mine_failures.py` and `champion_benchmark.py` to verify that the 31
   deterministic failures improve without adding false positives.

Because there are currently 0 `model_or_scoring_addressable` failures, Steps 4,
6, and 7 have no immediate ROI until deterministic completion is tested and the
failure distribution changes.

Outcome: deterministic known-interface completion fixed the 31
`known_selector_table_miss` failures. The current held-out run has 18 failures,
all `selector_missing_from_bytecode` and all `not_addressable_from_bytecode`.
No ML/scoring work is justified by the current canonical failure distribution.

The discovered interface set is the Uniswap V4 hook callback family:

- 10 callback functions.
- Trigger selectors: `575e24b4` and `dc4c90d3`.
- Training evidence: 100% precision and 100% recall across 14 training
  contracts.
- Runtime source label: `known-interface-completion`.
- Runtime confidence: `medium`.

## Step 3: Build The Champion Benchmark Gate

Create `scripts/eval/champion_benchmark.py` once failure mining exists. It
should run the current branch against fixed datasets and compare results against
a committed champion baseline.

Bootstrap flow:

```bash
python scripts/eval/champion_benchmark.py --init
git add eval_output/champion_baseline.json
```

The first `--init` run creates the committed champion baseline from the current
state. That file is the regression reference for later runs and should not be
manually edited. Subsequent benchmark runs compare the current branch against
`eval_output/champion_baseline.json`; if a new version should become champion,
update the baseline only as an explicit release/benchmark decision.

Benchmark slices:

- Canonical held-out verified-contract split.
- Hard-case failures from `eval_output/failures.json`.
- Deterministic regression fixtures.
- Synthetic fixtures once Step 4 exists.
- Offline/no-network mode.

Metrics:

- Exact parameter-type accuracy.
- Function-name accuracy.
- Selector coverage.
- Accuracy by confidence bucket.
- Accuracy by source tier: signature, evmole, known-selector table, ML, selector.
- Addressable-failure accuracy.
- Regression count versus champion.

Gate behavior:

- Fail if canonical held-out accuracy regresses beyond a small configured
  tolerance.
- Fail if high-confidence precision regresses against the champion baseline
  beyond tolerance. Absolute precision targets belong in confidence calibration
  reports, not the default champion comparison.
- Fail if a deterministic regression fixture breaks.
- Report improvements and regressions by failure category, not just aggregate
  accuracy.

## Step 4: Targeted Synthetic Generators

Status: deferred. The current canonical failure distribution has no
`model_or_scoring_addressable` failures. Revisit this step only if future
failure mining finds type ambiguity, arity, collision, or scoring failures.

Only build generators for failure classes found by Step 1. Start with simple
template-based Solidity generation, not a broad contract generator.

Recommended toolchain:

- Python string templates for small Solidity contracts.
- `solc` or `solc-select` for compiler version control.
- Config matrix for optimizer enabled/disabled, optimizer runs, and via-IR where
  supported.
- JSONL/parquet output containing source, bytecode, ABI, compiler settings, and
  generator metadata.
- Graceful handling for compilation failures, with failures recorded rather than
  crashing the whole run.

Example targeted generators:

- `bytes32` vs `uint256` scalar variants.
- `address` vs `uint160` variants.
- Dynamic type handling: `bytes`, `string`, arrays.
- Tuple and struct calldata variants.
- Overloaded function variants.
- Selector collision fixtures.
- Dispatcher/proxy patterns only if Step 1 shows they are real failures.

Success is not "generate 1,000 contracts." Success is covering the dominant
addressable failure classes with enough variation to measure whether a change
fixes the class.

## Step 5: Leverage Existing ML And Fusion Work

Status: deferred for ML work. The current data supports deterministic
interface-completion work, not model training.

This plan should not replace the existing ML path. It should feed it better
data and better evaluation.

Existing assets to reuse:

- `FusionReconstructor` source tiers and candidate output.
- `choose_candidate()` as the current per-selector baseline.
- `MLReconstructor` as the Tier 3 fallback interface.
- Existing feature extraction and family/type classification code.
- Canonical evaluation split and reports.

Use synthetic data to answer specific questions:

- Can the ML fallback improve `model_or_scoring_addressable` failures?
- Does learned confidence beat the current heuristic confidence labels?
- Can candidate scoring reduce collision mis-picks without hurting simple cases?
- Which synthetic fixtures transfer to verified-contract held-out data?

## Step 6: Calibrate Confidence

Status: deferred. Calibration is still useful product work, but it is not needed
to close the current canonical gap because remaining failures are not
bytecode-addressable.

The current `confidence` field is heuristic. After failure mining and benchmark
gates exist, calibrate confidence against measured held-out outcomes.

Features to consider:

- Source tier.
- Candidate count.
- Exact or partial evmole agreement.
- Candidate rank.
- Selector frequency.
- Known-selector table provenance.
- Failure category history.
- Whole-contract consistency features, if available.

Output requirements:

- `eval_output/confidence_calibration.md` with precision and recall per bucket.
- A documented target for `high`, such as `>=99%` exact parameter-type precision
  on canonical held-out data.
- Clear behavior when the target is not met: lower confidence labels rather than
  over-claiming.

## Step 7: Whole-Contract Scoring Only If Failure Data Justifies It

Status: deferred. Current failure mining found 0 collision mis-picks and 0
type-ambiguity failures, so there is no evidence yet that whole-contract scoring
would improve the canonical split.

Do not start with a vague search layer. First prove that per-selector
`choose_candidate()` misses cases where whole-contract evidence would help.

If justified, begin with additive scoring rather than complex search:

- Per-selector candidate score from signature DB rank and evmole agreement.
- Contract-family score from selector co-occurrence.
- Interface-pattern score for ERC and callback sets.
- Penalties for inconsistent overload/type motifs.

Only consider beam search or more complex assignment algorithms if additive
scoring improves hard-case fixtures but still leaves clear global-consistency
errors.

## Deliverables

| Deliverable | Purpose |
|---|---|
| `scripts/eval/mine_failures.py` | Classify current misses before new model work |
| `eval_output/failures.json` | Machine-readable failure data |
| `eval_output/failure_summary.md` | Human-readable failure distribution |
| `scripts/eval/champion_benchmark.py` | Regression gate against the current best reconstructor |
| `eval_output/champion_baseline.json` | Committed champion baseline created by `--init` |
| `scripts/eval/build_external_eval_set.py` | Resumable Sourcify/RPC external dataset builder |
| `eval_output/external_generalization.md` | Report-only external generalization comparison |
| `scripts/eval/build_external_known_table.py` | Build training-only/external interface-completion tables |
| `data/known_interface_sets.json` | Conservative known-interface completion data |
| `scripts/synth/generate_targeted_contracts.py` | Generate fixtures for proven addressable failure classes |
| `scripts/synth/compile_contracts.py` | Compile generated Solidity across selected compiler settings |
| `eval_output/champion_benchmark.md` | Main benchmark report |
| `eval_output/confidence_calibration.md` | Measured confidence precision/recall |

## Non-Goals

- Do not introduce literal reinforcement learning.
- Do not build a broad synthetic generator before failure mining identifies the
  target classes.
- Do not count verified ABI lookup as reconstruction accuracy unless reported as
  a separate Tier 0 lookup path.
- Do not train against `not_addressable_from_bytecode` failures as if bytecode
  reconstruction could solve them.
- Do not optimize only for headline accuracy. Track addressable failures,
  confidence calibration, and regressions.

## Success Criteria

- `mine_failures.py` explains the current held-out gap by category and
  addressability.
- The champion benchmark fails clearly on regressions and reports improvements
  by failure class.
- Synthetic generators exist for the top addressable failure classes, not for an
  arbitrary contract count.
- High-confidence output has measured precision, with documented behavior when
  calibration misses the target.
- Any ML or scoring change improves at least one addressable failure class
  without regressing canonical held-out accuracy beyond the benchmark tolerance.
