# Plan 08: Address Critical Packaging and Evaluation Issues

## Context

Criticism from a recent review identifies six categories of issues:

1. **Inconsistent evaluation numbers** — README says 96.4%, but holdout shows 96.7% and external validation shows 96.1%. No canonical evaluation story.
2. **Fragile ML model loading** — `FusionReconstructor.__init__` eagerly calls `MLReconstructor.get_instance()` which raises if `cache/family_model.pt` is missing. The "stubbed fallback" claim is fragile for fresh installs.
3. **Missing declared dependencies** — `torch`, `numpy`, and `eth_utils` are imported but absent from `pyproject.toml`.
4. **No canonical evaluation script** — No single command regenerates the headline number from scratch.
5. **No candidate provenance in output** — Reconstruction output shows `source` but not what alternatives were considered.
6. **No address-based reconstruction** — The tool only accepts bytecode, not contract address + chain.

---

## Step 0: Fix Fragile ML Loading (Quick Win)

**Problem:** `fusion.py:103` — `self.ml = MLReconstructor.get_instance()` is called eagerly in `__init__`. `MLReconstructor.get_instance()` calls `load_model()` which raises `FileNotFoundError` if the model is missing.

**Current flow:**
```
FusionReconstructor() → __init__ → self.ml = MLReconstructor.get_instance()
                                          → load_model("cache/family_model.pt")
                                          → raises if missing
```

**Fix:** Make ML a lazy property that gracefully degrades.

```python
# In FusionReconstructor.__init__, remove the eager self.ml = ...
# Instead:

@property
def ml(self) -> Optional[MLReconstructor]:
    if self._ml_instance is None:
        try:
            self._ml_instance = MLReconstructor.get_instance()
        except Exception as e:
            logger.debug("ML model not available: %s", e)
            self._ml_instance = None
    return self._ml_instance

# In reconstruct(), use: if self.ml and self.ml.model is not None:
```

ML is only called in Tier 3/4 (when Tiers 1+2 fail), so making it lazy and non-fatal is correct.

**Files:** `abi_reconstructor/fusion.py`

---

## Step 1: Fix Missing Dependencies

**Problem:** `torch`, `numpy`, `eth_utils` are imported in code but not declared in `pyproject.toml`.

**Fix:** Add to `pyproject.toml`:

```toml
dependencies = [
    "requests>=2.31.0",
    "evmole>=0.8.4",
    "python-dotenv>=1.0.0",
    "pandas>=2.0.0",
    "pyarrow>=14.0.0",
    "numpy>=1.24.0",
    "eth-utils>=2.0.0",
]

[project.optional-dependencies]
ml = ["torch>=2.0.0"]
dev = [...]
```

**Files:** `pyproject.toml`

---

## Step 2: Canonical Evaluation Script

**Problem:** No single command produces the headline accuracy number. Numbers in README (96.4%), holdout eval (96.7%), and external validation (96.1%) are inconsistent.

**Fix:** Create `scripts/eval/run_evaluation.py` that:
1. Uses a fixed seed and dataset size (500 contracts from contracts.parquet)
2. Runs 5-fold holdout with known-selector table from training split only
3. Reports one number: overall accuracy on held-out set
4. Writes `eval_output/canonical_results.md` with train/holdout split parameters

The canonical number should be the held-out (external) accuracy since that's the honest measure. Training accuracy should not be reported as a headline number.

**Target:** `eval_output/canonical_results.md` with a single bold number.

**Files:** `scripts/eval/run_evaluation.py`, `eval_output/canonical_results.md`

---

## Step 3: Expose Candidate Provenance in Output

**Problem:** `FusionReconstructor.reconstruct()` returns `source` (e.g., `"signature+evmole"`) but not what alternatives were considered or how confident the choice was.

**Fix:** Extend each function in the output to include:

```python
{
    "type": "function",
    "name": "transfer",
    "selector": "a9059cbb",
    "inputs": [{"type": "address", "name": "to"}, {"type": "uint256", "name": "amount"}],
    "source": "signature+evmole",        # which tier resolved it
    "candidates": [                       # what alternatives were available
        {"name": "transfer", "types": ["address", "uint256"], "source": "openchain"},
        {"name": "transfer", "types": ["address", "uint256"], "source": "4byte"},
    ],
    "confidence": "high",                # "high" | "medium" | "low"
}
```

The `choose_candidate()` function already has access to all candidates — just thread them through to the output.

**Files:** `abi_reconstructor/fusion.py`, `abi_reconstructor/reconstructor.py`

---

## Step 4: Address-Based Reconstruction CLI

**Problem:** Tool only accepts bytecode. Users want to pass `contract_address chain_id` and get ABI back.

**Fix:** Add `--address` flag to CLI:

```bash
abi-reconstruct fusion --address 0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D --chain-id 1
```

Uses eth_getCode RPC call (via `ETH_RPC_URL` env var) to fetch bytecode, then reconstructs. Falls back to the fusion path.

**Files:** `abi_reconstructor/cli.py`

---

## Step 5: Document Known Selector Table Provenance

**Problem:** The known selector table currently has entries sourced from the evaluation dataset. This is dataset contamination.

**Fix:**
1. Rename `data/known_selector_signatures.json` → `data/known_selector_signatures.json.bak`
2. Create `scripts/eval/build_external_known_table.py` that:
   - Sources selectors ONLY from training contracts (not eval)
   - Uses same min_occurrences=2 threshold
   - Documents clearly: "Built from training split only — no eval dataset leakage"
3. Update `eval_output/canonical_results.md` to explicitly note the table is built from training split

**Files:** `scripts/eval/build_external_known_table.py`, `data/known_selector_signatures.json`, `eval_output/canonical_results.md`

---

## Step 6: Update README with Canonical Number

**Problem:** README claims 96.4% but canonical eval may show a different number.

**Fix:** After Step 2 (canonical eval script) runs, update README to point to `eval_output/canonical_results.md` as the single source of truth, and update the headline number to match.

**Files:** `README.md`

---

## Implementation Order

```
Step 0: Fix fragile ML loading (quick win, ~30 min)
Step 1: Fix missing dependencies (5 min)
Step 2: Canonical eval script (1-2 hours)
Step 3: Expose candidates (1-2 hours)
Step 4: Address-based CLI (2 hours)
Step 5: Document known table provenance (1 hour)
Step 6: Update README (15 min)
```

---

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `abi_reconstructor/fusion.py` | Modify | Lazy ML loading, expose candidates in output |
| `abi_reconstructor/cli.py` | Modify | Add `--address` flag |
| `pyproject.toml` | Modify | Add numpy, eth-utils; add `[ml]` optional extra |
| `scripts/eval/run_evaluation.py` | Create | Single canonical eval script |
| `scripts/eval/build_external_known_table.py` | Create | Build known table from training only |
| `eval_output/canonical_results.md` | Create | Canonical evaluation report |
| `README.md` | Modify | Point to canonical results, update headline number |

---

## Success Metrics

| Issue | Metric | Target |
|-------|--------|--------|
| ML fragile loading | Fresh install without model | FusionReconstructor works, ML returns None |
| Missing deps | `pip install -e .` installs all deps | No import errors |
| Inconsistent numbers | README points to canonical_results.md | Single bold number |
| Candidate provenance | Each function output has `candidates` field | Always populated for Tier 1 |
| Address-based CLI | `abi-reconstruct fusion --address 0x... --chain-id 1` works | Returns valid ABI JSON |
| Known table provenance | Table built from training only | Documented in canonical_results.md |