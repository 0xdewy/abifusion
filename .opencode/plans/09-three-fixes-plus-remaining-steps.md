# Plan 09: Three Fixes + Remaining Steps (3/4)

## Priority Order

1. **Fix README/eval consistency** — add evmole baseline to canonical eval script
2. **Deprecate external_validation_report.md** — retire it; canonical_evaluation.md is the single source of truth
3. **Rename the lazy ML test** — sharpen intent without deleting the invariant it protects
4. **Step 3: Candidate provenance** — expose alternatives and confidence in reconstruct() output
5. **Step 4: Address-based CLI** — `abifusion fusion --address 0x... --chain-id 1`

---

## Fix 1: Add evmole baseline to canonical eval (Highest Priority)

**Problem:** README table shows "evmole baseline: 90.0%" but that number comes from a different dataset (fusion_eval.md), not the canonical eval script. Every number in the comparison table should come from the same evaluation path.

**Fix:** Add `--baseline` flag to `run_evaluation.py` that also evaluates evmole's reconstruction on the same split and writes the result to the canonical report.

Changes to `scripts/eval/run_evaluation.py`:

```python
# Add new import
from abifusion.reconstructor import ABIReconstructor  # evmole-based offline fallback

# In main(), after computing fusion heldout accuracy:
evmole_fusion = ABIReconstructor()
evmole_correct, evmole_total = 0, 0
for _, row in heldout_df.iterrows():
    truth = true_functions(row["abi"])
    recon = evmole_fusion.reconstruct(row["bytecode"])
    pred = {fn["selector"]: tuple(i["type"] for i in fn["inputs"]) for fn in recon["functions"]}
    for sel, info in truth.items():
        if pred.get(sel) == info["types"]:
            evmole_correct += 1
        evmole_total += 1

evmole_acc = pct(evmole_correct, evmole_total)

# In render_report(), add evmole row to the comparison table:
f"| evmole baseline | {evmole_acc:.1f}% |"
```

**Changes to `eval_output/canonical_evaluation.md`:**
- Add evmole baseline row to the headline comparison table
- Update methodology to note evmole was evaluated on the same held-out set

**Changes to `README.md:29-32`:**
- Keep the comparison table (it's useful)
- Ensure the evmole number matches canonical_evaluation.md

**File:** `scripts/eval/run_evaluation.py`, `eval_output/canonical_evaluation.md`, `README.md`

---

## Fix 2: Retire external_validation_report.md

**Problem:** `external_validation_report.md` and `canonical_evaluation.md` both show held-out accuracy. Duplicate reports are how stale numbers resurface.

**Fix:** Overwrite `external_validation_report.md` with a single line:

```markdown
# External Validation Report

**Deprecated.** See [canonical_evaluation.md](canonical_evaluation.md) for the
canonical accuracy figure. This report is retained for historical reference
only and will be removed in a future release.
```

Or delete it entirely. Overwriting with a deprecation note is safer for git history.

**File:** `eval_output/external_validation_report.md`

---

## Fix 3: Rename lazy ML test to sharpen intent

**Problem:** `test_init_does_not_load_ml_model` only checks `sys.modules`. The name doesn't convey *why* this matters — i.e., that importing FusionReconstructor should not pull in optional torch dependency.

**Fix:** Rename to `test_init_does_not_import_ml_dependencies`. The invariant being tested is: constructing FusionReconstructor must not eagerly import abifusion.ml_reconstructor (which would require torch).

The test itself is correct — it just needs a sharper name.

**File:** `tests/test_fusion.py:88`

---

## Step 3: Candidate Provenance in Output

**Problem:** `reconstruct()` output shows `source` (e.g., `"signature+evmole"`) but not what candidates were available or how confident the choice was.

**Fix:** Add two fields to each function in the output:

```python
{
    "type": "function",
    "name": "transfer",
    "selector": "a9059cbb",
    "inputs": [{"type": "address", "name": ""}, {"type": "uint256", "name": ""}],
    "source": "signature+evmole",
    "candidates": [                    # what was available before choosing
        {"name": "transfer", "types": ["address", "uint256"], "source": "openchain"},
        {"name": "transfer", "types": ["address", "uint256"], "source": "4byte"},
    ],
    "confidence": "high",             # high | medium | low
}
```

**Implementation:**

In `fusion.py`, `choose_candidate()` returns `(name, types)` today. Extend it to return `(name, types, confidence)` where confidence is derived from:
- `high`: exact evmole match (Branch 1)
- `medium`: same-arity overlap (Branch 2)
- `low`: first candidate fallback (Branch 3)

Also thread `candidates` through `reconstruct()` into the function dict. The `candidates` field should be populated for all Tier 1 resolutions (where candidates exist) and `None` for single-source tiers.

**Files:**
- `abifusion/fusion.py` — extend `choose_candidate()`, pass candidates to output
- `tests/test_fusion.py` — add test for `candidates` and `confidence` fields

---

## Step 4: Address-Based CLI

**Problem:** Tool only accepts bytecode. Users want `abifusion fusion --address 0x... --chain-id 1`.

**Fix:** Add `--address` and `--chain-id` flags to the fusion subcommand in `cli.py`.

```python
# In cli.py, add to fusion subparser:
parser.add_argument("--address", help="Contract address to reconstruct")
parser.add_argument("--chain-id", type=int, default=1, help="Chain ID (default: 1)")

# In the handler:
if args.address:
    rpc_url = os.environ.get("ETH_RPC_URL")
    if not rpc_url:
        raise SystemExit("ETH_RPC_URL environment variable required for --address")
    # eth_getCode RPC call
    code = requests.post(rpc_url, json={
        "jsonrpc": "2.0",
        "method": "eth_getCode",
        "params": [args.address, "latest"],
        "id": 1,
    }).json()
    bytecode = code["result"][2:] if code["result"].startswith("0x") else code["result"]
    result = reconstructor.reconstruct(bytecode)
else:
    bytecode = args.bytecode
```

**Files:**
- `abifusion/cli.py` — add address/chain-id flags to fusion subparser
- `tests/test_cli.py` — add test for address-based reconstruction (mock RPC response)

---

## Summary of Changes

| File | Action | Purpose |
|------|--------|---------|
| `scripts/eval/run_evaluation.py` | Modify | Add `--baseline` flag to compute evmole accuracy on same split |
| `eval_output/canonical_evaluation.md` | Modify | Add evmole row to comparison table |
| `README.md` | Modify | Ensure evmole number matches canonical eval |
| `eval_output/external_validation_report.md` | Overwrite | Deprecation notice pointing to canonical |
| `tests/test_fusion.py:88` | Rename | `test_init_does_not_load_ml_model` → `test_init_does_not_import_ml_dependencies` |
| `abifusion/fusion.py` | Modify | Extend `choose_candidate()` with confidence; thread candidates into output |
| `tests/test_fusion.py` | Add tests | Test `candidates` and `confidence` fields are populated |
| `abifusion/cli.py` | Modify | Add `--address` and `--chain-id` to fusion subcommand |
| `tests/test_cli.py` | Add tests | Test address-based reconstruction |