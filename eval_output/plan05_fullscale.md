# Plan 05 — Full-scale training run (2026-06-01)

500 verified mainnet contracts (discovered via Sourcify, fetched via Etherscan),
transformer encoder, 15 epochs. Both the function-name classifier and the
parameter model were trained into canonical `checkpoints/` with the inference
path now passing discriminating features (no train/serve skew).

## Dataset
- Contracts fetched: **500** (`data/contracts.parquet`)
- Parameter samples: **5,662** → train 4,075 / val 452 / test 1,131

## Results (held-out test set)

| Metric | Demo (20 contracts) | Full-scale (500) |
|---|---|---|
| Function name — Top-3 accuracy | 80.5% | **85.1%** |
| Parameter count accuracy | 26.0% | **41.9%** |
| Parameter type accuracy | 1.9% | **32.1%** |

Scaling 20 → 500 contracts improved type accuracy ~17× and lifted count and
function accuracy meaningfully. (Per-epoch *validation* type-accuracy reads
higher, ~0.5–0.62, because it scores all positions incl. padding via the loss
path; the test number above scores only real parameter positions and is the
meaningful figure.)

## Discriminating-features ablation (test type accuracy)
- with features:   **0.3207**
- zeroed features: **0.3302**
- degradation:     **−0.0095**

**The discriminating features do not help** — zeroing them at inference slightly
*improves* type accuracy. The ablation is clean (same trained model, same
dataset features vs zeros), so this is a real signal, not skew. The SigRec
R11–R18 features as currently extracted/fused add no value to type prediction at
this scale and arguably add a little noise.

## Assessment vs Plan 05 targets

| Target | Goal | Actual | Met? |
|---|---|---|---|
| Prototype F1 | ≥ 0.972 | n/a (ML type model) | — |
| Type accuracy | ≥ 0.995 | 0.321 | ✗ |
| Full exact match | ≥ 0.960 | — | ✗ |

The targets are **not met**. The learned parameter-type model (32%) is far below
the rule-based 4byte/heuristic type recovery the project already achieves
(~0.97 in earlier `eval_summary.md`). At 500 contracts the ML type head is not
competitive with the rule-based pipeline, and the discriminating features don't
close the gap.

## Follow-up: fixed feature extraction, re-ablated (2026-06-01)

The first ablation could have been blamed on broken extraction, so the features
were fixed: located each function's body via **evmole** (`bytecode_offset`,
opcode-aligned) and extracted the 7-vec there instead of from a misaligned
dispatcher window; revived the dead `address` dim. Feature health improved a lot
on the 500-contract dataset:

| | before (dispatcher window) | after (evmole body) |
|---|---|---|
| all-zero vectors | 63.8% | **30.2%** |
| uint AND-mask rate | 5.5% | **29.7%** |
| `address` dim | 0.0% (dead) | **6.5%** |

Re-trained (same 500 contracts, 15 epochs) and re-ablated:
- with features:   **0.1632**
- zeroed features: **0.1755**
- degradation:     **−0.0123**

**The features still don't help** — the ablation is negative in both runs
(−0.0095 broken, −0.0123 fixed). (Absolute type accuracy swung 32% → 16% between
runs; that is checkpoint-selection variance — "best" is chosen by val *loss*
while val type-accuracy bounces 0.40–0.61 per epoch — not an effect of the
features.)

**Why correctly-extracted features don't help:**
1. **No positional resolution** — a single global 7-vec ("an address mask exists
   somewhere in this function") can't tell the per-position head that param 0 is
   address and param 1 is uint256.
2. **Redundant with the input** — the model already ingests the tokenized
   bytecode containing the same opcodes; the boolean summary is a lossy
   compression of information already available to the transformer.
3. **evmole already does it better** — its CFG analysis recovers exact
   per-position types (`095ea7b3`→`address,uint256`), making crude flags a weak
   proxy.

**Conclusion:** the SigRec boolean discriminating-feature approach is a dead end
for this model. Use evmole's `arguments` as the type signal instead (or as the
type source directly), and stop investing in the opcode-flag features.

## Recommendations / next steps
1. **Keep the rule-based pipeline as the primary type source.** The ML
   parameter-type model is not yet competitive; treat it as experimental.
2. **Drop the SigRec opcode-flag features** — investigated and fixed (evmole
   body window), but they still don't help (see Follow-up). If a learned
   type-hint is wanted, feed **evmole's `arguments`** (exact per-position types)
   instead of boolean flags.
3. **Stabilise checkpoint selection** if pursuing the ML head: select "best" by
   validation *type accuracy*, not val loss (they diverge); the head is noisy
   (test type acc swung 16–32% across runs).
4. **More data + tuning** if pursuing the ML head: 2k–10k contracts,
   class-balanced sampling for types, longer training, and per-type F1 to see
   which types are learnable.
4. Function-name Top-3 at 85% is the most promising head and may be worth
   developing further (e.g., as a tie-breaker for 4byte selector collisions).

## Reproduce
```
python scripts/training/train_with_discriminating_features.py \
    --num-contracts 500 --epochs 15 --ablation --checkpoint-dir checkpoints
```
Checkpoints and `data/contracts.parquet` are gitignored.
