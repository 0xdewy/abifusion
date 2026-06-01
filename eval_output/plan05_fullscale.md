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

## Recommendations / next steps
1. **Keep the rule-based pipeline as the primary type source.** The ML
   parameter-type model is not yet competitive; treat it as experimental.
2. **Investigate the discriminating features** before relying on them: verify
   the 7-vector is discriminative (per-class stats), and reconsider fusion
   (currently a small projection concatenated to the encoder output). A neutral
   ablation suggests the signal isn't reaching the type head usefully.
3. **More data + tuning** if pursuing the ML head: 2k–10k contracts,
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
