# abifusion Evaluation

**20 contracts · 480 functions**

## Aggregate Metrics

| Metric | Value | SOTA Reference |
|--------|-------|----------------|
| Selector F1 | 0.9636 | Heimdall-rs: 0.996 (SCDBench, 2026) |
| Selector Precision | 0.9626 | Gigahorse: 0.991 |
| Selector Recall | 0.9646 | Gigahorse: 0.991 |
| Prototype F1 (type-level) | 0.9718 | Heimdall-rs: 0.777 (SCDBench, 2026) |
| Type Accuracy (per-parameter) | 0.9953 | — |
| Arity Accuracy (correct param count) | 0.9958 | — |
| Full-Signature Exact Match | 0.9563 | — |

## By Compiler Version

| Version | Contracts | Functions | Selector F1 | Prototype F1 |
|---------|-----------|-----------|-------------|-------------|
| v0.4 | 8 | 172 | 0.9913 | 0.9970 |
| v0.5 | 7 | 136 | 0.9964 | 1.0000 |
| v0.6 | 1 | 24 | 1.0000 | 0.8750 |
| v0.7 | 2 | 54 | 0.9259 | 0.9580 |
| v0.8 | 2 | 94 | 0.8776 | 0.9172 |
