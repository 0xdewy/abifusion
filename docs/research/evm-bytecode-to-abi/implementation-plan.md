# Implementation Plan — ABI Reconstructor

*Generated from `RESEARCH.md` (4 hypotheses tested, 16 papers, 7 in full text).*
*Each section traces to a specific research finding. Dependencies: Section 1 must complete first.*

---

## Section 1: The Measurement Apparatus

**Traces to:** H3 (parameter type recovery gap is real — SCDBench: selector F1 0.996 vs. prototype F1 0.777 = 22 pp gap). The IDEAS.md prior art identified this as the #1 architectural gap. The research confirms it: you cannot improve what you cannot measure, and no measurement of ABI reconstruction correctness exists in the project.

### Goal
Build a reproducible benchmark that measures selector F1, prototype/type F1, and full-signature exact-match F1 on a ground-truth dataset, stratified by compiler version.

### Success criteria
- Script runs (`python scripts/eval/benchmark.py`) and outputs:
  - `eval_results.json` — per-contract metrics
  - `eval_summary.md` — aggregate report with per-compiler-version breakdown
- N ≥ 200 verified contracts with paired bytecode + ground-truth ABI
- Covers at least Solidity v0.4, v0.5, v0.6, v0.7, v0.8
- Output includes the SCDBench-style decomposition: selector F1 vs. prototype F1 on the same contracts

### Metrics
| Metric | Measures | SOTA reference |
|---|---|---|
| Selector recall | Did we find all function selectors? | Gigahorse: 0.991 |
| Selector precision | Are the selectors we found correct? | Heimdall-rs: 0.996 |
| Prototype micro-F1 | Per-parameter type accuracy across all functions | Heimdall-rs: 0.777 |
| Full-signature exact match | Complete (selector + types) correctness per function | — |
| Per-compiler-version breakdown | Does accuracy degrade on newer compilers? | (Novel — no paper reports this) |

### Implementation steps
1. Fetch N ≥ 200 verified contracts from Etherscan/Sourcify with known source code
2. For each contract: extract ground-truth ABI from verified source, pair with runtime bytecode
3. Run existing `ABIReconstructorPipeline.reconstruct_complete_abi()` on each bytecode
4. Compute per-contract metrics: compare extracted selectors against ground truth, compare parameter types against ground truth
5. Aggregate by compiler version (extracted from contract metadata)
6. Write `eval_results.json` and `eval_summary.md`

### Key constraint from research
SigRec (Chen et al., 2021) reports 98.7% combined accuracy but does not decompose into selector-vs-type accuracy. Our benchmark must report these separately to match the SCDBench standard (Qin et al., 2026, which provides the cleanest decomposition: selector F1 0.996 vs. prototype F1 0.777).

---

## Section 2: Compiler Generalization (The Hold-Out Experiment)

**Traces to:** The Novel Insight — "Every ABI reconstruction method is ultimately a compiler model by another name." No paper in the corpus tests any method on contracts from an unseen compiler version. If our ML model generalizes across compiler versions, that is a genuinely novel result.

### Goal
Determine whether the ML models learn genuine EVM semantic invariants or merely fingerprint known Solidity compiler output patterns.

### Success criteria
- Hold-out experiment runs: train models on contracts compiled with Solidity v0.4–v0.7, evaluate on v0.8 contracts
- Report per-compiler-version accuracy degradation
- If degradation < 5 pp on prototype F1 → claim compiler generalization (novel)
- If degradation > 10 pp → justifies investment in compiler-agnostic feature extraction (Section 3)

### Implementation steps
1. Use the evaluation dataset from Section 1, annotated with compiler version per contract
2. Create a data splitter that separates contracts by compiler version: `train_split(v0.4-v0.7)` and `test_split(v0.8)`
3. Train both models (function name classifier, parameter predictor) on the v0.4–v0.7 split
4. Evaluate on the v0.8 split using the Section 1 benchmark
5. Compare against: (a) same models trained and tested on a random 80/20 split (baseline), (b) SigRec's reported cross-version robustness (accuracy never below 96% across v0.1.1–v0.8.0, but SigRec uses hand-crafted rules — our ML models are the first test of learned generalization)

### Key constraint from research
Wang et al. (2025) documents 8 distinct compiler-introduced code reuse patterns that could cause ML models to overfit to compiler-specific bytecode shapes. The hold-out experiment directly tests whether our feature extraction is learning meaningful EVM semantics or just pattern-matching compiler output quirks.

---

## Section 3: Focus on Parameter Type Recovery

**Traces to:** H2 (selector bottleneck — function ID is solved, type recovery is the bottleneck) and H3 (the 22 pp gap from SCDBench). SigRec bridges the gap with 31 hand-crafted rules; our opportunity is to learn equivalent features without hand-crafting them.

### Goal
Close at least half the 22 pp gap: push prototype F1 from baseline (current model, TBD by Section 1) toward ≥ 0.85.

### The discriminating instructions (from SigRec, Chen et al., 2021)
SigRec's 31 rules reveal which EVM instructions carry type information. These are the feature extraction targets:

| EVM pattern | What it reveals | SigRec rule |
|---|---|---|
| `AND` with a mask that zeros high-order bytes | `uint<M>` (e.g., uint32 masked by `0x0000...FFFFFFFF`) | R11 |
| `AND` with a mask that zeros low-order bytes | `bytes<M>` (e.g., bytes4 masked by `0xFFFFFFFF...0000`) | R12 |
| `SIGNEXTEND` instruction on parameter | `int<M>` (signed integer) | R13 |
| `SDIV` / `SMOD` / `SLT` / `SGT` on parameter | `int256` (signed math) | R15 |
| No arithmetic ops on 160-bit value | `address` (not uint160) | R16 |
| `BYTE` instruction on 256-bit word | `bytes32` (byte-level access) | R18 |
| `ISZERO` twice consecutively | `bool` | R14 |
| `CALLDATALOAD` + `CALLDATACOPY` with offset/length | Dynamic types (arrays, bytes, strings) | R5–R10 |

### Implementation steps
1. Add feature extraction for each discriminating instruction pattern in `features/bytecode_features.py`
   - Pattern: scan the bytes near each CALLDATALOAD for AND mask constants, SIGNEXTEND, BYTE, signed math opcodes
   - Output: a per-parameter feature vector encoding which discriminating instructions were observed
2. Add a dedicated type classifier head to `ParameterPredictionModel` 
   - Input: bytecode embedding + discriminating-instruction features
   - Output: per-parameter type logits over the type vocabulary (address, uint256, bytes32, int256, bool, etc.)
   - Separate from arity prediction (which predicts how many parameters exist)
3. Measure: run the Section 1 benchmark before and after these changes
4. Compare against Heimdall-rs prototype F1 0.777 (via SCDBench, Qin et al., 2026) as external baseline

### Key constraint from research
SigRec's Rule R4 is the information floor: "x is regarded as a uint256, if R1, R2 and R3 are not fulfilled." Some 256-bit words carry no type information because the contract never uses them in type-revealing ways (only stores/passes them through). The realistic ceiling is SigRec's 98.7% combined accuracy, and the irreducible error is the struct-vs-individual-params and int256-vs-uint256-without-signed-ops ambiguities that SigRec itself admits are bytecode-indistinguishable.

---

## Dependencies & Ordering

```
Section 1 (Measurement Apparatus)
    │
    ├──→ Section 2 (Compiler Generalization)
    │       Requires: eval dataset + compiler version metadata from Section 1
    │
    └──→ Section 3 (Parameter Type Recovery)
            Requires: baseline metrics from Section 1 to measure improvement
```

Section 1 is the blocker. Sections 2 and 3 can proceed in parallel once Section 1 completes.

---

## References

1. Chen et al. (2021). "SigRec: Automatic Recovery of Function Signatures in Smart Contracts." IEEE TSE. DOI: 10.1109/tse.2021.3078342
2. Qin et al. (2026). "SCDBench: A Benchmark for LLM-Based Smart Contract Decompilers." arXiv:2605.29059
3. Wang et al. (2025). "Building Reuse-Sensitive Control Flow Graphs (CFGs) for EVM Bytecode." arXiv:2505.14437
4. David et al. (2025). "Decompiling Smart Contracts with a Large Language Model." arXiv:2506.19624
5. He et al. (2023). "Neural-FEBI: Accurate function identification in EVM bytecode." JSS. DOI: 10.1016/j.jss.2023.111627
