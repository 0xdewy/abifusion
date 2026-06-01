# H4 Falsification Analysis

**REFUTATION STRENGTH: partial — the hypothesis is untestable from the evidence corpus as stated; its premises contain factual errors and it conflates categorically distinct tasks, but the core intuition about IR fidelity is not decisively refuted by any paper in the corpus.**

---

## 1. Direct Contradicting Evidence

### 1.1 The 42/600 number measures a fundamentally different task than the comparator tools

The hypothesis anchors its claim on "frontier models achieving only 42/600 perfect decompilations on SCDBench" and contrasts this with traditional static analysis tools. The SCDBench paper (full text) defines a "perfect" decompilation as one where *every* recovered public function passes semantic consistency replay — all return data, revert status, emitted logs, and touched storage-slot changes must match the ground truth. This is an end-to-end source-level decompilation evaluation requiring (a) format-complete output, (b) compilable Solidity, (c) full ABI recovery, and (d) semantic equivalence under differential replay.

The traditional tool with the highest reported accuracy in the corpus is **SigRec** (Chen et al., 2021, DOI: 10.1109/tse.2021.3078342, full text), which reports 98.7% accuracy. But SigRec recovers *only* function signatures — the 4-byte selector and parameter-type list. It does not produce compilable Solidity, does not recover function bodies, does not reconstruct control flow, and is not evaluated against semantic replay. These are categorically different tasks. Presenting SigRec's 98.7% as a comparator to SCDBench's 42/600 is a category error.

### 1.2 SCDBench does not supply "raw bytecode hex"

The hypothesis states that frontier models are "given raw bytecode hex." SCDBench (full text, Section 4.2) explicitly states: *"Each model receives the contract bytecode rendered as EVM assembly... The assembly is an exact textual rendering of the bytecode: it preserves the opcode sequence while presenting operations more clearly than a raw hexadecimal string."* The input already includes a degree of structured representation. The claim that the LLM gap stems from "how lossy the bytecode-to-text encoding is" is undercut by the fact that SCDBench *already* uses a non-lossy, opcode-level textual encoding.

### 1.3 Traditional decompilers' ABI performance on SCDBench is informative but incomplete

SCDBench Appendix F (full text) reports that **Gigahorse** (Grech et al., 2019) achieves selector micro-F1 0.991 and **Heimdall-rs** achieves selector micro-F1 0.996 and prototype micro-F1 0.777. This is substantially ahead of even the best LLM on typed ABI recovery (GPT-5.3-Codex† achieves 0.896 ABI F1). However, Appendix F also notes these tools "usually emit low-level intermediate representations or pseudocode rather than self-contained Solidity" and "their outputs cannot generally be compiled or passed through the semantic consistency stages." Crucially, prototype-level F1 for Heimdall-rs (0.777) shows that argument-type recovery remains difficult even for traditional tools, and the gap between selector-level (0.996) and prototype-level (0.777) recovery is large. This suggests IR fidelity is not the only bottleneck — *type inference itself* is hard regardless of representation.

### 1.4 The fine-tuned small model paper uses entirely different metrics

**David et al. (2025), "Decompiling Smart Contracts with a Large Language Model"** (full text, arXiv:2506.19624) fine-tunes Llama-3.2-3B on 238,446 TAC-to-Solidity pairs and reports average semantic similarity of 0.82 and edit distance metrics. It does **not** report ABI F1, does **not** evaluate against SCDBench, and does **not** measure semantic consistency via differential replay. There is no direct comparison possible between this paper's results and SCDBench's. The hypothesis's claim that "fine-tuned smaller models given a structured IR can outperform zero-shot frontier models given raw bytecode hex" is a conjuncture with no supporting evidence in the corpus — no paper runs both approaches on the same benchmark.

---

## 2. Confounds and Alternative Explanations

### 2.1 Task-scope confound

The primary confound is that the hypothesis compares success rates across *different tasks*. SigRec solves function signature recovery. SCDBench evaluates full source-level decompilation. Elipmoc (Grech et al., 2022, DOI: 10.1145/3527321, abstract only) reports 99.5% fully resolved operands — a structural decompilation metric, not semantic correctness. The "gap" may simply reflect that the tasks are of radically different difficulty, not that IR fidelity is the causal variable.

### 2.2 Semantic-vs-structural confound

SCDBench Table 4 (full text) shows that GPT-5.3-Codex† achieves ABI F1 of 0.896 but semantic consistency of only 28.0%. The 61.6-percentage-point gap between ABI recovery and semantic consistency *within the same model on the same benchmark* indicates that the bottleneck is semantic understanding, not interface reconstruction. If IR fidelity were the primary constraint, we would expect both ABI and semantic consistency to be similarly impaired. The fact that ABI recovery is far easier than semantic consistency for LLMs suggests the capability gap is substantive, not representational.

### 2.3 Compiler optimization confound

**Lin & Gao (2021), "When Function Signature Recovery Meets Compiler Optimization"** (IEEE S&P, DOI: 10.1109/sp40001.2021.00006, abstract only) directly addresses that compiler optimizations affect signature recovery. SigRec's rules (R1-R31, full text) depend on recognizing specific instruction patterns (e.g., AND masking for uint types, SIGNEXTEND for signed integers, nested LT loops for array dimensions). Compiler optimization levels can reorder, inline, or eliminate these patterns. Any comparison across tools must control for compiler version and optimization flags — which neither SCDBench (which samples across all compiler versions v0.4-v0.8) nor the hypothesis does.

### 2.4 Model-era confound

SigRec was published in 2021/2023. David et al. uses Llama-3.2-3B (released September 2024). SCDBench evaluates Claude Opus 4.7, GPT-5.3-Codex, and GLM-5 (all 2025-2026 models). These are separated by years of LLM capability growth. Attributing performance differences to IR rather than to model generation is a temporal confound.

### 2.5 Oracle database confound

Traditional tools like Gigahorse and Eveem rely on external signature databases (EFSD). SigRec explicitly claims independence from such databases (full text, Section 1). But SigRec still relies on Solidity/Vyper-specific heuristics (31 rules) that encode domain knowledge about compiler behavior. This is a form of "compiler oracle" — traditional tools succeed because they encode compiler-specific knowledge, not because bytecode is inherently more informative than text for neural models. The hypothesis conflates "IR fidelity" with "access to compiler-specific pattern knowledge."

---

## 3. Methodological Weaknesses

### 3.1 No controlled experiment varying IR format

No paper in the corpus runs an experiment where the same decompilation task is performed with (a) raw bytecode hex, (b) EVM assembly, (c) TAC, and (d) other IRs, holding model, dataset, and evaluation protocol constant. The hypothesis's causal claim ("IR quality predicts ABI accuracy more strongly than model parameter count") requires such a controlled experiment. Without it, the claim is speculation.

### 3.2 David et al. does not ablate IR format

The David et al. paper (full text) uses TAC consistently throughout and does not compare TAC against raw bytecode or EVM assembly input. The ablation study (Section 5.3) compares fine-tuned vs. base Llama-3.2-3B, not IR formats. The 37.4% degradation with the base model demonstrates that fine-tuning matters, but says nothing about whether IR format matters more than model size.

### 3.3 No replication of the core claim

The hypothesis claims "fine-tuned smaller models given a structured IR can outperform zero-shot frontier models." This is not tested anywhere in the corpus. David et al. (fine-tuned 3B) and SCDBench (zero-shot frontier) use different datasets, different metrics, and different evaluation protocols. No replication exists.

### 3.4 Small and skewed samples

SCDBench uses 600 contracts. SigRec evaluates on 119,404 contracts with 210,869 functions. David et al. trains on 238,446 function pairs and tests on 9,731. The sample sizes differ by orders of magnitude, and the sampling strategies differ qualitatively (SCDBench deliberately stratifies by difficulty; David et al. samples from verified contracts broadly). Cross-study comparison is unreliable.

### 3.5 SCDBench's difficulty scoring confound

SCDBench's difficulty score (Section 3.2, full text) weights bytecode scale at 30%, control flow at 20%, interface/source structure at 20%, state interaction at 15%, and low-level features at 15%. This means "hard" contracts in SCDBench are physically larger, interact with more state, and contain more complex control flow. These confounds make it impossible to isolate whether model failure on hard contracts is due to IR issues, context-length limitations, or semantic complexity.

---

## 4. What a Decisive Disconfirming Study Would Look Like

A study that could decisively refute (or confirm) H4 would need:

1. **Common benchmark**: Run both fine-tuned small models and zero-shot frontier models on the *same* contracts with the *same* evaluation protocol (ideally SCDBench or an extension thereof).

2. **IR ablation**: For each model class, vary the input representation: (a) raw hex bytecode, (b) EVM assembly/disassembly text, (c) three-address code (TAC), (d) a control-flow-graph-annotated IR. Measure ABI F1 and semantic consistency at each level.

3. **Model-size ablation**: Within the fine-tuned class, vary model size (e.g., 1B, 3B, 7B, 13B, 33B) while holding IR constant. Within the zero-shot class, vary model family (GPT, Claude, GLM) while holding IR constant.

4. **Compiler control**: Control for compiler version and optimization level. Stratify results to isolate whether IR effects persist across optimization boundaries.

5. **Disconfirming prediction**: If IR quality *does not* predict ABI accuracy more strongly than model size (i.e., a 33B model with raw hex outperforms a 3B model with TAC), the hypothesis is falsified. Conversely, if a 3B fine-tuned model with TAC outperforms a zero-shot frontier model with EVM assembly on the same benchmark, the hypothesis is supported.

No such study exists in the evidence corpus.

---

## Summary

The hypothesis is **not decisively refuted** because the evidence corpus lacks any study that tests it directly. However, it is **weakened** by: (1) a factual error about SCDBench's input format (it uses EVM assembly, not raw hex); (2) a category error in comparing SigRec's signature-recovery accuracy to SCDBench's end-to-end decompilation score; (3) the absence of any head-to-head comparison between fine-tuned small models and zero-shot frontier models on a common benchmark; (4) the strong confounding effect of task scope (ABI recovery vs. semantic decompilation) that independently explains much of the observed "gap"; and (5) the within-model gap of 61.6 pp between ABI F1 and semantic consistency on SCDBench, which suggests the LLM capability ceiling is substantive rather than representational. The core intuition — that structured IRs help — is plausible but untested from this corpus. The strongest counter-evidence is SCDBench Table 11: traditional decompilers achieve 0.991-0.996 selector-level F1 (near-perfect) yet cannot produce compilable Solidity at all, suggesting the gap between traditional tools and LLMs is primarily about task ambition, not IR fidelity.
