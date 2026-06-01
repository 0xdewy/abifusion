# H2 Falsification Analysis

## REFUTATION STRENGTH: partial — H2 correctly identifies a real gap between selector recovery and type recovery, but overstates the problem by claiming parameter type recovery is "substantially unsolved" and always requires external knowledge, when compiler-intrinsic signal is sufficient for most practical type distinctions.

---

## 1. Direct Contradicting Evidence

### SigRec: 98.7% type recovery from bytecode alone (full text)

**SigRec: Automatic Recovery of Function Signatures in Smart Contracts** (Chen et al., 2021, IEEE TSE, DOI: 10.1109/tse.2021.3078342) directly contradicts H2's central claim. SigRec achieves 98.7% accuracy in recovering complete function signatures — including parameter types — purely from EVM bytecode, without source code, without external signature databases (4byte.directory, Etherscan).

SigRec's Type-Aware Symbolic Execution (TASE) exploits compiler-intrinsic bytecode patterns that H2 claims are insufficient. Specific counterexamples to H2's claim that "bytecode-intrinsic signal is insufficient to distinguish semantically different types that share 256-bit EVM word encoding":

- **uint<N> vs bytes<N>**: Distinguished by AND-mask position. uint<N> masks retain lower-order bytes (zero-padded on left); bytes<N> masks retain higher-order bytes (zero-padded on right). Rule R11 vs R12 (sigrec, §3.4).
- **uint256 vs bytes32**: Distinguished by BYTE instruction usage. BYTE accesses individual bytes of bytes32; AND masks are used for uint256. Rule R18 (sigrec, §3.4).
- **uint256 vs int256**: Distinguished by SIGNEXTEND or signed arithmetic instructions (SDIV/SMOD/SLT/SGT). Rule R13, R15 (sigrec, §3.4).
- **uint256 vs address**: Distinguished by presence of mathematical operations on uint160 vs. address-typed values. Rule R16 (sigrec, §3.4).
- **bool**: Distinguished by two consecutive ISZERO instructions. Rule R14 (sigrec, §3.4).

SigRec also distinguishes arrays, bytes, strings, and nested structures by analyzing CALLDATALOAD/CALLDATACOPY patterns and loop structures (R1–R10, R19–R31). The paper reports accuracy never falling below 96% across all Solidity compiler versions from v0.1.1 to v0.8.0, optimization on or off (signal, §5.3).

**Crucially, SigRec does not use external databases.** The paper explicitly states prior approaches are inadequate "because they rely on ... incomplete databases or incomplete heuristic rules" and that SigRec works "without the need of source code and function signature databases" (sigrec, Abstract, §1). The entire rule set (31 rules) is derived from the bytecode's own instruction semantics, not from 4byte.directory lookups.

### SRIF: 94.76% F1 for type inference from bytecode context (full text)

**COBRA / SRIF** (Li et al., IEEE TDSC, DOI: not in corpus) reports 94.76% F1-score for function parameter inference using an LSTM-based seq2seq model trained on bytecode opcode sequences surrounding function entries. The model achieves this with only depth-1 basic block context, suggesting the entry-block bytecode patterns carry strong type signal. Performance stays ≥96% accuracy across 85 Solidity compiler versions (cobra, §IV.D, RQ3).

**Important caveat:** SRIF's ground-truth labels come from matching function selectors against the 4byte.directory database (cobra, §IV.A1: "These function signatures were then matched against the 4byte database"). So while SRIF *learns* from bytecode patterns, its labeled training data depends on external knowledge. This does not directly refute H2's claim that pure bytecode-intrinsic signal is insufficient — SRIF learns a mapping, but the mapping's source of truth is external.

### SCDBench: Selector recovery vs. prototype recovery gap is real but not "unsolved" (full text)

**SCDBench** (Qin et al., 2026, arXiv:2605.29059) provides one of the strongest empirical measurements. Traditional decompilers achieve:
- Gigahorse: selector micro F1 = **0.991** (function IDs nearly perfectly recovered)
- Heimdall-rs: selector micro F1 = **0.996**, typed prototype micro F1 = **0.777**

This confirms H2's claim that a gap exists between selector identification and full type recovery. However, a 77.7% prototype F1 is not "substantially unsolved" — it represents a mature capability with room for improvement, not a fundamental barrier.

For LLM-based decompilers, GPT-5.3-Codex† achieves ABI F1 of 0.896, with perfect ABI recovery for 413/600 contracts (scdbench, Table 3, Figure 2). This is well above what H2's framing would predict.

### Heimdall-rs: 77.7% typed prototype recovery from bytecode (full text)

Heimdall-rs recovers typed function prototypes with micro-F1 0.777 using symbolic execution + decompilation (scdbench, Appendix F, Table 11). This tool does not use LLMs or massive external databases for type inference — it uses program analysis on bytecode, further undermining H2's claim that external knowledge is required.

---

## 2. Confounds and Alternative Explanations

### Confound 1: Compiler-specific signal vs. information-theoretic necessity

**SigRec's success is a compiler artifact, not a proof of bytecode-intrinsic sufficiency.** Every rule in SigRec depends on specific code-generation patterns of the Solidity and Vyper compilers. If a compiler changes how it generates AND masks, loop structures, or CALLDATALOAD patterns, SigRec's rules would break. The paper's rule-generation methodology (sigrec, §3.1) is explicitly: "develop a tool to automatically construct smart contracts and compile them into bytecode, from which the rules will be learned." This is reverse-engineering of compiler conventions, not extraction of type information from an information-theoretically necessary signal.

H2's claim about "information-theoretic bottleneck" is thus partially correct at the formal level — there is no EVM-level type system that guarantees distinguishability. But it is incorrect at the practical level, because compilers reliably emit distinguishing patterns.

### Confound 2: SigRec's 98.7% includes cases where type recovery is fundamentally impossible

SigRec explicitly admits several type distinctions it cannot make:
- **struct vs. individual parameters**: "There is no sufficient hint from the bytecode to distinguish these two different situations" (sigrec, §2.3.1, struct). A function `f((uint256,uint256))` produces bytecode identical to `f(uint256,uint256)`.
- **int256 vs. uint256 without signed operations**: When no SDIV/SMOD/SLT/SGT instructions appear, SigRec falls back to labeling 256-bit values as uint256 (sigrec, Rules R4/R15). This is a heuristic default, not a type inference.
- **bytes vs. string**: Differentiated only by whether BYTE instructions appear to manipulate individual bytes — a behavioral difference that may not be present (sigrec, §2.3.1.4).

### Confound 3: The evaluation benchmark may over-represent easy cases

SigRec's evaluation corpus (119,404 unique open-source contracts) necessarily consists only of contracts whose source code was available for ground-truth labeling. These contracts may systematically differ from closed-source contracts in complexity, compiler optimization patterns, or coding conventions. The 98.7% accuracy may not generalize to the harder cases that H2 is concerned about (obfuscated contracts, MEV bots, malicious contracts with deliberately scrambled selectors).

### Confound 4: SCDBench's "ABI recovery" conflates function identification with type recovery

SCDBench's ABI F1 metric measures both whether the right functions are found AND whether they have correct types. The decomposition of precision vs. recall (scdbench, Table 3) shows recall (finding all functions) is the larger failure mode. Type errors vs. missing-function errors are not separately reported, so we cannot isolate the type recovery sub-problem's difficulty.

---

## 3. Methodological Weaknesses

### SigRec (Chen et al., 2021)

1. **No replication study.** The 98.7% figure comes from a single team evaluating their own tool. No independent replication exists in the corpus.

2. **Hand-crafted rules limit generalizability.** The 31 rules were manually summarized after observing compiler output (sigrec, §3.1, Step 5). This is vulnerable to human error in rule generalization and to silent incompleteness for edge cases.

3. **Symbolic execution coverage limits.** TASE stops when jump targets depend on unknown inputs (sigrec, §4.2). While the paper claims only 5 contracts in all of Ethereum trigger this, the claim is from 2021 and may not hold for newer obfuscated contracts.

4. **No measurement of per-type accuracy.** The 98.7% aggregate accuracy hides per-type breakdown. Are rare types (fixed-point decimal, nested struct, Vyper-specific types) equally accurate? The paper does not report precision/recall per type.

5. **Time-bound evaluation.** SigRec was evaluated against compiler versions up to v0.8.0 (2021). Newer Solidity versions (v0.8.20+) use the new IR pipeline (`--via-ir`), which may change bytecode generation patterns and break SigRec's hand-crafted rules.

### SCDBench (Qin et al., 2026)

1. **600 contracts is a small sample.** While deliberately chosen for evaluation cost, this limits statistical power for sub-analyses.

2. **LLM-only evaluation.** The main results are for frontier LLMs (Claude Opus 4.7, GPT-5.3-Codex, GLM-5). Traditional decompiler results are relegated to an appendix and rated only on ABI recovery, not compilability or semantic consistency. This limits direct comparison of type recovery approaches.

3. **No controlled experiment isolating type recovery from function identification.** The staged evaluation pipeline cannot cleanly separate "did the model find the right function?" from "did the model get the types right?" for ABI recovery.

### SRIF / COBRA (Li et al.)

1. **Circular dependency on external databases.** SRIF's training labels come from 4byte.directory matching (cobra, §IV.A1). The model may be learning to replicate the database's coverage patterns rather than extracting type information from bytecode. The 94.76% F1 cannot be cited as evidence for bytecode-intrinsic type signal, because the training signal itself depends on external knowledge.

2. **Small evaluation for Gigahorse comparison.** The comparison with Gigahorse uses only 12 contracts (120 functions) — insufficient for statistical conclusions (cobra, §IV.C).

3. **Type label space is only 17 types.** The evaluation considers only 17 Solidity parameter types (cobra, Fig. 4). This is a closed-world assumption that excludes novel or custom types.

---

## 4. What a Decisive Disconfirming Study Would Look Like

To decisively test H2, a study would need to:

1. **Control for external knowledge.** Create a benchmark of contracts whose compiled bytecode is the *only* available artifact — no source, no Etherscan verification, no 4byte.directory entries. Evaluate type recovery tools purely on bytecode-intrinsic signal.

2. **Measure type-level, not aggregate, accuracy.** Report precision and recall separately for each type category (uint256, address, bytes32, int256, bool, struct, array, etc.) to reveal which type distinctions are genuinely recoverable from bytecode and which are not.

3. **Include adversarial cases.** Include contracts designed to make types ambiguous — e.g., functions where uint256 and bytes32 are used identically (no BYTE instructions, no AND masking), or where int256 receives no signed operations.

4. **Isolate the type recovery sub-problem.** Build a pipeline where function identification is assumed solved (given ground-truth selectors and entry points) and evaluate only the type recovery component. This would directly measure whether H2's claim of a "fundamentally different" problem (b) holds.

5. **Cross-compiler stress testing.** Test recovery accuracy across compiler versions, optimization levels, and the new `--via-ir` pipeline to determine whether SigRec's 98.7% is a stable property of EVM bytecode or a fragile dependency on specific compiler versions.

6. **Information-theoretic lower bound.** Formally analyze how many bits of type information survive the Solidity→EVM compilation process and whether the 4-byte selector indeed creates a hard bottleneck independent of compiler conventions.

---

## Summary of Evidence

| Evidence | Supports H2? | Key Finding |
|----------|-------------|-------------|
| SigRec (full text) | **Against** | 98.7% type recovery from bytecode alone, no external DBs |
| SCDBench (full text) | **Mixed** | 0.777 prototype F1 for traditional tools; 0.896 ABI F1 for LLMs; confirms gap but not "unsolved" |
| SRIF/COBRA (full text) | **Weakly against** | 94.76% F1 for type inference, but labels depend on 4byte.directory |
| Heimdall-rs (via SCDBench, full text) | **Against** | 77.7% typed prototype F1 from symbolic execution alone |
| LLM Decompiler (David et al., 2025, full text) | **For** | "The EVM operates primarily on 256-bit words, with type information largely erased during compilation" |
| SigRec self-admitted limitations | **For** | Struct vs. individual params fundamentally indistinguishable; int256 vs uint256 ambiguous without signed ops |

**Bottom line:** H2 makes a useful distinction between function identification (well-solved) and type recovery (harder), and correctly identifies the 4-byte selector as a source of ambiguity. However, the claim that parameter type recovery is "substantially unsolved" and requires external knowledge is refuted by SigRec's 98.7% accuracy from bytecode alone and SCDBench's demonstration of 77.7% typed prototype recovery by traditional tools. The real picture is more nuanced: type recovery is *solved for the common case* given current compiler conventions, but remains *formally unsolvable in the information-theoretic sense* for certain type ambiguities (struct-vs-params, uint256-vs-int256 without signed ops, bytes-vs-string without byte manipulation). The 4-byte selector creates practical ambiguity (one hash maps to multiple signatures) but this ambiguity is primarily about function *names*, not parameter *types*, since the selector is computed over both name and types — given the selector, types constrain possible names and vice versa.
