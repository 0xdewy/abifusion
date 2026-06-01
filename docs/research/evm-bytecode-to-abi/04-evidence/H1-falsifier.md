# H1 Falsification Analysis

## REFUTATION STRENGTH: PARTIAL -- SigRec provides direct counter-evidence (96%+ accuracy regardless of optimization), but no study performs a clean stratified comparison of optimized vs. unoptimized ABI recovery, and the most architecturally decisive paper (Lin & Gao, 2021) is unavailable in full text.

---

## 1. Direct Contradicting Evidence

### SigRec: Automatic Recovery of Function Signatures in Smart Contracts
**DOI**: `10.1109/tse.2021.3078342` | **Citation tag**: (full text)

**What it found.** SigRec achieves 98.7% accuracy on function signature recovery and explicitly claims robustness to optimization. From Section 5.3 of the full text:

> "The experimental result shows that SigRec achieves an average accuracy of 98.7% (§5.2), and the accuracy never goes below 96% across all compilers (from V0.1.1 to V0.8.0) **with or without optimization** (§5.3)." (emphasis added)

This is a direct empirical contradiction of H1's central claim that optimization "imposes an irreversible information-loss ceiling on ABI recovery" and that "accuracy is dominated by compilation settings rather than method choice." SigRec's accuracy is dominated by its method (TASE), not by optimization status.

**Why it works despite H1's stated mechanism.** H1 claims the optimizer destroys PUSH4/EQ dispatch-pattern artifacts. SigRec acknowledges that it uses these to locate function IDs, but its *type inference* operates on an entirely different signal: the semantics of CALLDATALOAD/CALLDATACOPY instructions, bitmask AND operations, SIGNEXTEND operations, and nested-loop patterns for array bounds checks (Rules R1-R31, Figure 11). These parameter-handling instructions are "typically near a function's entry point" and are not subject to the same optimizations that might restructure dispatch tables. The two signals (dispatch location vs. parameter type) are independent.

### SRIF (from the COBRA framework)
**DOI**: None (from CoRR abs/2410.20712) | **Citation tag**: (full text)

**What it found.** SRIF, a seq2seq LSTM-based function signature inference model, achieves 94.76% F1-score on parameter type inference, and "maintains consistently high accuracy across all tested compiler versions" (85 versions, v0.4.11 through v0.8.30), with "no observed case falling below 96% across all 85 versions."

**Critical confound.** SRIF was evaluated on "a comprehensive dataset of **unoptimized**, open-source smart contracts" (Section IV-D, emphasis added). This is a fatal gap: SRIF's strong results cannot speak to H1 because the study deliberately excluded optimization effects from its evaluation. The consistent accuracy across compiler *versions* is evidence for robustness to compiler evolution, not to optimization.

### SCDBench: A Benchmark for LLM-Based Smart Contract Decompilers
**DOI**: None (from arXiv:2605.29059) | **Citation tag**: (full text)

**What it found.** Frontier LLMs (GPT-5.3-Codex, Claude Opus 4.7) achieve ABI recovery micro-F1 up to 0.896, recovering 413/600 perfect contracts on precision and 427/600 on recall. The dataset "cover[s] diverse application domains, coding styles, compiler versions, **optimization settings**, and complexity levels." The presence of optimization-varied inputs and non-zero ABI recovery strongly suggests optimization is not an absolute information-loss ceiling. However, the perfect semantic consistency rate is only 42/600 (7%), and results are not reported stratified by optimization level -- so we cannot quantify the optimization-specific degradation.

### Neural-FEBI: Accurate function identification in EVM bytecode
**DOI**: `10.1016/j.jss.2023.111627` | **Citation tag**: (abstract only)

**What it found.** A bi-LSTM + CRF neural method for function entry/boundary identification achieves F1-scores of 88.3 to 99.7 on "38,996 publicly available smart contracts collected as binary." It "does not rely on a fixed set of handcrafted rules" and "significantly outperforms state-of-the-art, often based on handcrafted heuristic rules." This suggests neural methods can learn signals beyond the PUSH4/EQ patterns H1 claims are destroyed by optimization, but the abstract does not mention optimization effects explicitly.

---

## 2. Confounds and Alternative Explanations

### Confound 1: H1 conflates two independent signals

H1 claims PUSH4/EQ dispatch-pattern artifacts are the primary signal for ABI recovery. However, SigRec's architecture reveals at least **three orthogonal signal classes**:

| Signal class | Optimization vulnerability | Recovery method |
|---|---|---|
| Function ID extraction (PUSH4 + EQ/JUMPI) | Moderate -- dead code elimination may remove unused dispatchers; constant folding shouldn't affect hash comparisons | Static pattern matching, symbolic execution |
| Parameter type inference (AND masking, SIGNEXTEND, CALLDATALOAD offsets) | Low -- parameter-handling code is essential for correctness and cannot be folded away | Type-aware symbolic execution (SigRec), seq2seq learning (SRIF) |
| Parameter count/structure inference (nested loops, MLOAD patterns, CALLDATACOPY lengths) | Low -- structural code around entry points is largely preserved | Control-flow analysis, deep learning (Neural-FEBI) |

SigRec loses at most one signal class (function IDs from dispatch) to optimization, but retains two more classes. The hypothesis would need to show that ALL three are destroyed for its ceiling claim to hold.

### Confound 2: The missing pivotal study

**"When Function Signature Recovery Meets Compiler Optimization"** by Lin & Gao (2021, IEEE S&P, DOI: `10.1109/sp40001.2021.00006`) is the paper architecturally designed to test H1. Its title and venue (IEEE S&P, the top security conference) make it the most likely source of decisive evidence. **Only the abstract is available** (status: `abstract_only` in manifest). Without its full text, we cannot determine whether this paper confirms or refutes H1. A null result from this study (showing no optimization effect) or a positive result (showing surmountable effects) would both weaken H1. If it shows a strong optimization effect, it would be the strongest support for H1 in the corpus. Its unavailability is a major evidence gap.

### Confound 3: The LLM Decompiler paper (David et al., 2025) provides mixed evidence

**Citation tag**: (full text, arXiv:2506.19624)

This paper explicitly acknowledges that "compiler optimizations further complicate the decompilation process. The Solidity compiler performs various optimizations including **constant folding**, **dead code elimination**, and **control flow optimization**. These transformations can significantly alter the structure of the code, making it difficult to recover the original source patterns." (Section 2.3, emphasis added). This *partially supports* H1's mechanistic claim.

However, the same paper achieves 0.82 average semantic similarity and 78.3% of functions above 0.8 similarity, despite these optimization challenges. The TAC intermediate representation "preserves essential semantic information" -- directly contradicting the "irreversible" part of H1. The model "uniquely recovers... precise function signatures." Information is degraded, not destroyed.

### Confound 4: Esuer paper shows optimization-caused CFG artifacts ARE resolvable

The "Building Reuse-Sensitive CFGs for EVM Bytecode" paper (Wang et al., 2025, full text) demonstrates that code reuse (a compiler optimization for size reduction) creates fake join nodes, fake loops, and polymorphic jump targets that degrade static analysis. However, Esuer resolves these via dynamic reuse-context tracking and achieves 99.94% execution trace coverage. This demonstrates that at least one class of compiler-induced signal degradation is **surmountable**, not irreversible.

---

## 3. Methodological Weaknesses

### SigRec (the strongest counter-evidence to H1)

1. **Rules derived from self-generated contracts, not learned.** SigRec's 31 rules (R1-R31) were "summariz[ed] manually" from automatically generated contracts. This means the rules *encode* the compiler's own code generation patterns. If a new compiler version or optimization pass changes these patterns, accuracy could drop silently. This limits generalizability to novel optimization strategies.

2. **No separate reporting by optimization level.** SigRec's claim "accuracy never goes below 96%... with or without optimization" reports aggregate performance. The paper does not provide a table showing accuracy separately for optimized and unoptimized contracts. We cannot rule out the possibility that accuracy is 98.7% on unoptimized contracts and 96.0% on optimized ones -- still above 96%, but with a measurable optimization penalty.

3. **Fundamental ambiguity cannot be resolved.** SigRec explicitly acknowledges that "there is no sufficient hint from the bytecode to distinguish" a function with a struct parameter containing N fields from a function with N individual parameters (Section 2.3.1, struct). This is a genuine information-loss ceiling, but it is inherent to ABI encoding, not caused by optimization.

4. **Validation on open-source contracts only.** SigRec evaluates against ground truth from source code, meaning all tested contracts had available source. This avoids the hardest recovery challenges (obfuscated/deoptimized bytecode, unusual compiler versions).

### SRIF/COBRA

1. **No optimization testing whatsoever.** Section IV-D explicitly states "unoptimized, open-source smart contracts." This is a fatal omission for evaluating H1. Their 96%+ accuracy across 85 compiler versions tells us about compiler version robustness, not optimization robustness.

2. **Small test set for Gigahorse comparison.** Only 12 contracts (120 functions) were used to compare SRIF against Gigahorse, and the paper notes "sufficient randomness" but this sample size is too small to draw meaningful conclusions about relative performance under optimization.

### SCDBench

1. **Results not stratified by optimization.** While the benchmark includes diverse optimization settings, results are reported by difficulty level (easy/medium/hard) based on code characteristics, not by compilation settings. We cannot isolate the optimization effect.
2. **Small sample.** 600 contracts, while carefully curated, may miss edge cases in optimization-heavy contracts. The 7% perfect semantic consistency rate may partly reflect optimization effects, but also reflects general decompilation difficulty.

### Neural-FEBI -- abstract only

No methods, results, or limitations beyond the abstract are available. Cannot assess sample composition, optimization controls, or replication.

---

## 4. What a Decisive Disconfirming Study Would Look Like

A study that would decisively test H1 would need:

1. **Stratified within-subjects design**: Take N contracts (N >= 500), compile each **twice** -- once with `--optimize` and once without -- producing matched bytecode pairs.
2. **Multi-method comparison**: Apply at least three fundamentally different recovery approaches to BOTH sets:
   - A static/heuristic method (e.g., Gigahorse, Panoramix)
   - A symbolic-execution method (e.g., SigRec/TASE)
   - An ML/neural method (e.g., a trained LLM or Neural-FEBI)
3. **Stratified reporting**: Report precision, recall, and F1 for ABI recovery separately for optimized and unoptimized bytecode, for each method.
4. **Control variables**: Control for compiler version, contract size, function count, and Solidity language features.
5. **Null-hypothesis test**: Test whether the interaction term (method × optimization) is significant. If optimization dominates method choice, the interaction term should be negligible (or at least the optimization main effect should dwarf method effects).

**Does such a study exist in the corpus?** No.

The Lin & Gao (2021) IEEE S&P paper ("When Function Signature Recovery Meets Compiler Optimization") has the right architecture by title, but we cannot confirm from the abstract alone whether it contains a stratified comparison. If it does, access to the full text would either provide the disconfirming evidence or strengthen H1 considerably. This is the single most important piece of missing evidence.

SigRec's Section 5.3 comes closest, but without separate reporting by optimization level, it falls short of the decisive standard.

---

## 5. Summary Assessment

**What H1 gets right.** Compiler optimization *does* perform constant folding, dead-code elimination, and expression simplification. The LLM Decompiler paper (full text) confirms that "multiple different source code patterns might compile to identical or very similar bytecode, creating ambiguity." The SCDBench results show that even frontier models achieve only 7% perfect semantic consistency. There is genuine information loss.

**What H1 gets wrong.** The claim of an "irreversible information-loss ceiling" and that accuracy is "dominated by compilation settings rather than method choice" is contradicted by SigRec's explicit finding that accuracy stays above 96% "with or without optimization." Method choice clearly matters: SigRec (98.7%) dramatically outperforms Gigahorse/Eveem/OSD/EBD/JEB (by 22.5-80.5 percentage points) on the same contracts. If optimization dominated, these method-level differences would collapse.

**Bottom line.** H1 overstates its case. Optimization degrades signal, but not irreversibly, and method choice remains the dominant factor. However, the absence of a clean stratified comparison means the *magnitude* of the optimization penalty remains unknown. The hypothesis cannot be fully refuted without access to the Lin & Gao (2021) full text and without a proper within-subjects optimization study.
