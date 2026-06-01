# 02 — Hypotheses

*Culled from 12 candidate hypotheses across 4 stances (mechanistic, contrarian, cross-disciplinary, frontier). 4 survivors.*

---

## H1 — Compiler Optimization Ceiling *(state: tested — REFUTED)*

**Hypothesis:** Solidity compiler optimization (`--optimize` flag) imposes an irreversible information-loss ceiling on ABI recovery: the optimizer's non-injective transformations systematically destroy dispatch-pattern artifacts, and no known method can fully compensate. Accuracy is dominated by compilation settings rather than method choice.

**Verdict: REFUTED.** SigRec's full-text evaluation (119,404 contracts, 210,869 functions, IEEE TSE 2021) directly tests and falsifies the central prediction: accuracy stays ≥96% "with or without optimization," and the gap between methods (SigRec vs. competitors: 22.5–80.5 pp) dwarfs any optimization effect. Optimization degrades *some* signal (bound-check patterns, code reuse creates ambiguous CFGs per Wang et al. 2025), but method choice — not compilation settings — dominates accuracy. The hypothesis conflates loss of dispatch-pattern artifacts with loss of all recoverable signal; SigRec demonstrates that parameter-handling semantics (CALLDATALOAD patterns, AND masking, SIGNEXTEND), which are robust to optimization, provide sufficient signal for near-perfect recovery. The most potentially supportive paper (Lin & Gao 2021, IEEE S&P) was abstract-only and yielded zero findings.

**Key evidence:** SigRec §5.3 (full text); Wang et al. "Building Reuse-Sensitive CFGs" (full text); David et al. "Decompiling Smart Contracts with a Large Language Model" (full text).
**Confidence:** High — direct experimental falsification from the highest-quality evidence in the corpus.

---

## H2 — Selector Bottleneck Separates Identification from Type Recovery *(state: tested — SUPPORTED)*

**Hypothesis:** The 4-byte keccak256 function selector creates an information-theoretic bottleneck that splits ABI recovery into two fundamentally different problems: (a) function *identification* (locating selectors and entry points) which is well-solved and converges across methods (~95%+ accuracy), and (b) parameter *type* recovery (determining uint256 vs. address vs. bytes32) which depends on external knowledge and remains substantially harder, because bytecode-intrinsic signal is insufficient to distinguish semantically different types that share 256-bit EVM word encoding.

**Verdict: SUPPORTED (refined).** Multiple independent lines of evidence converge: SCDBench (full text) provides direct within-benchmark comparison — Heimdall-rs selector F1 0.996 vs. prototype F1 0.777 (22 pp gap). SigRec (full text) explicitly documents intrinsic ambiguity: structs are indistinguishable from individual parameters, int256/uint256 without signed ops are ambiguous, bytes vs. string are indistinguishable without BYTE instructions. Its fallback rule R4 defaults unknown 32-byte values to uint256 — an explicit admission of the information floor. The Falsifier correctly notes that SigRec achieves 98.7% type recovery without external databases like 4byte.directory, but this is achieved via 31 compiler-specific rules that encode external knowledge about *how* Solidity/Vyper compilers emit type-revealing patterns. Compiler models *are* external knowledge — a new compiler version or obfuscation can break these rules. **Refinement:** Type recovery is "solved for known compilers given intensive compiler-specific engineering, but remains formally unsolvable for genuinely ambiguous type pairs (struct vs. params, int256 vs. uint256 without signed ops)." The hypothesis's claim that type recovery "depends on external knowledge" is correct when "external" includes compiler models; the claim that it is "substantially unsolved" overstates — it is "solved for the common case, genuinely ambiguous at the margin."

**Key evidence:** SCDBench Table 11 (full text); SigRec §2.3.1, Rules R4, R15, R18 (full text); David et al. §2.3 (full text).
**Confidence:** Moderate-High — strong convergence across methodologies, but nuance required about compiler-specific vs. information-theoretic limitations.

---

## H3 — Parameter Type Recovery Is the Neglected Hard Problem *(state: tested — SUPPORTED)*

**Hypothesis:** Across all published methods, parameter type recovery accuracy is substantially lower than function identification accuracy, and this gap is a genuine hardness difference: function identification reduces to pattern matching on dispatch sequences, while parameter type recovery requires resolving the semantic ambiguity of the EVM's 256-bit word.

**Verdict: SUPPORTED.** The quantitative evidence is decisive: SCDBench (full text) shows Heimdall-rs selector F1 0.996 vs prototype F1 0.777 — a 22 pp gap the authors explicitly attribute to type recovery being "harder." SigRec's architecture confirms the mechanism asymmetry: function IDs require one dispatch pattern; type inference requires 31 rules spanning three categories (CALLDATALOAD, CALLDATACOPY, other instructions) with symbolic execution. The Falsifier correctly notes that SigRec achieves 98.7% combined accuracy, narrowing the gap to ~1-2 pp — but this required 31 hand-crafted rules, precisely demonstrating the hardness. SigRec's own admissions (R4 default-to-uint256 catch-all, struct-vs-params indistinguishable) confirm the fundamental ambiguity. The gap is real and structural: it widens to 22 pp when compiler-specific engineering is shallower (Heimdall-rs) and narrows only with intensive, brittle rule engineering (SigRec). The H3 claim that "current methods cannot reliably capture" the disambiguating context is supported for the general case — SigRec succeeds on known compilers by encoding compiler-specific knowledge, not by solving the general type inference problem.

**Key evidence:** SCDBench Appendix F (full text); SigRec R4, R11-R18 rules (full text); David et al. §2.3 (full text); Neural-FEBI (abstract only).
**Confidence:** Moderate-High — strong mechanistic and quantitative evidence from multiple sources, moderated by the absence of a clean per-type accuracy decomposition in any study.

---

## H4 — LLM Performance Gap Is IR Fidelity, Not Model Scale *(state: tested — INCONCLUSIVE)*

**Hypothesis:** The apparent gap between LLM-based decompilers (frontier models achieving only 42/600 perfect decompilations on SCDBench) and traditional static analysis tools is primarily an artifact of intermediate representation (IR) fidelity — not a fundamental LLM capability ceiling.

**Verdict: INCONCLUSIVE.** The corpus provides strong circumstantial evidence but no dispositive experiment. In favor: (a) Traditional tools with structured IR (SigRec 98.7%, Gigahorse 0.991 selector F1) consistently outperform zero-shot frontier LLMs on ABI-related metrics. (b) David et al.'s 3B-parameter model with TAC achieves 0.82 semantic similarity — competitive with what frontier models achieve on the zero-shot SCDBench task (though different metrics prevent direct comparison). (c) Small neural models with structured input (COBRA/SRIF: 0.5M-param LSTM, 94.76% F1) demonstrate that modest models can extract strong signal from structured IRs. Against: (a) The 42/600 SCDBench number measures end-to-end source-level decompilation, not just ABI recovery — comparing it to SigRec's signature recovery is a category error. (b) SCDBench already provides EVM assembly (not raw hex), so the "lossy encoding" gap is smaller than claimed. (c) The within-model gap on SCDBench (ABI F1 0.896 vs. semantic consistency 28%) suggests the bottleneck is semantic capability, not just representation. (d) No study in the corpus varies IR while holding model constant. The hypothesis is directionally plausible — structured IRs demonstrably improve all forms of bytecode analysis — but the specific causal claim (IR fidelity explains the LLM gap) is untested. A controlled experiment varying IR format (raw hex → assembly → TAC → CFG-annotated) on the same models and benchmark is needed.

**Key evidence:** SCDBench (full text); David et al. (full text); COBRA/SRIF (full text); SigRec (full text); EthIR (full text).
**Confidence:** Low — the hypothesis is plausible but untestable from this corpus due to categorical differences in tasks and metrics across the available papers.
