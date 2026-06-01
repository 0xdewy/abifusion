# Approaches for Reconstructing ABIs from EVM Bytecode

> **TL;DR:** Function identification from EVM bytecode is solved (F1 > 0.99 across all methods), but parameter type recovery remains substantially harder — a 22% F1 gap confirmed in controlled benchmarks. Static symbolic analysis (SigRec: 98.7% accuracy) currently dominates, but all successful methods depend on modeling compiler-specific code-generation patterns, not on the EVM's type-absent architecture. The field's central open question is whether these methods recover genuine semantic invariants or merely fingerprint known compilers. *Moderate confidence — strong convergence on core findings, but limited by paywalled key papers and absent compiler-holdout studies.*

*16 papers · 7 read in full · 4 hypotheses tested · 2026-05-31*

---

## What We Found

### 1. Function identification is solved; the real bottleneck is parameter type recovery *(strong confidence)*

Function identification — locating selectors and entry points in EVM bytecode — has reached ceiling performance. SCDBench (Qin et al., 2026, arXiv:2605.29059, **full text**) reports selector-level F1 of 0.996 for Heimdall-rs and 0.991 for Gigahorse on 600 contracts. SigRec (Chen et al., 2021, IEEE TSE, DOI: 10.1109/tse.2021.3078342, **full text**) achieves combined signature recovery accuracy of 98.7% across 119,404 contracts and 210,869 functions. Neural-FEBI (He et al., 2023, JSS, DOI: 10.1016/j.jss.2023.111627, **abstract only**) converges at F1 88.3–99.7 for function entry identification using neural methods. These converge because every Solidity and Vyper contract implements the same dispatch mechanism: CALLDATALOAD at offset 0 to read the selector, DIV/SHR to extract the first 4 bytes, and EQ/JUMPI comparisons to route to function bodies. This is a single well-defined pattern that is robust to optimization and compiler version.

Parameter type recovery is structurally harder. On the same 600-contract benchmark, Heimdall-rs drops from selector F1 0.996 to prototype F1 0.777 — a 22-percentage-point gap (SCDBench Appendix F, **full text**). The mechanism is well-understood: the EVM operates on 256-bit words with no runtime type tags, meaning `address`, `uint256`, `bytes32`, and `int256` are bytecode-indistinguishable without observing how the contract *uses* each value — AND masking reveals uint types, SIGNEXTEND reveals signed integers, BYTE instructions reveal bytes32, and the *absence* of arithmetic on a 160-bit value suggests address (SigRec Rules R4, R11–R18, **full text**). SigRec's own fallback rule R4, which defaults unknown 32-byte parameters to `uint256`, is an explicit acknowledgment of the underlying ambiguity. David et al. (2025, arXiv:2506.19624, **full text**) confirm that "type information [is] largely erased during compilation" and that reconstructing it "requires sophisticated analysis of how the value is used throughout the contract" — analysis that even their fine-tuned 3B-parameter model fails to fully achieve (case study on DeFi staking contracts: 0.52 semantic similarity).

The practical posture: function identification works out of the box; parameter type recovery works only if you encode enough compiler-specific knowledge. SigRec's 31 hand-crafted rules bridge most of the gap (to ~98%), but less-engineered tools (Heimdall-rs: 77.7%) leave a 22% gap open in practice.

### 2. Compiler optimization is not the dominant accuracy constraint *(strong confidence)*

A pervasive concern in the literature is that Solidity's `--optimize` flag destroys the bytecode patterns ABI recovery depends on. This concern has a plausible mechanism — David et al. (**full text**) document that constant folding, dead code elimination, and control flow optimization create non-injective source-to-bytecode mappings — but does not translate to a practical accuracy ceiling. SigRec's large-scale evaluation (**full text**, §5.3) explicitly tests the hypothesis: accuracy stays ≥96% "with or without optimization" across all compiler versions from v0.1.1 to v0.8.0. The gap between tools (SigRec vs. competitors: 22.5–80.5 percentage points) dwarfs any optimization effect.

This finding comes with two caveats. First, Wang et al. (2025, arXiv:2505.14437, **full text**) documents that compiler-introduced code reuse patterns create real structural ambiguities — fake join nodes, fake loops, and polymorphic jump targets — that degrade CFG construction, a prerequisite for accurate analysis. Their Esuer tool achieves 99.94% trace coverage by dynamically resolving reuse, compared to 23.99% for naive static tools. Second, the most architecturally relevant paper — Lin & Gao (2021), "When Function Signature Recovery Meets Compiler Optimization" (IEEE S&P, DOI: 10.1109/sp40001.2021.00006) — is paywalled and yielded zero findings. It could contain evidence specific to non-SigRec methods that this inquiry could not evaluate. With that acknowledged, the available evidence strongly suggests that method quality dominates compilation settings.

### 3. The LLM vs. traditional tool gap is about task scope, not just IR fidelity *(moderate confidence)*

The 42/600 perfect decompilations achieved by frontier LLMs on SCDBench (**full text**) is widely cited as evidence of a capability gap. However, this number measures end-to-end source-level decompilation — compilable Solidity, full ABI recovery, and semantic equivalence under differential replay — not ABI recovery alone. SigRec's 98.7% signature recovery and SCDBench's 42/600 full decompilation are categorically different tasks.

The structured-IR hypothesis — that providing LLMs with three-address code (TAC) or CFG-annotated bytecode instead of raw EVM assembly would close the gap — has plausible supporting evidence but no definitive test. David et al. (**full text**) fine-tunes Llama-3.2-3B on 238,446 TAC-to-Solidity pairs and achieves 0.82 semantic similarity. COBRA/SRIF (Li et al., 2024, arXiv:2410.20712, **full text**) uses a 0.5M-parameter LSTM on CFG-extracted basic blocks to achieve 94.76% F1 for signature inference. Traditional tools like SigRec (zero ML parameters) achieve 98.7% from symbolic rules. These results demonstrate that structural encoding matters immensely — but none compares models of different sizes on the same input format, and the SCDBench within-model gap (ABI F1 0.896 vs. semantic consistency 28%) suggests that the harder problem is semantic competence, not interface reconstruction.

### 4. All ABI recovery methods depend on compiler-specific knowledge *(moderate confidence)*

This is the pattern that unifies the inquiry. SigRec encodes 31 rules reverse-engineered from Solidity and Vyper output. SRIF's training labels come from matching selectors against the 4byte.directory, populated from verified source contracts compiled with known compilers. David et al.'s fine-tuned model trains on 238,446 pairs of TAC and Solidity source — all from known compiler versions. SCDBench's frontier LLMs receive 4-byte selector hints from the same databases. Heimdall-rs's prototype recovery (F1 0.777) is achieved without the intensive compiler-specific engineering of SigRec (98.7%), and the 22 pp gap is exactly the difference between shallow and deep compiler modeling. No paper in the corpus evaluates any method on contracts from an unseen compiler or an obfuscation technique that deliberately changes code-generation patterns. Until such a study exists, published accuracy figures should be understood as conditional on known compiler conventions.

---

## The Novel Insight

**Every ABI reconstruction method in the literature — symbolic, neural, and LLM-based — is ultimately a compiler model masquerading as a type inference engine. The EVM itself provides no type information; what these methods recover is not "the ABI of this contract" but "the compiler conventions that happen to have generated this bytecode."**

This reframes the problem. The field's axis of difficulty is not static-vs-dynamic-vs-ML, but rather *how much* compiler-specific information a method encodes and *how fragile* that encoding proves when the compiler changes. SigRec encodes 31 explicit rules — maximal compiler knowledge, maximal accuracy (98.7%). Heimdall-rs encodes less — 22 pp lower precision. Zero-shot LLMs encode whatever compilers they have seen in their training distribution — they achieve 0.896 ABI F1 but collapse to 28% semantic consistency. The correlation suggests that "ABI recovery accuracy" measures compiler coverage, not analytical capability. A genuinely novel compiler with unknown code-generation conventions could plausibly reduce every method in the corpus to near-chance performance. The most valuable experiment no one has run — and the one that would clarify whether this field is mature or in its infancy — is a clean compiler hold-out: train on Solidity v0.4–v0.7 and test on v0.8, or train on Solidity and test on Vyper, measuring per-type degradation.

The connection also explains the persistent abstraction gap between ABI recovery and full decompilation. Recovering an ABI is compiler fingerprinting; recovering semantically correct Solidity is understanding what the contract *does*. These are different tasks that the evidence treats as a single spectrum.

---

## Hypotheses Tested

| Hypothesis | Verdict | Key evidence |
|---|---|---|
| H1: Compiler optimization imposes an irreversible accuracy ceiling | **Refuted** | SigRec §5.3 (full text): accuracy ≥96% with/without optimization on 119K contracts. Method gap (22.5–80.5 pp) dominates. |
| H2: Selector bottleneck separates function ID (solved) from type recovery (harder) | **Supported** | SCDBench (full text): selector F1 0.996 vs. prototype F1 0.777. SigRec (full text): 31 rules for types vs. 1 for IDs. |
| H3: Parameter type recovery is the neglected hard subproblem | **Supported** | SigRec's 31 rules vs. 1 dispatch pattern (full text). SCDBench 22 pp gap (full text). R4 catch-all proves fundamental ambiguity. |
| H4: LLM decompilation gap is IR fidelity, not model capacity | **Inconclusive** | Triangulation supports directionally; no controlled IR-ablation study exists in corpus. Within-model gap (ABI 0.896 vs. semantic 28%) suggests semantic ceiling. |

---

## Open Questions

**Are current ABI recovery methods learning genuine semantic invariants of the EVM's type system, or are they merely learning compiler-specific code generation patterns?**

This is the single most valuable unresolved question. It determines whether existing tools can be trusted on adversarial contracts, MEV bots, novel compilers (Huff, Fe), or Solidity's `--via-ir` pipeline (≥v0.8.20). It determines whether published accuracy numbers are measurements of a general capability or measurements of how many compiler versions were in the training data. It determines whether the field's next investment should be in better type inference algorithms or in larger compiler-coverage datasets. A hold-out study — training on one compiler and testing on another, with per-type accuracy decomposition — would resolve it. SigRec (119K contracts) and SCDBench (600 contracts, stratified by compiler version) each have the data infrastructure to run this study but neither does.

---

## References

1. Chen T, Li Z, Luo X, et al. (2021). "SigRec: Automatic Recovery of Function Signatures in Smart Contracts." *IEEE Transactions on Software Engineering.* DOI: 10.1109/tse.2021.3078342 **[full text]**
2. Qin K, Song D, Gervais A. (2026). "SCDBench: A Benchmark for LLM-Based Smart Contract Decompilers." *arXiv:2605.29059.* **[full text]**
3. David I, Zhou L, Song D, Gervais A, Qin K. (2025). "Decompiling Smart Contracts with a Large Language Model." *arXiv:2506.19624.* **[full text]**
4. Li Z, et al. (2024). "Interaction-Aware Vulnerability Detection in Smart Contract Bytecodes." *arXiv:2410.20712.* **[full text]**
5. He J, Li S, Wang X, et al. (2023). "Neural-FEBI: Accurate function identification in Ethereum Virtual Machine bytecode." *Journal of Systems and Software.* DOI: 10.1016/j.jss.2023.111627 **[abstract only]**
6. Grech N, et al. (2022). "Elipmoc: advanced decompilation of Ethereum smart contracts." *Proceedings of the ACM on Programming Languages (PACMPL/OOPSLA).* DOI: 10.1145/3527321 **[abstract only]**
7. Wang Y, et al. (2025). "Building Reuse-Sensitive Control Flow Graphs (CFGs) for EVM Bytecode." *arXiv:2505.14437.* **[full text]**
8. Lin Y, Gao D. (2021). "When Function Signature Recovery Meets Compiler Optimization." *IEEE Symposium on Security and Privacy (S&P).* DOI: 10.1109/sp40001.2021.00006 **[abstract only]**
9. Lagouvardos S, et al. (2020). "Precise static modeling of Ethereum 'memory'." *Proceedings of the ACM on Programming Languages (PACMPL).* DOI: 10.1145/3428258 **[abstract only]**
10. Arceri V, et al. (2025). "EVMLiSA: Sound Static Control-Flow Graph Construction for EVM Bytecode." *Blockchain: Research and Applications.* DOI: 10.1016/j.bcra.2025.100384 **[abstract only]**
11. Ruaro N, et al. (2024). "Not your Type! Detecting Storage Collision Vulnerabilities in Ethereum Smart Contracts." *NDSS 2024.* DOI: 10.14722/ndss.2024.24713 **[abstract only]**
12. Albert E, et al. (2018). "EthIR: A Framework for High-Level Analysis of Ethereum Bytecode." *arXiv:1805.07208.* **[full text]**
13. Qian Y, et al. (2022). "Nimbus: Toward Speed Up Function Signature Recovery via Input Resizing and Multi-Task Learning." *QRS 2022.* DOI: 10.1109/qrs57517.2022.00053 **[abstract only]**
14. Anonymous. (2023). "Abusing the Ethereum Smart Contract Verification Services for Fun and Profit." *arXiv.* **[full text]**
15. Bartel A, et al. (2012). "Dexpler." *Proceedings of the ACM SIGPLAN International Workshop on State of the Art in Java Program analysis.* DOI: 10.1145/2259051.2259056 **[abstract only]**
16. Gagnon E, et al. (2000). "Efficient Inference of Static Types for Java Bytecode." *Lecture Notes in Computer Science.* DOI: 10.1007/978-3-540-45099-3_11 **[abstract only]**
