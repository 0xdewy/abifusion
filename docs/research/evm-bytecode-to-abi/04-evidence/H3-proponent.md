# H3 Proponent Analysis

**SUPPORT STRENGTH: strong** — The SCDBench paper provides direct, quantitative, side-by-side evidence that in traditional decompilers, selector-level recovery achieves F1 of 0.996 while typed-prototype recovery drops to 0.777; SigRec's architecture reinforces the mechanism by requiring 31 rules for type inference versus a single dispatch pattern for function IDs; and multiple papers independently describe the semantic opacity of the EVM's 256-bit word as the root cause.

---

## 1. Direct Quantitative Evidence of the Gap

### SCDBench (Qin, Song & Gervais, 2026) — (full text)
URL: http://arxiv.org/abs/2605.29059v1

Appendix F of SCDBench (lines 753–776) provides the most decisive evidence in the corpus. The paper evaluates traditional decompilers (Gigahorse, Heimdall-rs) on the same 600-contract benchmark and reports two metrics:

> **Selector-level F1 (function ID only):** Gigahorse = 0.991, Heimdall-rs = 0.996
> **Prototype-level F1 (including parameter types):** Heimdall-rs = 0.777

The authors state this conclusion explicitly: *"argument-type recovery is harder than recovering four-byte selectors alone"* and *"selector recovery is comparatively mature for traditional decompilers."* This is a clean, within-study comparison on the same contracts, same tools, same evaluation framework. The gap is 21.9 percentage points between recovering a function's 4-byte selector and recovering the typed prototype.

**Quality assessment:** Sample of 600 real-world Solidity contracts stratified into easy/medium/hard difficulty tiers, spanning all Solidity compiler versions v0.4–v0.8. Replayable evaluation artifacts. High quality — this is the strongest single piece of evidence in the corpus for H3.

---

## 2. Function Identification as Pattern Matching

### SigRec (Chen et al., 2021) — (full text)
DOI: 10.1109/tse.2021.3078342

SigRec (Section 2.2, lines 326–334) describes how function IDs are extracted: the callee executes `CALLDATALOAD` with offset 0 to read 32 bytes from the call data, then applies `DIV` or `SHR` to shift the 4-byte function id to the lowest 4 bytes. This is a single, well-defined dispatch pattern. The paper notes that function IDs are the first 4 bytes of the Keccak-256 hash of the signature string and can be matched against public databases (though incompletely).

In contrast, SigRec requires **31 rules (R1–R31)** for parameter type inference (Section 3, lines 1087–1189), spanning three categories: rules for CALLDATALOAD (R1–R4, R19–R25), rules for CALLDATACOPY (R5–R10, R23), and rules for other instructions (R11–R18, R20, R26–R31). The paper's own fallback rule, **R4** (lines 1160–1164), is directly quoted for its mechanistic significance:

> *"R4: x is regarded as a uint256, if R1, R2 and R3 are not fulfilled. R4 means that without sufficient hints we just know that the length of x is 32 bytes and thus regard a 32-bytes parameter as a uint256."*

This is the paper's own admission of the fundamental ambiguity: when no distinguishing operations exist, a 32-byte slot is type-unknown. R4 is the explicit acknowledgment that {address, uint256, bytes32, int256} cannot be disambiguated from the 256-bit word alone.

**Quality assessment:** 119,404 unique open-source contracts, 210,869 public/external functions. SigRec achieves 98.7% overall accuracy, but the paper's architecture reveals the asymmetry: function ID extraction (one dispatch pattern) vs. type inference (31 rules requiring symbolic execution). The R4 default-to-uint256 fallback is direct mechanistic evidence for H3. High quality.

---

## 3. Function Identification Accuracy Without Type Recovery

### Neural-FEBI (He et al., 2023) — (abstract only)
DOI: 10.1016/j.jss.2023.111627

Neural-FEBI focuses exclusively on function entry and boundary identification — it does not address parameter type recovery at all. Despite this, it achieves F1-scores of **88.3% to 99.7%** for function entry identification across 38,996 smart contracts. The paper describes the function identification problem as distinct from signature recovery and demonstrates that neural methods can bring it near ceiling. The fact that a dedicated paper can publish on *only* function identification, achieving near-perfect scores, while parameter type recovery remains a separate, harder problem, is circumstantial evidence for the gap. However, no type recovery results are reported, so direct head-to-head comparison within this paper is absent.

**Quality assessment:** 38,996 contracts. Two-level bi-LSTM + CRF architecture. But no type recovery component — supports H3 by demonstrating function identification is achievable in isolation. Moderate quality for H3 specifically.

---

## 4. The Semantic Ambiguity Mechanism

### Decompiling Smart Contracts with a Large Language Model (David et al., 2025) — (full text)
URL: http://arxiv.org/abs/2506.19624v1

This paper (Section 2.3, lines 266–275) describes the core mechanism directly:

> *"A fundamental challenge lies in the recovery of high-level type information. The EVM operates primarily on 256-bit words, with type information largely erased during compilation. Reconstructing whether a value represents an address, a timestamp, or a financial amount requires sophisticated analysis of how the value is used throughout the contract."*

The paper's empirical evaluation (9,731 test functions, 238,446 training pairs) shows that even a fine-tuned Llama-3.2-3B model achieving 0.82 average semantic similarity struggles with complex type-dependent patterns, and the ablation study (Section 5.3) confirms that domain-specific fine-tuning is essential — the base model shows *"fundamental misunderstandings of Solidity patterns"* and *"no grasp of smart contract conventions or token standards."* In the staking rewards case study (Section 5.2), the model completely collapses on type-dependent DeFi arithmetic, losing fixed-point precision operations entirely. This suggests that even modern ML approaches do not reliably capture the deep semantic context needed to disambiguate 256-bit word types.

### SigRec (Chen et al., 2021) — (full text)

SigRec's rules for disambiguating the ambiguous types reveal the mechanism:

- **int256 vs uint256** (R15): Requires detecting `SDIV`/`SMOD`/`SLT`/`SGT` instructions operating on the parameter — signed math operations. Without these, the types are indistinguishable.
- **address vs uint160** (R16): Requires detecting that the parameter is *not involved in any MATH operation* — absence of evidence, not presence.
- **bytes32 vs uint256** (R18): Requires detecting `BYTE` instruction used on the parameter; otherwise indistinguishable.
- **uint32 vs bytes4** (R11 vs R12): Both use `AND` masking but differ in *which bytes are retained* (leading vs trailing zeros in the mask constant).

All these rules require analyzing how the parameter is *subsequently used* in the bytecode — not examining the parameter value itself. This is the deepest possible semantic context. SigRec notes (lines 450–451) that `int256` and `uint256` are *not extended* in the call data, so their "layout and accessing pattern" are identical — only subsequent signed/unsigned usage distinguishes them.

---

## 5. Vulnerability Evidence that Type Ambiguity Has Real-World Consequences

### Not your Type! (Ruaro et al., 2024) — (abstract only)
DOI: 10.14722/ndss.2024.24713

This paper (14,237,696 contracts analyzed) studies storage collision vulnerabilities: when two contracts sharing storage via `delegatecall` have *different understandings of the types/semantics* of shared storage, leading to denial of service, privilege escalation, and theft. The fact that a dedicated security venue paper addresses type misunderstandings as a class of vulnerabilities confirms that type information is both critical and lossy in the EVM model. However, this paper studies storage layout type collisions, not parameter type recovery per se, so evidence is indirect. The large scale (14M+ contracts, $6M+ in novel financial damage identified) does support the claim that the EVM's typeless design creates real ambiguity problems.

**Quality assessment:** 14,237,696 contracts, 956 end-to-end exploits synthesized. Large-scale, but addresses a different (though related) type problem. Weak-to-moderate for H3.

---

## 6. Honest Assessment of Strengths and Limitations

### What the evidence strongly supports:

1. **A quantitative gap exists between function ID and parameter type recovery.** SCDBench Appendix F provides the cleanest empirical evidence: selector F1 = 0.996 vs. prototype F1 = 0.777 on the same benchmark. This is a direct, measured gap in a well-controlled study.

2. **Function ID recovery is pattern-based and solved/near-solved.** SigRec shows it reduces to one dispatch pattern. Neural-FEBI confirms F1 scores up to 99.7%. SCDBench confirms Gigahorse selector F1 = 0.991. Multiple convergent findings across different methods.

3. **The EVM's 256-bit word erases type information.** David et al. describe this explicitly. SigRec's architecture with R4 as the default-to-uint256 fallback is the mechanistic realization of this claim.

4. **Disambiguation of {address, uint256, bytes32, int256} requires analysis of subsequent operations, not the type encoding itself.** SigRec's R11–R18 rules are all about *how the value is used*, not how it is stored. Without specific usage patterns (signed math, BYTE instruction, address-specific checking), types are bytecode-indistinguishable.

### Where the evidence is weaker or missing:

1. **No study in the corpus reports separate accuracy numbers for "function ID only" vs. "parameter types only" on the SAME dataset with the SAME tool.** SCDBench comes closest by reporting selector vs. prototype F1 for Heimdall-rs, but this is one tool. SigRec reports 98.7% overall accuracy but does not break this down into function ID vs. type-recovery accuracy — its near-perfect score for *combined* signature recovery complicates the claim that type recovery is "substantially lower."

2. **The evidence that LLMs specifically fail at type disambiguation is qualitative, not quantitative.** David et al. describe the challenge and show case studies where type-dependent logic is lost, but SCDBench does not break down ABI recovery errors by whether they're function-id errors or parameter-type errors. It is possible that some ABI recovery failures are due to missing entire functions rather than incorrect parameter types.

3. **SigRec's 98.7% accuracy.** If SigRec truly achieves near-perfect type recovery through its 31-rule system, then the problem may be *solvable* through deep semantic analysis (symbolic execution that traces how values are used), even if it is *harder* than function ID. H3's claim that "current methods cannot reliably capture this" is partly contradicted by SigRec's result — SigRec IS a current method, and it achieves 98.7%. The qualification would be that SigRec's default-to-uint256 rule (R4) may over-claim accuracy when ground truth types are actually uint256 (the most common type), and that its 98.7% may mask lower accuracy on the genuinely ambiguous subset. But the corpus does not provide evidence on this point.

4. **Absence of Nimbus evidence.** The Nimbus paper (Qian et al., 2022, DOI: 10.1109/qrs57517.2022.00053) is abstract-only, so its specific accuracy numbers are unavailable beyond the abstract, which does not report them. Similarly, "When Function Signature Recovery Meets Compiler Optimization" (Lin & Gao, 2021, DOI: 10.1109/sp40001.2021.00006) has an empty abstract in the corpus, yielding no usable evidence.

### Bottom line:

The evidence case for H3 is **strong** overall. The mechanism is well-described (256-bit word erases type information; disambiguation requires semantic analysis of subsequent usage). The quantitative gap is demonstrated directly by SCDBench (selector vs. prototype F1). Function identification is confirmed as pattern-based and near-solved by multiple papers. The primary limitation is that no study in the corpus provides a clean within-tool breakdown of function ID accuracy vs. parameter type accuracy, and SigRec's 98.7% combined accuracy complicates the narrative somewhat.

---

## References (all from corpus, none retracted)

1. Qin K, Song D, Gervais A. "SCDBench: A Benchmark for LLM-Based Smart Contract Decompilers." arXiv, 2026. URL: http://arxiv.org/abs/2605.29059v1. **(full text)**
2. Chen T, Li Z, Luo X, et al. "SigRec: Automatic Recovery of Function Signatures in Smart Contracts." IEEE Transactions on Software Engineering, 2021. DOI: 10.1109/tse.2021.3078342. **(full text)**
3. He J, Li S, Wang X, et al. "Neural-FEBI: Accurate function identification in Ethereum Virtual Machine bytecode." Journal of Systems and Software, 2023. DOI: 10.1016/j.jss.2023.111627. **(abstract only)**
4. David I, Zhou L, Song D, Gervais A, Qin K. "Decompiling Smart Contracts with a Large Language Model." arXiv, 2025. URL: http://arxiv.org/abs/2506.19624v1. **(full text)**
5. Ruaro N, Gritti F, McLaughlin R, Grishchenko I, Kruegel C, Vigna G. "Not your Type! Detecting Storage Collision Vulnerabilities in Ethereum Smart Contracts." NDSS, 2024. DOI: 10.14722/ndss.2024.24713. **(abstract only)**
