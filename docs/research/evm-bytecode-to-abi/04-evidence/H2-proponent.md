# H2 Proponent Analysis

**SUPPORT STRENGTH: strong — multiple independent lines of evidence from distinct methodologies (symbolic execution, neural decompilation, LLM-based decompilation, traditional decompiler benchmarks) converge on a common finding: selector recovery is quantitatively mature (F1 > 0.99) while typed-prototype recovery and semantic consistency remain substantially unsolved, and the mechanism for this asymmetry (256-bit word encoding that erases type distinctions) is explicitly characterised in the literature.**

---

## 1. Evidence for Part (a): Selector/Entry-Point Identification is Well-Solved

### SigRec (full text)

SigRec (Chen et al., 2021, DOI: 10.1109/tse.2021.3078342) demonstrates that extracting function ids from bytecode is a solved problem. The method exploits the fact that contract dispatchers always read the call data at offset 0 using CALLDATALOAD, then shift the first 4 bytes via DIV or SHR to obtain the function id. SigRec reports 98.7% accuracy across 119,404 unique open-source smart contracts containing 210,869 public/external functions, with accuracy never below 96% across all Solidity compiler versions from v0.1.1 to v0.8.0, with or without optimisation. The method takes 0.074 seconds on average per function signature.

### Neural-FEBI (abstract only)

Neural-FEBI (He et al., 2023, DOI: 10.1016/j.jss.2023.111627) specifically studies function identification in EVM bytecode using a neural network framework (bi-LSTM + CRF). Across 38,996 contracts, it achieves F1-scores of 88.3 to 99.7 for function entry identification, and 79.4% to 97.1% for function boundary identification. This converges with SigRec's findings: entry-point detection is robust across both heuristic and neural methods, reaching ceiling performance.

### SCDBench traditional decompiler baselines (full text)

SCDBench (Qin et al., 2026) provides the most direct quantitative evidence for the asymmetry. Table 11 reports traditional decompilers on 600 contracts:
- **Gigahorse**: selector micro-F1 = **0.991** (function identification only)
- **Heimdall-rs**: selector micro-F1 = **0.996** (function identification only)
- **Heimdall-rs**: prototype micro-F1 = **0.777** (when argument types must match)

This is direct head-to-head comparison on the same dataset: selector recovery is near-perfect; type recovery drops by 22 percentage points.

### Convergence across methods

The three approaches — heuristic (SigRec), neural (Neural-FEBI), and traditional-decompiler (Gigahorse/Heimdall-rs) — converge on >95% selector accuracy. This cross-methodological convergence strongly supports claim (a).

---

## 2. Evidence for Part (b): Type Recovery is Substantially Unsolved

### 2.1 SigRec documents the intrinsic ambiguity (full text)

SigRec is the most thorough paper on EVM type inference and explicitly characterises the fundamental ambiguity:

**Identical bytecode for semantically different types.** The paper documents that the call-data layout and EVM instructions for reading `struct {uint256 a; uint256 b;}` are *indistinguishable* from those for two separate `uint256` parameters:

> "there is no sufficient hint from the bytecode to distinguish these two different situations" (SigRec §2.3.1, struct discussion)

This is true for both Solidity and Vyper compilers. SigRec's fallback rule R4 explicitly encodes this: "x is regarded as a uint256, if R1, R2 and R3 are not fulfilled. R4 means that without sufficient hints we just know that the length of x is 32 bytes and thus regard a 32-bytes parameter as a uint256." The default is the 256-bit word type because that is all the EVM reveals at the word level.

**Collapsed type distinctions at the EVM level.** SigRec documents that `address` (20 bytes) has identical call-data layout to `uint160`. Distinguishing them requires observing whether the value participates in mathematical operations — if not, it could be either. Similarly, `bytes` and `string` have identical layouts; distinguishing them requires observing BYTE/MSTORE8 operations on individual bytes (`string` does not support per-byte access). The entire type inference depends on 31 handcrafted rules that observe *indirect evidence* — masking patterns, sign-extension opcodes, arithmetic ops, comparison ops — rather than any intrinsic type tag in the bytecode.

**Compiler dependence as evidence of insufficient intrinsic signal.** SigRec's rules apply specifically to Solidity-compiled and Vyper-compiled bytecode. Different compilers produce different patterns for the same types (e.g., Solidity uses AND masking for address validation while Vyper uses LT comparisons). If bytecode carried sufficient intrinsic type information, compiler-specific heuristics would be unnecessary.

### 2.2 SCDBench: Type recovery lags far behind selector recovery (full text)

SCDBench evaluates three frontier LLMs (Claude Opus 4.7, GPT-5.3-Codex, GLM-5) on 600 contracts with staged metrics:

| Stage | Best result |
|-------|-------------|
| Selector micro-F1 (traditional) | 0.996 |
| Typed prototype micro-F1 (traditional) | 0.777 |
| ABI recovery F1 (LLM, best) | 0.896 |
| Semantic consistency (LLM, best) | 29.4% of functions |
| Perfect end-to-end recovery | 42/600 contracts |

The progressive collapse from near-perfect selector recovery to 29.4% semantic consistency quantifies the severity of the type-recovery problem. Even when an LLM scores 0.896 ABI F1 (meaning it gets most type signatures correct), only 29.4% of individual functions pass replay-based semantic checks. And at the whole-contract level, only 42 of 600 (7.0%) are perfectly recovered.

Critically, the best model for compilation and ABI recovery (GPT-5.3-Codex†) is *not* the best for semantic consistency (Opus 4.7†), showing that getting types correct does not guarantee correctness — and that models may be guessing types that compile but do not preserve the original semantics.

The paper explicitly notes the information-theoretic bottleneck: "Mappings from common selectors to candidate signatures are publicly available, but they are non-unique and uncertain: one selector may correspond to multiple possible signatures." The prompt provides these ambiguous 4byte.directory hints as optional input, acknowledging that the 4-byte selector is lossy.

### 2.3 David et al. (2025): Even domain-specific fine-tuning leaves a large gap (full text)

David et al. (2025, arXiv:2506.19624) fine-tune Llama-3.2-3B on 238,446 TAC-to-Solidity function pairs. Despite extensive domain-specific training data containing both bytecode structure and type information, they achieve an average semantic similarity of only 0.82 with original source.

The paper explicitly identifies the type-recovery challenge as fundamental: "The EVM operates primarily on 256-bit words, with type information largely erased during compilation. Reconstructing whether a value represents an address, a timestamp, or a financial amount requires sophisticated analysis of how the value is used throughout the contract" (§2.3).

Their ablation study (§5.3) shows that the base (non-fine-tuned) model drops to 55% of fine-tuned performance on semantic similarity — demonstrating that the external knowledge embedded in the fine-tuning corpus is essential. Without it, the model "struggled with interface compliance, breaking contract compatibility through malformed function signatures and parameter types."

Their Case Study 2 (§5.2) on a DeFi staking rewards function is particularly revealing: even the fine-tuned model achieves only 0.52 semantic similarity because it cannot recover the precise type distinctions (fixed-point arithmetic semantics, nested storage accesses that encode type-level meaning).

### 2.4 Storage collision vulnerabilities as evidence of ambiguous types (abstract only)

CRUSH (Ruaro et al., 2024, DOI: 10.14722/ndss.2024.24713) studies storage collision vulnerabilities where "two contracts have different understandings of the types/semantics of their shared storage." The existence of this vulnerability class — affecting 14,891 of 14,237,696 contracts studied — demonstrates that type information is genuinely ambiguous at the bytecode level: two pieces of code can interpret the same 256-bit words as different types, and neither interpretation is provably "wrong" from bytecode alone. This is consistent with the claim that bytecode-intrinsic signal is insufficient to disambiguate types.

---

## 3. Mechanism: Why the 4-Byte Selector Creates an Information-Theoretic Bottleneck

The literature collectively describes a two-layer information bottleneck:

**Layer 1 — Selector collision (4 bytes of a 32-byte hash).** The function selector is the first 4 bytes of keccak256(function_name + "(" + type_list + ")"). With a 4-byte output, the selector space is 2^32 ≈ 4.3 billion possible values. While sufficient for approximate disambiguation in practice, it is lossy: different function signatures can and do hash to the same 4-byte prefix (SCDBench §4.2 notes that the 4byte.directory provides "non-unique" hints). Function identification is possible because selectors appear as literal constants in the bytecode dispatcher, not because they uniquely identify the type signature.

**Layer 2 — 256-bit word encoding erases type identities.** The EVM stack and memory operate on 256-bit words. All fixed-size Solidity types (uint256, int256, bytes32, address, bool — the last two being padded) occupy a single 256-bit stack slot. The EVM provides no runtime type tags. SigRec's entire contribution is recovering types by observing *which EVM opcodes* operate on each word (AND for uint masking, SIGNEXTEND for ints, BYTE for bytes, etc.). These are indirect signals that depend on compiler behaviour, not intrinsic type annotations — and they fail entirely when the code does not include type-revealing operations.

The papers collectively show that multiple semantically distinct high-level types compile to identical bytecode sequences at the parameter-handling level, making the recovery problem fundamentally underdetermined from bytecode alone.

---

## 4. Quality Assessment of the Evidence

| Paper | Quality | Sample | Relevance |
|-------|---------|--------|-----------|
| SigRec (full text) | High. IEEE TSE. Thorough methodology. | 119,404 contracts, 210,869 functions | Direct evidence for both (a) and (b). Documents intrinsic ambiguity. |
| SCDBench (full text) | High. Systematic benchmark. Rigorous staged evaluation. | 600 contracts, 14,553 functions, 227,383 test cases | Direct quantitative evidence: selector F1 0.996 vs. prototype F1 0.777. |
| David et al. 2025 (full text) | Moderate-High. arXiv preprint but methodologically sound. | 238,446 training pairs, 9,731 test functions | Strong evidence that external knowledge (fine-tuning) is necessary but insufficient. |
| Neural-FEBI (abstract only) | Moderate. JSS publication, solid methodology. | 38,996 contracts | Supports (a) with cross-methodological convergence. |
| CRUSH/Not your Type! (abstract only) | Moderate. NDSS 2024. | 14.2M contracts | Indirect evidence: existence of type-confusion vulnerabilities supports the ambiguity claim. |

---

## 5. Honest Assessment

The supporting case for H2 is **strong**. The evidence is not merely suggestive — it provides quantitative comparisons of selector vs. type recovery on identical datasets, explicit documentation of the mechanisms causing the asymmetry, and cross-methodological convergence across symbolic execution, neural networks, traditional decompilation, and LLM-based approaches.

The strongest single piece of evidence is SCDBench's Table 11, which directly compares selector recovery (F1 = 0.996) against prototype recovery (F1 = 0.777) on the same 600 contracts using the same tool (Heimdall-rs). This is a clean, within-tool, within-dataset measurement that isolates the type-recovery difficulty from all other confounds. The fact that even frontier LLMs with access to 4-byte signature hints top out at 29.4% function-level semantic consistency reinforces this.

The mechanism is well-characterised (Layer 1: 4-byte selector is a lossy compression; Layer 2: 256-bit EVM words lack runtime type tags, and multiple high-level types produce identical bytecode patterns). SigRec explicitly documents specific type pairs that are bytecode-indistinguishable.

The main caveat in calling type recovery "substantially unsolved" is that SigRec achieves ~98% accuracy when the *compiler is known* and its rules are tuned to that compiler. The claim should therefore be refined: type recovery is substantially unsolved *without external knowledge* (compiler-specific heuristics, LLM fine-tuning on verified contracts, or 4byte.directory lookups). The bytecode-intrinsic signal IS sufficient for robust type inference when augmented with compiler models — but this is external knowledge about the compilation process, not information present in the bytecode itself. When a new compiler, new optimization pass, or obfuscated contract appears, the type-recovery performance collapses to the low rates observed for closed-source and synthesized contracts in SigRec's comparison against existing tools (Eveem: 58.1% on closed-source, 18.3% on synthesized).
