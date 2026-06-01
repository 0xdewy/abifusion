# H3 Falsification Analysis

## REFUTATION STRENGTH: PARTIAL — SigRec (full text) demonstrates that with compiler-specific heuristic rules, parameter-type recovery accuracy can approach function-ID accuracy (98.7% combined), narrowing the gap to ~1–2 pp. However, SCDBench (full text) reveals a persistent ~22 pp gap (selector F1 0.996 vs prototype F1 0.777) when type recovery operates without the same depth of compiler-specific rule engineering, and its authors explicitly confirm the hardness asymmetry. The gap is real in the general case but is largely an engineering artifact when the compiler is known, not an irreducible semantic ambiguity.

---

## 1. Direct Contradicting Evidence

### 1.1 SigRec achieves 98.7% on the joint task — narrowing the gap to near-zero

SigRec (Chen et al., 2021, DOI: `10.1109/tse.2021.3078342`, *full text*) claims "an unprecedented 98.7% accuracy" for recovering full function signatures — defined by the authors as *function id + parameter types* — from 119,404 open-source contracts encompassing 210,869 public/external functions (abstract, lines 12–24). The paper explicitly defines its target as the joint task: "we just consider function id, parameter number and parameter types" (p. 1, footnote 1). If function-ID extraction from dispatch sequences is nearly perfect (Gigahorse achieves selector F1 0.991; SCDBench Table 11, *full text*), then SigRec's 98.7% combined accuracy implies parameter-type accuracy is at worst ~98%. A gap of ≤2 pp between function-ID and type accuracy does not meet the threshold of "substantially lower."

SigRec operates via **type-aware symbolic execution (TASE)**, which is *static heuristic analysis* — precisely the class of methods H3 claims "cannot reliably capture" the disambiguating context. TASE's 31 hand-crafted rules (R1–R31; §3–§4) exploit how Solidity and Vyper compilers emit distinct bytecode for different parameter types:
- `AND` with leading-zero-byte mask → unsigned integer `uint<M>` (R11)
- `SIGNEXTEND` → signed integer `int<M>` (R13)
- `SDIV/SMOD/SLT/SGT` → `int256` (R15)
- No arithmetic on 160-bit value → `address` (R16)
- `BYTE` instruction on a 256-bit word → `bytes32` (R18)

These rules demonstrate that the four types H3 identifies as "bytecode-indistinguishable" (`address`, `uint256`, `bytes32`, `int256`) *are* distinguishable in the vast majority of Solidity/Vyper functions, because the compiler emits type-specific operations (masking, sign extension, arithmetic, byte access) near the function entry point. This is not "deep semantic context" — it is shallow, compiler-specific pattern matching within the first few basic blocks of each function's dispatch handler.

### 1.2 SRIF achieves 94.76% F1 — but this is not genuine type inference

The COBRA framework (Li et al., 2024, *full text*, lines 920–955) reports that its SRIF component achieves 94.76% F1-score, 93.49% precision, and 96.06% recall for "function signature inference" (Table V). However, this number is misleading with respect to H3: SRIF's training data was constructed by extracting function IDs from bytecode and matching them against the 4byte directory database (line 831–834). This is fundamentally a *selector-to-signature dictionary lookup*, not a type-inference-from-bytecode-semantics task. SRIF never addresses how to distinguish `uint256` from `address` when the 4byte database returns ambiguous matches. This result does not constitute evidence for or against the hypothesis.

---

## 2. Confounds and Alternative Explanations

### 2.1 SigRec's 98.7% is a combined metric that masks the type-specific accuracy

SigRec reports "an average accuracy of 98.7%" for *signature recovery* (§5.2) but the experimental results section (which should contain the per-type breakdown) is truncated in our full-text copy (60,000 char limit reached at preliminary section boundaries). Without a published decomposition into (a) function-ID extraction accuracy and (b) parameter-type inference accuracy, the 98.7% figure cannot refute the core H3 claim that *type-specific* accuracy is substantially lower. If function-ID extraction is trivially near 100% and contributes to the average, SigRec's effective type accuracy could be materially lower than 98.7% while still yielding the reported combined figure. The paper's claim that "the accuracy never goes below 96% across all compilers" (§5.3) does not isolate type recovery either.

### 2.2 SigRec is compiler-couple — it does not generalize

SigRec's rules are learned from auto-generated contracts compiled with *specific*, known Solidity and Vyper versions (§3.1, steps 1–5). The paper acknowledges this: "our work support[s] two mainstream compilers, Solidity and Vyper" (§2.3). This means SigRec does not solve the *general* type-recovery problem; it solves the *compiler-specific* type-recovery problem. If a contract is compiled with an unknown or obfuscated compiler, or if inline assembly is used, the rules may fail. This directly aligns with H3's claim about genuine hardness: parameter-type recovery is hard *in the absence of known compiler patterns*, precisely because the 256-bit word is semantically ambiguous.

### 2.3 SigRec itself confirms fundamental ambiguity for specific types

SigRec explicitly acknowledges irresolvable ambiguity in two cases:

- **Struct vs. individual parameters**: "there is no sufficient hint from the bytecode to distinguish these two different types" (§2.3, lines 764–766) — the calldata layout of `func((uint256,uint256))` is identical to `func(uint256,uint256)`.

- **uint256 as a catch-all**: Rule R4 defaults an unknown 32-byte parameter to `uint256` — "without sufficient hints we just know that the length of x is 32 bytes and thus regard a 32-bytes parameter as a uint256. We will refine it to a specific type after using other rules to get more hints" (§3.2, R4). If the function body never invokes the relevant discriminating instructions (e.g., the parameter is only stored or passed through without math/byte access), SigRec cannot distinguish `address` from `bytes32` from `uint256`.

These admissions confirm the *genuineness* of the hardness at the bytecode level. SigRec's high accuracy merely reflects that real-world contracts happen to use their parameters in ways that trigger discriminating instructions, not that the type ambiguity is resolvable in principle.

### 2.4 The SCDBench gap may be partly implementation, not intractability

Heimdall-rs achieves Prototype F1 0.777 vs Selector F1 0.996 — a 22 pp gap. But Heimdall-rs is *one* implementation. SigRec's result (98.7%) with a different methodology suggests the gap could be closed with more extensive compiler-specific rule engineering. If the gap were a "genuine hardness difference" as H3 claims, we would expect different method classes to converge on a similar ceiling — but SigRec breaks through Heimdall-rs's ceiling by ~21 pp. This is consistent with an *engineering gap* (research priorities) rather than a *fundamental hardness gap*.

---

## 3. Methodological Weaknesses

### 3.1 SigRec: opaque combined metric, no per-type breakdown

As noted in §2.1, SigRec's 98.7% is a joint metric that conflates the trivially solved sub-task (function-ID extraction) with the hard sub-task (parameter-type inference). A rigorous evaluation would report per-parameter accuracy, per-type accuracy, and function-ID accuracy separately. The lack of this decomposition prevents us from quantifying the residual gap.

### 3.2 SigRec: no hold-out compiler or obfuscation evaluation

SigRec evaluates across Solidity v0.1.1 to v0.8.0 and Vyper (98.7% accuracy; never below 96%), but this is an *interpolation* test — all compiler versions were studied during rule generation (§3.1). A true *extrapolation* test would evaluate on a held-out compiler version or an obfuscated variant, which would directly measure whether the rules capture genuine semantics or merely compile-time artifacts. No such test is reported.

### 3.3 SRIF: conflates database lookup with type inference

SRIF evaluates "function signature inference" by matching selectors to a pre-existing 4byte database (lines 831–834). This is database coverage testing, not type inference. SRIF's 94.76% F1 is an upper bound on the quality of the 4byte database, not a measure of the solvability of type recovery from bytecode.

### 3.4 SCDBench: ABI recovery metric conflates interface-level with type-level correctness

SCDBench's ABI recovery metric (§4.1) measures whether recovered public function *signatures* match the ground truth — a signature is a string like `transfer(address,uint256)`. If a model recovers the selector `0xa9059cbb` but assigns it type `(uint256,uint256)`, both the function-ID *and* parameter-type components are wrong in a single binary judgment. The metric does not decompose errors into function-ID errors vs. type errors, so we cannot measure the type-specific gap from SCDBench's LLM ABI numbers alone. The traditional-decompiler breakdown (Selector F1 vs Prototype F1 in Table 11) *does* provide this decomposition and confirms the gap.

### 3.5 Small sample sizes in cross-method comparisons

SRIF's comparison against Gigahorse uses only 12 contracts / 120 functions (lines 956–967). SCDBench uses 600 contracts. SigRec uses 119,404 contracts but only for self-evaluation, without direct head-to-head comparison on a shared benchmark against ML/LLM methods.

---

## 4. What a Decisive Disconfirming Study Would Look Like

A study that could decisively disconfirm H3 would need:

1. **Decomposed metrics.** Report *separately*: function-ID accuracy (selector extraction), parameter-count accuracy, and per-parameter type accuracy (is `address` correctly distinguished from `uint256` from `bytes32` from `int256` in each position?). Use a per-parameter micro-average to avoid function-ID-inflation.

2. **Shared benchmark.** Evaluate at least: (a) a heuristic/symbolic method (SigRec or reimplementation), (b) an ML classifier (e.g., SRIF-style LSTM but trained on bytecode, not 4byte database), and (c) a frontier LLM (GPT-5.3-Codex / Opus 4.7) — all on the same corpus.

3. **Compiler-holdout design.** Train rules/models on contracts compiled with Solidity v0.4–v0.7; test on v0.8 contracts. Train on Solidity; test on Vyper. Train on unoptimized; test on optimized. This would isolate whether the gap is compiler-artifact or genuine semantic.

4. **Address the indistinguishable-type quartet directly.** For every parameter typed `address` in the ground truth, report how often each method outputs `address` vs `uint256` vs `bytes32` vs `int256`. This is the tightest test of H3's core ambiguity claim.

5. **Obfuscation stress test.** Evaluate on contracts with deliberately stripped type-hinting instructions (e.g., a contract where every parameter is first stored to storage via `SSTORE` without any `AND`/`SIGNEXTEND`/`BYTE`/`SDIV`). If SigRec's accuracy degrades to near-chance on this set while function-ID extraction remains perfect, H3's hardness claim is strongly supported. If an LLM maintains accuracy (having learned deeper contextual patterns), H3 is refuted for LLMs specifically.

---

## 5. Summary

| Claim in H3 | Status after falsification attempt |
|---|---|
| *Parameter type accuracy is substantially lower than function ID accuracy* | **Partially falsified.** SigRec narrows the gap to ~1–2 pp on known compilers. However, SCDBench's traditional-tool data (22 pp gap) confirms the gap is real under weaker assumptions, and SCDBench's own authors call type recovery "harder." |
| *The gap is not an accident of research priorities* | **Not fully refutable from available evidence.** SigRec's 21 pp improvement over Heimdall-rs suggests more careful engineering (a research-priority effect) can close much of the gap. But SigRec's approach is fundamentally *engineering heavy* — 31 hand-crafted rules — which itself suggests the problem is hard enough that it attracted concentrated effort. |
| *Function ID reduces to pattern matching on dispatch sequences (a solved problem)* | **Supported by the evidence.** Gigahorse (0.991) and Heimdall-rs (0.996) confirm selector extraction is near-solved for traditional tools. |
| *address, uint256, bytes32, int256 are bytecode-indistinguishable without deep semantic context* | **Partially falsified for Solidity/Vyper.** SigRec's rules demonstrate they are distinguishable via compiler-specific patterns (AND, SIGNEXTEND, BYTE, SDIV). **Supported in the general case.** SigRec's R4 catch-all and struct-vs-params ambiguity confirm that without these patterns, the types are indeed indistinguishable. |
| *Current methods (static heuristics, ML classifiers, even LLMs) cannot reliably capture this context* | **Falsified for static heuristics on known compilers.** SigRec achieves 98.7% with static heuristics. **Not falsified for ML/LLMs.** SCDBench's LLM ABI F1 (0.896) and the 2025 LLM decompilation paper's explicit acknowledgment that "a fundamental challenge lies in the recovery of high-level type information" (David et al., *full text*, §2.3) suggest LLMs have not yet reached SigRec-level type accuracy. |

**Bottom line:** H3 overstates the irreducibility of the gap. With enough compiler-specific rule engineering (SigRec's approach), parameter-type recovery can approach function-ID accuracy within 1–2 pp. However, the gap is structurally real — it widens to 22 pp when compiler knowledge is shallower (Heimdall-rs), and the fundamental ambiguity H3 identifies (256-bit words with no type tags) is explicitly confirmed by SigRec's own catch-all rules. The "genuine hardness" is genuine but *contingent on how much compiler-specific knowledge is encoded in the recovery method*, not an absolute barrier.
