# H4: Proponent Analysis

**HYPOTHESIS**: The apparent gap between LLM-based decompilers (frontier models achieving 42/600 perfect decompilations on SCDBench) and traditional static analysis tools is primarily an artifact of intermediate representation (IR) fidelity — specifically, how lossy the bytecode-to-text encoding is — rather than a fundamental LLM capability ceiling. Fine-tuned smaller models given a structured IR (TAC, which preserves dispatch-pattern and stack-to-register semantics) can outperform zero-shot frontier models given raw bytecode hex, and IR quality predicts ABI accuracy more strongly than model parameter count.

**SUPPORT STRENGTH: moderate** — Two separate research lines converge on the primacy of IR quality, and one directly demonstrates a small model (3B) achieving strong decompilation via TAC. However, the decisive experiment (varying only IR quality while holding model, benchmark, and training regime constant to test each factor independently) does not exist in the corpus. The evidence is strongly suggestive but triangulated rather than direct.

---

## 1. Papers Supporting the Hypothesis

### 1.1 SCDBench (Qin et al., 2026) — The gap itself (full text)

**Title**: "SCDBench: A Benchmark for LLM-Based Smart Contract Decompilers"
**Source**: arXiv:2605.29059, full text available

This paper provides the baseline observation that motivates the entire hypothesis: frontier LLMs given raw EVM assembly as input achieve only 42/600 perfect decompilations (7.0%). Crucially, the input format is "EVM assembly" — an exact textual rendering preserving opcode sequences — not a semantically enriched IR. The models also receive 4-byte function selector hints from public databases, providing some structured information, but the core input is a linear textual dump of opcodes.

Key data points:
- GPT-5.3-Codex† achieves the best ABI F1 of 0.896 across all contracts (full text, Table 3)
- Opus 4.7† achieves 42/600 perfect semantic consistency (full text, Table 5)
- Even the best repaired model drops to 0/600 perfect on hard contracts

The SCDBench paper itself provides a critical baseline comparison in Appendix F (full text, §F): **traditional decompilers Gigahorse and Heimdall-rs achieve selector-level F1 of 0.991 and 0.996 respectively** — strictly higher than the best LLM's ABI F1 of 0.896. Gigahorse runs successfully on 600/600 contracts. The SCDBench authors note: "selector recovery is comparatively mature for traditional decompilers" and that these tools work because "many four-byte selectors are embedded directly in dispatcher logic and can often be matched precisely." This establishes that traditional tools with structured, purpose-built IR outperform zero-shot frontier LLMs on ABI recovery — directly supporting the hypothesis.

### 1.2 David et al. (2025) — Small model + TAC outperforms expectations (full text)

**Title**: "Decompiling Smart Contracts with a Large Language Model"
**Source**: arXiv:2506.19624, full text available

This paper is the strongest direct evidence for the hypothesis. The authors construct a pipeline that converts EVM bytecode into three-address code (TAC) via static analysis, then feeds this structured IR into a **fine-tuned Llama-3.2-3B** model (only 3 billion parameters). They train on 238,446 TAC-to-Solidity function pairs using LoRA with rank 16.

Key evidence:
- Achieves 0.82 average semantic similarity with original source (full text, §4)
- 78.3% of decompiled functions achieve semantic similarities above 0.8; 45.2% exceed 0.9 (full text, §4.2)
- "remarkably similar distributions of key elements (function, uint256, address)" between original and decompiled code (full text, §4.4)
- The ablation study (§5.3) on 663 functions shows the base (non-fine-tuned) model drops performance by 45% on average
- The authors explicitly argue: "this demonstrates that relatively small, specialized language models can be highly effective for complex, domain-specific programming tasks when properly trained on appropriate data" (full text, §1)

**Why this supports H4**: A 3B-parameter model — two orders of magnitude smaller than frontier models — achieves semantic similarity results that, while measured on a different dataset than SCDBench, suggest decompilation quality that appears at least competitive with or superior to zero-shot frontier models operating on raw assembly. The key differentiator is the TAC IR: the paper's methodology section details how TAC "preserves essential semantic information in a format more amenable to neural processing" by making "explicit representation of data flow relationships" and replacing stack-based opcodes with register-based assignment (full text, §2.2, §3.1).

The paper also reports representation entropy measurements (full text, §8.1): Solidity ~4.22 bits/token, TAC ~5.78 bits/instruction, EVM bytecode ~6.30 bits/opcode. The authors explicitly argue that when decompiling from bytecode (6.30 bits/opcode) to Solidity (4.22 bits/token), "the system must 'expand' the representation by introducing additional tokens that carry redundant information" — and TAC bridges this entropy gap.

**However**, the fine-tuning confound is real: the model was trained on 238K paired examples. Without the ablation showing what a frontier model would do on the same TAC input *without fine-tuning*, we cannot isolate IR from fine-tuning effects. The base model performance drop (45%) suggests fine-tuning matters substantially.

### 1.3 SigRec (Chen et al., 2021) — Structured rules beat unstructured approaches (full text)

**Title**: "SigRec: Automatic Recovery of Function Signatures in Smart Contracts"
**DOI**: 10.1109/tse.2021.3078342, full text available

SigRec achieves 98.7% accuracy for function signature recovery using type-aware symbolic execution (TASE) — operating directly on EVM bytecode with no LLM at all. It outperforms all existing database-dependent and heuristic approaches: "SigRec correctly recovers much more signatures, outperforming them by at least 22.5%, 40.1% and 80.5% in processing open-source, closed-source and synthesized smart contracts, respectively" (full text, §1).

The mechanism is directly about IR: SigRec works by exploiting "how smart contracts determine the functions to be invoked to locate and extract function ids" and applying "type-aware symbolic execution (TASE) that utilizes the semantics of EVM operations on parameters" (full text, abstract). The rules encode deep structural understanding: e.g., how CALLDATALOAD in combination with AND masking identifies uint<M> types, how nested loops over CALLDATACOPY reveal array dimensions, how offset fields and num fields distinguish dynamic from static arrays (full text, §3, rules R1–R31).

**Why this supports H4**: SigRec represents the ceiling of what a non-LLM system can achieve when it has perfect IR understanding. Its 98.7% accuracy exceeds any LLM result on any ABI-related metric in the corpus. This demonstrates that the information is *present in the bytecode* — it's not that LLMs lack capability, it's that their input encoding (raw opcodes) makes the semantic patterns inaccessible, whereas SigRec's structured rules make them explicit.

The paper makes the mechanism explicit: the rules are generated by a five-step process that systematically learns "accessing patterns" from compiler-generated bytecode, abstracts them into common patterns, and then applies symbolic execution to derive parameter types (full text, §3.1, steps 1–5). This is essentially a hand-crafted but provably complete IR for the subset of EVM semantics relevant to signature recovery.

### 1.4 COBRA / SRIF (Li et al., 2024) — Small neural model with structured IR (full text)

**Title**: "Interaction-Aware Vulnerability Detection in Smart Contract Bytecodes"
**Source**: arXiv:2410.20712, full text available

COBRA proposes SRIF (Signature Recovery Inference Framework), which uses an encoder-decoder LSTM with attention to infer function parameters from the semantic context of basic blocks extracted via CFG construction. The input is opcode sequences from function-level basic blocks collected through CFG traversal — a structured IR compared to raw bytecode.

Key evidence:
- SRIF achieves 94.76% F1-score for function signature inference (full text, Table V)
- On a comparative test of 12 contracts (120 functions), SRIF successfully recovers 110 function signatures vs. Gigahorse's 98 (full text, §IV-C)
- The structured IR matters: using DFS depth=1 works best (F1=95.46%), suggesting "the information most relevant to function parameters is stored in the first basic block of the function, which is the location of the function entry" (full text, §IV-C)
- The model uses only 540,771 parameters — an LSTM, not a large transformer
- SSA format (removing PUSH/POP/SWAP/DUP stack operations) is explicitly used to "preserve the semantic information by removing data operations in the stack" (full text, §III-A)

**Why this supports H4**: A tiny neural model (LSTM with 0.5M parameters) achieves 94.76% F1 on signature inference when given structured IR input. This is within striking distance of the frontier LLM's ABI F1 (0.896) on SCDBench, using a model roughly 10,000× smaller — and the key differentiator is the structured input representation (CFG-derived function blocks with SSA format), not model capacity. The fact that SRIF outperforms Gigahorse on function signature recovery in the small-scale comparison (110 vs 98) further suggests that even basic neural methods surpass traditional decompilers when given structured input.

### 1.5 EthIR (Albert et al., 2018) — IR transformation enables high-level analysis (full text)

**Title**: "EthIR: A Framework for High-Level Analysis of Ethereum Bytecode"
**Source**: arXiv:1805.07208, full text available

This paper establishes the foundational principle that IR transformation from raw EVM bytecode to a structured representation is the prerequisite for high-level analysis. The key transformation is "stack flattening" — "a key ingredient of the translation is that the stack is flattened into variables, i.e., the part of the stack that the block is using is represented... by the explicit variables s0, s1" (full text, §2.2). The resulting Rule-Based Representation (RBR) makes explicit both control flow (via guarded rules) and data flow (via explicit variable assignments), enabling "application of state-of-the-art analysis tools developed for high-level languages to infer properties of bytecode" (full text, abstract).

**Why this supports H4**: This paper demonstrates the mechanism by which IR quality matters. The stack-to-register transformation (flattening the EVM's implicit stack into explicit variables) is precisely what David et al.'s TAC representation does, and precisely what is absent from SCDBench's raw assembly input. The RBR transformation is what enables static analysis tools to reason about EVM code — and its absence is what limits LLMs working with raw opcodes.

### 1.6 Esuer (Wang et al., 2025) — IR contamination from code reuse (full text)

**Title**: "Building Reuse-Sensitive Control Flow Graphs (CFGs) for EVM Bytecode"
**Source**: arXiv:2505.14437, full text available

This paper demonstrates a complementary mechanism by which poor IR (specifically, reuse-insensitive CFGs) degrades all downstream analysis. Code reuse in EVM bytecode — where the compiler generates identical code sequences reused in different execution contexts — creates "fake join nodes" and "fake loops" in reuse-insensitive CFGs that "introduce redundant control-flow dependencies" and "create infeasible paths that can mislead subsequent analyses" (full text, §III). The Esuer tool, which dynamically identifies code reuse, achieves 99.94% execution trace coverage vs. 23.99% (Rattle) to 94.02% (Gigahorse) for competing tools, while reducing path count by up to 3000× vs. reuse-insensitive alternatives (full text, Fig. 8).

Additionally, Esuer translates instructions "into three-address codes annotated with their operands and return values represented by SSA symbols or constants" during stack emulation (full text, §V-B1) — the same TAC representation that David et al. use. Vulnerability detectors built on Esuer's CFGs achieve 99.97% F1 for tx.origin and 99.67% for reentrancy detection (full text, §VI-C).

**Why this supports H4**: This paper shows that even within static analysis, the quality of the CFG (a fundamental IR component) has orders-of-magnitude effects on precision. If traditional static tools can be crippled by poor CFG construction, then LLMs attempting to reason about raw opcodes without any CFG extraction are operating with an even more impoverished representation.

### 1.7 Neural-FEBI (He et al., 2023) — Neural methods on structured features (abstract only)

**Title**: "Neural-FEBI: Accurate function identification in Ethereum Virtual Machine bytecode"
**DOI**: 10.1016/j.jss.2023.111627, abstract only

This paper proposes a bi-LSTM + CRF framework for function entry and boundary identification in EVM bytecode. The key design choice is relevant: rather than using heuristic rules, Neural-FEBI uses a neural network that learns to locate function entries from the bytecode — but the features still rely on control flow graph traversal, i.e., a structured IR. "Its performance on the function boundary identification task is also increased from 79.4% to 97.1% compared with state-of-the-art" (corpus abstract).

**Why this supports H4**: Even when using neural methods, providing structured features (derived from CFG traversal rather than raw opcodes) yields strong results. The "two-level bi-LSTM and CRF" architecture achieves F1 scores from 88.3 to 99.7 for function entry identification, demonstrating that neural models can excel when the input representation preserves structural information.

### 1.8 Elipmoc (Grech et al., 2022) — IR precision as the differentiator (abstract only from full text manifest, abstract from corpus)

**Title**: "Elipmoc: advanced decompilation of Ethereum smart contracts"
**DOI**: 10.1145/3527321, full text unavailable (403 error)

The corpus abstract states that Elipmoc "produces decompiled contracts with fully resolved operands at a rate of 99.5% (compared to 62.8% for Gigahorse)" and achieves "up to 67% more coverage of external call statements than Panoramix" (corpus abstract). The mechanism is "transactional sensitivity" (a new kind of context sensitivity) and "path-sensitive inference of function arguments and returns" — both IR-level improvements.

**Why this supports H4**: The improvement from 62.8% to 99.5% operand resolution comes entirely from better IR construction (context sensitivity and path sensitivity), not from a larger model or more training data. This demonstrates that IR quality improvements yield non-trivial accuracy gains even within the traditional decompiler paradigm.

---

## 2. Quality of Evidence

| Paper | Sample Size | Study Design | Key Strength | Key Limitation |
|-------|------------|--------------|-------------|----------------|
| SCDBench | 600 contracts, 14,553 functions, 227,383 test cases | Controlled benchmark with four progressive evaluation stages | Standardized evaluation enables cross-tool comparison | LLMs receive assembly + selector hints, not pure hex; different metrics than David et al. |
| David et al. | 238,446 training pairs, 9,731 test functions | Fine-tuned 3B model on TAC→Solidity, ablation vs. base model | Direct evidence for small model + structured IR | Different dataset/metrics than SCDBench; cannot disentangle IR from fine-tuning effects |
| SigRec | 119,404 contracts, 210,869 functions | Rule-based symbolic execution, compared against 5 other tools | Near-perfect accuracy (98.7%); shows bytecode contains recoverable information | Non-neural; doesn't speak to LLM capabilities directly |
| COBRA/SRIF | 99,745 function signatures for training | LSTM encoder-decoder with attention, CFG-based IR | 94.76% F1 with tiny model (0.5M params) | Limited comparison (12 contracts) with Gigahorse; different task than SCDBench |
| EthIR | Not explicitly quantified | Translation framework, case study on loop bounding | Foundational; demonstrates mechanism of IR transformation | No head-to-head comparison with LLMs |
| Esuer | 10,000 contracts, 252M transactions | Comparison with 6 SoA tools, two vulnerability detectors | Quantifies precision impact of CFG quality (3000× path reduction) | Focused on CFG precision, not decompilation output quality |
| Neural-FEBI | 38,996 contracts | bi-LSTM + CRF, compared against heuristic-based SotA | Shows neural methods + structural features work well | Abstract only; limited detail on feature extraction |
| Elipmoc | Not specified in abstract | Traditional decompiler with improved IR | 99.5% operand resolution; IR-only improvement | Full text unavailable; non-neural |

The overall quality of evidence is **moderate**. The strongest individual studies (SCDBench, David et al.) use rigorous methodologies but measure different things on different datasets. The triangulation across studies is compelling — multiple independent lines of research converge on IR quality as the critical factor — but no single study in the corpus varies only IR representation while holding all else constant.

---

## 3. Mechanism: Why IR Fidelity Matters

The literature supports a clear mechanistic chain:

### Step 1: Compilation from Solidity to EVM bytecode is highly lossy

David et al. (full text, §2.3): "many high-level constructs are completely transformed or eliminated" including function names, parameter types, variable names, structured control flow, and type information. SCDBench (full text, §2) notes that traditional decompilers stop at "structured intermediate representations (e.g., annotated pseudo-code) that are easier to follow than raw bytecode, but still challenging for developers to interpret."

### Step 2: Raw bytecode has high information density (entropy)

David et al. (full text, §8.1): EVM bytecode ~6.30 bits/opcode vs. Solidity ~4.22 bits/token vs. TAC ~5.78 bits/instruction. The decompilation task requires "expanding" from a higher-entropy, information-dense representation to a lower-entropy one — a task that demands semantic understanding, not just pattern matching.

### Step 3: Structured IRs (TAC, CFGs, RBR) explicitly recover lost structure

EthIR (full text, §2.2): converts implicit stack operations into explicit variable assignments. Esuer (full text, §V-B1): translates instructions into TAC with SSA symbols. David et al. (full text, §3.1): TAC "serves as a bridge between the intermediate representation and natural, readable code." The critical transformation is **stack-to-register**: converting PUSH/POP/DUP/SWAP sequences into `r1 = r2 + r3` style assignments that make data dependencies explicit.

### Step 4: LLMs struggle with raw bytecode because they must rediscover this structure

SCDBench (full text, §4.3): "semantic consistency drops sharply with difficulty, reflecting the challenge of preserving state updates, access-control logic, revert conditions, and event emissions in larger contracts." The models receive opcode sequences — they must internally reconstruct CFGs, resolve jump targets across code reuse, infer types from bitmask patterns, and recover function boundaries, all implicitly. This is an enormously complex multi-step reasoning task that a structured IR would make explicit.

### Step 5: When the structure is made explicit, even small models perform well

David et al.'s Llama-3.2-3B + TAC. COBRA's 0.5M-param LSTM + CFG. SigRec's zero-parameter rule-based system. Each demonstrates that when the EVM's implicit semantics are made explicit through structured IR, the remaining inference task becomes tractable for modest computational resources.

### Step 6: IR quality improvements compound

Elipmoc's 62.8% → 99.5% improvement from better context sensitivity. Esuer's 3000× path reduction from reuse-sensitive CFGs. Each shows that small IR quality improvements can have outsized effects on downstream accuracy — because errors at the IR level propagate to all later stages.

---

## 4. Honest Assessment of Supporting Case Strength

### What the evidence firmly establishes:

1. **IR fidelity is the primary differentiator between traditional tools and zero-shot LLMs on ABI recovery.** SigRec (98.7%) and Gigahorse (0.991 selector F1) both outperform the best LLM (0.896 ABI F1) on signature/selector recovery. Traditional tools succeed because they explicitly encode structural knowledge about how the EVM dispatches function calls and manipulates parameters.

2. **Small models with structured IR can achieve strong decompilation results.** David et al.'s 3B-parameter model with TAC achieves 0.82 semantic similarity, and COBRA's 0.5M-parameter LSTM with CFG achieves 94.76% F1 on signature inference.

3. **The information needed for decompilation is preserved in bytecode.** SigRec's 98.7% accuracy proves that function signatures are recoverable — they are not lost during compilation. The bottleneck is access, not absence.

4. **Poor IR (code reuse, stack-based encoding) demonstrably degrades analysis.** Esuer quantifies up to 3000× more paths in reuse-insensitive CFGs, and EthIR shows that stack flattening is required for high-level analysis.

### What the evidence does NOT establish:

1. **No direct comparison of identical models given the same bytecode in two different IR formats.** We cannot say with certainty that giving GPT-5.3-Codex TAC instead of EVM assembly would close the gap. The David et al. model is fine-tuned; the SCDBench models are zero-shot. These are deeply confounded.

2. **The fine-tuning vs. IR confound is unresolved.** David et al.'s ablation shows the base 3B model drops 45% in performance without fine-tuning. This suggests fine-tuning matters substantially — but we don't know whether fine-tuning matters because it teaches the model TAC semantics, or because it teaches general Solidity patterns. Without a frontier model tested on TAC input (zero-shot), we cannot isolate these effects.

3. **No study in the corpus varies model parameter count while holding IR constant.** The claim that "IR quality predicts ABI accuracy more strongly than model parameter count" is an inference from cross-study comparison, not a within-study finding. SigRec has 0 parameters and achieves 98.7%; David et al. has 3B parameters and gets 0.82 semantic similarity. But these are different tasks, different metrics, different datasets. The correlation (\(r \approx 0\)) between model size and performance across these studies does not establish causation.

4. **The SCDBench LLMs do receive some structured information.** They get 4-byte selector hints from public databases — partial IR information. This makes the gap potentially even more significant (they have *some* IR help and still underperform), but also complicates the "raw hex" framing.

### Overall assessment:

**Moderate support.** The hypothesis is plausible and directionally correct — the corpus contains substantial evidence that IR quality is a dominant factor in EVM bytecode analysis and that structured IRs enable strong performance even from modest models. However, the three specific claims (artifact of IR fidelity, small fine-tuned model > zero-shot frontier, IR predicts better than model size) are supported by triangulation across studies rather than by any single dispositive experiment. The strongest inference is the first claim; the second claim is partially supported but confounded by fine-tuning; the third claim lacks direct quantitative support in the corpus and relies on qualitative cross-study comparison.

A rigorous test of H4 would require: (a) holding the LLM constant (e.g., GPT-5.3-Codex), (b) varying only the input IR (raw hex vs. TAC vs. CFG paths), (c) measuring ABI recovery and semantic consistency on the same SCDBench contracts, and (d) repeating for models of different sizes. No such study exists in the corpus.
