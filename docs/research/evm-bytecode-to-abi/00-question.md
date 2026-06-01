# 00 — Research Question (INPUT)

**Original user topic:** "how to successfully decompile a bytecode evm contract into proper abi?"

**Narrowed research question:**

> What approaches exist for accurately reconstructing or recovering Application Binary Interfaces (ABIs) from Ethereum Virtual Machine (EVM) bytecode, and what is the comparative effectiveness of static analysis, dynamic analysis, and machine-learning-based methods?

**Narrowing decisions:**
- Focused on *ABI reconstruction* (function signatures with parameter types) rather than full decompilation to Solidity source code. Full decompilation is a superset problem; ABI recovery is the necessary first step and a well-defined sub-problem.
- Scope includes both on-chain deployed contracts (no source available) and the general case of signature disambiguation — distinguishing `transfer(address,uint256)` from `transfer(uint256,address)` when only the selector (4 bytes) is visible.
- Excluded: fully known-ABI cases (e.g., recompiling from verified source), decompilation to readable Solidity/Yul, and tool-specific benchmarks not published as research.

**Search query strings (Phase 1):**

1. `"EVM bytecode ABI reconstruction function signature recovery"`
2. `"smart contract decompilation selector extraction Ethereum ABI"`
3. `"evmole static bytecode analysis function signature inference"`
4. `"limitations failure automated ABI recovery blockchain decompilation incomplete"` *(adversarial — targets null/negative/no-effect results)*

**Mode:** Lean (narrow, focused question; verdicts folded into hypotheses file)
