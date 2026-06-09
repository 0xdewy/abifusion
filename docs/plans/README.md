# Plan 06 Implementation: Reintroducing ML for Zero-Shot ABI Reconstruction

For the next improvement loop, see
[Plan 07: Closed-Loop Failure-Driven Improvement](07-closed-loop-failure-driven-improvement.md).

## What Was Built

This implementation provides the foundation for ML-based zero-shot ABI reconstruction, targeting the 3.6% residual error (244 functions) that signature-based approaches cannot resolve.

### Components

#### 1. `scripts/eval/build_known_selector_table.py` (Step 0)
Expands `data/known_selector_signatures.json` from verified contract sources.

**What it does:**
- Loads the 29 hard-case repeated selectors (Uniswap V3 callbacks, NFT setters, SVG generators)
- Scans `data/contracts.parquet` for contracts containing these selectors
- Verifies each entry against ground-truth ABI before adding
- Merges new entries with existing table (preserving already-verified entries)
- Updates count fields for expanded coverage

**Usage:**
```bash
cd /home/user/code/abifusion
python scripts/eval/build_known_selector_table.py
```

**Expected output:**
- Lists which selectors were added vs updated
- Shows count increases for expanded coverage
- Total entries should cover the 29 hard-case selectors

#### 2. `scripts/eval/verify_calldataload_offset.py` (Step 2 Phase 0)
Phase 0 verification script — validates CALLDATALOAD offset analysis.

**What it does:**
- Loads first 10 contracts from `data/contracts.parquet`
- For each function, gets bytecode_offset from evmole
- Scans function body for `PUSHn <offset> CALLDATALOAD` patterns
- Compares offset count to evmole-reported arity
- Reports match rate percentage

**Usage:**
```bash
cd /home/user/code/abifusion
python scripts/eval/verify_calldataload_offset.py
```

**Success criteria:**
- `>90%` match rate → **Phase 0 PASSED** — per-position features are viable
- `<90%` match rate → **Phase 0 FAILED** — fallback to global features

#### 3. `abifusion/features/per_position_features.py` (Step 2 foundation)
Per-position feature extractor for ML input representation.

**Phase 0 result: FAILED (0% match)** — CALLDATALOAD offset analysis does not
work for these contracts. Modern Solidity with optimization does not emit
standard PUSHN+CALLDATALOAD patterns at predictable bytecode positions.

The module uses the global-feature fallback: a single bytecode window (the
function body starting at bytecode_offset) is analyzed once, and those features
are replicated for each parameter position with the parameter count as
additional signal.

**Key functions:**
- `extract_global_features(bytecode_hex, bytecode_offset)` — Extracts 7-dim global SigRec vector from function body
- `extract_per_position_features(bytecode_hex, bytecode_offset, num_params)` — Returns global features replicated per parameter
- `build_feature_matrix(...)` — Produces `[num_params × 8]` feature matrix (7 SigRec features + 1 arity)

**Usage:**
```python
from abifusion.features.per_position_features import build_feature_matrix

feature_matrix = build_feature_matrix(bytecode_hex, bytecode_offset, num_params)
# feature_matrix[i] = [f1, f2, ..., f7, num_params] for parameter i
```

#### 4. `abifusion/ml_reconstructor.py` (Step 4 stub)
MLReconstructor stub with expected model interface.

**Interface:**
```python
from abifusion.ml_reconstructor import MLReconstructor

ml = MLReconstructor.get_instance()
ml.load_model("path/to/model.pt")  # TODO: train model first
result = ml.predict(bytecode)
# result = {"source": "ml_stub", "name": None, "name_top3": [], "types": [], ...}
```

**Expected model architecture (for Step 3):**
- Bytecode encoder: 1D CNN over tokenized bytecode bytes
- Per-position features: `[num_params × 8]` matrix concatenated with bytecode embeddings
- Two heads: function name (Top-3 over ~2000-name vocabulary) + parameter types (4-class: bytes-like, uint-like, address, other)

**Current status:** `predict()` returns stub predictions. `load_model()` loads a PyTorch checkpoint but inference is NotImplementedError until Step 3 builds the training pipeline.

#### 5. Updated `fusion.py` with Tier 3 ML fallback
Added ML model as third-tier fallback in `FusionReconstructor.reconstruct()`.

**Tier order:**
1. 4byte/openchain signature
2. Known-selector table (Step 0 expansion)
3. ML model prediction (stubbed)
4. `function_{selector}` fallback

## What Remains for Step 3 (Training Pipeline)

The training pipeline is **NOT** built in this iteration. The following are TODO:

1. **Training data preparation** — Extract per-position features from 500 contracts in `data/contracts.parquet` using `build_feature_matrix()`
2. **Model training** — Train a PyTorch model (1D CNN bytecode encoder + two classification heads) on 5,667 training functions
3. **Test evaluation** — Evaluate on 57 unique-selector functions (not the full dataset)
4. **Threshold check** — >60% Top-3 function name accuracy required for integration

**The critical empirical unknown:** Whether a model trained on known-selector functions can generalize to the 57 unique selectors never seen during training.

## Running the Verification

```bash
# Step 0: Expand known-selector table
python scripts/eval/build_known_selector_table.py

# Step 2 Phase 0: Verify CALLDATALOAD offset analysis
python scripts/eval/verify_calldataload_offset.py

# Verify the updated fusion still works
python -c "
from abifusion.fusion import FusionReconstructor
fr = FusionReconstructor()
result = fr.reconstruct('0x608060405234801561001057600080fd5b50600436106100935760003560e01c8063313ce56711610066578063313ce5671461013457806370a082311461015257806395d89b4114610182578063a9059cbb146101a0578063dd62ed3e146101d057610093565b806306fdde0314610098578063095ea7b3146100b657806318160ddd146100e657806323b872dd14610104575b600080fd5b6100a0610200565b6040516100ad9190610aaa565b60405180910390f35b6100d060048036038101906100cb9190610b65565b610292565b6040516100dd9190610bc0565b60405180910390f35b6100ee6102b5565b6040516100fb9190610bea565b60405180910390f35b61011e60048036038101906101199190610c05565b6102bf565b60405161012b9190610bc0565b60405180910390f35b61013c6102ee565b6040516101499190610c74565b60405180910390f35b61016c60048036038101906101679190610c8f565b6102f7565b6040516101799190610bea565b60405180910390f35b61018a61033f565b6040516101979190610aaa565b60405180910390f35b6101ba60048036038101906101b59190610b65565b6103d1565b6040516101c79190610bc0565b6040516101f79190610bea565b60606003805461020f90610d2b565b80601f016020809104026020016040519081016040528092919081815260200182805461023b90610d2b565b80156102885780601f1061025d57610100808354040283529160200191610288565b820191906000526020600020905b81548152906001019060200180831161026b57829003601f168201915b5050505050905090565b60008061029d61047b565b90506102aa818585610483565b600191505092915050565b6000600254905090565b6000806102ca61047b565b90506102d7858285610495565b6102e285858561052a565b60019150509392505050565b60006012905090565b60008060008373ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff168152602001908152602001600020549050919050565b60606004805461034e90610d2b565b80601f016020809104026020016040519081016040528092919081815260200182805461037a90610d2b565b80156103c75780601f1061039c576101008083540402835291602001916103c7565b820191906000526020600020905b8154815290600101906020018083116103aa57829003601f168201915b5050505050905090565b6000806103dc61047b565b90506103e981858561052a565b600191505092915050565b6000600160008473ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff16815260200190815260200160002060008373ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff16815260200190815260200160002054905092915050565b600033905090565b610490838383600161061e565b505050565b60006104a184846103f4565b90507fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff8110156105245781811015610514578281836040517ffb8f41b200000000000000000000000000000000000000000000000000000000815260040161050b93929190610d6b565b60405180910390fd5b6105238484848403600061061e565b5b50505050565b600073ffffffffffffffffffffffffffffffffffffffff168373ffffffffffffffffffffffffffffffffffffffff160361059c5760006040517f96c6fd1e0000000000000000000000000000000000000000000000000000000081526004016105939190610da2565b60405180910390fd5b600073ffffffffffffffffffffffffffffffffffffffff168273ffffffffffffffffffffffffffffffffffffffff160361060e5760006040517fec442f050000000000000000000000000000000000000000000000000000000081526004016106059190610da2565b60405180910390fd5b6106198383836107f5565b505050565b600073ffffffffffffffffffffffffffffffffffffffff168473ffffffffffffffffffffffffffffffffffffffff16036106905760006040517fe602df050000000000000000000000000000000000000000000000000000000081526004016106879190610da2565b60405180910390fd5b600073ffffffffffffffffffffffffffffffffffffffff168373ffffffffffffffffffffffffffffffffffffffff16036107025760006040517f94280d620000000000000000000000000000000000000000000000000000000081526004016106f99190610da2565b60405180910390fd5b81600160008673ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff16815260200190815260200160002060008573ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff1681526020019081526020016000208190555080156107ef578273ffffffffffffffffffffffffffffffffffffffff168473ffffffffffffffffffffffffffffffffffffffff167f8c5be1e5ebec7d5bd14f71427d1e84f3dd0314c0f7b2291e5b200ac8c7c3b925846040516107e69190610bea565b60405180910390a35b50505050565b600073ffffffffffffffffffffffffffffffffffffffff168373ffffffffffffffffffffffffffffffffffffffff160361084757806002600082825461083b9190610dec565b9250508190555061091a565b60008060008573ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff168152602001908152602001600020549050818110156108d3578381836040517fe450d38c0000000000000000000000000000000000000000000000000000000081526004016108ca93929190610d6b565b60405180910390fd5b8181036000808673ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff16815260200190815260200160002081905550505b600073ffffffffffffffffffffffffffffffffffffffff168273ffffffffffffffffffffffffffffffffffffffff160361096357806002600082825403925050819055506109b0565b806000808473ffffffffffffffffffffffffffffffffffffffff1673ffffffffffffffffffffffffffffffffffffffff168152602001908152602001600020600082825401925050819055505b8173ffffffffffffffffffffffffffffffffffffffff168373ffffffffffffffffffffffffffffffffffffffff167fddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef83604051610a0d9190610bea565b60405180910390a3505050565b600081519050919050565b600082825260208201905092915050565b60005b83811015610a54578082015181840152602081019050610a39565b60008484015250505050565b6000601f19601f8301169050919050565b6000610a7c82610a1a565b610a868185610a25565b9350610a96818560208601610a36565b610a9f81610a60565b840191505092915050565b60006020820190508181036000830152610ac48184610a71565b905092915050565b600080fd5b600073ffffffffffffffffffffffffffffffffffffffff82169050919050565b6000610afc82610ad1565b9050919050565b610b0c81610af1565b8114610b1757600080fd5b50565b600081359050610b2981610b03565b92915050565b6000819050919050565b610b4281610b2f565b8114610b4d57600080fd5b50565b600081359050610b5f81610b39565b92915050565b60008060408385031215610b7c57610b7b610acc565b5b6000610b8a85828601610b1a565b9250506020610b9b85828601610b50565b9150509250929050565b60008115159050919050565b610bba81610ba5565b82525050565b6000602082019050610bd56000830184610bb1565b92915050565b610be481610b2f565b82525050565b6000602082019050610bff6000830184610bdb565b92915050565b600080600060608486031215610c1e57610c1d610acc565b5b6000610c2c86828701610b1a565b9350506020610c3d86828701610b1a565b9250506040610c4e86828701610b50565b9150509250925092565b600060ff82169050919050565b610c6e81610c58565b82525050565b6000602082019050610c896000830184610c65565b92915050565b600060208284031215610ca557610ca4610acc565b5b6000610cb384828501610b1a565b91505092915050565b60008060408385031215610cd357610cd2610acc565b5b6000610ce185828601610b1a565b9250506020610cf285828601610b1a565b9150509250929050565b7f4e487b7100000000000000000000000000000000000000000000000000000000600052602260045260246000fd5b60006002820490506001821680610d4357607f821691505b602082108103610d5657610d55610cfc565b5b50919050565b610d6581610af1565b82525050565b6000606082019050610d806000830186610d5c565b610d8d6020830185610bdb565b610d9a6040830184610bdb565b949350505050565b6000602082019050610db76000830184610d5c565b92915050565b7f4e487b7100000000000000000000000000000000000000000000000000000000600052601160045260246000fd5b6000610df782610b2f565b9150610e0283610b2f565b9250828201905080821115610e1a57610e19610dbd565b5b9291505056fea2646970667358221220bd8441f759c9541a7ca904a1e6605aa88b22bd920a603bd4d05edc3176b9b44864736f6c634300081e0033')
print(f'Functions: {result[\"metadata\"][\"function_count\"]}')
for f in result['functions'][:5]:
    print(f'  {f[\"selector\"]}: {f[\"name\"]} ({f[\"source\"]})')
"
```
