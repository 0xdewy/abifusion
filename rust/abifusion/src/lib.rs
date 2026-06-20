//! Adapters + orchestrator for the abifusion Rust port.
//!
//! Wires the bytecode analyzer (evmole) and signature provider (HTTP) into the
//! pure [`abifusion_core::fuse_abi`]. Mirrors `abifusion/fusion.py::ABIFusion`.
//! The ML tier is intentionally omitted (the model is PyTorch-only with no ONNX
//! export); `fuse_abi` simply receives empty predictions.

pub mod analyzer;
pub mod signatures;
pub mod tables;

use std::collections::{BTreeSet, HashMap};

use abifusion_core::{fuse_abi, Abi, MlPrediction};

pub use analyzer::{BytecodeAnalysis, BytecodeAnalyzer, EvmoleAnalyzer};
pub use signatures::{HttpSignatureProvider, SignatureProvider};

/// Reconstruct an ABI from bytecode using evmole + live signature databases.
pub fn reconstruct(bytecode: &str) -> Abi {
    reconstruct_with(bytecode, &EvmoleAnalyzer, &HttpSignatureProvider)
}

/// Reconstruct with injectable adapters (useful for offline tests).
pub fn reconstruct_with(
    bytecode: &str,
    analyzer: &dyn BytecodeAnalyzer,
    sig: &dyn SignatureProvider,
) -> Abi {
    let analysis = analyzer.analyze(bytecode);

    let mut all: BTreeSet<String> = BTreeSet::new();
    all.extend(analysis.evmole_types.keys().cloned());
    all.extend(analysis.extra_selectors.iter().cloned());

    let mut candidates_by_selector: HashMap<String, Vec<(String, Vec<String>)>> = HashMap::new();
    for sel in &all {
        candidates_by_selector.insert(sel.clone(), sig.candidates(sel));
    }

    let ml_predictions: HashMap<String, MlPrediction> = HashMap::new();

    fuse_abi(
        &analysis.evmole_types,
        &analysis.extra_selectors,
        &candidates_by_selector,
        tables::known_selectors(),
        tables::known_interfaces(),
        analysis.v4_before_swap_present,
        analysis.v4_pool_manager_present,
        &ml_predictions,
    )
}
