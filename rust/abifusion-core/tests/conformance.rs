//! Cross-language conformance: run the shared `tests/conformance/vectors.json`
//! (the same file the Python suite uses) through the Rust `fuse_abi` and assert
//! the output matches each vector's frozen `expected` value exactly.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use abifusion_core::{fuse_abi, KnownInterface, KnownSelector, MlPrediction};
use serde::Deserialize;

#[derive(Deserialize)]
struct VectorInput {
    evmole_types: HashMap<String, Option<Vec<String>>>,
    extra_selectors: HashSet<String>,
    candidates_by_selector: HashMap<String, Vec<(String, Vec<String>)>>,
    known_selectors: HashMap<String, KnownSelector>,
    known_interfaces: Vec<KnownInterface>,
    v4_before_swap_present: bool,
    v4_pool_manager_present: bool,
    ml_predictions: HashMap<String, MlPrediction>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    input: VectorInput,
    expected: serde_json::Value,
}

fn vectors_path() -> PathBuf {
    // rust/abifusion-core -> repo root -> tests/conformance/vectors.json
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/conformance/vectors.json")
}

#[test]
fn conformance_vectors_match() {
    let raw = std::fs::read_to_string(vectors_path())
        .expect("read tests/conformance/vectors.json");
    let cases: Vec<Case> = serde_json::from_str(&raw).expect("parse vectors.json");
    assert!(!cases.is_empty(), "no conformance vectors found");

    for case in &cases {
        let i = &case.input;
        let abi = fuse_abi(
            &i.evmole_types,
            &i.extra_selectors,
            &i.candidates_by_selector,
            &i.known_selectors,
            &i.known_interfaces,
            i.v4_before_swap_present,
            i.v4_pool_manager_present,
            &i.ml_predictions,
        );
        let got = serde_json::to_value(&abi).expect("serialize abi");
        assert_eq!(
            got, case.expected,
            "conformance vector `{}` diverged",
            case.name
        );
    }
}
