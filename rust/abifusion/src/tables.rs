//! Embedded curated tables — single source of truth shared with Python.
//!
//! The JSON lives at the repo-root `data/` directory (validated by
//! `schema/known-*.schema.json`) and is compiled in via `include_str!`.

use std::collections::HashMap;

use abifusion_core::{KnownInterface, KnownSelector};
use once_cell::sync::Lazy;

static KNOWN_SELECTORS: Lazy<HashMap<String, KnownSelector>> = Lazy::new(|| {
    let raw = include_str!("../../../data/known_selector_signatures.json");
    serde_json::from_str(raw).expect("parse data/known_selector_signatures.json")
});

static KNOWN_INTERFACES: Lazy<Vec<KnownInterface>> = Lazy::new(|| {
    let raw = include_str!("../../../data/known_interface_sets.json");
    serde_json::from_str(raw).expect("parse data/known_interface_sets.json")
});

pub fn known_selectors() -> &'static HashMap<String, KnownSelector> {
    &KNOWN_SELECTORS
}

pub fn known_interfaces() -> &'static [KnownInterface] {
    &KNOWN_INTERFACES
}
