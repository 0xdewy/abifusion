//! Pure, I/O-free fusion core — the Rust port of `abifusion/core/fuse.py`.
//!
//! Given already-resolved inputs (evmole argument structure, signature-database
//! candidates, curated tables, optional ML predictions), [`fuse_abi`] performs
//! the deterministic tiered fusion that builds a Solidity ABI. It imports no
//! evmole, no HTTP, no filesystem — those live in the `abifusion` crate's
//! adapters. The shared conformance vectors (`tests/conformance/vectors.json`,
//! also exercised by the Python suite) lock this port to the Python spec.

use std::collections::{BTreeSet, HashMap, HashSet};

use serde::{Deserialize, Serialize};

/// Uniswap V4 hook callback selectors emitted by the post-pass.
pub const V4_HOOK_SELECTORS: [&str; 10] = [
    "bc29bafc", "468ead2c", "5cb32d10", "606e8192", "b6d4944a", "b772b8cc",
    "c13d1c69", "1f29cf9d", "d950bd74", "ebe1cdaf",
];
pub const TRIGGER_BEFORE_SWAP: &str = "575e24b4";
pub const TRIGGER_POOL_MANAGER: &str = "dc4c90d3";

// ---------------------------------------------------------------------------
// Output types (serialize to schema/abi-output.schema.json)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct Input {
    #[serde(rename = "type")]
    pub ty: String,
    pub name: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Candidate {
    pub name: String,
    pub types: Vec<String>,
    pub source: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Function {
    #[serde(rename = "type")]
    pub ty: String,
    pub name: String,
    pub selector: String,
    pub inputs: Vec<Input>,
    pub source: String,
    pub confidence: String,
    // Parity risk #1: interface-completion entries omit this key entirely;
    // other entries emit it as an array or `null`. `None` -> field omitted;
    // `Some(None)` -> `null`; `Some(Some(v))` -> array.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidates: Option<Option<Vec<Candidate>>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Metadata {
    pub function_count: usize,
    pub evmole_selectors: usize,
    pub extra_selectors: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct Abi {
    pub functions: Vec<Function>,
    pub metadata: Metadata,
}

// ---------------------------------------------------------------------------
// Table / prediction input types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct KnownSelector {
    pub name: String,
    pub types: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct KnownFunction {
    pub selector: String,
    pub name: String,
    pub types: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Triggers {
    pub selector_set: Vec<String>,
    #[serde(default = "default_min_matches")]
    pub min_matches: usize,
}

fn default_min_matches() -> usize {
    2
}

#[derive(Debug, Clone, Deserialize)]
pub struct KnownInterface {
    pub name: String,
    pub functions: Vec<KnownFunction>,
    pub triggers: Triggers,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct MlPrediction {
    pub family: Option<String>,
    pub family_top3: Vec<String>,
    pub types: Vec<String>,
}

// ---------------------------------------------------------------------------
// Pure helpers (ports of split_args / parse_signature / choose_candidate)
// ---------------------------------------------------------------------------

/// Split a Solidity argument list, respecting nested tuple parens.
pub fn split_args(arg_str: &str) -> Vec<String> {
    let s = arg_str.trim();
    if s.is_empty() {
        return Vec::new();
    }
    let mut out: Vec<String> = Vec::new();
    let mut depth: i32 = 0;
    let mut cur = String::new();
    for ch in s.chars() {
        match ch {
            '(' => {
                depth += 1;
                cur.push(ch);
            }
            ')' => {
                depth -= 1;
                cur.push(ch);
            }
            ',' if depth == 0 => {
                out.push(cur.trim().to_string());
                cur.clear();
            }
            _ => cur.push(ch),
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

/// `"transfer(address,uint256)"` -> `("transfer", ["address","uint256"])`.
pub fn parse_signature(text_sig: &str) -> Option<(String, Vec<String>)> {
    let i = text_sig.find('(')?;
    let name = text_sig[..i].to_string();
    // Mirror Python `text_sig[i+1 : text_sig.rfind(")")]`, where rfind == -1
    // yields a slice that drops the final character.
    let end = match text_sig.rfind(')') {
        Some(r) => r,
        None => text_sig.len().saturating_sub(1),
    };
    let inner = if i < end { &text_sig[i + 1..end] } else { "" };
    Some((name, split_args(inner)))
}

/// Pick the candidate that best matches evmole's type structure.
///
/// Returns `(name, types, confidence)`; `None` if there are no candidates.
/// Parity risk #2: on a tie in the medium tier we keep the *first* maximal
/// candidate (Python `max` semantics), not the last.
pub fn choose_candidate(
    candidates: &[(String, Vec<String>)],
    evmole_types: Option<&Vec<String>>,
) -> Option<(String, Vec<String>, String)> {
    if candidates.is_empty() {
        return None;
    }
    if let Some(ev) = evmole_types {
        for (name, types) in candidates {
            if types == ev {
                return Some((name.clone(), types.clone(), "high".to_string()));
            }
        }
        let mut best: Option<&(String, Vec<String>)> = None;
        let mut best_score: i64 = -1;
        for c in candidates {
            if c.1.len() == ev.len() {
                let score: i64 = c
                    .1
                    .iter()
                    .zip(ev.iter())
                    .filter(|(a, b)| a == b)
                    .count() as i64;
                if best.is_none() || score > best_score {
                    best = Some(c);
                    best_score = score;
                }
            }
        }
        if let Some(b) = best {
            return Some((b.0.clone(), b.1.clone(), "medium".to_string()));
        }
    }
    let first = &candidates[0];
    Some((first.0.clone(), first.1.clone(), "low".to_string()))
}

// ---------------------------------------------------------------------------
// fuse_abi
// ---------------------------------------------------------------------------

fn inputs_from_types(types: &[String]) -> Vec<Input> {
    types
        .iter()
        .map(|t| Input {
            ty: t.clone(),
            name: String::new(),
        })
        .collect()
}

/// Fuse evmole structure + signature-DB candidates into a Solidity ABI.
/// Pure port of `abifusion.core.fuse.fuse_abi`.
#[allow(clippy::too_many_arguments)]
pub fn fuse_abi(
    evmole_types: &HashMap<String, Option<Vec<String>>>,
    extra_selectors: &HashSet<String>,
    candidates_by_selector: &HashMap<String, Vec<(String, Vec<String>)>>,
    known_selectors: &HashMap<String, KnownSelector>,
    known_interfaces: &[KnownInterface],
    v4_before_swap_present: bool,
    v4_pool_manager_present: bool,
    ml_predictions: &HashMap<String, MlPrediction>,
) -> Abi {
    let mut functions: Vec<Function> = Vec::new();

    // sorted(set(evmole_types) | extra_selectors), lexicographic on hex strings.
    let mut all: BTreeSet<String> = BTreeSet::new();
    all.extend(evmole_types.keys().cloned());
    all.extend(extra_selectors.iter().cloned());

    let empty_candidates: Vec<(String, Vec<String>)> = Vec::new();

    for selector in &all {
        // Parity risk #3: missing key OR a stored `None` both mean "no signal".
        let ev: Option<&Vec<String>> = evmole_types.get(selector).and_then(|o| o.as_ref());
        let candidates = candidates_by_selector
            .get(selector)
            .unwrap_or(&empty_candidates);
        let chosen = choose_candidate(candidates, ev);

        let name: String;
        let types: Vec<String>;
        let source: String;
        let confidence: String;
        let func_candidates: Option<Option<Vec<Candidate>>>;

        if let Some((n, t, conf)) = chosen {
            name = n;
            types = t;
            source = if ev.is_some() {
                "signature+evmole".to_string()
            } else {
                "signature".to_string()
            };
            confidence = conf;
            let fc: Vec<Candidate> = candidates
                .iter()
                .map(|c| Candidate {
                    name: c.0.clone(),
                    types: c.1.clone(),
                    source: "4byte".to_string(),
                })
                .collect();
            func_candidates = Some(Some(fc));
        } else if let Some(ev_types) = ev {
            name = format!("function_{selector}");
            types = ev_types.clone();
            source = "evmole".to_string();
            confidence = "high".to_string();
            func_candidates = Some(None);
        } else if let Some(known) = known_selectors.get(selector) {
            name = known.name.clone();
            types = known.types.clone();
            source = "known-selector-table".to_string();
            confidence = "high".to_string();
            func_candidates = Some(None);
        } else {
            // Tier 3: ML prediction (data-in; the Rust adapter never populates it).
            let ml = ml_predictions.get(selector);
            let mut resolved = false;
            if let Some(ml) = ml {
                if let Some(fam) = &ml.family {
                    name = fam.clone();
                    types = ml.types.clone();
                    source = "ml".to_string();
                    confidence = "low".to_string();
                    func_candidates = Some(None);
                    resolved = true;
                } else if !ml.family_top3.is_empty() {
                    name = ml.family_top3[0].clone();
                    types = ml.types.clone();
                    source = "ml".to_string();
                    confidence = "low".to_string();
                    func_candidates = Some(None);
                    resolved = true;
                } else {
                    name = format!("function_{selector}");
                    types = Vec::new();
                    source = "selector".to_string();
                    confidence = "low".to_string();
                    func_candidates = Some(None);
                }
            } else {
                name = format!("function_{selector}");
                types = Vec::new();
                source = "selector".to_string();
                confidence = "low".to_string();
                func_candidates = Some(None);
            }
            let _ = resolved;
        }

        functions.push(Function {
            ty: "function".to_string(),
            name,
            selector: selector.clone(),
            inputs: inputs_from_types(&types),
            source,
            confidence,
            candidates: func_candidates,
        });
    }

    // Interface completion.
    let bytecode_selectors: HashSet<String> = all.iter().cloned().collect();
    let mut results_by_selector: HashSet<String> =
        functions.iter().map(|f| f.selector.clone()).collect();
    let completed = complete_interfaces(
        &bytecode_selectors,
        &mut results_by_selector,
        known_interfaces,
    );
    functions.extend(completed);

    // V4 hook post-pass.
    let existing: HashSet<String> = functions.iter().map(|f| f.selector.clone()).collect();
    let post = emit_undiscovered_v4_hooks(
        &existing,
        known_selectors,
        v4_before_swap_present,
        v4_pool_manager_present,
    );
    functions.extend(post);

    let evmole_selectors = evmole_types.len();
    let extra_count = extra_selectors
        .iter()
        .filter(|s| !evmole_types.contains_key(*s))
        .count();
    let function_count = functions.len();

    Abi {
        functions,
        metadata: Metadata {
            function_count,
            evmole_selectors,
            extra_selectors: extra_count,
        },
    }
}

fn complete_interfaces(
    bytecode_selectors: &HashSet<String>,
    results_by_selector: &mut HashSet<String>,
    known_interfaces: &[KnownInterface],
) -> Vec<Function> {
    let mut completed: Vec<Function> = Vec::new();
    for iface in known_interfaces {
        let matching = iface
            .triggers
            .selector_set
            .iter()
            .filter(|s| bytecode_selectors.contains(*s))
            .count();
        if matching < iface.triggers.min_matches {
            continue;
        }
        for func in &iface.functions {
            if results_by_selector.contains(&func.selector) {
                continue;
            }
            completed.push(Function {
                ty: "function".to_string(),
                name: func.name.clone(),
                selector: func.selector.clone(),
                inputs: inputs_from_types(&func.types),
                source: "known-interface-completion".to_string(),
                confidence: "medium".to_string(),
                candidates: None, // omit the key entirely
            });
            results_by_selector.insert(func.selector.clone());
        }
    }
    completed
}

fn emit_undiscovered_v4_hooks(
    existing: &HashSet<String>,
    known_selectors: &HashMap<String, KnownSelector>,
    before_swap_present: bool,
    pool_manager_present: bool,
) -> Vec<Function> {
    // Mirrors Python's `if not (has_before_swap and not has_pool_manager)`.
    if !before_swap_present || pool_manager_present {
        return Vec::new();
    }
    let mut completed: Vec<Function> = Vec::new();
    for sel in V4_HOOK_SELECTORS {
        if existing.contains(sel) {
            continue;
        }
        let Some(known) = known_selectors.get(sel) else {
            continue;
        };
        completed.push(Function {
            ty: "function".to_string(),
            name: known.name.clone(),
            selector: sel.to_string(),
            inputs: inputs_from_types(&known.types),
            source: "known-interface-completion-post".to_string(),
            confidence: "medium".to_string(),
            candidates: Some(None), // -> null
        });
    }
    completed
}
