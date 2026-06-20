//! BytecodeAnalyzer port: evmole static analysis + PUSH4+EQ fallback scan.
//!
//! Mirrors `abifusion/adapters.py::EvmoleBytecodeAnalyzer` and the selector
//! scan in `abifusion/reconstructor.py::BytecodeParser.extract_selectors`.

use std::collections::{HashMap, HashSet};

use abifusion_core::{TRIGGER_BEFORE_SWAP, TRIGGER_POOL_MANAGER};
use evmole::{contract_info, ContractInfoArgs};

const MIN_SELECTOR_VALUE: u32 = 0x0000_0100;

pub struct BytecodeAnalysis {
    pub evmole_types: HashMap<String, Option<Vec<String>>>,
    pub extra_selectors: HashSet<String>,
    pub v4_before_swap_present: bool,
    pub v4_pool_manager_present: bool,
}

pub trait BytecodeAnalyzer {
    fn analyze(&self, bytecode: &str) -> BytecodeAnalysis;
}

#[derive(Default)]
pub struct EvmoleAnalyzer;

impl BytecodeAnalyzer for EvmoleAnalyzer {
    fn analyze(&self, bytecode: &str) -> BytecodeAnalysis {
        let cleaned = clean_hex(bytecode);

        let mut evmole_types: HashMap<String, Option<Vec<String>>> = HashMap::new();
        if let Ok(code) = hex::decode(&cleaned) {
            if !code.is_empty() {
                let args = ContractInfoArgs::new(&code)
                    .with_selectors()
                    .with_arguments();
                let info = contract_info(args);
                if let Some(funcs) = info.functions {
                    for f in funcs {
                        let sel = hex::encode(f.selector);
                        // Python stores split_args(f.arguments or "") — always a
                        // tuple (possibly empty), never None.
                        let types: Vec<String> = match f.arguments {
                            Some(a) => a.iter().map(|t| t.sol_type_name().to_string()).collect(),
                            None => Vec::new(),
                        };
                        evmole_types.insert(sel, Some(types));
                    }
                }
            }
        }

        let lower = bytecode.to_lowercase();
        BytecodeAnalysis {
            extra_selectors: extract_selectors(&cleaned),
            v4_before_swap_present: lower.contains(TRIGGER_BEFORE_SWAP),
            v4_pool_manager_present: lower.contains(TRIGGER_POOL_MANAGER),
            evmole_types,
        }
    }
}

/// Strip a leading `0x`, drop non-hex chars, lowercase, drop a trailing
/// half-byte. Mirrors `BytecodeParser.clean_bytecode`.
fn clean_hex(bytecode: &str) -> String {
    let s = bytecode.strip_prefix("0x").unwrap_or(bytecode);
    let mut out: String = s
        .chars()
        .filter(|c| c.is_ascii_hexdigit())
        .map(|c| c.to_ascii_lowercase())
        .collect();
    if !out.len().is_multiple_of(2) {
        out.pop();
    }
    out
}

/// PUSH4 + EQ selector scan. Returns the lowercased selector set (the only thing
/// fusion needs; confidence/position from the Python version are irrelevant).
fn extract_selectors(cleaned: &str) -> HashSet<String> {
    let bc = cleaned.as_bytes();
    let n = bc.len();
    let mut seen: HashSet<String> = HashSet::new();
    if n < 10 {
        return seen;
    }

    let mut i = 0usize;
    while i < n - 10 {
        if &cleaned[i..i + 2] != "63" {
            i += 2;
            continue;
        }
        let selector = &cleaned[i + 2..i + 10];

        if seen.contains(selector)
            || selector == "00000000"
            || selector.bytes().all(|c| c == selector.as_bytes()[0])
        {
            i += 2;
            continue;
        }
        match u32::from_str_radix(selector, 16) {
            Ok(v) if v >= MIN_SELECTOR_VALUE => {}
            _ => {
                i += 2;
                continue;
            }
        }

        let search_end = std::cmp::min(n, i + 24);
        let mut eq_found = false;
        let mut j = i + 10;
        while j < search_end {
            if &cleaned[j..j + 2] == "14" {
                eq_found = true;
                break;
            }
            j += 2;
        }
        if eq_found {
            seen.insert(selector.to_string());
        }
        i += 2;
    }
    seen
}
