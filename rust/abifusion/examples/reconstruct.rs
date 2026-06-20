//! Reconstruct an ABI from bytecode hex and print it as JSON.
//!
//! Usage: cargo run -p abifusion --example reconstruct -- <bytecode-hex>
//!
//! Set ABIFUSION_NO_SIGS=1 to skip signature-DB lookups (deterministic, offline)
//! — useful for parity testing the evmole adapter + fusion tiers against Python.

use std::env;

use abifusion::{reconstruct, reconstruct_with, EvmoleAnalyzer, SignatureProvider};

struct NoSignatures;
impl SignatureProvider for NoSignatures {
    fn candidates(&self, _selector: &str) -> Vec<(String, Vec<String>)> {
        Vec::new()
    }
}

fn main() {
    let bytecode = match env::args().nth(1) {
        Some(b) => b,
        None => {
            eprintln!("usage: reconstruct <bytecode-hex>");
            std::process::exit(2);
        }
    };
    let abi = if env::var("ABIFUSION_NO_SIGS").is_ok() {
        reconstruct_with(&bytecode, &EvmoleAnalyzer, &NoSignatures)
    } else {
        reconstruct(&bytecode)
    };
    println!("{}", serde_json::to_string_pretty(&abi).expect("serialize abi"));
}
