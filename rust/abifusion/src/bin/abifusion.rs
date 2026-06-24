//! `abifusion` CLI — reconstruct a Solidity ABI from EVM bytecode.
//!
//! Usage:
//!   abifusion <bytecode-hex>
//!   abifusion --no-sigs <bytecode-hex>   # offline mode, no HTTP lookups
//!   abifusion --file contract.hex
//!
//! Set `ABIFUSION_NO_SIGS=1` as an env-var alternative to `--no-sigs`.

use std::env;
use std::io::Read;
use std::path::PathBuf;

use abifusion::{reconstruct, reconstruct_with, EvmoleAnalyzer, SignatureProvider};

struct NoSignatures;
impl SignatureProvider for NoSignatures {
    fn candidates(&self, _selector: &str) -> Vec<(String, Vec<String>)> {
        Vec::new()
    }
}

fn main() {
    let mut args = env::args().skip(1).collect::<Vec<_>>();
    let mut no_sigs = env::var("ABIFUSION_NO_SIGS").is_ok();
    let mut file: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--no-sigs" => { no_sigs = true; args.remove(i); }
            "--file" => {
                args.remove(i);
                if i >= args.len() {
                    eprintln!("error: --file requires a path");
                    std::process::exit(2);
                }
                file = Some(PathBuf::from(&args[i]));
                args.remove(i);
            }
            _ => i += 1,
        }
    }

    let bytecode = if let Some(f) = file {
        let mut s = String::new();
        std::fs::File::open(&f)
            .unwrap_or_else(|e| { eprintln!("error: cannot open {}: {}", f.display(), e); std::process::exit(1); })
            .read_to_string(&mut s)
            .unwrap_or_else(|e| { eprintln!("error: cannot read {}: {}", f.display(), e); std::process::exit(1); });
        s.trim().to_string()
    } else if let Some(b) = args.first() {
        b.clone()
    } else {
        eprintln!("usage: abifusion [--no-sigs] [--file <path>] <bytecode-hex>");
        std::process::exit(2);
    };

    let abi = if no_sigs {
        reconstruct_with(&bytecode, &EvmoleAnalyzer, &NoSignatures)
    } else {
        reconstruct(&bytecode)
    };

    println!("{}", serde_json::to_string_pretty(&abi).expect("serialize abi"));
}
