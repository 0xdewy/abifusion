//! SignatureProvider port: openchain (preferred) then 4byte.directory.
//!
//! Mirrors `abifusion/utils/signature_lookup.py` + `ABIFusion._candidates`.

use abifusion_core::parse_signature;

pub trait SignatureProvider {
    /// Candidate `(name, types)` signatures for a selector, in API order.
    fn candidates(&self, selector: &str) -> Vec<(String, Vec<String>)>;
}

const OPENCHAIN_API: &str = "https://api.openchain.xyz/signature-database/v1/lookup";
const FOURBYTE_API: &str = "https://www.4byte.directory/api/v1/signatures/";

#[derive(Default)]
pub struct HttpSignatureProvider;

impl SignatureProvider for HttpSignatureProvider {
    fn candidates(&self, selector: &str) -> Vec<(String, Vec<String>)> {
        let mut texts = lookup_openchain(selector);
        if texts.is_empty() {
            texts = lookup_4byte(selector);
        }
        texts.iter().filter_map(|t| parse_signature(t)).collect()
    }
}

fn fetch_json(req: ureq::Request) -> Option<serde_json::Value> {
    req.call().ok()?.into_json().ok()
}

fn lookup_openchain(selector: &str) -> Vec<String> {
    let func = format!("0x{selector}");
    let v = match fetch_json(
        ureq::get(OPENCHAIN_API)
            .query("function", &func)
            .query("filter", "true"),
    ) {
        Some(v) => v,
        None => return Vec::new(),
    };
    let mut out = Vec::new();
    if let Some(arr) = v
        .get("result")
        .and_then(|x| x.get("function"))
        .and_then(|x| x.get(&func))
        .and_then(|x| x.as_array())
    {
        for item in arr {
            if let Some(name) = item.get("name").and_then(|n| n.as_str()) {
                if !name.is_empty() {
                    out.push(name.to_string());
                }
            }
        }
    }
    out
}

fn lookup_4byte(selector: &str) -> Vec<String> {
    let hex_sig = format!("0x{selector}");
    let v = match fetch_json(ureq::get(FOURBYTE_API).query("hex_signature", &hex_sig)) {
        Some(v) => v,
        None => return Vec::new(),
    };
    let mut out = Vec::new();
    if let Some(arr) = v.get("results").and_then(|x| x.as_array()) {
        for item in arr {
            if let Some(ts) = item.get("text_signature").and_then(|n| n.as_str()) {
                out.push(ts.to_string());
            }
        }
    }
    out
}
