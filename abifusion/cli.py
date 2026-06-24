#!/usr/bin/env python3
"""CLI for abifusion."""

import argparse
import json
import sys
from pathlib import Path


def _apply_confidence_filter(result: dict, min_confidence: str) -> dict:
    ordering = {"low": 0, "medium": 1, "high": 2}
    threshold = ordering[min_confidence]
    original_count = len(result["functions"])
    filtered = [
        f for f in result["functions"]
        if ordering.get(f.get("confidence", "low"), 0) >= threshold
    ]
    result = dict(result)
    result["functions"] = filtered
    result["metadata"] = dict(result.get("metadata", {}))
    result["metadata"]["function_count"] = len(filtered)
    result["metadata"]["filtered_function_count"] = original_count - len(filtered)
    result["metadata"]["min_confidence"] = min_confidence
    return result


def cmd_fusion(args) -> int:
    """Build an ABI by fusing openchain/4byte signatures with evmole.

    Highest-accuracy path (~98.5% held-out exact parameter types). Requires
    network for signature lookups (cached).
    """
    import os
    from abifusion.fusion import ABIFusion

    if args.address:
        rpc_url = os.environ.get("ETH_RPC_URL")
        if not rpc_url:
            print("Error: ETH_RPC_URL environment variable required when using --address", file=sys.stderr)
            return 1
        try:
            import requests
            resp = requests.post(
                rpc_url,
                json={
                    "jsonrpc": "2.0",
                    "method": "eth_getCode",
                    "params": [args.address, "latest"],
                    "id": 1,
                },
                timeout=30,
            )
            resp.raise_for_status()
            result_json = resp.json()
            if result_json.get("error"):
                print(f"RPC error: {result_json['error']}", file=sys.stderr)
                return 1
            bytecode = result_json["result"]
            if bytecode in ("0x", "0x0", ""):
                print(f"No code at address {args.address}", file=sys.stderr)
                return 1
            bytecode = bytecode[2:] if bytecode.startswith("0x") else bytecode
        except Exception as e:
            print(f"Failed to fetch bytecode: {e}", file=sys.stderr)
            return 1
    elif args.bytecode_file:
        with open(args.bytecode_file, "r") as f:
            bytecode = f.read().strip()
    else:
        bytecode = args.bytecode

    result = ABIFusion().reconstruct(bytecode)

    output = result
    if args.min_confidence != "low":
        output = _apply_confidence_filter(output, args.min_confidence)

    if args.output_format == "abi":
        output = [
            {"type": f.get("type", "function"), "name": f["name"], "inputs": f["inputs"]}
            for f in output["functions"]
        ]

    if args.output:
        args.output.write_text(json.dumps(output, indent=2))
    else:
        print(json.dumps(output, indent=2))
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="abifusion - Build Solidity ABIs from EVM bytecode by fusing signature databases with evmole"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    fusion_parser = subparsers.add_parser(
        "fusion",
        help="Fuse openchain/4byte signatures with evmole (recommended, ~98.5%% held-out)",
    )
    fusion_parser.add_argument("--bytecode", help="Bytecode as hex string")
    fusion_parser.add_argument("--bytecode-file", help="File containing bytecode")
    fusion_parser.add_argument(
        "--address",
        help="Contract address to recover (requires ETH_RPC_URL env var)",
    )
    fusion_parser.add_argument(
        "--output-format",
        choices=["annotated", "abi"],
        default="annotated",
        help="Output format: 'annotated' includes source/confidence/candidates (default), 'abi' is plain Solidity ABI array",
    )
    fusion_parser.add_argument(
        "--min-confidence",
        choices=["low", "medium", "high"],
        default="low",
        help="Minimum confidence level to include",
    )
    fusion_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file (stdout if not specified)",
    )

    args = parser.parse_args()

    handlers = {
        "fusion": cmd_fusion,
    }
    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 1
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
