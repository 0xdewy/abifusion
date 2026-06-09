#!/usr/bin/env python3
"""CLI for ABI Reconstructor."""

import argparse
import json
import sys
from pathlib import Path

from abi_reconstructor import ABIReconstructor, FourByteDatabase
from abi_reconstructor.reconstructor import BytecodeParser


def cmd_reconstruct(args) -> int:
    """Reconstruct ABI from bytecode."""
    if args.bytecode_file:
        with open(args.bytecode_file, "r") as f:
            bytecode = f.read().strip()
    else:
        bytecode = args.bytecode

    reconstructor = ABIReconstructor(db_path=args.db)

    if args.selector:
        result = reconstructor.reconstruct_function(bytecode, args.selector)
        print(json.dumps(result, indent=2))
    else:
        result = reconstructor.reconstruct_abi(bytecode, max_selectors=args.max_selectors)
        print(reconstructor.to_json(result))

    reconstructor.close()
    return 0


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
    """Reconstruct an ABI by fusing openchain/4byte signatures with evmole.

    Highest-accuracy path (~98.5% held-out exact parameter types). Requires
    network for signature lookups (cached).
    """
    import os
    from abi_reconstructor.fusion import FusionReconstructor

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

    result = FusionReconstructor().reconstruct(bytecode)

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


def cmd_extract_selectors(args) -> int:
    """Extract function selectors from bytecode."""
    if args.bytecode_file:
        with open(args.bytecode_file, "r") as f:
            bytecode = f.read().strip()
    else:
        bytecode = args.bytecode

    parser = BytecodeParser()
    selectors = parser.extract_selectors(bytecode)

    print(f"Found {len(selectors)} selectors:")
    for sel in selectors:
        print(f"  0x{sel.selector} at pos {sel.position} (confidence: {sel.confidence:.2f}, source: {sel.source})")

    return 0


def cmd_populate(args) -> int:
    """Populate database from 4byte.directory API.

    When called without selectors argument, paginates through all signatures
    in 4byte.directory and adds them to the local database.
    """
    db = FourByteDatabase(db_path=args.db)
    print("Populating database from 4byte.directory API (this may take a while)...")

    selectors = None if not args.selectors else [s.strip() for s in args.selectors.split(",")]
    if selectors:
        print(f"Fetching {len(selectors)} specific selectors...")
    else:
        print("Fetching all signatures via pagination...")

    results = db.populate_from_api(selectors)

    print(f"Population complete: {results['success']} success, {results['failed']} failed")
    if results.get("fetched", 0) > 0:
        print(f"Total new signatures fetched: {results['fetched']}")
    return 0


def cmd_stats(args) -> int:
    """Show database statistics."""
    db = FourByteDatabase(db_path=args.db)
    stats = db.get_stats()

    print("4byte Database Statistics:")
    print(f"  Database path: {stats['db_path']}")
    print(f"  Total signatures: {stats['total_signatures']}")
    print(f"  Unique selectors: {stats['unique_selectors']}")
    print(f"  Standard signatures: {stats['standard_signatures']}")

    return 0


def cmd_export(args) -> int:
    """Export database to JSON."""
    db = FourByteDatabase(db_path=args.db)
    data = db.export_json()

    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Exported {len(data)} selectors to {args.output}")
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ABI Reconstructor - Reconstruct ABIs from EVM bytecode"
    )

    parser.add_argument(
        "--db",
        default="./cache/4byte.db",
        help="Path to 4byte SQLite database"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    fusion_parser = subparsers.add_parser(
        "fusion",
        help="Reconstruct ABI by fusing openchain/4byte signatures with evmole (recommended, ~98.5 pct held-out)",
    )
    fusion_parser.add_argument("--bytecode", help="Bytecode as hex string")
    fusion_parser.add_argument("--bytecode-file", help="File containing bytecode")
    fusion_parser.add_argument(
        "--address",
        help="Contract address to reconstruct (requires ETH_RPC_URL env var)",
    )
    fusion_parser.add_argument(
        "--chain-id",
        type=int,
        default=1,
        help="Chain ID for RPC lookup (default: 1, mainnet)",
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

    reconstruct_parser = subparsers.add_parser("reconstruct", help="Reconstruct ABI from bytecode (rule-based 4byte)")
    reconstruct_parser.add_argument("--bytecode", help="Bytecode as hex string")
    reconstruct_parser.add_argument("--bytecode-file", help="File containing bytecode")
    reconstruct_parser.add_argument("--selector", help="Specific selector to reconstruct (8 hex chars)")
    reconstruct_parser.add_argument("--max-selectors", type=int, default=50, help="Maximum selectors to extract")

    extract_parser = subparsers.add_parser("extract", help="Extract selectors from bytecode")
    extract_parser.add_argument("--bytecode", help="Bytecode as hex string")
    extract_parser.add_argument("--bytecode-file", help="File containing bytecode")

    populate_parser = subparsers.add_parser("populate", help="Populate database from API")
    populate_parser.add_argument("--selectors", help="Comma-separated list of selectors to fetch, or omit for full pagination")

    subparsers.add_parser("stats", help="Show database statistics")

    export_parser = subparsers.add_parser("export", help="Export database to JSON")
    export_parser.add_argument("--output", required=True, help="Output JSON file path")

    args = parser.parse_args()

    if args.command == "fusion":
        return cmd_fusion(args)
    elif args.command == "reconstruct":
        return cmd_reconstruct(args)
    elif args.command == "extract":
        return cmd_extract_selectors(args)
    elif args.command == "populate":
        return cmd_populate(args)
    elif args.command == "stats":
        return cmd_stats(args)
    elif args.command == "export":
        return cmd_export(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
