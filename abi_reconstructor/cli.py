#!/usr/bin/env python3
"""CLI for ABI Reconstructor."""

import argparse
import json
import sys

from abi_reconstructor import ABIReconstructor, FourByteDatabase
from abi_reconstructor.reconstructor import BytecodeParser


def cmd_reconstruct(args) -> int:
    """Reconstruct ABI from bytecode."""
    if args.bytecode_file:
        with open(args.bytecode_file, "r") as f:
            bytecode = f.read().strip()
    else:
        bytecode = args.bytecode

    reconstructor = ABIReconstructor(
        db_path=args.db,
        model_path=args.model_path,
        use_cuda=False,
    )

    if args.selector:
        result = reconstructor.reconstruct_function(bytecode, args.selector, use_ml=args.use_ml)
        print(json.dumps(result, indent=2))
    else:
        result = reconstructor.reconstruct_abi(bytecode, max_selectors=args.max_selectors, use_ml=args.use_ml)
        print(reconstructor.to_json(result))

    reconstructor.close()
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

    reconstruct_parser = subparsers.add_parser("reconstruct", help="Reconstruct ABI from bytecode")
    reconstruct_parser.add_argument("--bytecode", help="Bytecode as hex string")
    reconstruct_parser.add_argument("--bytecode-file", help="File containing bytecode")
    reconstruct_parser.add_argument("--selector", help="Specific selector to reconstruct (8 hex chars)")
    reconstruct_parser.add_argument("--max-selectors", type=int, default=50, help="Maximum selectors to extract")
    reconstruct_parser.add_argument("--use-ml", action="store_true", help="Use ML-based parameter inference")
    reconstruct_parser.add_argument("--model-path", help="Path to ML model checkpoints")

    extract_parser = subparsers.add_parser("extract", help="Extract selectors from bytecode")
    extract_parser.add_argument("--bytecode", help="Bytecode as hex string")
    extract_parser.add_argument("--bytecode-file", help="File containing bytecode")

    populate_parser = subparsers.add_parser("populate", help="Populate database from API")
    populate_parser.add_argument("--selectors", help="Comma-separated list of selectors to fetch, or omit for full pagination")

    subparsers.add_parser("stats", help="Show database statistics")

    export_parser = subparsers.add_parser("export", help="Export database to JSON")
    export_parser.add_argument("--output", required=True, help="Output JSON file path")

    args = parser.parse_args()

    if args.command == "reconstruct":
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
