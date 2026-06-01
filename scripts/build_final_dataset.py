#!/usr/bin/env python3
"""Build final dataset from Sourcify tables using efficient pandas joins."""

import pandas as pd
import json
import os
from tqdm import tqdm
import argparse
import requests


def extract_abi_from_compiled(compiled_df):
    """Extract ABI from compiled contracts DataFrame."""
    print("Extracting ABIs from compiled contracts...")

    abi_map = {}

    for _, row in tqdm(
        compiled_df.iterrows(), total=len(compiled_df), desc="Extracting ABIs"
    ):
        try:
            runtime_hash = row.get("runtime_code_hash")
            if not isinstance(runtime_hash, bytes) or len(runtime_hash) != 32:
                continue

            artifacts = row.get("compilation_artifacts")
            if isinstance(artifacts, str):
                try:
                    artifacts_data = json.loads(artifacts)
                    if isinstance(artifacts_data, dict) and "abi" in artifacts_data:
                        abi = artifacts_data["abi"]
                        if isinstance(abi, list):
                            abi_map[runtime_hash] = abi
                except:
                    pass
        except:
            continue

    print(f"Extracted {len(abi_map)} ABIs")
    return abi_map


def download_additional_code_tables():
    """Download additional code tables if needed."""
    import requests
    import io

    code_tables = []

    # Load the first code table we already have
    code_path = "data/sourcify/code.parquet"
    if os.path.exists(code_path):
        code_df = pd.read_parquet(code_path)
        code_tables.append(code_df)
        print(f"  Loaded existing code table: {len(code_df):,} rows")

    # Try to download additional code tables
    base_url = "https://export.sourcify.dev/v2/code/"
    ranges = [
        ("code_100000_200000.parquet", "data/sourcify/code_100000_200000.parquet"),
        ("code_200000_300000.parquet", "data/sourcify/code_200000_300000.parquet"),
        ("code_300000_400000.parquet", "data/sourcify/code_300000_400000.parquet"),
    ]

    for url_suffix, local_path in ranges:
        url = base_url + url_suffix
        if not os.path.exists(local_path):
            print(f"  Downloading {url_suffix}...")
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    with open(local_path, "wb") as f:
                        f.write(response.content)
                    print(f"    Downloaded {len(response.content):,} bytes")
                else:
                    print(f"    Not available (status: {response.status_code})")
                    continue
            except Exception as e:
                print(f"    Error downloading: {e}")
                continue

        if os.path.exists(local_path):
            try:
                df = pd.read_parquet(local_path)
                code_tables.append(df)
                print(f"  Loaded {url_suffix}: {len(df):,} rows")
            except Exception as e:
                print(f"    Error loading {local_path}: {e}")

    if code_tables:
        return pd.concat(code_tables, ignore_index=True)
    else:
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(
        description="Build final dataset from Sourcify tables"
    )
    parser.add_argument(
        "--max-contracts",
        type=int,
        default=100000,
        help="Maximum number of contracts to include",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/sourcify_100k_final.parquet",
        help="Output file path",
    )
    parser.add_argument(
        "--test-mode", action="store_true", help="Test mode with small sample"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading additional code tables",
    )
    args = parser.parse_args()

    if args.test_mode:
        args.max_contracts = 100
        args.output = "data/sourcify_test.parquet"
        print("TEST MODE: Using small sample")

    print(f"Building dataset with up to {args.max_contracts} contracts")
    print(f"Output: {args.output}")

    # Load tables
    print("\nLoading tables...")

    # Load code tables (may download additional ones)
    if args.skip_download:
        code_df = pd.read_parquet("data/sourcify/code.parquet")
        print(f"  Code: {len(code_df):,} rows (single table)")
    else:
        code_df = download_additional_code_tables()
        print(f"  Code: {len(code_df):,} rows (combined tables)")

    deployments_df = pd.read_parquet("data/sourcify/contract_deployments.parquet")
    compiled_df = pd.read_parquet("data/sourcify/compiled_contracts.parquet")
    contracts_df = pd.read_parquet("data/sourcify/contracts.parquet")

    print(f"  Code: {len(code_df):,} rows")
    print(f"  Deployments: {len(deployments_df):,} rows")
    print(f"  Compiled: {len(compiled_df):,} rows")
    print(f"  Contracts: {len(contracts_df):,} rows")

    # Step 1: Join deployments with contracts
    print("\nStep 1: Joining deployments with contracts...")

    # Convert contract_id to string for proper joining
    deployments_df["contract_id_str"] = deployments_df["contract_id"].astype(str)
    contracts_df["id_str"] = contracts_df["id"].astype(str)

    # Join deployments with contracts
    deployments_with_contracts = pd.merge(
        deployments_df[["address", "contract_id_str", "chain_id"]],
        contracts_df[["id_str", "runtime_code_hash"]],
        left_on="contract_id_str",
        right_on="id_str",
        how="inner",
    )

    print(
        f"  Joined deployments with contracts: {len(deployments_with_contracts):,} rows"
    )

    # Step 2: Join with code table
    print("\nStep 2: Joining with code table...")

    # Convert runtime_code_hash to bytes for joining
    # Note: contracts.runtime_code_hash matches code.code_hash, not code.code_hash_keccak
    deployments_with_code = pd.merge(
        deployments_with_contracts,
        code_df[["code_hash", "code"]],
        left_on="runtime_code_hash",
        right_on="code_hash",
        how="inner",
    )

    print(f"  Joined with code table: {len(deployments_with_code):,} rows")

    # Step 3: Extract ABIs
    print("\nStep 3: Extracting ABIs...")
    abi_map = extract_abi_from_compiled(compiled_df)

    # Step 4: Create final dataset
    print("\nStep 4: Creating final dataset...")

    contracts = []
    processed = 0

    for _, row in tqdm(
        deployments_with_code.iterrows(),
        total=min(len(deployments_with_code), args.max_contracts * 2),
        desc="Processing",
    ):
        if processed >= args.max_contracts:
            break

        try:
            # Get address
            address_bytes = row["address"]
            if not isinstance(address_bytes, bytes) or len(address_bytes) != 20:
                continue

            address = "0x" + address_bytes.hex()

            # Get bytecode
            code_bytes = row["code"]
            if not isinstance(code_bytes, bytes) or len(code_bytes) < 100:
                continue

            bytecode = "0x" + code_bytes.hex()

            # Get runtime hash for ABI lookup
            runtime_hash = row["runtime_code_hash"]

            # Get ABI if available
            abi = abi_map.get(runtime_hash, [])

            # Create contract record
            contract = {
                "address": address,
                "bytecode": bytecode,
                "bytecode_length": len(bytecode) - 2,  # Subtract "0x"
                "abi": abi,
                "has_abi": len(abi) > 0,
                "abi_length": len(abi),
                "chain_id": int(row["chain_id"])
                if not pd.isna(row["chain_id"])
                else None,
                "runtime_code_hash": runtime_hash.hex()
                if isinstance(runtime_hash, bytes)
                else None,
                "contract_id": row["contract_id_str"],
            }

            contracts.append(contract)
            processed += 1

        except Exception as e:
            continue

    print(f"\nCreated {len(contracts)} contract records")

    if contracts:
        # Create DataFrame
        contracts_df = pd.DataFrame(contracts)

        # Save to file
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        contracts_df.to_parquet(args.output, index=False)
        print(f"Saved dataset to {args.output}")

        # Print summary
        print(f"\nDataset summary:")
        print(f"  Total contracts: {len(contracts_df):,}")
        print(f"  Contracts with ABI: {contracts_df['has_abi'].sum():,}")
        print(
            f"  Contracts without ABI: {len(contracts_df) - contracts_df['has_abi'].sum():,}"
        )
        print(
            f"  Average bytecode length: {contracts_df['bytecode_length'].mean():.1f}"
        )
        print(f"  Average ABI length: {contracts_df['abi_length'].mean():.1f}")

        # Check bytecode quality
        print(f"\nBytecode verification:")
        real_bytecode = (
            contracts_df["bytecode"]
            .apply(lambda x: isinstance(x, str) and x.startswith("0x60806040"))
            .sum()
        )
        print(f"  Has EVM preamble (60806040): {real_bytecode:,}")

        # Sample output
        print(f"\nSample contracts:")
        for i in range(min(3, len(contracts_df))):
            contract = contracts_df.iloc[i]
            print(f"  Contract {i}:")
            print(f"    Address: {contract['address'][:20]}...")
            print(f"    Bytecode: {contract['bytecode_length']} bytes")
            print(f"    Has ABI: {contract['has_abi']}")
            print(f"    ABI length: {contract['abi_length']}")
    else:
        print("No contracts created")


if __name__ == "__main__":
    main()
