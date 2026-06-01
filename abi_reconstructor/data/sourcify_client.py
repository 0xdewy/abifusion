"""Sourcify API client for fetching verified contract data."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


class SourcifyClient:
    """Client for Sourcify API."""

    def __init__(
        self,
        base_url: str = "https://sourcify.dev/server",
        rate_limit_delay: float = 0.1,
    ):
        self.base_url = base_url
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()

    def _make_request(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make API request with rate limiting."""
        url = f"{self.base_url}/{endpoint}"

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            raise Exception(f"Sourcify API request failed: {e}") from e

        finally:
            # Rate limiting
            time.sleep(self.rate_limit_delay)

    def get_contract_files(
        self,
        address: str,
        chain_id: int = 1,  # Mainnet
    ) -> Optional[Dict[str, Any]]:
        """Get contract files from Sourcify.

        Args:
            address: Contract address
            chain_id: Blockchain chain ID

        Returns:
            Contract files or None if not found
        """
        endpoint = f"files/{chain_id}/{address}"

        try:
            data = self._make_request(endpoint)
            return data
        except Exception:
            return None

    def get_contract_metadata(
        self,
        address: str,
        chain_id: int = 1,
    ) -> Optional[Dict[str, Any]]:
        """Get contract metadata from Sourcify.

        Args:
            address: Contract address
            chain_id: Blockchain chain ID

        Returns:
            Contract metadata or None if not found
        """
        endpoint = f"metadata/{chain_id}/{address}"

        try:
            data = self._make_request(endpoint)
            return data
        except Exception:
            return None

    def check_contract_verification(
        self,
        address: str,
        chain_id: int = 1,
    ) -> Dict[str, Any]:
        """Check if contract is verified on Sourcify.

        Args:
            address: Contract address
            chain_id: Blockchain chain ID

        Returns:
            Verification status
        """
        endpoint = f"check/{chain_id}/{address}"

        try:
            data = self._make_request(endpoint)
            return {
                "status": "success",
                "verified": True,
                "data": data,
            }
        except Exception as e:
            # Check if it's a 404 (not found)
            if "404" in str(e):
                return {
                    "status": "not_found",
                    "verified": False,
                    "error": "Contract not verified on Sourcify",
                }
            else:
                return {
                    "status": "error",
                    "verified": False,
                    "error": str(e),
                }

    def get_contract(
        self,
        address: str,
        chain_id: int = 1,
    ) -> Optional[Dict[str, Any]]:
        """Get complete contract information from Sourcify.

        Args:
            address: Contract address
            chain_id: Blockchain chain ID

        Returns:
            Complete contract data or None if not found
        """
        # Check verification status
        status = self.check_contract_verification(address, chain_id)
        if not status["verified"]:
            return None

        # Get metadata
        metadata = self.get_contract_metadata(address, chain_id)
        if not metadata:
            return None

        # Get files
        files = self.get_contract_files(address, chain_id)

        # Extract ABI from metadata
        abi = []
        if metadata and "output" in metadata and "abi" in metadata["output"]:
            abi = metadata["output"]["abi"]

        # Extract bytecode if available
        bytecode = ""
        if metadata and "output" in metadata and "bytecode" in metadata["output"]:
            bytecode = metadata["output"]["bytecode"]["object"]
        elif metadata and "deployedBytecode" in metadata:
            bytecode = metadata["deployedBytecode"]["object"]

        return {
            "address": address,
            "chain_id": chain_id,
            "verified": True,
            "metadata": metadata,
            "files": files,
            "abi": abi,
            "bytecode": bytecode,
            "contract_name": metadata.get("contractName", "") if metadata else "",
            "compiler_version": metadata.get("compiler", {}).get("version", "")
            if metadata
            else "",
        }

    def fetch_contracts(
        self,
        addresses: List[str],
        output_dir: str = "data",
        chain_id: int = 1,
    ) -> Dict[str, Any]:
        """Fetch multiple contracts from Sourcify.

        Args:
            addresses: List of contract addresses
            output_dir: Directory to save results
            chain_id: Blockchain chain ID

        Returns:
            Dictionary with fetch statistics
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        contracts = []
        failed = []

        for i, address in enumerate(addresses):
            print(f"Fetching contract {i + 1}/{len(addresses)}: {address}")

            try:
                contract = self.get_contract(address, chain_id)
                if contract:
                    contracts.append(contract)
                    print("  ✓ Success (verified)")
                else:
                    failed.append(address)
                    print("  ✗ Failed (not verified)")

            except Exception as e:
                failed.append(address)
                print(f"  ✗ Error: {e}")

        # Save results
        if contracts:
            # Save full data
            full_output_file = output_path / "sourcify_contracts_full.json"
            with open(full_output_file, "w") as f:
                json.dump(contracts, f, indent=2)

            # Save simplified data for training
            simplified_data = []
            for contract in contracts:
                simplified_data.append(
                    {
                        "address": contract["address"],
                        "bytecode": contract["bytecode"],
                        "abi": contract["abi"],
                        "contract_name": contract["contract_name"],
                        "compiler_version": contract["compiler_version"],
                        "chain_id": contract["chain_id"],
                        "verified": contract["verified"],
                    }
                )

            simplified_output_file = output_path / "sourcify_contracts.json"
            with open(simplified_output_file, "w") as f:
                json.dump(simplified_data, f, indent=2)

            print(f"\nSaved {len(contracts)} contracts to {simplified_output_file}")
            print(f"Full data saved to {full_output_file}")

        return {
            "total": len(addresses),
            "successful": len(contracts),
            "failed": len(failed),
            "failed_addresses": failed,
        }
