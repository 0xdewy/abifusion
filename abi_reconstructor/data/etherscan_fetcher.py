"""Etherscan API client for fetching contract data."""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


class EtherscanAPIError(Exception):
    """Etherscan API error with response data."""
    def __init__(self, message: str, response: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.response = response


@dataclass
class EtherscanContract:
    """Contract data from Etherscan."""

    address: str
    bytecode: str
    abi: List[Dict[str, Any]]
    source_code: Optional[str] = None
    contract_name: Optional[str] = None
    compiler_version: Optional[str] = None


class EtherscanClient:
    """Client for Etherscan API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.etherscan.io/v2/api",
        chain_id: str = "1",
        rate_limit_delay: float = 0.2,
    ):
        self.api_key = api_key or os.getenv("ETHERSCAN_API_KEY")
        self.base_url = base_url
        self.chain_id = chain_id
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()

    def _make_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make API request with rate limiting."""
        params["apikey"] = self.api_key
        params.setdefault("chainid", self.chain_id)

        try:
            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()

            if data.get("status") == "0":
                msg = str(data.get("message", ""))
                res = str(data.get("result", ""))
                combined = f"{msg} {res}".lower()
                NON_ERROR_SUBSTRINGS = (
                    "no transactions found", "no contract found",
                    "contract source code not verified", "max rate",
                )
                if any(m in combined for m in NON_ERROR_SUBSTRINGS):
                    return data
                raise EtherscanAPIError(
                    f"Etherscan API error: {data.get('message', 'Unknown error')}",
                    response=data,
                )

            return data

        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {e}") from e

        finally:
            # Rate limiting
            time.sleep(self.rate_limit_delay)

    def get_contract_abi(
        self,
        address: str,
    ) -> Optional[List[Dict[str, Any]]]:
        """Get contract ABI from Etherscan.

        Args:
            address: Contract address

        Returns:
            Contract ABI or None if not found
        """
        params = {
            "module": "contract",
            "action": "getabi",
            "address": address,
        }

        data = self._make_request(params)

        if data.get("status") == "1":
            abi_json = data.get("result", "[]")
            try:
                return json.loads(abi_json)
            except json.JSONDecodeError:
                return None

        return None

    def get_contract_source_code(
        self,
        address: str,
    ) -> Optional[Dict[str, Any]]:
        """Get contract source code from Etherscan.

        Args:
            address: Contract address

        Returns:
            Source code information or None if not found
        """
        params = {
            "module": "contract",
            "action": "getsourcecode",
            "address": address,
        }

        data = self._make_request(params)

        if data.get("status") == "1" and data.get("result"):
            result = data["result"][0]
            return {
                "source_code": result.get("SourceCode", ""),
                "contract_name": result.get("ContractName", ""),
                "compiler_version": result.get("CompilerVersion", ""),
                "optimization_used": result.get("OptimizationUsed", ""),
                "runs": result.get("Runs", ""),
                "constructor_arguments": result.get("ConstructorArguments", ""),
                "evm_version": result.get("EVMVersion", ""),
                "library": result.get("Library", ""),
                "license_type": result.get("LicenseType", ""),
                "proxy": result.get("Proxy", ""),
                "implementation": result.get("Implementation", ""),
                "swarm_source": result.get("SwarmSource", ""),
            }

        return None

    def get_contract_bytecode(
        self,
        address: str,
    ) -> Optional[str]:
        """Get contract runtime bytecode from Etherscan.

        Args:
            address: Contract address

        Returns:
            Contract bytecode (0x-prefixed hex) or None if not found
        """
        params = {
            "module": "proxy",
            "action": "eth_getCode",
            "address": address,
            "tag": "latest",
        }

        data = self._make_request(params)

        # v2 proxy response: {"jsonrpc":"2.0","result":"0x..."} or {"jsonrpc":"2.0","result":"0x"}
        result = data.get("result", "")
        if result and result != "0x":
            return result

        return None

    def get_contract(
        self,
        address: str,
        include_source_code: bool = False,
    ) -> Optional[EtherscanContract]:
        """Get complete contract information.

        Args:
            address: Contract address
            include_source_code: Whether to fetch source code

        Returns:
            EtherscanContract object or None if not found
        """
        # Get ABI
        abi = self.get_contract_abi(address)
        if abi is None:
            return None

        # Get bytecode
        bytecode = self.get_contract_bytecode(address)
        if not bytecode or bytecode == "0x":
            return None

        # Get source code if requested
        source_info = None
        if include_source_code:
            source_info = self.get_contract_source_code(address)

        return EtherscanContract(
            address=address,
            bytecode=bytecode,
            abi=abi,
            source_code=source_info["source_code"] if source_info else None,
            contract_name=source_info["contract_name"] if source_info else None,
            compiler_version=source_info["compiler_version"] if source_info else None,
        )

    def fetch_contracts(
        self,
        addresses: List[str],
        output_dir: str = "data",
        include_source_code: bool = False,
    ) -> Dict[str, Any]:
        """Fetch multiple contracts.

        Args:
            addresses: List of contract addresses
            output_dir: Directory to save results
            include_source_code: Whether to fetch source code

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
                contract = self.get_contract(address, include_source_code)
                if contract:
                    contracts.append(contract)
                    print("  ✓ Success")
                else:
                    failed.append(address)
                    print("  ✗ Failed (not found or no bytecode)")

            except Exception as e:
                failed.append(address)
                print(f"  ✗ Error: {e}")

        # Save results
        if contracts:
            contracts_data = []
            for contract in contracts:
                contracts_data.append(
                    {
                        "address": contract.address,
                        "bytecode": contract.bytecode,
                        "abi": contract.abi,
                        "contract_name": contract.contract_name,
                        "compiler_version": contract.compiler_version,
                    }
                )

            output_file = output_path / "etherscan_contracts.json"
            with open(output_file, "w") as f:
                json.dump(contracts_data, f, indent=2)

            print(f"\nSaved {len(contracts)} contracts to {output_file}")

        return {
            "total": len(addresses),
            "successful": len(contracts),
            "failed": len(failed),
            "failed_addresses": failed,
        }
