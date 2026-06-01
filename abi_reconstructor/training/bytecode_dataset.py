"""Dataset creation for bytecode + selector → signature training."""

import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

# Try to import proper Ethereum keccak
try:
    from eth_utils import keccak

    HAS_KECCAK = True
except ImportError:
    HAS_KECCAK = False


logger = logging.getLogger(__name__)


class BytecodeDataset:
    """Create training dataset for bytecode + selector → signature model."""

    def __init__(self, data_dir: str = "data"):
        """Initialize dataset.

        Args:
            data_dir: Root data directory
        """
        self.data_dir = data_dir

    def calculate_selector(self, signature: str) -> str:
        """Calculate Ethereum function selector from signature.

        Args:
            signature: Function signature

        Returns:
            8-character hex selector
        """
        if HAS_KECCAK:
            hash_bytes = keccak(text=signature)
            return hash_bytes.hex()[:8]
        else:
            raise RuntimeError(
                "keccak hash unavailable - cannot compute function selectors. "
                "Install eth_utils: pip install eth_utils"
            )

    def build_signature(self, func: Dict[str, Any]) -> str:
        """Build function signature from ABI function.

        Args:
            func: ABI function dictionary

        Returns:
            Function signature string
        """
        name = func.get("name", "unknown")
        inputs = func.get("inputs", [])
        param_types = [inp["type"] for inp in inputs]
        return f"{name}({','.join(param_types)})"

    def _filter_real_functions(
        self, functions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter out events and errors mislabeled as functions.

        Args:
            functions: List of ABI items marked as "function"

        Returns:
            Filtered list of real functions
        """
        real_functions = []

        # Common event names that are often mislabeled as functions
        common_event_names = {
            "upgraded",
            "adminchanged",
            "beaconupgraded",
            "transfer",
            "approval",
            "roleadminchanged",
            "rolegranted",
            "rolerevoked",
            "paused",
            "unpaused",
            "ownershiptransferred",
            "pauserchanged",
            "mint",
            "burn",
            "swap",
            "sync",
            "transferownership",
            "renounceownership",
            "deposit",
            "withdraw",
            "claim",
            "stake",
            "unstake",
            "vote",
            "propose",
        }

        for func in functions:
            name = func.get("name", "").lower()

            # Skip functions with common event names (case-insensitive)
            if name in common_event_names:
                continue

            # Skip functions that look like error names (often start with uppercase or contain "Error")
            if (
                name.endswith("error")
                or name.endswith("failure")
                or name.endswith("exception")
            ):
                continue

            # Skip functions with no inputs AND no outputs (likely events or errors)
            inputs = func.get("inputs", [])
            outputs = func.get("outputs", [])

            # Handle numpy arrays
            if hasattr(inputs, "size"):
                has_inputs = inputs.size > 0
            else:
                has_inputs = len(inputs) > 0

            if hasattr(outputs, "size"):
                has_outputs = outputs.size > 0
            else:
                has_outputs = len(outputs) > 0

            if not has_inputs and not has_outputs:
                continue

            real_functions.append(func)

        return real_functions

    def find_selector_in_bytecode(
        self, bytecode_hex: str, func: Dict[str, Any]
    ) -> Tuple[str, int]:
        """Find function selector in bytecode using multiple strategies.

        Args:
            bytecode_hex: Contract bytecode as hex string
            func: ABI function dictionary

        Returns:
            Tuple of (selector_hex, position) or (None, -1) if not found
        """

        if not bytecode_hex or bytecode_hex == "0x":
            return None, -1

        bytecode = bytecode_hex.lower().replace("0x", "")

        # Strategy 1: Use selector from ABI if available
        selector_from_abi = func.get("selector")
        if selector_from_abi:
            selector_hex = selector_from_abi.lower()
            pos = bytecode.find(selector_hex)
            if pos != -1:
                return selector_hex, pos

        # Strategy 2: Calculate selector from signature
        signature = self.build_signature(func)
        calculated_selector = self.calculate_selector(signature)
        selector_hex = calculated_selector.lower()
        pos = bytecode.find(selector_hex)
        if pos != -1:
            return selector_hex, pos

        # Strategy 3: Try without "0x" prefix if present
        if selector_from_abi and selector_from_abi.startswith("0x"):
            selector_hex = selector_from_abi[2:].lower()
            pos = bytecode.find(selector_hex)
            if pos != -1:
                return selector_hex, pos

        return None, -1

    def load_contracts(self) -> List[Dict[str, Any]]:
        """Load all contracts from unified parquet file.

        Returns:
            List of contract dictionaries
        """
        contracts = []

        # Load from unified parquet file in data directory
        # Try comprehensive dataset first, then combined, then original
        parquet_paths = [
            os.path.join(
                self.data_dir if self.data_dir else "data",
                "all_contracts_comprehensive.parquet",
            ),
            os.path.join(
                self.data_dir if self.data_dir else "data", "contracts_combined.parquet"
            ),
            os.path.join(
                self.data_dir if self.data_dir else "data", "contracts.parquet"
            ),
        ]

        parquet_path = None
        for path in parquet_paths:
            if os.path.exists(path):
                parquet_path = path
                break

        if parquet_path and os.path.exists(parquet_path):
            try:
                import numpy as np
                import pandas as pd

                logger.info(f"Loading contracts from unified parquet: {parquet_path}")
                df = pd.read_parquet(parquet_path)

                # Get file size for info
                file_size = os.path.getsize(parquet_path)
                file_size_mb = file_size / (1024 * 1024)
                logger.info(f"  File size: {file_size_mb:.1f} MB")
                logger.info(f"  Number of rows: {len(df)}")

                # Convert DataFrame to list of dictionaries using vectorized operations
                # This is much faster than iterrows()
                contracts = df.to_dict("records")

                logger.info(f"  Converted to {len(contracts)} contract dictionaries")

                # Process ABI fields in batch
                valid_contracts = []
                missing_fields = 0
                empty_fields = 0

                for contract in contracts:
                    # Check required fields
                    has_required = all(
                        field in contract for field in ["address", "bytecode", "abi"]
                    )

                    if not has_required:
                        missing_fields += 1
                        continue

                    # Check field values
                    bytecode_valid = bool(contract["bytecode"])
                    abi_valid = False

                    # Check if ABI is not empty
                    abi_value = contract["abi"]
                    if hasattr(abi_value, "__len__"):
                        try:
                            abi_valid = len(abi_value) > 0
                        except Exception:
                            abi_valid = False
                    elif abi_value:
                        abi_valid = True

                    has_values = bytecode_valid and abi_valid
                    if not has_values:
                        empty_fields += 1
                        continue

                    # Convert numpy arrays to lists for ABI
                    if hasattr(contract["abi"], "__iter__"):
                        try:
                            # Convert numpy array to list if needed
                            if isinstance(contract["abi"], np.ndarray):
                                contract["abi"] = contract["abi"].tolist()
                            elif not isinstance(contract["abi"], list):
                                contract["abi"] = list(contract["abi"])
                        except Exception:
                            contract["abi"] = []

                    valid_contracts.append(contract)

                logger.info(
                    f"Loaded {len(valid_contracts)} valid contracts from unified parquet"
                )

                if missing_fields > 0:
                    logger.info(
                        f"  Warning: {missing_fields} contracts missing required fields"
                    )
                if empty_fields > 0:
                    logger.info(
                        f"  Warning: {empty_fields} contracts have empty bytecode or ABI"
                    )

                return valid_contracts

            except Exception as e:
                logger.error(f"Error loading unified parquet file: {e}")
                logger.info("Please run scripts/consolidate_all_data.py first")
                return []
        else:
            logger.info(f"Unified parquet file not found at {parquet_path}")
            logger.info("Please run scripts/consolidate_all_data.py first")
            return []

    def extract_training_samples(
        self, contracts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract training samples from contracts.

        Each sample: (bytecode, selector, signature)

        Args:
            contracts: List of contract dictionaries

        Returns:
            List of training samples
        """
        samples = []

        for contract in contracts:
            try:
                # Get bytecode
                bytecode_hex = contract.get("bytecode", "")
                if not bytecode_hex or bytecode_hex == "0x":
                    continue

                # Get ABI functions
                abi = contract.get("abi", [])
                functions = [item for item in abi if item.get("type") == "function"]

                # Filter out events and errors mislabeled as functions
                functions = self._filter_real_functions(functions)

                if not functions:
                    continue

                # Create samples for each function
                for func in functions:
                    # Build signature
                    signature = self.build_signature(func)

                    # Calculate selector
                    selector = self.calculate_selector(signature)

                    # Create sample
                    sample = {
                        "bytecode": bytecode_hex,
                        "selector": selector,
                        "signature": signature,
                        "contract_address": contract.get("address", "unknown"),
                        "contract_name": contract.get("contract_name", "unknown"),
                        "function_name": func.get("name", "unknown"),
                    }

                    samples.append(sample)

            except Exception as e:
                logger.error(f"Error processing contract: {e}")
                continue

        logger.info(f"Extracted {len(samples)} training samples")
        return samples

    def create_negative_samples(
        self, positive_samples: List[Dict[str, Any]], negative_ratio: float = 1.0
    ) -> List[Dict[str, Any]]:
        """Create negative samples by pairing wrong selectors with signatures.

        Args:
            positive_samples: List of positive samples
            negative_ratio: Ratio of negative to positive samples

        Returns:
            List of negative samples
        """
        if not positive_samples:
            return []

        # Group by contract for within-contract negative samples
        contract_to_samples = defaultdict(list)
        for sample in positive_samples:
            contract_to_samples[sample["contract_address"]].append(sample)

        negative_samples = []
        n_negative = int(len(positive_samples) * negative_ratio)

        for _ in range(n_negative):
            # Pick a random contract
            contract_addr = np.random.choice(list(contract_to_samples.keys()))
            contract_samples = contract_to_samples[contract_addr]

            if len(contract_samples) < 2:
                continue

            # Pick two different samples from same contract
            idx1, idx2 = np.random.choice(len(contract_samples), 2, replace=False)
            sample1 = contract_samples[idx1]
            sample2 = contract_samples[idx2]

            # Create negative sample: bytecode from sample1, selector from sample2,
            # signature from sample1
            negative_sample = {
                "bytecode": sample1["bytecode"],
                "selector": sample2["selector"],  # Wrong selector!
                "signature": sample1["signature"],
                "contract_address": sample1["contract_address"],
                "contract_name": sample1["contract_name"],
                "function_name": sample1["function_name"],
                "is_negative": True,
            }

            negative_samples.append(negative_sample)

        logger.info(f"Created {len(negative_samples)} negative samples")
        return negative_samples




def test_dataset():
    """Test the dataset creation."""
    logger.info("Testing BytecodeDataset...")

    dataset = BytecodeDataset()

    try:
        contracts = dataset.load_contracts()
        logger.info(f"Loaded {len(contracts)} contracts")

        if contracts:
            samples = dataset.extract_training_samples(contracts[:2])
            logger.info(f"Extracted {len(samples)} samples")

            if samples:
                negative_samples = dataset.create_negative_samples(
                    samples, negative_ratio=0.5
                )
                logger.info(f"Created {len(negative_samples)} negative samples")

                logger.info("prepare_dataset is deprecated - skipping")

    except Exception as e:
        logger.error(f"Error during testing: {e}")
        import traceback

        traceback.print_exc()

    logger.info("Test completed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    test_dataset()
