"""Dataset builder for ABI reconstruction training data."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """Build training datasets from contract data."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

    def load_contracts(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """Load contracts from parquet file.

        Args:
            filepath: Path to parquet file. If None, uses default location.

        Returns:
            DataFrame with contract data
        """
        if filepath is None:
            filepath = self.data_dir / "contracts.parquet"

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Contract file not found: {filepath}")

        return pd.read_parquet(filepath)

    def extract_training_examples(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract training examples from contracts dataframe.

        Args:
            df: DataFrame with contract data

        Returns:
            List of training examples
        """
        training_examples = []

        for _, row in df.iterrows():
            try:
                address = row.get("address", "")
                bytecode = row.get("bytecode", "")
                abi = row.get("abi", [])

                if not bytecode or not abi:
                    continue

                # Extract function information from ABI
                for item in abi:
                    if item.get("type") == "function":
                        example = {
                            "address": address,
                            "bytecode": bytecode,
                            "function_name": item.get("name", ""),
                            "inputs": item.get("inputs", []),
                            "outputs": item.get("outputs", []),
                            "stateMutability": item.get("stateMutability", ""),
                            "type": item.get("type", ""),
                        }
                        training_examples.append(example)

            except Exception as e:
                logger.error(f"Error processing contract: {e}")
                continue

        return training_examples

    def save_training_data(
        self,
        examples: List[Dict[str, Any]],
        filename: str = "training_data.json",
    ) -> str:
        """Save training examples to JSON file.

        Args:
            examples: List of training examples
            filename: Output filename

        Returns:
            Path to saved file
        """
        output_path = self.data_dir / filename

        with open(output_path, "w") as f:
            json.dump(examples, f, indent=2)

        return str(output_path)

    def build_dataset(
        self,
        input_file: Optional[str] = None,
        output_file: str = "training_data.json",
        max_examples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Build complete dataset from contracts.

        Args:
            input_file: Input contracts file
            output_file: Output training data file
            max_examples: Maximum number of examples to extract

        Returns:
            Dictionary with dataset statistics
        """
        logger.info("Loading contracts...")
        df = self.load_contracts(input_file)

        logger.info(f"Loaded {len(df)} contracts")

        logger.info("Extracting training examples...")
        examples = self.extract_training_examples(df)

        if max_examples:
            examples = examples[:max_examples]

        logger.info(f"Extracted {len(examples)} training examples")

        logger.info("Saving training data...")
        output_path = self.save_training_data(examples, output_file)

        # Calculate statistics
        stats = {
            "total_contracts": len(df),
            "total_examples": len(examples),
            "output_file": output_path,
            "unique_functions": len({ex["function_name"] for ex in examples}),
        }

        logger.info(f"Dataset built successfully: {stats}")

        return stats
