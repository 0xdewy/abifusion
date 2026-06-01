"""Dataset preparation for parameter prediction ML model."""

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from abi_reconstructor.training.bytecode_dataset import BytecodeDataset
from abi_reconstructor.utils.signature_lookup import SignatureLookup


class ParameterDataset:
    """Dataset for parameter prediction with parameter-level annotations."""

    PADDING_TOKEN = 256

    COMMON_TYPES = [
        "address",
        "uint256",
        "bool",
        "bytes32",
        "string",
        "bytes",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "uint128",
        "int256",
        "int8",
        "int16",
        "int32",
        "int64",
        "int128",
        "address[]",
        "uint256[]",
        "bytes32[]",
        "string[]",
        "bytes[]",
        "bool[]",
    ]

    # Function name vocabulary (most common names from training data)
    FUNCTION_NAME_VOCAB = [
        "transfer",
        "approve",
        "balanceOf",
        "totalSupply",
        "allowance",
        "transferFrom",
        "name",
        "symbol",
        "decimals",
        "ownerOf",
        "safeTransferFrom",
        "setApprovalForAll",
        "isApprovedForAll",
        "tokenURI",
        "mint",
        "burn",
        "pause",
        "unpause",
        "renounceOwnership",
        "transferOwnership",
        "upgradeTo",
        "upgradeToAndCall",
        "admin",
        "changeAdmin",
        "implementation",
        "getBalance",
        "sendCoin",
        "convert",
        "execute",
        "close",
        "update",
        "set",
        "get",
        "add",
        "remove",
        "create",
        "delete",
        "deploy",
        "initialize",
        "withdraw",
        "deposit",
        "claim",
        "stake",
        "unstake",
        "vote",
        "propose",
    ]

    def __init__(self, data_dir: str = "data"):
        """Initialize parameter dataset.

        Args:
            data_dir: Root data directory
        """
        self.data_dir = data_dir
        self.bytecode_dataset = BytecodeDataset(data_dir)
        self.signature_lookup = SignatureLookup(
            cache_dir=os.path.join(data_dir, "cache", "signatures")
        )

        # Build type to index mapping
        self.type_to_idx = {t: i for i, t in enumerate(self.COMMON_TYPES)}
        self.idx_to_type = dict(enumerate(self.COMMON_TYPES))

        # Build function name to index mapping
        self.func_to_idx = {f: i for i, f in enumerate(self.FUNCTION_NAME_VOCAB)}
        self.idx_to_func = dict(enumerate(self.FUNCTION_NAME_VOCAB))

        # Add unknown/other types
        self.type_to_idx["unknown"] = len(self.COMMON_TYPES)
        self.idx_to_type[len(self.COMMON_TYPES)] = "unknown"

        self.func_to_idx["unknown"] = len(self.FUNCTION_NAME_VOCAB)
        self.idx_to_func[len(self.FUNCTION_NAME_VOCAB)] = "unknown"

    def load_parameter_samples(
        self, max_samples: int = 10000, context_size: int = 500
    ) -> List[Dict[str, Any]]:
        """Load training samples with parameter-level annotations.

        Args:
            max_samples: Maximum number of samples to load
            context_size: Bytes of bytecode context to extract around selector

        Returns:
            List of parameter samples
        """
        print("Loading parameter prediction samples...")

        # Load contracts from bytecode dataset
        contracts = self.bytecode_dataset.load_contracts()

        samples = []
        processed = 0

        for contract in contracts:
            try:
                # Get bytecode
                bytecode_hex = contract.get("bytecode", "")
                if not bytecode_hex or bytecode_hex == "0x":
                    continue

                # Clean bytecode
                bytecode = bytecode_hex.lower().replace("0x", "")

                # Get ABI functions
                abi = contract.get("abi", [])
                functions = [item for item in abi if item.get("type") == "function"]

                # Filter out events and errors mislabeled as functions
                functions = self.bytecode_dataset._filter_real_functions(functions)

                if not functions:
                    continue

                # Create samples for each function
                for func in functions:
                    # Extract function information
                    name = func.get("name", "unknown")
                    inputs = func.get("inputs", [])
                    outputs = func.get("outputs", [])

                    # Build true signature
                    # Handle numpy arrays and regular lists
                    if hasattr(inputs, "dtype") and isinstance(inputs, np.ndarray):
                        param_types = [inp["type"] for inp in inputs.tolist()]
                    else:
                        param_types = [inp["type"] for inp in inputs]
                    true_signature = f"{name}({','.join(param_types)})"

                    # Find selector in bytecode using multiple strategies
                    selector_hex, pos = self.bytecode_dataset.find_selector_in_bytecode(
                        bytecode_hex, func
                    )
                    if pos == -1:
                        # Selector not found in this bytecode (might be different version)
                        continue

                    # Calculate selector for reference (might be different from found selector)
                    calculated_selector = self.bytecode_dataset.calculate_selector(
                        true_signature
                    )

                    # Extract context window around selector
                    start = max(0, pos - context_size)
                    end = min(len(bytecode), pos + len(selector_hex) + context_size)
                    context = bytecode[start:end]
                    selector_offset = pos - start

                    # Prepare parameter annotations
                    parameters = []
                    # Convert inputs to list if it's a numpy array
                    if hasattr(inputs, "dtype") and isinstance(inputs, np.ndarray):
                        inputs_list = inputs.tolist()
                    else:
                        inputs_list = inputs

                    for i, inp in enumerate(inputs_list):
                        param_type = inp.get("type", "unknown")
                        param_name = inp.get("name", f"param{i}")

                        # Map type to common type index
                        type_idx = self.type_to_idx.get(
                            self._normalize_type(param_type),
                            self.type_to_idx["unknown"],
                        )

                        parameters.append(
                            {
                                "position": i,
                                "type": param_type,
                                "type_idx": type_idx,
                                "name": param_name,
                                "is_generic": self._is_generic_name(param_name),
                            }
                        )

                    # Create sample
                    sample = {
                        "bytecode": bytecode_hex,
                        "bytecode_context": context,
                        "selector": selector_hex,
                        "selector_offset": selector_offset,
                        "function_name": name,
                        "function_name_idx": self.func_to_idx.get(
                            self._normalize_function_name(name),
                            self.func_to_idx["unknown"],
                        ),
                        "parameters": parameters,
                        "parameter_count": min(len(parameters), 12),
                        "true_signature": true_signature,
                        "contract_address": contract.get("address", "unknown"),
                        "contract_name": contract.get("contract_name", "unknown"),
                        "has_outputs": len(outputs) > 0,
                    }

                    samples.append(sample)
                    processed += 1

                    if max_samples > 0 and processed >= max_samples:
                        print(
                            f"Loaded {len(samples)} parameter samples (max_samples limit reached)"
                        )
                        return samples

            except Exception as e:
                print(f"Error processing contract: {e}")
                continue

        print(f"Loaded {len(samples)} parameter samples")
        return samples

    def _normalize_type(self, type_str: str) -> str:
        """Normalize Solidity type string to common form."""
        # Handle arrays
        if type_str.endswith("[]"):
            base_type = type_str[:-2]
            normalized_base = self._normalize_base_type(base_type)
            return f"{normalized_base}[]"

        # Handle mappings (simplify)
        if type_str.startswith("mapping("):
            return "address"  # Most mappings are address→something

        # Normalize base types
        return self._normalize_base_type(type_str)

    def _normalize_base_type(self, type_str: str) -> str:
        """Normalize base Solidity type."""
        # Integer types
        if type_str.startswith("uint"):
            if type_str == "uint" or type_str == "uint256":
                return "uint256"
            elif type_str in ["uint8", "uint16", "uint32", "uint64", "uint128"]:
                return type_str
            else:
                return "uint256"  # Default to uint256

        if type_str.startswith("int"):
            if type_str == "int" or type_str == "int256":
                return "int256"
            elif type_str in ["int8", "int16", "int32", "int64", "int128"]:
                return type_str
            else:
                return "int256"  # Default to int256

        # Bytes types
        if type_str.startswith("bytes"):
            if type_str == "bytes":
                return "bytes"
            elif type_str == "bytes32":
                return "bytes32"
            else:
                return "bytes32"  # Default to bytes32

        # Other common types
        if type_str in ["address", "bool", "string"]:
            return type_str

        # Address payable
        if type_str == "address payable":
            return "address"

        # Default to unknown
        return "unknown"

    def _normalize_function_name(self, name: str) -> str:
        """Normalize function name to common form."""
        name_lower = name.lower()

        # Exact match first
        for func_name in self.FUNCTION_NAME_VOCAB:
            if name_lower == func_name.lower():
                return func_name

        # Check for common suffixes
        common_suffixes = ["_", "Of", "From", "To", "By", "For", "With"]
        for suffix in common_suffixes:
            if name_lower.endswith(suffix.lower()):
                base_name = name_lower[: -len(suffix)].rstrip("_")
                for func_name in self.FUNCTION_NAME_VOCAB:
                    if base_name == func_name.lower():
                        return func_name

        return "unknown"

    def _is_generic_name(self, name: str) -> bool:
        """Check if parameter name is generic (param0, param1, etc.)."""
        name_lower = name.lower()
        return (
            name_lower.startswith("param") and name_lower[5:].isdigit()
        ) or name_lower in ["", "arg", "argument", "input", "value"]

    def compute_type_class_weights(self, samples: List[Dict[str, Any]]) -> np.ndarray:
        """Compute class weights for parameter types based on inverse frequency.

        Args:
            samples: List of parameter samples

        Returns:
            Numpy array of class weights (num_types,)
        """
        from collections import Counter

        type_counts = Counter()
        for sample in samples:
            for param in sample["parameters"]:
                type_counts[param["type_idx"]] += 1

        num_types = len(self.COMMON_TYPES) + 1  # +1 for unknown
        weights = np.zeros(num_types)

        total_samples = sum(type_counts.values())
        for type_idx, count in type_counts.items():
            # Inverse frequency weighting with smoothing
            weights[type_idx] = total_samples / (count + 1)

        # Normalize weights so they sum to num_types
        weights = weights / weights.mean()

        return weights

    def compute_count_class_weights(
        self, samples: List[Dict[str, Any]], max_params: int = 12
    ) -> np.ndarray:
        """Compute class weights for parameter counts based on inverse frequency.

        Args:
            samples: List of parameter samples
            max_params: Maximum number of parameters

        Returns:
            Numpy array of class weights (max_params + 1,)
        """
        from collections import Counter

        count_counts = Counter()
        for sample in samples:
            param_count = min(sample["parameter_count"], max_params)
            count_counts[param_count] += 1

        num_classes = max_params + 1
        weights = np.zeros(num_classes)

        total_samples = sum(count_counts.values())
        for count, count_val in count_counts.items():
            weights[count] = total_samples / (count_val + 1)

        # Normalize weights
        weights = weights / weights.mean()

        return weights

    def filter_noisy_samples(
        self, samples: List[Dict[str, Any]], min_params: int = 0, max_params: int = 12
    ) -> List[Dict[str, Any]]:
        """Filter out samples with potentially noisy labels.

        Removes samples where:
        - Parameter types contain selector-like patterns (e.g., bytes4 from selectors)
        - Parameter types are too generic (only 'unknown')
        - Function name is 'unknown' for most common functions

        Args:
            samples: List of parameter samples
            min_params: Minimum number of parameters
            max_params: Maximum number of parameters

        Returns:
            Filtered list of samples
        """
        noisy_patterns = [
            "0x",
            "0000",
            "selector",
        ]

        filtered_samples = []
        removed = 0

        for sample in samples:
            # Check parameter count
            param_count = sample["parameter_count"]
            if param_count < min_params or param_count > max_params:
                removed += 1
                continue

            # Check if all parameters have 'unknown' type (noisy label)
            if len(sample["parameters"]) > 0:
                types = [p["type_idx"] for p in sample["parameters"]]
                unknown_idx = self.type_to_idx.get("unknown", len(self.COMMON_TYPES))

                # If all types are unknown, it's likely a poorly labeled sample
                if all(t == unknown_idx for t in types) and len(types) > 0:
                    removed += 1
                    continue

            # Check for selector-like patterns in types
            has_noisy_type = False
            for param in sample["parameters"]:
                type_str = param.get("type", "")
                for pattern in noisy_patterns:
                    if pattern.lower() in type_str.lower():
                        has_noisy_type = True
                        break

            if has_noisy_type:
                removed += 1
                continue

            # Keep sample
            filtered_samples.append(sample)

        if removed > 0:
            print(
                f"  Filtered out {removed} noisy samples ({len(filtered_samples)} kept)"
            )

        return filtered_samples

    def prepare_training_data(
        self,
        samples: List[Dict[str, Any]],
        test_split: float = 0.2,
        val_split: float = 0.1,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Prepare training, validation, and test splits.

        Args:
            samples: List of parameter samples
            test_split: Fraction of data for testing
            val_split: Fraction of training data for validation

        Returns:
            Tuple of (train_data, val_data, test_data) dictionaries
        """
        print("Preparing training data splits...")

        # Shuffle samples
        np.random.shuffle(samples)

        # Split indices
        n_samples = len(samples)
        n_test = int(n_samples * test_split)
        n_val = int((n_samples - n_test) * val_split)
        n_train = n_samples - n_test - n_val

        train_samples = samples[:n_train]
        val_samples = samples[n_train : n_train + n_val]
        test_samples = samples[n_train + n_val :]

        print(f"  Training samples: {n_train}")
        print(f"  Validation samples: {n_val}")
        print(f"  Test samples: {n_test}")

        # Prepare data dictionaries
        train_data = self._prepare_data_dict(train_samples, "train")
        val_data = self._prepare_data_dict(val_samples, "val")
        test_data = self._prepare_data_dict(test_samples, "test")

        return train_data, val_data, test_data

    def _prepare_data_dict(
        self, samples: List[Dict[str, Any]], split_name: str
    ) -> Dict[str, Any]:
        """Prepare data dictionary for a split."""
        # Extract features and labels
        bytecode_contexts = []
        selectors = []
        selector_offsets = []
        function_names = []
        function_name_idxs = []
        parameter_counts = []
        parameter_types = []
        parameter_masks = []

        for sample in samples:
            bytecode_contexts.append(sample["bytecode_context"])
            selectors.append(sample["selector"])
            selector_offsets.append(sample["selector_offset"])
            function_names.append(sample["function_name"])
            function_name_idxs.append(sample["function_name_idx"])

            # Parameter count
            param_count = sample["parameter_count"]
            parameter_counts.append(param_count)

            # Parameter types (padded to max 6 parameters)
            max_params = 12
            param_types = np.full(
                max_params, len(self.COMMON_TYPES)
            )  # Fill with unknown
            param_mask = np.zeros(max_params)

            for i, param in enumerate(sample["parameters"][:max_params]):
                param_types[i] = param["type_idx"]
                param_mask[i] = 1.0

            parameter_types.append(param_types)
            parameter_masks.append(param_mask)

        data_dict = {
            "split": split_name,
            "samples": samples,
            "bytecode_contexts": bytecode_contexts,
            "selectors": selectors,
            "selector_offsets": selector_offsets,
            "function_names": function_names,
            "function_name_idxs": np.array(function_name_idxs),
            "parameter_counts": np.array(parameter_counts),
            "parameter_types": np.array(parameter_types),
            "parameter_masks": np.array(parameter_masks),
            "num_types": len(self.COMMON_TYPES) + 1,  # +1 for unknown
            "num_functions": len(self.FUNCTION_NAME_VOCAB) + 1,
            "max_parameters": 12,
        }

        # Extract discriminating features for each sample (SigRec R11-R18)
        try:
            from abi_reconstructor.features.discriminating_features import (
                DiscriminatingFeatureExtractor,
            )
            disc_features = []
            extractor = DiscriminatingFeatureExtractor()
            for sample in samples:
                ctx = sample.get("bytecode_context", "")
                sel = sample.get("selector", "")
                if ctx and sel:
                    features = extractor.extract_from_context(ctx, ctx)
                    disc_features.append(features.to_vector())
                else:
                    disc_features.append([0.0] * 7)
            data_dict["discriminating_features"] = np.array(disc_features, dtype=np.float32)
        except Exception:
            data_dict["discriminating_features"] = np.zeros(
                (len(samples), 7), dtype=np.float32
            )

        return data_dict

    @staticmethod
    def extract_discriminating_features(
        bytecode_context: str,
        selector: str = "",
    ) -> List[float]:
        """Extract SigRec R11-R18 discriminating instruction features.

        Args:
            bytecode_context: Hex bytecode context string.
            selector: Optional selector for context location.

        Returns:
            7-element feature vector (see DiscriminatingFeatures.to_vector).
        """
        try:
            from abi_reconstructor.features.discriminating_features import (
                DiscriminatingFeatureExtractor,
            )
            extractor = DiscriminatingFeatureExtractor()
            features = extractor.extract_from_context(bytecode_context, bytecode_context)
            return features.to_vector()
        except Exception:
            return [0.0] * 7

    def save_dataset(self, data_dict: Dict[str, Any], filepath: str):
        """Save dataset to file."""
        # Convert numpy arrays to lists for JSON serialization
        save_dict = data_dict.copy()

        # Convert numpy arrays
        for key in [
            "function_name_idxs",
            "parameter_counts",
            "parameter_types",
            "parameter_masks",
            "discriminating_features",
        ]:
            if key in save_dict and isinstance(save_dict[key], np.ndarray):
                save_dict[key] = save_dict[key].tolist()

        # Save to file
        with open(filepath, "w") as f:
            json.dump(save_dict, f, indent=2)

        print(f"Saved dataset to {filepath}")

    def load_dataset(self, filepath: str) -> Dict[str, Any]:
        """Load dataset from file."""
        with open(filepath, "r") as f:
            data_dict = json.load(f)

        # Convert lists back to numpy arrays
        for key in [
            "function_name_idxs",
            "parameter_counts",
            "parameter_types",
            "parameter_masks",
            "discriminating_features",
        ]:
            if key in data_dict and isinstance(data_dict[key], list):
                data_dict[key] = np.array(data_dict[key])

        return data_dict


class ParameterDatasetTorch(Dataset):
    """PyTorch Dataset for parameter prediction."""

    def __init__(
        self,
        data_dict: Dict[str, Any],
        tokenize_bytecode: bool = True,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
    ):
        """Initialize PyTorch dataset.

        Args:
            data_dict: Data dictionary from ParameterDataset
            tokenize_bytecode: Whether to tokenize bytecode into integers
            cache_dir: Directory for caching tokenized data
            use_cache: Whether to use caching
        """
        self.data_dict = data_dict
        self.tokenize_bytecode = tokenize_bytecode
        self.use_cache = use_cache

        # Set up cache
        if cache_dir is None:
            cache_dir = "cache/tokenized_bytecode"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Generate cache key from data
        self.cache_key = self._generate_cache_key(data_dict)

        # Extract data
        self.bytecode_contexts = data_dict["bytecode_contexts"]
        self.function_name_idxs = torch.LongTensor(data_dict["function_name_idxs"])
        self.parameter_counts = torch.LongTensor(data_dict["parameter_counts"])
        self.parameter_types = torch.LongTensor(data_dict["parameter_types"])
        self.parameter_masks = torch.FloatTensor(data_dict["parameter_masks"])

        # Discriminating features (SigRec R11-R18) — may not be present in older datasets
        disc_arr = data_dict.get("discriminating_features")
        if disc_arr is not None and len(disc_arr) == len(self):
            self.discriminating_features = torch.FloatTensor(disc_arr)
        else:
            self.discriminating_features = torch.zeros(len(self), 7, dtype=torch.float32)

        # Tokenize bytecode if requested
        if tokenize_bytecode:
            self.bytecode_tokens = self._load_or_tokenize_bytecode()
        else:
            self.bytecode_tokens = None

    def _tokenize_bytecode_contexts(self) -> List[torch.Tensor]:
        """Tokenize bytecode contexts into integer sequences."""
        from abi_reconstructor.utils.bytecode_utils import (
            hex_to_tensor_batch_vectorized,
        )

        # Use vectorized batch conversion
        tokens_tensor = hex_to_tensor_batch_vectorized(
            self.bytecode_contexts,
            max_len=512,  # Default max length
        )

        # Convert to list of tensors (one per sample)
        tokens_list = [tokens_tensor[i] for i in range(tokens_tensor.size(0))]

        return tokens_list

    def _generate_cache_key(self, data_dict: Dict[str, Any]) -> str:
        """Generate cache key from data dictionary."""
        # Create a hash of the bytecode contexts and metadata
        hash_data = {
            "bytecode_contexts": data_dict["bytecode_contexts"],
            "split": data_dict.get("split", "unknown"),
            "num_samples": len(data_dict["bytecode_contexts"]),
            "tokenize_bytecode": self.tokenize_bytecode,
        }

        # Convert to string and hash
        hash_str = json.dumps(hash_data, sort_keys=True)
        cache_key = hashlib.md5(hash_str.encode()).hexdigest()[:16]

        return cache_key

    def _get_cache_path(self) -> Path:
        """Get cache file path."""
        return self.cache_dir / f"{self.cache_key}.pkl"

    def _load_or_tokenize_bytecode(self) -> List[torch.Tensor]:
        """Load tokenized bytecode from cache or tokenize."""
        if not self.use_cache:
            return self._tokenize_bytecode_contexts()

        cache_path = self._get_cache_path()

        # Try to load from cache
        if cache_path.exists():
            try:
                print(f"Loading tokenized bytecode from cache: {cache_path}")
                with open(cache_path, "rb") as f:
                    cached_data = torch.load(f, weights_only=True)

                # Verify cache integrity
                if len(cached_data) == len(self.bytecode_contexts) and all(
                    isinstance(t, torch.Tensor) for t in cached_data
                ):
                    print(f"  Cache hit: {len(cached_data)} samples loaded")
                    return cached_data
                else:
                    print("  Cache invalid, re-tokenizing")
            except Exception as e:
                print(f"  Cache load error: {e}, re-tokenizing")

        # Tokenize and cache
        print("Tokenizing bytecode (cache miss)...")
        tokens_list = self._tokenize_bytecode_contexts()

        # Save to cache
        try:
            with open(cache_path, "wb") as f:
                torch.save(tokens_list, f)
            print(f"  Cached {len(tokens_list)} samples to {cache_path}")
        except Exception as e:
            print(f"  Warning: Failed to cache tokenized data: {e}")

        return tokens_list

    def __len__(self) -> int:
        return len(self.bytecode_contexts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item for training.

        Returns dictionary with:
            - bytecode_tokens: Tokenized bytecode context
            - function_name_idx: Function name class index
            - parameter_count: Number of parameters (0-6)
            - parameter_types: Type indices for each parameter position (padded)
            - parameter_mask: Mask indicating valid parameter positions
        """
        if self.tokenize_bytecode:
            bytecode_tokens = self.bytecode_tokens[idx]
        else:
            # Return raw hex string
            bytecode_tokens = self.bytecode_contexts[idx]

        return {
            "bytecode_tokens": bytecode_tokens,
            "function_name_idx": self.function_name_idxs[idx],
            "parameter_count": self.parameter_counts[idx],
            "parameter_types": self.parameter_types[idx],
            "parameter_mask": self.parameter_masks[idx],
            "discriminating_features": self.discriminating_features[idx],
            "sample_idx": idx,
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate function for DataLoader.

        Handles variable-length bytecode sequences by padding.
        """
        # Separate batch elements
        bytecode_tokens = [item["bytecode_tokens"] for item in batch]
        function_name_idxs = torch.stack([item["function_name_idx"] for item in batch])
        parameter_counts = torch.stack([item["parameter_count"] for item in batch])
        parameter_types = torch.stack([item["parameter_types"] for item in batch])
        parameter_masks = torch.stack([item["parameter_mask"] for item in batch])
        sample_idxs = torch.tensor([item["sample_idx"] for item in batch])
        discriminating_features = torch.stack([item["discriminating_features"] for item in batch])

        # Pad bytecode tokens to same length
        PADDING_TOKEN = 256
        if isinstance(bytecode_tokens[0], torch.Tensor):
            padded_tokens = torch.nn.utils.rnn.pad_sequence(
                bytecode_tokens,
                batch_first=True,
                padding_value=PADDING_TOKEN,
            )
            attention_mask = (padded_tokens != PADDING_TOKEN).float()
        else:
            # String mode: return as list
            padded_tokens = bytecode_tokens
            attention_mask = None

        return {
            "bytecode_tokens": padded_tokens,
            "attention_mask": attention_mask,
            "function_name_idx": function_name_idxs,
            "parameter_count": parameter_counts,
            "parameter_types": parameter_types,
            "parameter_mask": parameter_masks,
            "discriminating_features": discriminating_features,
            "sample_idx": sample_idxs,
        }


def test_parameter_dataset():
    """Test the parameter dataset."""
    print("Testing ParameterDataset...")

    # Create dataset
    dataset = ParameterDataset()

    # Load samples
    samples = dataset.load_parameter_samples(max_samples=100)
    print(f"Loaded {len(samples)} samples")

    if samples:
        # Show sample statistics
        param_counts = [s["parameter_count"] for s in samples]
        print("Parameter count distribution:")
        for count in range(0, 7):
            count_samples = sum(1 for c in param_counts if c == count)
            print(f"  {count} parameters: {count_samples} samples")

        # Show type distribution
        all_types = []
        for sample in samples:
            for param in sample["parameters"]:
                all_types.append(param["type"])

        from collections import Counter

        type_counts = Counter(all_types)
        print("\nTop 10 parameter types:")
        for type_name, count in type_counts.most_common(10):
            print(f"  {type_name}: {count}")

        # Prepare training data
        train_data, val_data, test_data = dataset.prepare_training_data(samples)
        print("\nData splits prepared:")
        print(f"  Train: {len(train_data['samples'])} samples")
        print(f"  Val: {len(val_data['samples'])} samples")
        print(f"  Test: {len(test_data['samples'])} samples")

        # Test PyTorch dataset
        torch_dataset = ParameterDatasetTorch(train_data, tokenize_bytecode=True)
        print(f"\nPyTorch dataset created with {len(torch_dataset)} samples")

        # Get a batch
        dataloader = DataLoader(
            torch_dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=ParameterDatasetTorch.collate_fn,
        )

        batch = next(iter(dataloader))
        print("\nBatch shapes:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            elif isinstance(value, list):
                print(f"  {key}: list of {len(value)} items")

        return True
    else:
        print("No samples loaded")
        return False


if __name__ == "__main__":
    test_parameter_dataset()
