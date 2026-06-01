"""ABI Reconstruction Pipeline - Orchestrates the complete ABI reconstruction process."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from abi_reconstructor.features.bytecode_features import BytecodeFeatureExtractor
from abi_reconstructor.features.selector_extractor import NeuralSelectorExtractor
from abi_reconstructor.models.function_name_classifier import FunctionNameClassifier
from abi_reconstructor.models.parameter_prediction_model import ParameterPredictionModel
from abi_reconstructor.utils.bytecode_utils import (
    clean_bytecode,
    extract_selector_context,
)
from abi_reconstructor.utils.signature_lookup import SignatureLookup

logger = logging.getLogger(__name__)


class ABIReconstructorPipeline:
    """Main pipeline for ABI reconstruction from bytecode."""

    def __init__(
        self,
        function_classifier: FunctionNameClassifier,
        parameter_predictor: ParameterPredictionModel,
        selector_extractor: Optional[NeuralSelectorExtractor] = None,
        signature_lookup: Optional[SignatureLookup] = None,
        feature_extractor: Optional[BytecodeFeatureExtractor] = None,
        use_cuda: bool = True,
        verbose: bool = False,
        function_vocab: Optional[Dict[str, int]] = None,
        type_vocab: Optional[Dict[str, int]] = None,
    ):
        """Initialize the ABI reconstruction pipeline.

        Args:
            function_classifier: Trained function name classifier
            parameter_predictor: Trained parameter prediction model
            selector_extractor: Optional selector extractor
            signature_lookup: Optional signature lookup service
            feature_extractor: Optional bytecode feature extractor
            use_cuda: Whether to use CUDA/GPU
            verbose: Whether to print debug information
            function_vocab: Function name to index mapping (for parameter predictor)
            type_vocab: Type name to index mapping (for parameter predictor)
        """
        self.function_classifier = function_classifier
        self.parameter_predictor = parameter_predictor
        self.selector_extractor = selector_extractor
        self.signature_lookup = signature_lookup or SignatureLookup()
        self.feature_extractor = feature_extractor
        self.use_cuda = use_cuda
        self.verbose = verbose

        # Vocabulary mappings for parameter predictor
        if function_vocab is None:
            # Default vocabulary from ParameterDataset
            from abi_reconstructor.training.parameter_dataset import ParameterDataset

            dataset = ParameterDataset()
            self.function_to_idx = dataset.func_to_idx
            self.idx_to_function = dataset.idx_to_func
            self.type_to_idx = dataset.type_to_idx
            self.idx_to_type = dataset.idx_to_type
        else:
            self.function_to_idx = function_vocab
            self.idx_to_function = {v: k for k, v in function_vocab.items()}
            self.type_to_idx = type_vocab or {}
            self.idx_to_type = {v: k for k, v in (type_vocab or {}).items()}

        # Move models to device through DeviceManager
        from abi_reconstructor.device import get_device_manager
        dm = get_device_manager()
        if dm.is_cuda:
            self.function_classifier.to(dm.device)
            self.parameter_predictor.to(dm.device)

        # Set models to evaluation mode
        self.function_classifier.eval()
        self.parameter_predictor.eval()

        # Initialize components if not provided
        self.selector_extractor = selector_extractor or NeuralSelectorExtractor(
            use_cuda=use_cuda
        )
        self.signature_lookup = signature_lookup or SignatureLookup()
        self.feature_extractor = feature_extractor or BytecodeFeatureExtractor()

        # Configuration
        self.min_selector_confidence = 0.3
        self.min_function_confidence = 0.1
        self.min_parameter_confidence = 0.2

        # Function-specific type priors - common parameter type patterns
        # Format: function_name -> list of expected types in order
        # Empty list means no parameters (0-param function)
        self.FUNCTION_TYPE_PRIORS = {
            # ERC20 functions
            "transfer": ["address", "uint256"],
            "transferFrom": ["address", "address", "uint256"],
            "approve": ["address", "uint256"],
            "totalSupply": [],
            "allowance": ["address", "address"],
            "name": [],
            "symbol": [],
            "decimals": [],
            # NOTE: "balanceOf" and "safeTransferFrom" are overloaded across token
            # standards (ERC20/721 vs ERC1155) and are resolved by predicted arity
            # via FUNCTION_TYPE_PRIOR_OVERLOADS below, not here.
            # ERC721 functions
            "ownerOf": ["uint256"],
            "setApprovalForAll": ["address", "bool"],
            "isApprovedForAll": ["address", "address"],
            "tokenURI": ["uint256"],
            # AccessControl functions
            "hasRole": ["bytes32", "address"],
            "grantRole": ["bytes32", "address"],
            "revokeRole": ["bytes32", "address"],
            "renounceRole": ["bytes32", "address"],
            "getRoleAdmin": ["bytes32"],
            "DEFAULT_ADMIN_ROLE": [],
            # ERC1155 functions
            "safeBatchTransferFrom": [
                "address",
                "address",
                "uint256[]",
                "uint256[]",
                "bytes",
            ],
            # Pool functions (Curve-like)
            "add_liquidity": ["uint256[]", "uint256"],
            "remove_liquidity": ["uint256", "uint256[]"],
            "exchange": ["int128", "int128", "uint256", "uint256"],
            "get_dy": ["int128", "int128", "uint256"],
            "calc_token_amount": ["uint256[]", "bool"],
            "get_virtual_price": [],
            # General functions
            "mint": ["address", "uint256"],
            "burn": ["uint256"],
            "pause": [],
            "unpause": [],
            "renounceOwnership": [],
            "transferOwnership": ["address"],
            "upgradeTo": ["address"],
            "upgradeToAndCall": ["address", "bytes"],
            "getBalance": ["address"],
            "sendCoin": ["address", "uint256"],
            "execute": ["bytes"],
            "castVote": ["uint256", "bool"],
            "propose": ["address[]", "uint256[]", "bytes[]", "string"],
            "vote": ["uint256", "bool"],
            "stake": ["uint256"],
            "unstake": ["uint256"],
            "deposit": ["uint256"],
            "withdraw": ["uint256"],
            "claim": [],
            "initialize": [],
            # Zero-param view functions (common in multi-sig, governance, etc.)
            "domainSeparator": [],
            "version": [],
            "nonce": [],
            "getThreshold": [],
            "getOwners": [],
            "getImplementation": [],
            "getAdmin": [],
            "paused": [],
        }

        # Overloaded standard functions: the same name maps to different
        # signatures across token standards. Keys are lowercase; candidates are
        # ordered most-common-first and selected by predicted parameter count at
        # lookup time (see predict_parameters).
        self.FUNCTION_TYPE_PRIOR_OVERLOADS = {
            # ERC20/721 balanceOf(address) vs ERC1155 balanceOf(address,uint256)
            "balanceof": [["address"], ["address", "uint256"]],
            # ERC721 safeTransferFrom(address,address,uint256) vs
            # ERC1155 safeTransferFrom(address,address,uint256,uint256,bytes)
            "safetransferfrom": [
                ["address", "address", "uint256"],
                ["address", "address", "uint256", "uint256", "bytes"],
            ],
        }

    @classmethod
    def load_pipeline(
        cls,
        checkpoint_dir: str = "cache/checkpoints",
        use_cuda: bool = True,
        verbose: bool = False,
    ) -> "ABIReconstructorPipeline":
        """Load a trained pipeline from checkpoint directory.

        Args:
            checkpoint_dir: Directory containing model checkpoints
            use_cuda: Whether to use CUDA/GPU
            verbose: Whether to print debug information

        Returns:
            Initialized ABIReconstructorPipeline
        """
        checkpoint_dir = Path(checkpoint_dir)

        # Load function classifier
        function_classifier_path = checkpoint_dir / "function_classifier_final.pth"
        if not function_classifier_path.exists():
            # Try alternative name
            function_classifier_path = checkpoint_dir / "final_function_classifier.pth"

        logger.info(f"Loading function classifier from {function_classifier_path}")
        function_classifier = FunctionNameClassifier.load_model(
            str(function_classifier_path), use_cuda=use_cuda
        )

        # Load parameter predictor
        parameter_predictor_path = checkpoint_dir / "parameter_predictor_final.pth"
        if not parameter_predictor_path.exists():
            # Try alternative name
            parameter_predictor_path = checkpoint_dir / "final_parameter_predictor.pth"

        logger.info(f"Loading parameter predictor from {parameter_predictor_path}")

        parameter_predictor = ParameterPredictionModel.load_model(
            str(parameter_predictor_path),
            use_cuda=use_cuda,
        )

        # Load vocabulary from ParameterDataset
        from abi_reconstructor.training.parameter_dataset import ParameterDataset

        dataset = ParameterDataset()
        function_vocab = dataset.func_to_idx
        type_vocab = dataset.type_to_idx

        return cls(
            function_classifier=function_classifier,
            parameter_predictor=parameter_predictor,
            use_cuda=use_cuda,
            verbose=verbose,
            function_vocab=function_vocab,
            type_vocab=type_vocab,
        )

    def extract_selectors(
        self, bytecode: str, max_selectors: int = 100
    ) -> List[Dict[str, Any]]:
        """Extract function selectors from bytecode.

        Args:
            bytecode: Contract bytecode (hex string)
            max_selectors: Maximum number of selectors to return (0 for unlimited)

        Returns:
            List of dictionaries with selector information
        """
        bytecode = clean_bytecode(bytecode)
        selectors = []

        # Common known selectors (ERC20, ERC721, ERC1155, common patterns)
        common_selectors = [
            # ERC20
            "a9059cbb",  # transfer(address,uint256)
            "095ea7b3",  # approve(address,uint256)
            "70a08231",  # balanceOf(address)
            "23b872dd",  # transferFrom(address,address,uint256)
            "18160ddd",  # totalSupply()
            "06fdde03",  # name()
            "95d89b41",  # symbol()
            "313ce567",  # decimals()
            "dd62ed3e",  # allowance(address,address)
            # ERC721
            "6352211e",  # ownerOf(uint256)
            "42842e0e",  # safeTransferFrom(address,address,uint256)
            "b88d4fde",  # safeTransferFrom(address,address,uint256,bytes)
            "081812fc",  # getApproved(uint256)
            "a22cb465",  # setApprovalForAll(address,bool)
            "e985e9c5",  # isApprovedForAll(address,address)
            "150b7a02",  # onERC721Received(address,address,uint256,bytes)
            # ERC1155
            "f242432a",  # safeTransferFrom(address,address,uint256,uint256,bytes)
            "2eb2c2d6",  # safeBatchTransferFrom(address,address,uint256[],uint256[],bytes)
            "00fdd58e",  # balanceOf(address,uint256)
            "4e1273f4",  # balanceOfBatch(address[],uint256[])
            "a22cb465",  # setApprovalForAll(address,bool) - also ERC721
            "e985e9c5",  # isApprovedForAll(address,address) - also ERC721
            # Common functions
            "40c10f19",  # mint(address,uint256)
            "d0e30db0",  # deposit()
            "2e1a7d4d",  # withdraw(uint256)
            "a85d3939",  # deposit(uint256)
            "3ccfd60b",  # withdraw(uint256)
            "4cd88b76",  # receive()
            "1a4d01d2",  # fallback()
            # Ownership
            "8da5cb5b",  # owner()
            "f2fde38b",  # transferOwnership(address)
            "715018a6",  # renounceOwnership()
            # Pausable
            "8456cb59",  # pause()
            "3f4ba83a",  # unpause()
            "5c975abb",  # paused()
            # Access Control
            "a217fddf",  # DEFAULT_ADMIN_ROLE()
            "248a9ca3",  # getRoleAdmin(bytes32)
            "2f2ff15d",  # grantRole(bytes32,address)
            "d547741f",  # revokeRole(bytes32,address)
            "91d14854",  # renounceRole(bytes32,address)
            "36568abe",  # hasRole(bytes32,address)
            # Special
            "7fffffff",  # fallback (simplified)
            "00000000",  # constructor
        ]

        # First pass: Look for known selectors (high confidence)
        for selector in common_selectors:
            if selector == "00000000" or selector == "7fffffff":
                continue
            pos = bytecode.find(selector)
            if pos != -1 and pos % 2 == 0:
                context_start = max(0, pos - 64)
                context_end = min(len(bytecode), pos + 128)
                context = bytecode[context_start:context_end]

                selectors.append(
                    {
                        "selector": selector,
                        "position": pos,
                        "context": context,
                        "confidence": 0.95,
                        "source": "known_selector",
                    }
                )

        # Second pass: Look for PUSH4 (0x63) followed by selector that appears
        # before an EQ instruction (0x14) - indicating function comparison
        # Pattern: PUSH4 <selector> ... EQ
        for i in range(0, len(bytecode) - 11, 2):
            if bytecode[i : i + 2] == "63":  # PUSH4
                selector = bytecode[i + 2 : i + 10]
                if len(selector) == 8:
                    # Filter out obvious non-selectors
                    if selector == "00000000":
                        continue
                    if all(
                        c == selector[0] for c in selector
                    ):  # All same char (e.g., 0x11111111)
                        continue

                    # Check if selector is in valid range (most function selectors are > 0x10000000)
                    try:
                        sel_int = int(selector, 16)
                        if sel_int < 0x10000000:
                            continue
                    except ValueError:
                        continue

                    # Look for EQ (0x14) after the PUSH4 within reasonable distance
                    # This indicates the selector is being compared (function dispatch)
                    eq_pos = -1
                    search_end = min(len(bytecode), i + 24)  # Search within ~12 bytes
                    for j in range(i + 10, search_end, 2):
                        if bytecode[j : j + 2] == "14":  # EQ opcode
                            eq_pos = j
                            break

                    if eq_pos == -1:
                        continue  # No EQ found - likely not a function selector

                    # Extract context
                    context_start = max(0, i - 64)
                    context_end = min(len(bytecode), i + 128)
                    context = bytecode[context_start:context_end]

                    selectors.append(
                        {
                            "selector": selector,
                            "position": i + 2,
                            "context": context,
                            "confidence": 0.7,
                            "source": "push4_eq_pattern",
                        }
                    )

        # Third pass: Look for selector patterns in function dispatchers
        # This is more conservative than the previous pattern-based approach
        # We look for sequences that indicate a function selector is being used

        # Common function dispatcher patterns in Solidity:
        # 1. PUSH4 selector EQ/JUMPI
        # 2. DUP1 PUSH4 selector EQ/JUMPI
        # 3. Various combinations with DUP, SWAP, etc.

        # We'll search for PUSH4 followed by selector, then look for dispatch patterns
        for i in range(0, len(bytecode) - 15, 2):  # Need more bytes for patterns
            # Check for PUSH4 (0x63)
            if bytecode[i : i + 2] == "63" and i + 10 < len(bytecode):
                selector = bytecode[i + 2 : i + 10]
                if len(selector) != 8:
                    continue

                # Skip if we already found this selector
                if any(s["selector"] == selector for s in selectors):
                    continue

                # Basic validation (same as before)
                if selector == "00000000":
                    continue
                if all(c == selector[0] for c in selector):
                    continue

                try:
                    sel_int = int(selector, 16)
                    if sel_int < 0x10000000:  # Reasonable selector range
                        continue
                except ValueError:
                    continue

                # Now check for dispatch patterns after the selector
                # Look for common patterns within next 20 bytes
                pattern_found = False
                pattern_confidence = 0.5  # Medium confidence for pattern-based

                # Check next 20 bytes (10 instructions) for dispatch patterns
                for j in range(i + 10, min(len(bytecode), i + 30), 2):
                    opcode = bytecode[j : j + 2]

                    # Pattern 1: EQ (0x14) followed by ISZERO (0x15) or JUMPI (0x57)
                    if opcode == "14":  # EQ
                        # Check next opcode
                        if j + 2 < len(bytecode):
                            next_op = bytecode[j + 2 : j + 4]
                            if next_op in ["15", "57"]:  # ISZERO or JUMPI
                                pattern_found = True
                                pattern_confidence = 0.7
                                break

                    # Pattern 2: Direct JUMPI after selector (less common but possible)
                    elif opcode == "57":  # JUMPI
                        pattern_found = True
                        pattern_confidence = 0.6
                        break

                    # Pattern 3: DUP1 (0x80) before PUSH4 - check before current position
                    # This handles DUP1 PUSH4 selector patterns
                    elif j == i + 10 and i >= 2:  # First opcode after selector
                        prev_op = bytecode[i - 2 : i]
                        if prev_op == "80":  # DUP1
                            pattern_found = True
                            pattern_confidence = 0.65
                            break

                if pattern_found:
                    context_start = max(0, i - 64)
                    context_end = min(len(bytecode), i + 128)
                    context = bytecode[context_start:context_end]

                    selectors.append(
                        {
                            "selector": selector,
                            "position": i + 2,
                            "context": context,
                            "confidence": pattern_confidence,
                            "source": "dispatch_pattern",
                        }
                    )

        # Deduplicate by selector, keeping highest confidence
        unique_selectors = {}
        for sel in selectors:
            selector = sel["selector"]
            if (
                selector not in unique_selectors
                or sel["confidence"] > unique_selectors[selector]["confidence"]
            ):
                unique_selectors[selector] = sel

        result = list(unique_selectors.values())
        if max_selectors > 0:
            result = result[:max_selectors]
        return result

    def lookup_signature_candidates(self, selector: str) -> List[Dict[str, Any]]:
        """Look up possible signatures for a selector from 4byte.directory.

        Args:
            selector: Function selector (4-byte hex string)

        Returns:
            List of signature candidates with metadata
        """
        signatures = self.signature_lookup.lookup_signatures(selector)

        candidates = []
        for sig_dict in signatures:
            # sig_dict is a dictionary with keys like "text_signature", "source", etc.
            text_sig = sig_dict.get("text_signature", "")
            source = sig_dict.get("source", "unknown")

            # Parse signature to extract function name and parameters
            # Format: "functionName(type1,type2,...)"
            if text_sig and "(" in text_sig and ")" in text_sig:
                name_part = text_sig.split("(")[0]
                params_part = text_sig.split("(")[1].rstrip(")")
                param_types = (
                    [p.strip() for p in params_part.split(",")] if params_part else []
                )

                candidates.append(
                    {
                        "signature": text_sig,
                        "function_name": name_part,
                        "parameter_types": param_types,
                        "source": source,
                        "confidence": 0.5,  # Placeholder, will be updated by classifier
                        "original_dict": sig_dict,  # Keep original for debugging
                    }
                )

        # If no signatures found, create a placeholder
        if not candidates:
            candidates.append(
                {
                    "signature": f"unknown_{selector}()",
                    "function_name": f"unknown_{selector}",
                    "parameter_types": [],
                    "source": "placeholder",
                    "confidence": 0.1,
                }
            )

        return candidates

    def extract_bytecode_features(
        self, bytecode: str, selector: str, position: int
    ) -> Dict[str, Any]:
        """Extract features from bytecode for model inference.

        Args:
            bytecode: Contract bytecode (hex string)
            selector: Function selector
            position: Position of selector in hex characters (not bytes)

        Returns:
            Dictionary of features for model input
        """
        # Clean bytecode
        bytecode = clean_bytecode(bytecode)

        # Extract context around selector
        context = extract_selector_context(bytecode, position, context_size=500)

        # Convert context to tokenized features
        # Simple approach: convert hex to integers (0-255)
        tokens = []
        for i in range(0, len(context), 2):
            if i + 1 < len(context):
                hex_byte = context[i : i + 2]
                try:
                    token = int(hex_byte, 16)
                    tokens.append(token)
                except ValueError:
                    tokens.append(0)  # Padding

        # Pad or truncate to fixed length
        max_length = 512
        PADDING_TOKEN = 256
        num_real_tokens = len(tokens)
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
            num_real_tokens = max_length
        else:
            tokens = tokens + [PADDING_TOKEN] * (max_length - len(tokens))

        # Convert to tensor
        tokens_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(
            0
        )  # Add batch dimension

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.zeros(max_length, dtype=torch.float)
        attention_mask[:num_real_tokens] = 1.0
        attention_mask = attention_mask.unsqueeze(0)  # Add batch dimension

        if self.use_cuda and torch.cuda.is_available():
            from abi_reconstructor.device import batch_to_device
            moved = batch_to_device(
                {"tokens": tokens_tensor, "mask": attention_mask}
            )
            tokens_tensor = moved["tokens"]
            attention_mask = moved["mask"]

        return {
            "bytecode_tokens": tokens_tensor,
            "attention_mask": attention_mask,
            "selector": selector,
            "position": position,
            "context": context,
        }

    def classify_function_name(
        self, features: Dict[str, Any], candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Use function classifier to select the best signature candidate.

        Args:
            features: Bytecode features
            candidates: List of signature candidates

        Returns:
            Best candidate with updated confidence from classifier
        """
        # Prepare input for function classifier
        bytecode_tokens = features["bytecode_tokens"]
        attention_mask = features.get("attention_mask")

        # Get predictions from function classifier
        with torch.no_grad():
            outputs = self.function_classifier(bytecode_tokens, attention_mask)
            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=-1)

            # Get top predictions
            top_probs, top_indices = torch.topk(probs, k=min(10, probs.size(-1)))
            top_probs = top_probs[0].cpu().numpy().tolist()
            top_indices = top_indices[0].cpu().numpy().tolist()

        # Function name vocabulary from training
        # This should match the vocabulary used during training
        # The classifier has 47 output classes, so we need 47 items
        # Last item might be for "unknown" or "other"
        function_name_vocab = [
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
            "unknown",  # 47th class for unknown functions
        ]

        # If we have candidates from 4byte.directory, use classifier to rank them
        if candidates:
            best_candidate = None
            best_confidence = 0.0

            for candidate in candidates:
                function_name = candidate["function_name"].lower()
                confidence = candidate["confidence"]
                source = candidate.get("source", "")

                # If this is a placeholder candidate (unknown_...), use classifier to predict function name
                if source == "placeholder" and function_name.startswith("unknown_"):
                    # Use classifier to predict function name
                    if len(top_indices) > 0:
                        best_idx = top_indices[0]
                        best_prob = top_probs[0]

                        if best_idx < len(function_name_vocab):
                            predicted_name = function_name_vocab[best_idx]
                            # Update candidate with predicted name
                            candidate["function_name"] = predicted_name
                            candidate["signature"] = f"{predicted_name}()"
                            candidate["source"] = "classifier"
                            confidence = best_prob
                        else:
                            # Keep placeholder but update confidence
                            confidence = (
                                best_prob * 0.5
                            )  # Lower confidence for unknown function
                else:
                    matched = False
                    for i, vocab_name in enumerate(function_name_vocab):
                        if function_name == vocab_name.lower():
                            if i in top_indices:
                                idx = list(top_indices).index(i)
                                classifier_confidence = top_probs[idx]
                                confidence = classifier_confidence
                                matched = True
                                break

                    if not matched:
                        # No match in vocabulary, use original confidence
                        pass

                candidate["confidence"] = float(min(confidence, 1.0))

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_candidate = candidate

            return best_candidate
        else:
            # No candidates from 4byte.directory, use classifier to predict function name
            if len(top_indices) > 0:
                best_idx = top_indices[0]
                best_prob = top_probs[0]

                if best_idx < len(function_name_vocab):
                    predicted_name = function_name_vocab[best_idx]
                else:
                    predicted_name = f"func_{best_idx}"

                # Create a candidate with the predicted function name
                return {
                    "signature": f"{predicted_name}()",
                    "function_name": predicted_name,
                    "parameter_types": [],
                    "source": "classifier",
                    "confidence": float(best_prob),
                }
            else:
                return None

    def predict_parameters(
        self, features: Dict[str, Any], function_name: str
    ) -> Dict[str, Any]:
        """Use parameter predictor to guess function parameters.

        Args:
            features: Bytecode features
            function_name: Selected function name

        Returns:
            Parameter predictions
        """
        # Map function name to index using vocabulary
        # Normalize function name (lowercase for matching, but keep original for display)
        normalized_name = function_name.strip()

        # For matching, we need to handle case-insensitive comparison
        # Build lowercase version of vocabulary for matching
        if not hasattr(self, "_lowercase_vocab"):
            self._lowercase_vocab = {
                k.lower(): v for k, v in self.function_to_idx.items()
            }

        if self.verbose:
            logger.info(f"  Mapping function name: '{function_name}'")

        # Try to find exact match first (case-insensitive)
        lowercase_name = normalized_name.lower()
        if lowercase_name in self._lowercase_vocab:
            function_idx = self._lowercase_vocab[lowercase_name]
        else:
            found = False
            for vocab_lower, idx in self._lowercase_vocab.items():
                if lowercase_name == vocab_lower:
                    function_idx = idx
                    found = True
                    if self.verbose:
                        original_vocab_name = next(
                            k for k, v in self.function_to_idx.items() if v == idx
                        )
                        logger.info(f"    Mapped to '{original_vocab_name}' (idx: {idx})")
                    break

            if not found:
                # Use "unknown" index
                function_idx = self.function_to_idx.get("unknown", 0)
                if self.verbose:
                    logger.info(f"    No match, using 'unknown' (idx: {function_idx})")

        function_name_idx = torch.tensor([function_idx], dtype=torch.long)
        if self.use_cuda and torch.cuda.is_available():
            function_name_idx = function_name_idx.cuda()

        # Get predictions from parameter predictor
        with torch.no_grad():
            outputs = self.parameter_predictor(
                features["bytecode_tokens"],
                function_ids=function_name_idx,
                attention_mask=features.get("attention_mask"),
            )

        # Parse predictions
        type_logits = outputs["type_logits"]
        mask_logits = outputs["mask_logits"]
        count_logits = outputs.get("count_logits")

        # Get type predictions
        type_probs = torch.softmax(type_logits, dim=-1)
        max_type_probs, type_preds = torch.max(type_probs, dim=-1)

        # Get mask predictions (parameter existence)
        mask_probs = torch.sigmoid(mask_logits)
        mask_preds = (mask_probs > self.min_parameter_confidence).float()

        # Get count prediction
        if count_logits is not None:
            count_probs = torch.softmax(count_logits, dim=-1)
            _, count_pred = torch.max(count_probs, dim=-1)
            param_count = count_pred.item()
        else:
            # Estimate from mask predictions
            param_count = int(torch.sum(mask_preds > 0.5).item())

        # Check for function-specific type priors (case-insensitive lookup)
        # Build lowercase version of priors for matching
        if not hasattr(self, "_lowercase_priors"):
            self._lowercase_priors = {
                k.lower(): v for k, v in self.FUNCTION_TYPE_PRIORS.items()
            }

        prior_types = self._lowercase_priors.get(lowercase_name, None)

        # Overloaded standard functions (e.g. balanceOf, safeTransferFrom): pick
        # the candidate whose arity matches the predicted parameter count, falling
        # back to the most-common variant when no candidate matches.
        if prior_types is None:
            overloads = self.FUNCTION_TYPE_PRIOR_OVERLOADS.get(lowercase_name)
            if overloads:
                prior_types = next(
                    (c for c in overloads if len(c) == param_count), overloads[0]
                )

        # Map type indices to type names using vocabulary
        parameters = []

        if prior_types:
            # Use function-specific priors when available
            for i, prior_type in enumerate(prior_types):
                # Map type string to type index
                type_idx = self.type_to_idx.get(
                    prior_type, self.type_to_idx.get("unknown", 23)
                )
                parameters.append(
                    {
                        "index": i,
                        "type": prior_type,
                        "confidence": 0.95,  # High confidence for known patterns
                        "exists": True,
                        "source": "function_prior",
                    }
                )
            param_count = len(prior_types)
        else:
            # Use ML predictions for unknown functions
            for i in range(min(param_count, len(type_preds[0]))):
                if i < mask_preds.size(1) and mask_preds[0, i] > 0.5:
                    type_idx = type_preds[0, i].item()
                    type_name = self.idx_to_type.get(type_idx, "unknown")
                    confidence = max_type_probs[0, i].item()

                    parameters.append(
                        {
                            "index": i,
                            "type": type_name,
                            "confidence": confidence,
                            "exists": True,
                        }
                    )

        return {
            "parameter_count": param_count,
            "parameters": parameters,
            "type_confidences": type_probs.cpu()
            .numpy()
            .tolist(),  # Full probability distributions, not just max
            "mask_confidences": mask_probs.cpu().numpy().tolist(),
        }

    def reconstruct_function(
        self,
        bytecode: str,
        selector: str,
        position: int,
    ) -> Dict[str, Any]:
        """Reconstruct a single function from bytecode.

        Args:
            bytecode: Contract bytecode (hex string)
            selector: Function selector
            position: Position of selector in bytecode

        Returns:
            Reconstructed function with both classifier+4byte and ML parameter predictions
        """
        # Extract features
        features = self.extract_bytecode_features(bytecode, selector, position)

        # Look up signature candidates
        candidates = self.lookup_signature_candidates(selector)

        # Classify to select best candidate
        best_candidate = self.classify_function_name(features, candidates)

        if not best_candidate:
            return {
                "selector": selector,
                "success": False,
                "error": "No suitable signature candidate found",
            }

        # Always predict parameters for comparison/validation
        parameter_predictions = self.predict_parameters(
            features, best_candidate["function_name"]
        )

        # Build classifier + 4byte prediction (includes parameter info from 4byte if available)
        classifier_4byte_prediction = self._build_classifier_4byte_prediction(
            best_candidate, parameter_predictions
        )

        # Build ML parameter prediction (always shows ML results)
        ml_parameter_prediction = self._build_ml_parameter_prediction(
            parameter_predictions, best_candidate["function_name"]
        )

        return {
            "selector": selector,
            "classifier_4byte_prediction": classifier_4byte_prediction,
            "ml_parameter_prediction": ml_parameter_prediction,
            "success": True,
            "metadata": {
                "position": position,
                "source": best_candidate["source"],
            },
        }

    def _build_classifier_4byte_prediction(
        self, best_candidate: Dict[str, Any], parameter_predictions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build classifier + 4byte prediction including parameters from 4byte if available."""
        candidate_param_types = best_candidate.get("parameter_types", [])
        candidate_has_params = len(candidate_param_types) > 0

        # Build signature with parameters
        if candidate_has_params:
            # Use 4byte parameter types if available
            param_str = ", ".join(candidate_param_types)
            parameters = []
            for i, param_type in enumerate(candidate_param_types):
                parameters.append(
                    {
                        "index": i,
                        "type": param_type,
                        "confidence": 0.9,  # High confidence for 4byte.directory
                        "exists": True,
                        "source": "4byte.directory",
                    }
                )
            param_count = len(candidate_param_types)
        else:
            # Use ML predictions when no 4byte parameters
            parameters = parameter_predictions["parameters"]
            param_count = parameter_predictions["parameter_count"]
            param_str = ", ".join([p["type"] for p in parameters])
            # Mark parameters as from prediction
            for p in parameters:
                p["source"] = "prediction"

        signature = f"{best_candidate['function_name']}({param_str})"

        return {
            "function_name": best_candidate["function_name"],
            "signature": signature,
            "confidence": best_candidate["confidence"],
            "source": best_candidate["source"],
            "parameter_count": param_count,
            "parameters": parameters,
        }

    def _build_ml_parameter_prediction(
        self, parameter_predictions: Dict[str, Any], function_name: str
    ) -> Dict[str, Any]:
        """Build ML parameter prediction with full confidence information."""
        parameters = parameter_predictions["parameters"]
        param_count = parameter_predictions["parameter_count"]
        type_confidences = parameter_predictions.get("type_confidences", [])
        mask_confidences = parameter_predictions.get("mask_confidences", [])

        # Extract ML predicted parameter types for display
        ml_predicted_types = []
        type_names = [
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
            "unknown",
        ]

        if type_confidences and len(type_confidences) > 0:
            for pos in range(min(param_count, len(type_confidences[0]), len(mask_confidences[0]) if mask_confidences else 0)):
                if pos < len(mask_confidences[0]) and mask_confidences[0][pos] > 0.5:
                    type_probs = type_confidences[0][pos]
                    if len(type_probs) > 0:
                        max_prob = max(type_probs)
                        type_idx = type_probs.index(max_prob)
                        type_name = (
                            type_names[type_idx]
                            if type_idx < len(type_names)
                            else "unknown"
                        )
                        ml_predicted_types.append(
                            {
                                "position": pos,
                                "type": type_name,
                                "confidence": max_prob,
                            }
                        )

        return {
            "method": "Parameter Prediction Model",
            "function_name": function_name,
            "parameter_count": param_count,
            "parameters": [
                {
                    "index": p["index"],
                    "type": p["type"],
                    "confidence": p["confidence"],
                    "exists": True,
                    "source": "prediction",
                }
                for p in parameters
            ],
            "type_confidences": type_confidences,
            "mask_confidences": mask_confidences,
            "ml_predicted_types": ml_predicted_types,
        }

    def reconstruct_abi(
        self,
        bytecode: str,
        selector_hex: Optional[str] = None,
        return_all_predictions: bool = False,
        max_selectors: int = 0,  # 0 means unlimited
    ) -> Dict[str, Any]:
        """Reconstruct ABI from bytecode.

        Args:
            bytecode: Contract bytecode (hex string or file path)
            selector_hex: Optional specific selector to reconstruct
            return_all_predictions: Whether to return all predictions or just best
            max_selectors: Maximum number of selectors to process (when selector_hex is None)

        Returns:
            Dictionary with reconstructed ABI
        """
        import time

        start_time = time.time()

        # Check if bytecode is a file path
        if os.path.exists(bytecode):
            with open(bytecode, "r") as f:
                bytecode = f.read().strip()

        # Clean bytecode
        bytecode = clean_bytecode(bytecode)

        if selector_hex:
            # Single selector reconstruction
            selector_hex = selector_hex.lower().replace("0x", "")
            if len(selector_hex) != 8:
                return {
                    "success": False,
                    "error": f"Invalid selector length: {selector_hex}. Expected 8 hex characters.",
                }

            # Find selector in bytecode
            position = bytecode.find(selector_hex)
            if position == -1:
                return {
                    "success": False,
                    "error": f"Selector {selector_hex} not found in bytecode",
                }

            result = self.reconstruct_function(bytecode, selector_hex, position)

            # Check if result has new format (classifier_4byte_prediction)
            if result["success"] and "classifier_4byte_prediction" in result:
                # Return new format
                return {
                    "success": True,
                    "classifier_4byte_prediction": result[
                        "classifier_4byte_prediction"
                    ],
                    "ml_parameter_prediction": result["ml_parameter_prediction"],
                    "metadata": {
                        "bytecode_size_bytes": len(bytecode) // 2,
                        "reconstruction_time_seconds": time.time() - start_time,
                        "selector_provided": True,
                    },
                }
            else:
                # Return old format for backward compatibility
                return {
                    "success": result["success"],
                    "reconstructed_signature": result if result["success"] else None,
                    "error": result.get("error"),
                    "metadata": {
                        "bytecode_size_bytes": len(bytecode) // 2,
                        "reconstruction_time_seconds": time.time() - start_time,
                        "selector_provided": True,
                    },
                }
        else:
            # Complete ABI reconstruction - extract all selectors
            selectors = self.extract_selectors(bytecode, max_selectors=max_selectors)

            if not selectors:
                return {
                    "success": False,
                    "error": "No function selectors found in bytecode",
                    "metadata": {
                        "bytecode_size_bytes": len(bytecode) // 2,
                        "reconstruction_time_seconds": time.time() - start_time,
                    },
                }

            # Reconstruct each selector
            reconstructed_functions = []
            errors = []

            for sel_info in selectors:
                try:
                    result = self.reconstruct_function(
                        bytecode,
                        sel_info["selector"],
                        sel_info["position"],
                    )

                    if result["success"]:
                        # Check if result has new format
                        if "classifier_4byte_prediction" in result:
                            reconstructed_functions.append(result)
                        else:
                            # Old format - convert to new format
                            reconstructed_functions.append(
                                {
                                    "classifier_4byte_prediction": result.get(
                                        "reconstructed_signature", {}
                                    ),
                                    "ml_parameter_prediction": {
                                        "method": "Parameter Prediction Model",
                                        "function_name": result.get(
                                            "reconstructed_signature", {}
                                        ).get("function_name", "unknown"),
                                        "parameter_count": result.get(
                                            "reconstructed_signature", {}
                                        ).get("parameter_count", 0),
                                        "parameters": result.get(
                                            "reconstructed_signature", {}
                                        ).get("parameters", []),
                                        "type_confidences": [],
                                        "mask_confidences": [],
                                        "ml_predicted_types": [],
                                    },
                                }
                            )
                    else:
                        errors.append(
                            {
                                "selector": sel_info["selector"],
                                "error": result.get("error", "Unknown error"),
                            }
                        )
                except Exception as e:
                    errors.append(
                        {
                            "selector": sel_info["selector"],
                            "error": str(e),
                        }
                    )

            return {
                "success": len(reconstructed_functions) > 0,
                "abi": reconstructed_functions,
                "errors": errors,
                "metadata": {
                    "bytecode_size_bytes": len(bytecode) // 2,
                    "reconstruction_time_seconds": time.time() - start_time,
                    "selectors_found": len(selectors),
                    "functions_reconstructed": len(reconstructed_functions),
                    "errors_count": len(errors),
                },
            }

    def reconstruct_complete_abi(
        self,
        bytecode: str,
        use_4byte: bool = True,
        max_selectors: int = 0,  # 0 means unlimited
        min_confidence: float = 0.1,
    ) -> Dict[str, Any]:
        """Reconstruct complete ABI from bytecode (alias for reconstruct_abi).

        Args:
            bytecode: Contract bytecode
            use_4byte: Whether to use 4byte.directory lookup
            max_selectors: Maximum number of selectors to process
            min_confidence: Minimum confidence threshold

        Returns:
            Dictionary with reconstructed ABI
        """
        # Update configuration
        self.min_selector_confidence = min_confidence

        return self.reconstruct_abi(bytecode, max_selectors=max_selectors)
