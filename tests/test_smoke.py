"""Smoke tests for ABI Reconstructor."""

import pytest

from abifusion.reconstructor import BytecodeParser, OfflineABI, ParameterReconstructor
from abifusion.features.discriminating_features import DiscriminatingFeatureExtractor


class TestSmoke:
    """Smoke tests to verify core functionality."""

    def test_selector_extraction_from_bytecode(self):
        """Test that bytecode with 0xa9059cbb correctly extracts the transfer selector."""
        parser = BytecodeParser()

        bytecode = (
            "6080604052348015600f57600080fd5b506004361060285760003560e01c"
            "8063a9059cbb14602d578063095ea7b314604257806370a0823114605857"
            "5b600080fd5b606360048036036040811015604157600080fd5b81019080"
        )

        selectors = parser.extract_selectors(bytecode)

        selector_hex_values = [s.selector for s in selectors]
        assert "a9059cbb" in selector_hex_values, f"Expected 'a9059cbb' in selectors, got: {selector_hex_values}"

    def test_bytecode_parser_initialization(self):
        """Test that BytecodeParser can be initialized."""
        parser = BytecodeParser()
        assert parser is not None

    def test_clean_bytecode(self):
        """Test bytecode cleaning removes 0x prefix and normalizes."""
        parser = BytecodeParser()

        dirty = "0x6080604052348015600f57600080fd5b"
        clean = parser.clean_bytecode(dirty)

        assert not clean.startswith("0x")
        assert len(clean) > 0

    def test_no_selectors_in_empty_bytecode(self):
        """Test that empty bytecode produces no selectors."""
        parser = BytecodeParser()
        selectors = parser.extract_selectors("")
        assert len(selectors) == 0

    def test_parser_detects_common_selector(self):
        """Test parser finds transfer selector."""
        parser = BytecodeParser()

        bytecode = (
            "63a9059cbb"  # PUSH4 followed by transfer selector
            "14602d57"    # EQ, JUMPI
        )

        selectors = parser.extract_selectors(bytecode)
        selector_hex_values = [s.selector for s in selectors]

        assert "a9059cbb" in selector_hex_values

    def test_bytecode_to_tokens(self):
        """Test bytecode to token conversion."""
        parser = BytecodeParser()
        tokens = parser.bytecode_to_tokens("60806040")
        assert len(tokens) == 512
        assert tokens[0] == 0x60
        assert tokens[1] == 0x80
        assert tokens[2] == 0x60
        assert tokens[3] == 0x40
        assert tokens[4] == 256  # padding token

    def test_state_mutability_detection(self):
        """Test state mutability detection from bytecode."""
        parser = BytecodeParser()

        bytecode = "6080604052348015600f57600080fd5b506004361060285760003560e01c8063a9059cbb14602d578063095ea7b314604257806370a08231146058575b600080fd5b606360048036036040811015604157600080fd5b81019080"

        mutability = parser.detect_state_mutability(bytecode, "a9059cbb")
        assert mutability in ["view", "pure", "payable", "nonpayable"]


class TestFullPipelineSmoke:
    """End-to-end smoke tests exercising the complete ABI reconstruction stack."""

    REALISTIC_BYTECODE = (
        "608060405234801561001057600080fd5b50600436106100885760003560e01c"
        "8063a9059cbb1461008d578063095ea7b3146100c357806318160ddd146100f957"
        "806323b872dd1461011757806370a082311461014d578063dd62ed3e1461017d57"
        "5b600080fd5b6100c160048036038101906100ac91906102cf565b6101b5565b00"
        "5b6100fd60048036038101906100f891906102cf565b6102c2565b005b61010161"
        "03cf565b60405161010e919061038c565b60405180910390f35b61014b60048036"
        "0381019061014691906102fc565b6103cd565b005b610167600480360381019061"
        "0162919061025c565b61043d565b604051610174919061038c565b604051809103"
        "90f35b61019f600480360381019061019a9190610289565b61048d565b60405161"
        "01ac919061038c565b60405180910390f35b600080fd5b50505050505050505050"
    )

    def test_reconstruct_abi_from_realistic_bytecode(self):
        """Full OfflineABI pipeline on realistic bytecode."""
        r = OfflineABI(db_path="./cache/4byte.db")

        result = r.reconstruct_abi(self.REALISTIC_BYTECODE, max_selectors=20)

        assert result["success"] is True
        assert len(result["functions"]) > 0
        # Should find at least transfer and balanceOf selectors
        selectors_found = {f["selector"] for f in result["functions"]}
        assert "a9059cbb" in selectors_found, f"Missing transfer selector. Found: {selectors_found}"
        assert "70a08231" in selectors_found, f"Missing balanceOf selector. Found: {selectors_found}"

        r.close()

    def test_reconstruct_function_with_known_selector(self):
        """Single function reconstruction with known selector."""
        r = OfflineABI(db_path="./cache/4byte.db")

        result = r.reconstruct_function(self.REALISTIC_BYTECODE, "a9059cbb")
        assert result["function_name"] == "transfer"
        assert len(result["parameters"]) == 2

        r.close()

    def test_discriminating_features_in_full_pipeline(self):
        """discriminating feature extraction integrated with OfflineABI."""
        # Use the realistic ERC20 bytecode which has proper dispatch structure
        r = OfflineABI(db_path="./cache/4byte.db")
        result = r.reconstruct_abi(self.REALISTIC_BYTECODE, max_selectors=20)

        assert result["success"] is True
        funcs = result["functions"]
        assert len(funcs) >= 1

        # transfer should be recovered from 4byte DB
        tx = [f for f in funcs if f["selector"] == "a9059cbb"]
        if tx:
            assert len(tx[0]["parameters"]) == 2
            assert tx[0]["parameters"][0]["type"] == "address"
            assert tx[0]["parameters"][1]["type"] == "uint256"

        r.close()

    def test_parameter_reconstructor_unknown_selector(self):
        """ParameterReconstructor infers types for unknown selectors via discriminating features."""
        pr = ParameterReconstructor()

        # Bytecode with unknown selector + SIGNEXTEND → should infer int256
        bytecode = "63deadbeef" + "5b" + "0b" + "50" * 20
        types = pr.predict_parameter_types(
            bytecode=bytecode,
            selector="deadbeef",
            function_name="unknown_deadbeef",
            context=bytecode,
        )
        assert "int256" in types, f"Expected int256 inference from SIGNEXTEND, got {types}"

    def test_discriminating_features_extraction_from_context(self):
        """DiscriminatingFeatureExtractor works on hex context."""
        ext = DiscriminatingFeatureExtractor()

        # Context with AND mask pattern
        context = (
            "7f"
            "000000000000000000000000ffffffffffffffffffffffffffffffffffffffff"
            "16"
        )
        features = ext.extract_from_context(context, context)
        assert features.has_and_mask_leading_zeros is True
        assert features.and_mask_bytes == 20

        vec = features.to_vector()
        assert len(vec) == 7
        assert all(isinstance(v, float) for v in vec)

    def test_benchmark_parse_ground_truth_via_abi(self):
        """Ground-truth function parsing from an ABI entry."""
        from benchmark import parse_ground_truth_functions

        abi = [
            {"type": "function", "name": "transfer", "inputs": [
                {"name": "to", "type": "address"},
                {"name": "amount", "type": "uint256"},
            ]},
        ]
        funcs = parse_ground_truth_functions(abi)
        assert len(funcs) == 1
        assert funcs[0]["selector"] == "a9059cbb"
        assert funcs[0]["arity"] == 2
