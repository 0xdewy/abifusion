"""Tests for the rule-based OfflineABI, BytecodeParser, and FourByteDatabase.

These cover the primary (non-ML) reconstruction path in ``reconstructor.py``,
which previously had no dedicated tests. All cases run offline: the database
uses an in-memory SQLite seeded with standard signatures, and the 4byte API
fallback is patched out.
"""

from unittest.mock import patch

import pytest

from tooling.abifusion_legacy.database import FourByteDatabase
from tooling.abifusion_legacy.reconstructor import OfflineABI, BytecodeParser

# Selectors present in the shared `sample_bytecode` fixture (all ERC20 standard).
TRANSFER = "a9059cbb"
APPROVE = "095ea7b3"
BALANCE_OF = "70a08231"


@pytest.fixture
def no_network():
    """Prevent any 4byte.directory API calls during reconstruction."""
    with patch.object(FourByteDatabase, "fetch_from_4byte_api", return_value=[]):
        yield


class TestBytecodeParser:
    def test_extracts_known_selectors(self, sample_bytecode):
        parser = BytecodeParser()
        selectors = parser.extract_selectors(sample_bytecode)
        found = {s.selector for s in selectors}
        assert {TRANSFER, APPROVE, BALANCE_OF}.issubset(found)

    def test_empty_bytecode_returns_no_selectors(self):
        parser = BytecodeParser()
        assert parser.extract_selectors("") == []

    def test_selectors_carry_position_and_confidence(self, sample_bytecode):
        parser = BytecodeParser()
        selectors = parser.extract_selectors(sample_bytecode)
        assert selectors, "expected at least one selector"
        for sel in selectors:
            assert sel.position >= 0
            assert 0.0 <= sel.confidence <= 1.0


class TestFourByteDatabase:
    def test_standard_signature_resolves_offline(self):
        db = FourByteDatabase(db_path=":memory:")
        sigs = db.lookup(TRANSFER, fetch_api=False)
        assert any(s.text_signature == "transfer(address,uint256)" for s in sigs)
        db.close()

    def test_unknown_selector_without_api_is_empty(self):
        db = FourByteDatabase(db_path=":memory:")
        assert db.lookup("deadbeef", fetch_api=False) == []
        db.close()


class TestOfflineABI:
    def test_reconstruct_abi_on_erc20_bytecode(self, sample_bytecode, no_network):
        recon = OfflineABI(db_path=":memory:")
        result = recon.reconstruct_abi(sample_bytecode)

        assert result["success"] is True
        names = {
            f.get("function_name") or f.get("name") or f.get("signature", "")
            for f in result["functions"]
        }
        # The standard ERC20 selectors should resolve to named functions.
        joined = " ".join(str(n) for n in names)
        assert "transfer" in joined
        assert result["metadata"]["functions_reconstructed"] >= 1
        recon.close()

    def test_reconstruct_abi_result_is_json_serializable(
        self, sample_bytecode, no_network
    ):
        recon = OfflineABI(db_path=":memory:")
        result = recon.reconstruct_abi(sample_bytecode)
        text = recon.to_json(result)
        assert isinstance(text, str)
        assert "functions" in text
        recon.close()

    def test_reconstruct_function_known_selector(self, sample_bytecode, no_network):
        recon = OfflineABI(db_path=":memory:")
        func = recon.reconstruct_function(sample_bytecode, TRANSFER)
        assert isinstance(func, dict)
        recon.close()
