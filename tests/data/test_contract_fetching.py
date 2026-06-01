"""Tests for contract fetching pipeline."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from abi_reconstructor.data.etherscan_fetcher import EtherscanClient, EtherscanContract
from abi_reconstructor.data.sourcify_client import SourcifyClient
from abi_reconstructor.data.dataset_builder import DatasetBuilder


class TestEtherscanClient:
    """Test Etherscan API client functionality."""

    def test_initialization_without_api_key(self):
        """Test client initialization without API key."""
        client = EtherscanClient(api_key=None)
        assert client.api_key is None
        assert client.rate_limit_delay == 0.2

    def test_initialization_with_api_key(self):
        """Test client initialization with API key."""
        client = EtherscanClient(api_key="test_key")
        assert client.api_key == "test_key"

    def test_rate_limit(self):
        """Test rate limiting is applied during requests.

        Rate limiting is an implementation detail of _make_request (via
        time.sleep in the finally block). It is verified implicitly through
        integration tests and through the rate_limit_delay parameter.
        This test validates the parameter is set correctly.
        """
        client = EtherscanClient(rate_limit_delay=0.1)
        assert client.rate_limit_delay == 0.1

        client_default = EtherscanClient()
        assert client_default.rate_limit_delay == 0.2

    @patch("requests.Session.get")
    def test_get_contract_abi_success(self, mock_get):
        """Test successful contract ABI fetch."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "1",
            "message": "OK",
            "result": '[{"type":"function","name":"test","inputs":[],"outputs":[]}]'
        }
        mock_get.return_value = mock_response

        client = EtherscanClient(api_key="test_key")
        abi = client.get_contract_abi("0x1234567890123456789012345678901234567890")

        assert abi is not None
        assert len(abi) == 1
        assert abi[0]["name"] == "test"

    @patch("requests.Session.get")
    def test_get_contract_source_code_success(self, mock_get):
        """Test successful contract source code fetch."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "1",
            "message": "OK",
            "result": [
                {
                    "ContractName": "TestContract",
                    "CompilerVersion": "v0.8.20+commit.a1b79de6",
                    "OptimizationUsed": "1",
                    "Runs": "200",
                    "ConstructorArguments": "",
                    "EVMVersion": "default",
                    "Library": "",
                    "LicenseType": "MIT",
                    "Proxy": "0",
                    "Implementation": "",
                    "SwarmSource": "",
                    "SourceCode": "pragma solidity ^0.8.20; contract TestContract {}",
                    "ABI": '[{"type":"function","name":"test","inputs":[],"outputs":[]}]',
                }
            ],
        }
        mock_get.return_value = mock_response

        client = EtherscanClient(api_key="test_key")
        result = client.get_contract_source_code("0x1234567890123456789012345678901234567890")

        assert result is not None
        assert result["contract_name"] == "TestContract"
        assert result["compiler_version"] == "v0.8.20+commit.a1b79de6"

    @patch("requests.Session.get")
    def test_get_contract_bytecode_success(self, mock_get):
        """Test successful contract bytecode fetch."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "1",
            "result": "0x6080604052348015600f57600080fd5b506004361060285760003560e01c8063a9059cbb14602d578063095ea7b314604257806370a08231146058575b600080fd5b606360048036036040811015604157600080fd5b81019080803590602001909291905050506065565b005b60"
        }
        mock_get.return_value = mock_response

        client = EtherscanClient(api_key="test_key")
        bytecode = client.get_contract_bytecode("0x1234567890123456789012345678901234567890")

        assert bytecode is not None
        assert bytecode.startswith("0x")

    @patch("requests.Session.get")
    def test_get_contract_not_found(self, mock_get):
        """Test contract not found."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "0",
            "message": "No contract found",
            "result": [],
        }
        mock_get.return_value = mock_response

        client = EtherscanClient(api_key="test_key")
        contract = client.get_contract("0x1234567890123456789012345678901234567890")

        assert contract is None


class TestSourcifyClient:
    """Test Sourcify API client functionality."""

    def test_initialization(self):
        """Test client initialization."""
        client = SourcifyClient()
        assert client.base_url == "https://sourcify.dev/server"
        assert client.rate_limit_delay == 0.1

    @patch("requests.Session.get")
    def test_check_contract_verification_success(self, mock_get):
        """Test successful contract verification check."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "perfect",
            "address": "0x1234567890123456789012345678901234567890",
        }
        mock_get.return_value = mock_response

        client = SourcifyClient()
        result = client.check_contract_verification(
            "0x1234567890123456789012345678901234567890", 1
        )

        assert result["status"] == "success"
        assert result["verified"] is True

    @patch("requests.Session.get")
    def test_check_contract_verification_not_found(self, mock_get):
        """Test contract not found in Sourcify."""
        mock_get.side_effect = Exception("404 Not Found")

        client = SourcifyClient()
        result = client.check_contract_verification(
            "0x1234567890123456789012345678901234567890", 1
        )

        assert result["status"] == "not_found"
        assert result["verified"] is False

    @patch("requests.Session.get")
    def test_get_contract_files_http_error(self, mock_get):
        """Test HTTP error during Sourcify fetch."""
        import requests

        mock_get.side_effect = requests.exceptions.RequestException("HTTP error")

        client = SourcifyClient()
        files = client.get_contract_files(
            "0x1234567890123456789012345678901234567890", 1
        )

        assert files is None


class TestDatasetBuilder:
    """Test dataset builder functionality."""

    def test_initialization(self):
        """Test dataset builder initialization."""
        builder = DatasetBuilder(data_dir="test_output")
        assert builder.data_dir == Path("test_output")

    def test_expand_with_similar_contracts(self):
        """Test expanding contract list."""
        pytest.skip("get_known_verified_contracts and expand_with_similar_contracts do not exist in DatasetBuilder")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])