"""Common test fixtures for ABI Reconstructor tests."""

import pytest
import tempfile
import os


@pytest.fixture
def sample_bytecode():
    """Sample ERC20-like bytecode for testing."""
    return (
        "6080604052348015600f57600080fd5b506004361060285760003560e01c"
        "8063a9059cbb14602d578063095ea7b314604257806370a0823114605857"
        "5b600080fd5b606360048036036040811015604157600080fd5b81019080"
        "80359060200190929190803590602001909291905050506065565b005b60"
        "6360048036036040811015605757600080fd5b8101908080359060200190"
        "929190803590602001909291905050506084565b005b60a9600480360360"
        "20811015609a57600080fd5b81019080803590602001909190505060ab56"
        "5b005b505056fea2646970667358221220deadbeefcafebabe1234567890"
    )


@pytest.fixture
def sample_contract_data():
    """Sample contract data for testing."""
    return {
        "address": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
        "chain": "ethereum",
        "contract_name": "WrappedBTC",
        "compiler_version": "v0.8.19+commit.7dd6d404",
        "optimization_used": True,
        "runs": 200,
        "constructor_arguments": "",
        "evm_version": "london",
        "library": "",
        "license_type": "MIT",
        "proxy": "0",
        "implementation": "",
        "swarm_source": "",
        "abi": [
            {
                "type": "function",
                "name": "name",
                "inputs": [],
                "outputs": [{"type": "string", "name": ""}],
                "stateMutability": "view",
            },
            {
                "type": "function",
                "name": "symbol",
                "inputs": [],
                "outputs": [{"type": "string", "name": ""}],
                "stateMutability": "view",
            },
            {
                "type": "function",
                "name": "decimals",
                "inputs": [],
                "outputs": [{"type": "uint8", "name": ""}],
                "stateMutability": "view",
            },
            {
                "type": "function",
                "name": "totalSupply",
                "inputs": [],
                "outputs": [{"type": "uint256", "name": ""}],
                "stateMutability": "view",
            },
            {
                "type": "function",
                "name": "balanceOf",
                "inputs": [{"type": "address", "name": "account"}],
                "outputs": [{"type": "uint256", "name": ""}],
                "stateMutability": "view",
            },
            {
                "type": "function",
                "name": "transfer",
                "inputs": [
                    {"type": "address", "name": "to"},
                    {"type": "uint256", "name": "amount"},
                ],
                "outputs": [{"type": "bool", "name": ""}],
                "stateMutability": "nonpayable",
            },
            {
                "type": "function",
                "name": "approve",
                "inputs": [
                    {"type": "address", "name": "spender"},
                    {"type": "uint256", "name": "amount"},
                ],
                "outputs": [{"type": "bool", "name": ""}],
                "stateMutability": "nonpayable",
            },
        ],
        "source_code": "// SPDX-License-Identifier: MIT\npragma solidity ^0.8.19;",
        "bytecode": "0x6080604052348015600f57600080fd5b506004361060285760003560e01c8063a9059cbb14602d578063095ea7b314604257806370a08231146058575b600080fd5b606360048036036040811015604157600080fd5b81019080803590602001909291908035906020019092919050505060655b005b606360048036036040811015605757600080fd5b81019080803590602001909291908035906020019092919050505060845b005b60a960048036036020811015609a57600080fd5b81019080803590602001909190505060ab565b005b505056fea2646970667358221220deadbeefcafebabe1234567890abcdef1234567890abcdef1234567890abcdef64736f6c63430008180033",
    }



@pytest.fixture
def mock_4byte_response():
    """Mock response from 4byte.directory API."""
    return {
        "count": 2,
        "results": [
            {
                "id": 1,
                "created_at": "2023-01-01T00:00:00Z",
                "text_signature": "transfer(address,uint256)",
                "hex_signature": "0xa9059cbb",
                "bytes_signature": "\\xa9\\x05\\x9c\\xbb",
            },
            {
                "id": 2,
                "created_at": "2023-01-01T00:00:00Z",
                "text_signature": "workMyDirefulOwner(uint256,uint256)",
                "hex_signature": "0xa9059cbb",
                "bytes_signature": "\\xa9\\x05\\x9c\\xbb",
            },
        ],
    }




@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = os.path.join(temp_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        yield cache_dir
