"""Light schema test for build_external_known_table.py output."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_known_interface_sets_schema():
    result = subprocess.run(
        [sys.executable, "scripts/eval/build_external_known_table.py"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Script failed: {result.stderr}"

    output_path = Path("data/known_interface_sets.json")
    assert output_path.exists(), f"Output file not found: {output_path}"

    with open(output_path) as f:
        data = json.load(f)

    assert isinstance(data, list), f"Expected list, got {type(data)}"

    required_keys = {"name", "functions", "triggers", "precision", "recall", "contract_count"}
    trigger_required_keys = {"selector_set", "min_matches"}

    for i, iface in enumerate(data):
        assert isinstance(iface, dict), f"Entry {i} is not a dict: {type(iface)}"

        missing = required_keys - set(iface.keys())
        assert not missing, f"Entry {i} missing keys: {missing}"

        assert isinstance(iface["name"], str), f"Entry {i} name is not str"
        assert isinstance(iface["functions"], list), f"Entry {i} functions is not list"
        assert isinstance(iface["triggers"], dict), f"Entry {i} triggers is not dict"
        assert isinstance(iface["precision"], (int, float)), f"Entry {i} precision is not number"
        assert isinstance(iface["recall"], (int, float)), f"Entry {i} recall is not number"
        assert isinstance(iface["contract_count"], int), f"Entry {i} contract_count is not int"

        for j, func in enumerate(iface["functions"]):
            assert isinstance(func, dict), f"Entry {i} func {j} is not dict"
            assert "selector" in func, f"Entry {i} func {j} missing selector"
            assert "name" in func, f"Entry {i} func {j} missing name"
            assert "types" in func, f"Entry {i} func {j} missing types"
            assert isinstance(func["types"], list), f"Entry {i} func {j} types is not list"

        trigger_missing = trigger_required_keys - set(iface["triggers"].keys())
        assert not trigger_missing, f"Entry {i} triggers missing keys: {trigger_missing}"
        assert isinstance(iface["triggers"]["selector_set"], list), f"Entry {i} selector_set is not list"
        assert isinstance(iface["triggers"]["min_matches"], int), f"Entry {i} min_matches is not int"