"""Tests for the fusion ABI reconstructor.

The core is choose_candidate: picking the correct 4byte signature out of a
spam-laden candidate set using evmole's recovered type structure.
"""

import sys
from unittest.mock import patch

from abifusion.fusion import (
    ABIFusion,
    choose_candidate,
    parse_signature,
    split_args,
)
from abifusion.utils.signature_lookup import SignatureLookup


def _push4_eq_bytecode(selector: str) -> str:
    """Build minimal PUSH4+EQ bytecode for a selector."""
    sel = selector.replace("0x", "")
    return f"63{sel}14600157"


HOOK_CALLBACKS = {
    "bc29bafc": "afterAddLiquidity",
    "1f29cf9d": "afterDonate",
    "b6d4944a": "afterInitialize",
    "606e8192": "afterRemoveLiquidity",
    "c13d1c69": "afterSwap",
    "b772b8cc": "beforeAddLiquidity",
    "d950bd74": "beforeDonate",
    "ebe1cdaf": "beforeInitialize",
    "5cb32d10": "beforeRemoveLiquidity",
    "468ead2c": "beforeSwap",
}

TRIGGER_A = "575e24b4"
TRIGGER_B = "dc4c90d3"


def _trigger_only_bytecode(trigger_a: str, trigger_b: str) -> str:
    """Bytecode with only trigger selectors, no hook callback selectors."""
    return _push4_eq_bytecode(trigger_a) + _push4_eq_bytecode(trigger_b)


def _hook_bytecode_with_triggers(trigger_a: str, trigger_b: str) -> str:
    bc = _push4_eq_bytecode(trigger_a) + _push4_eq_bytecode(trigger_b)
    for sel in HOOK_CALLBACKS:
        bc += _push4_eq_bytecode(sel)
    return bc


class TestParsing:
    def test_split_simple(self):
        assert split_args("address,uint256") == ("address", "uint256")

    def test_split_empty(self):
        assert split_args("") == ()

    def test_split_respects_tuple_parens(self):
        assert split_args("(address,uint256),bytes") == ("(address,uint256)", "bytes")

    def test_parse_signature(self):
        assert parse_signature("transfer(address,uint256)") == (
            "transfer",
            ("address", "uint256"),
        )


class TestChooseCandidate:
    def test_picks_exact_evmole_match_over_spam(self):
        cands = [
            ("workMyDirefulOwner", ("uint256", "uint256")),
            ("transfer", ("address", "uint256")),
        ]
        name, types, confidence = choose_candidate(cands, ("address", "uint256"))
        assert name == "transfer"
        assert types == ("address", "uint256")
        assert confidence == "high"

    def test_4byte_supplies_exact_type_evmole_cannot(self):
        cands = [("commit", ("bytes32",))]
        _, types, confidence = choose_candidate(cands, ("uint256",))
        assert types == ("bytes32",)
        assert confidence == "medium"

    def test_falls_back_to_first_when_no_evmole(self):
        cands = [("a", ("address",)), ("b", ("uint256",))]
        name, types, confidence = choose_candidate(cands, None)
        assert (name, types, confidence) == ("a", ("address",), "low")

    def test_none_when_no_candidates(self):
        assert choose_candidate([], ("address",)) is None

    def test_max_agreement_on_partial_match(self):
        cands = [
            ("x", ("uint256", "bytes32", "bytes32")),
            ("y", ("address", "uint256", "bytes32")),
        ]
        name, _, confidence = choose_candidate(cands, ("address", "uint256", "uint256"))
        assert name == "y"
        assert confidence == "medium"


class TestCandidateSource:
    def test_prefers_openchain_over_4byte(self):
        sl = SignatureLookup.__new__(SignatureLookup)  # no network/init
        with patch.object(sl, "lookup_openchain", return_value=[{"text_signature": "transfer(address,uint256)"}]) as oc, \
             patch.object(sl, "lookup_4byte", return_value=[{"text_signature": "spam(uint256,uint256)"}]) as fb:
            cands = ABIFusion(sl)._candidates("a9059cbb")
        assert cands == [("transfer", ("address", "uint256"))]
        oc.assert_called_once()
        fb.assert_not_called()  # openchain hit → 4byte never queried

    def test_falls_back_to_4byte_when_openchain_empty(self):
        sl = SignatureLookup.__new__(SignatureLookup)
        with patch.object(sl, "lookup_openchain", return_value=[]), \
             patch.object(sl, "lookup_4byte", return_value=[{"text_signature": "foo(bytes32)"}]):
            cands = ABIFusion(sl)._candidates("deadbeef")
        assert cands == [("foo", ("bytes32",))]


class TestReconstruct:
    def test_does_not_import_torch(self):
        """The ML subsystem was removed; nothing in the fusion path may pull
        torch. Guards against an ML dependency creeping back in."""
        sys.modules.pop("torch", None)
        ABIFusion().reconstruct("63deadbeef1457")
        assert "torch" not in sys.modules

    def test_unresolved_selector_falls_back_to_selector(self):
        """A selector with no candidates, no evmole structure, and no known-table
        entry falls back to a bare function_<selector>."""
        bytecode = "63deadbeef1457"
        with patch.object(ABIFusion, "_candidates", return_value=[]):
            result = ABIFusion().reconstruct(bytecode)

        by_sel = {f["selector"]: f for f in result["functions"]}
        assert by_sel["deadbeef"]["name"] == "function_deadbeef"
        assert by_sel["deadbeef"]["source"] == "selector"

    def test_reconstruct_shape_and_resolution(self, sample_bytecode):
        fake = {
            "a9059cbb": [{"text_signature": "transfer(address,uint256)"}],
            "095ea7b3": [{"text_signature": "approve(address,uint256)"}],
            "70a08231": [{"text_signature": "balanceOf(address)"}],
        }
        with patch.object(
            ABIFusion, "_candidates", autospec=True
        ) as mock_c:
            def _c(self, selector):
                from abifusion.fusion import parse_signature
                return [
                    parse_signature(s["text_signature"])
                    for s in fake.get(selector, [])
                ]
            mock_c.side_effect = _c
            result = ABIFusion().reconstruct(sample_bytecode)

        assert "functions" in result and "metadata" in result
        for fn in result["functions"]:
            assert {"type", "name", "selector", "inputs", "source"} <= set(fn)
            assert "confidence" in fn
            assert "candidates" in fn
        by_sel = {f["selector"]: f for f in result["functions"]}
        if "a9059cbb" in by_sel:
            fn = by_sel["a9059cbb"]
            assert [i["type"] for i in fn["inputs"]] == [
                "address",
                "uint256",
            ]
            assert fn["confidence"] == "low"
            assert fn["candidates"] == [
                {"name": "transfer", "types": ["address", "uint256"], "source": "4byte"}
            ]


class TestInterfaceCompletion:
    """Tests for known-interface completion in ABIFusion.

    The interface completion mechanism emits known interface functions (e.g., Uniswap V4
    hook callbacks) when trigger selectors are found in bytecode, even though those
    interface selectors themselves are not extractable via PUSH4+EQ.
    """

    def test_both_triggers_emits_all_10_callbacks(self):
        """When both trigger selectors are in bytecode, all 10 hook callbacks are emitted."""
        bc = _trigger_only_bytecode(TRIGGER_A, TRIGGER_B)
        result = ABIFusion().reconstruct(bc)

        completed = {
            f["selector"]: f
            for f in result["functions"]
            if f["source"] == "known-interface-completion"
        }
        assert len(completed) == 10, f"Expected 10, got {len(completed)}"

        emitted_names = {f["name"] for f in completed.values()}
        expected_names = set(HOOK_CALLBACKS.values())
        assert emitted_names == expected_names, f"Missing: {expected_names - emitted_names}"

    def test_single_trigger_no_completion(self):
        """With only TRIGGER_A (beforeSwap) in bytecode, the post-processing
        pass emits all 10 V4 hook callbacks. TRIGGER_B (poolManager) alone
        does NOT trigger the post-processing pass (poolManager appears in
        non-V4-hook contracts)."""
        bc_a_only = _trigger_only_bytecode(TRIGGER_A, TRIGGER_A)  # 575e24b4 only, no dc4c90d3
        result_a = ABIFusion().reconstruct(bc_a_only)
        post_completed_a = {
            f["selector"]: f
            for f in result_a["functions"]
            if f["source"] == "known-interface-completion-post"
        }
        assert len(post_completed_a) == 10, f"Expected 10 (post-processing fires on 575e24b4 alone), got {len(post_completed_a)}"

        bc_b_only = _trigger_only_bytecode(TRIGGER_B, TRIGGER_B)  # dc4c90d3 only, no 575e24b4
        result_b = ABIFusion().reconstruct(bc_b_only)
        post_completed_b = {
            f["selector"]: f
            for f in result_b["functions"]
            if f["source"] == "known-interface-completion-post"
        }
        assert len(post_completed_b) == 0, f"Expected 0 (post-processing does NOT fire on dc4c90d3 alone), got {len(post_completed_b)}"

    def test_each_trigger_alone_insufficient(self):
        """_complete_interfaces (min_matches=2) requires both triggers. The
        post-processing pass fires on TRIGGER_A alone but not TRIGGER_B alone.
        This test verifies _complete_interfaces behavior: neither trigger alone
        should trigger it (post-processing is tested separately)."""
        for trigger in [TRIGGER_A, TRIGGER_B]:
            bc = _trigger_only_bytecode(trigger, trigger)
            result = ABIFusion().reconstruct(bc)
            completed_from_complete_interfaces = {
                f["selector"]: f
                for f in result["functions"]
                if f["source"] == "known-interface-completion"
            }
            assert len(completed_from_complete_interfaces) == 0, \
                f"_complete_interfaces should not fire with only {trigger}: {list(completed_from_complete_interfaces.keys())}"

    def test_completed_functions_have_correct_source_and_confidence(self):
        """All interface-completed functions have source=known-interface-completion and confidence=medium."""
        bc = _trigger_only_bytecode(TRIGGER_A, TRIGGER_B)
        result = ABIFusion().reconstruct(bc)

        completed = [
            f for f in result["functions"]
            if f["source"] == "known-interface-completion"
        ]
        assert len(completed) == 10

        for fn in completed:
            assert fn["source"] == "known-interface-completion", f"Wrong source: {fn['source']}"
            assert fn["confidence"] == "medium", f"Wrong confidence: {fn['confidence']}"

    def test_interface_completion_does_not_override_bytecode_functions(self):
        """If a function is already found via bytecode extraction, interface completion skips it."""
        bc = _trigger_only_bytecode(TRIGGER_A, TRIGGER_B)
        bc += _push4_eq_bytecode("a9059cbb")  # add transfer selector via bytecode
        result = ABIFusion().reconstruct(bc)

        by_src = {f["source"] for f in result["functions"]}

        assert "known-interface-completion" in by_src
        # transfer should be from bytecode, not from interface completion
        transfer_fn = next((f for f in result["functions"] if f["selector"] == "a9059cbb"), None)
        assert transfer_fn is not None
        assert transfer_fn["source"] != "known-interface-completion"
