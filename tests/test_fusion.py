"""Tests for the fusion ABI reconstructor.

The core is choose_candidate: picking the correct 4byte signature out of a
spam-laden candidate set using evmole's recovered type structure.
"""

from unittest.mock import patch

from abi_reconstructor.fusion import (
    FusionReconstructor,
    choose_candidate,
    parse_signature,
    split_args,
)


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
        # 4byte order puts spam first; evmole structure selects the real one.
        cands = [
            ("workMyDirefulOwner", ("uint256", "uint256")),
            ("transfer", ("address", "uint256")),
        ]
        name, types = choose_candidate(cands, ("address", "uint256"))
        assert name == "transfer"
        assert types == ("address", "uint256")

    def test_4byte_supplies_exact_type_evmole_cannot(self):
        # evmole says uint256; 4byte's same-arity candidate says bytes32 → trust 4byte.
        cands = [("commit", ("bytes32",))]
        _, types = choose_candidate(cands, ("uint256",))
        assert types == ("bytes32",)

    def test_falls_back_to_first_when_no_evmole(self):
        cands = [("a", ("address",)), ("b", ("uint256",))]
        assert choose_candidate(cands, None) == ("a", ("address",))

    def test_none_when_no_candidates(self):
        assert choose_candidate([], ("address",)) is None

    def test_max_agreement_on_partial_match(self):
        cands = [
            ("x", ("uint256", "bytes32", "bytes32")),  # 0/3 agreement
            ("y", ("address", "uint256", "bytes32")),  # 2/3 agreement
        ]
        name, _ = choose_candidate(cands, ("address", "uint256", "uint256"))
        assert name == "y"


class TestReconstruct:
    def test_reconstruct_shape_and_resolution(self, sample_bytecode):
        # Mock 4byte so the test is offline and deterministic.
        fake = {
            "a9059cbb": [{"text_signature": "transfer(address,uint256)"}],
            "095ea7b3": [{"text_signature": "approve(address,uint256)"}],
            "70a08231": [{"text_signature": "balanceOf(address)"}],
        }
        with patch.object(
            FusionReconstructor, "_candidates", autospec=True
        ) as mock_c:
            def _c(self, selector):
                from abi_reconstructor.fusion import parse_signature
                return [
                    parse_signature(s["text_signature"])
                    for s in fake.get(selector, [])
                ]
            mock_c.side_effect = _c
            result = FusionReconstructor().reconstruct(sample_bytecode)

        assert "functions" in result and "metadata" in result
        for fn in result["functions"]:
            assert {"type", "name", "selector", "inputs", "source"} <= set(fn)
        by_sel = {f["selector"]: f for f in result["functions"]}
        if "a9059cbb" in by_sel:
            assert [i["type"] for i in by_sel["a9059cbb"]["inputs"]] == [
                "address",
                "uint256",
            ]
