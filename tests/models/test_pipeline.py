"""Tests for ABIReconstructorPipeline parameter prediction.

The pipeline (1200+ lines) previously had no dedicated tests. These focus on
the function-prior resolution logic in ``predict_parameters`` — in particular
the arity-aware handling of overloaded standard functions (balanceOf,
safeTransferFrom), which regressed when those names collided as duplicate dict
keys in FUNCTION_TYPE_PRIORS.

A fake parameter predictor is injected so the predicted parameter count is
deterministic, isolating the prior-selection logic from untrained weights.
"""

import torch
import torch.nn as nn

from abi_reconstructor.models.abi_reconstructor_pipeline import ABIReconstructorPipeline

FUNCTION_VOCAB = {"balanceOf": 0, "safeTransferFrom": 1, "transfer": 2, "unknown": 3}
TYPE_VOCAB = {"address": 0, "uint256": 1, "bytes": 2, "unknown": 3}
MAX_PARAMS = 12


class _FakePredictor(nn.Module):
    """Parameter predictor stub whose argmax count is fixed to `count`."""

    def __init__(self, count: int):
        super().__init__()
        self.count = count

    def forward(
        self, tokens, function_ids=None, attention_mask=None, discriminating_features=None
    ):
        batch = 1
        num_types = len(TYPE_VOCAB)
        count_logits = torch.full((batch, MAX_PARAMS + 1), -10.0)
        count_logits[0, self.count] = 10.0
        mask_logits = torch.full((batch, MAX_PARAMS), -10.0)
        mask_logits[0, : self.count] = 10.0
        type_logits = torch.zeros(batch, MAX_PARAMS, num_types)
        return {
            "type_logits": type_logits,
            "mask_logits": mask_logits,
            "count_logits": count_logits,
        }


def _make_pipeline(count: int) -> ABIReconstructorPipeline:
    return ABIReconstructorPipeline(
        function_classifier=nn.Module(),
        parameter_predictor=_FakePredictor(count),
        selector_extractor=nn.Module(),
        feature_extractor=nn.Module(),
        use_cuda=False,
        function_vocab=FUNCTION_VOCAB,
        type_vocab=TYPE_VOCAB,
    )


def _features():
    return {"bytecode_tokens": torch.zeros(1, 10, dtype=torch.long)}


class TestOverloadedPriors:
    def test_balanceof_resolves_to_erc20_form_when_arity_one(self):
        pipe = _make_pipeline(count=1)
        result = pipe.predict_parameters(_features(), "balanceOf")
        types = [p["type"] for p in result["parameters"]]
        assert types == ["address"]
        assert all(p["source"] == "function_prior" for p in result["parameters"])

    def test_balanceof_resolves_to_erc1155_form_when_arity_two(self):
        pipe = _make_pipeline(count=2)
        result = pipe.predict_parameters(_features(), "balanceOf")
        types = [p["type"] for p in result["parameters"]]
        assert types == ["address", "uint256"]

    def test_safetransferfrom_resolves_by_arity(self):
        erc721 = _make_pipeline(count=3).predict_parameters(
            _features(), "safeTransferFrom"
        )
        assert [p["type"] for p in erc721["parameters"]] == [
            "address",
            "address",
            "uint256",
        ]
        erc1155 = _make_pipeline(count=5).predict_parameters(
            _features(), "safeTransferFrom"
        )
        assert [p["type"] for p in erc1155["parameters"]] == [
            "address",
            "address",
            "uint256",
            "uint256",
            "bytes",
        ]


class TestPriorTableInvariants:
    """Regression guards for the duplicate-key bug (ruff F601)."""

    def test_overloaded_names_not_in_single_value_priors(self):
        pipe = _make_pipeline(count=1)
        lowered = {k.lower() for k in pipe.FUNCTION_TYPE_PRIORS}
        assert "balanceof" not in lowered
        assert "safetransferfrom" not in lowered

    def test_overloads_table_covers_both_arities(self):
        pipe = _make_pipeline(count=1)
        overloads = pipe.FUNCTION_TYPE_PRIOR_OVERLOADS
        assert [len(c) for c in overloads["balanceof"]] == [1, 2]
        assert [len(c) for c in overloads["safetransferfrom"]] == [3, 5]
