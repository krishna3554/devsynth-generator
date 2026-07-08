"""Tests for the quality evaluation module."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from devsynth_generator.generator import ScenarioGenerator
from devsynth_generator.quality import QualityConfig, QualityDimension, QualityEvaluator, QualityResult


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

def make_conversation():
    """Create a valid conversation for testing."""
    return ScenarioGenerator(seed=1).generate(1)[0]


def make_eval_response(
    accuracy: int = 8,
    helpfulness: int = 7,
    clarity: int = 8,
    realism: int = 7,
) -> dict[str, Any]:
    """Build a well-formed evaluator response payload."""
    return {
        "dimensions": [
            {"name": "technical_accuracy", "score": accuracy, "reasoning": "Technically sound advice."},
            {"name": "helpfulness", "score": helpfulness, "reasoning": "Addresses the problem well."},
            {"name": "clarity", "score": clarity, "reasoning": "Well structured and concise."},
            {"name": "realism", "score": realism, "reasoning": "Feels like a real interaction."},
        ]
    }


def make_mock_client(response_payload: dict[str, Any] | None = None) -> MagicMock:
    """Create a mock OpenRouterClient that returns JSON text."""
    client = MagicMock()
    payload = response_payload or make_eval_response()
    client.generate_text.return_value = json.dumps(payload)
    client.model = "test-model"
    return client


# ---------------------------------------------------------------------------
# QualityDimension tests
# ---------------------------------------------------------------------------

class TestQualityDimension:
    def test_valid_dimension(self):
        dim = QualityDimension(name="clarity", score=8, reasoning="Clear and concise.")
        assert dim.normalized_score == pytest.approx(7 / 9)

    def test_minimum_score_normalized(self):
        dim = QualityDimension(name="realism", score=1, reasoning="Bad.")
        assert dim.normalized_score == pytest.approx(0.0)

    def test_maximum_score_normalized(self):
        dim = QualityDimension(name="helpfulness", score=10, reasoning="Perfect.")
        assert dim.normalized_score == pytest.approx(1.0)

    def test_invalid_dimension_name(self):
        with pytest.raises(ValueError, match="Unknown dimension"):
            QualityDimension(name="unknown", score=5, reasoning="Some reasoning.")

    def test_score_out_of_range_low(self):
        with pytest.raises(ValueError):
            QualityDimension(name="clarity", score=0, reasoning="Too low.")

    def test_score_out_of_range_high(self):
        with pytest.raises(ValueError):
            QualityDimension(name="clarity", score=11, reasoning="Too high.")


# ---------------------------------------------------------------------------
# QualityConfig tests
# ---------------------------------------------------------------------------

class TestQualityConfig:
    def test_default_config(self):
        config = QualityConfig()
        assert config.threshold == 0.7
        assert abs(sum(config.dimension_weights.values()) - 1.0) < 1e-6
        assert config.max_eval_retries == 2
        assert config.temperature == 0.2

    def test_invalid_threshold_too_high(self):
        with pytest.raises(ValueError, match="threshold"):
            QualityConfig(threshold=1.5)

    def test_invalid_threshold_negative(self):
        with pytest.raises(ValueError, match="threshold"):
            QualityConfig(threshold=-0.1)

    def test_invalid_weights_sum(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            QualityConfig(dimension_weights={
                "technical_accuracy": 0.5,
                "helpfulness": 0.5,
                "clarity": 0.5,
                "realism": 0.5,
            })

    def test_missing_dimension_weight(self):
        with pytest.raises(ValueError, match="Missing weight"):
            QualityConfig(dimension_weights={
                "technical_accuracy": 0.5,
                "helpfulness": 0.5,
            })

    def test_negative_retries(self):
        with pytest.raises(ValueError, match="max_eval_retries"):
            QualityConfig(max_eval_retries=-1)


# ---------------------------------------------------------------------------
# QualityResult tests
# ---------------------------------------------------------------------------

class TestQualityResult:
    def test_result_passed(self):
        result = QualityResult(
            conversation_id="test-001",
            dimensions=[
                QualityDimension(name="technical_accuracy", score=8, reasoning="Good."),
                QualityDimension(name="helpfulness", score=7, reasoning="Good."),
                QualityDimension(name="clarity", score=8, reasoning="Good."),
                QualityDimension(name="realism", score=7, reasoning="Good."),
            ],
            overall_score=0.75,
            passed=True,
            threshold=0.7,
        )
        assert result.passed is True
        assert result.overall_score == 0.75

    def test_result_failed(self):
        result = QualityResult(
            conversation_id="test-002",
            dimensions=[
                QualityDimension(name="technical_accuracy", score=3, reasoning="Bad."),
                QualityDimension(name="helpfulness", score=2, reasoning="Bad."),
                QualityDimension(name="clarity", score=3, reasoning="Bad."),
                QualityDimension(name="realism", score=2, reasoning="Bad."),
            ],
            overall_score=0.18,
            passed=False,
            threshold=0.7,
        )
        assert result.passed is False

    def test_to_dict_roundtrip(self):
        result = QualityResult(
            conversation_id="test-003",
            dimensions=[
                QualityDimension(name="technical_accuracy", score=9, reasoning="Excellent."),
                QualityDimension(name="helpfulness", score=8, reasoning="Great."),
                QualityDimension(name="clarity", score=9, reasoning="Crystal clear."),
                QualityDimension(name="realism", score=8, reasoning="Natural."),
            ],
            overall_score=0.88,
            passed=True,
            threshold=0.7,
        )
        data = result.to_dict()
        assert data["conversation_id"] == "test-003"
        assert data["passed"] is True
        assert len(data["dimensions"]) == 4


# ---------------------------------------------------------------------------
# QualityEvaluator tests
# ---------------------------------------------------------------------------

class TestQualityEvaluator:
    def test_evaluate_passing_conversation(self):
        client = make_mock_client(make_eval_response(accuracy=8, helpfulness=7, clarity=8, realism=7))
        evaluator = QualityEvaluator(client=client, config=QualityConfig(threshold=0.5))

        conversation = make_conversation()
        result = evaluator.evaluate(conversation)

        assert result.passed is True
        assert result.conversation_id == conversation.id
        assert len(result.dimensions) == 4
        assert result.overall_score > 0.5

    def test_evaluate_failing_conversation(self):
        client = make_mock_client(make_eval_response(accuracy=2, helpfulness=2, clarity=2, realism=2))
        evaluator = QualityEvaluator(client=client, config=QualityConfig(threshold=0.9))

        conversation = make_conversation()
        result = evaluator.evaluate(conversation)

        assert result.passed is False
        assert result.overall_score < 0.9

    def test_evaluate_uses_correct_model(self):
        client = make_mock_client()
        config = QualityConfig(model="eval-model-v1")
        evaluator = QualityEvaluator(client=client, config=config)

        conversation = make_conversation()
        evaluator.evaluate(conversation)

        call_kwargs = client.generate_text.call_args
        assert call_kwargs.kwargs.get("model") == "eval-model-v1"

    def test_evaluate_uses_low_temperature(self):
        client = make_mock_client()
        config = QualityConfig(temperature=0.15)
        evaluator = QualityEvaluator(client=client, config=config)

        conversation = make_conversation()
        evaluator.evaluate(conversation)

        call_kwargs = client.generate_text.call_args
        assert call_kwargs.kwargs.get("temperature") == 0.15

    def test_evaluate_many(self):
        client = make_mock_client(make_eval_response(accuracy=9, helpfulness=8, clarity=9, realism=8))
        evaluator = QualityEvaluator(client=client, config=QualityConfig(threshold=0.5))

        conversations = ScenarioGenerator(seed=42).generate(3)
        results = evaluator.evaluate_many(conversations)

        assert len(results) == 3
        assert all(r.passed for r in results)

    def test_filter_separates_passed_and_failed(self):
        # Return different scores on successive calls
        high_response = json.dumps(make_eval_response(accuracy=9, helpfulness=9, clarity=9, realism=9))
        low_response = json.dumps(make_eval_response(accuracy=1, helpfulness=1, clarity=1, realism=1))

        client = MagicMock()
        client.model = "test-model"
        client.generate_text.side_effect = [high_response, low_response, high_response]

        evaluator = QualityEvaluator(client=client, config=QualityConfig(threshold=0.5))

        conversations = ScenarioGenerator(seed=42).generate(3)
        accepted, results = evaluator.filter(conversations)

        assert len(results) == 3
        assert len(accepted) == 2  # 1st and 3rd pass, 2nd fails

    def test_evaluate_weighted_scoring(self):
        """Verify the weighted average computation."""
        # All dimensions score 10 → normalized = 1.0 → overall = 1.0
        client = make_mock_client(make_eval_response(accuracy=10, helpfulness=10, clarity=10, realism=10))
        evaluator = QualityEvaluator(client=client, config=QualityConfig(threshold=0.5))

        conversation = make_conversation()
        result = evaluator.evaluate(conversation)

        assert result.overall_score == pytest.approx(1.0)

    def test_evaluate_minimum_scores(self):
        """All dimensions score 1 → normalized = 0.0 → overall = 0.0."""
        client = make_mock_client(make_eval_response(accuracy=1, helpfulness=1, clarity=1, realism=1))
        evaluator = QualityEvaluator(client=client, config=QualityConfig(threshold=0.5))

        conversation = make_conversation()
        result = evaluator.evaluate(conversation)

        assert result.overall_score == pytest.approx(0.0)
        assert result.passed is False

    def test_evaluate_retries_on_malformed_response(self):
        """Parser retries on bad JSON then succeeds on valid JSON."""
        valid_response = json.dumps(make_eval_response())

        client = MagicMock()
        client.model = "test-model"
        client.generate_text.side_effect = [
            "this is not json",
            valid_response,
        ]

        config = QualityConfig(max_eval_retries=1, threshold=0.5)
        evaluator = QualityEvaluator(client=client, config=config)

        conversation = make_conversation()
        result = evaluator.evaluate(conversation)

        assert result.passed is True
        assert client.generate_text.call_count == 2

    def test_evaluate_raises_after_exhausted_retries(self):
        """Raises QualityEvaluationError after all retries fail."""
        from devsynth_generator.quality.quality_evaluator import QualityEvaluationError

        client = MagicMock()
        client.model = "test-model"
        client.generate_text.return_value = "not json at all"

        config = QualityConfig(max_eval_retries=1, threshold=0.5)
        evaluator = QualityEvaluator(client=client, config=config)

        conversation = make_conversation()
        with pytest.raises(QualityEvaluationError):
            evaluator.evaluate(conversation)

    def test_evaluate_rejects_missing_dimensions(self):
        """Response with missing dimensions triggers parse error and retry."""
        from devsynth_generator.quality.quality_evaluator import QualityEvaluationError

        incomplete_response = json.dumps({
            "dimensions": [
                {"name": "technical_accuracy", "score": 8, "reasoning": "Good."},
            ]
        })

        client = MagicMock()
        client.model = "test-model"
        client.generate_text.return_value = incomplete_response

        config = QualityConfig(max_eval_retries=0, threshold=0.5)
        evaluator = QualityEvaluator(client=client, config=config)

        conversation = make_conversation()
        with pytest.raises(QualityEvaluationError):
            evaluator.evaluate(conversation)

    def test_evaluate_rejects_duplicate_dimensions(self):
        """Response with duplicate dimension names is rejected."""
        from devsynth_generator.quality.quality_evaluator import QualityEvaluationError

        duplicate_response = json.dumps({
            "dimensions": [
                {"name": "technical_accuracy", "score": 8, "reasoning": "Good."},
                {"name": "technical_accuracy", "score": 7, "reasoning": "Also good."},
                {"name": "helpfulness", "score": 7, "reasoning": "Good."},
                {"name": "clarity", "score": 8, "reasoning": "Good."},
            ]
        })

        client = MagicMock()
        client.model = "test-model"
        client.generate_text.return_value = duplicate_response

        config = QualityConfig(max_eval_retries=0, threshold=0.5)
        evaluator = QualityEvaluator(client=client, config=config)

        conversation = make_conversation()
        with pytest.raises(QualityEvaluationError):
            evaluator.evaluate(conversation)

    def test_evaluate_boundary_threshold(self):
        """Score exactly at threshold should pass."""
        # Score 7 on all dimensions → normalized = 6/9 ≈ 0.6667
        client = make_mock_client(make_eval_response(accuracy=7, helpfulness=7, clarity=7, realism=7))
        threshold = 6 / 9  # Exactly what we expect
        evaluator = QualityEvaluator(client=client, config=QualityConfig(threshold=threshold))

        conversation = make_conversation()
        result = evaluator.evaluate(conversation)

        assert result.passed is True

    def test_quality_scores_persisted_in_metadata(self):
        """Verify quality scores dict is properly serializable."""
        client = make_mock_client()
        evaluator = QualityEvaluator(client=client, config=QualityConfig(threshold=0.5))

        conversation = make_conversation()
        result = evaluator.evaluate(conversation)
        data = result.to_dict()

        assert "conversation_id" in data
        assert "dimensions" in data
        assert "overall_score" in data
        assert "passed" in data
        assert "threshold" in data
        # Verify it's JSON serializable
        json.dumps(data)
