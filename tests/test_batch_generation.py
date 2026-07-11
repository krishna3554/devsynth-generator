import json

import pytest

from devsynth_generator.exporter import DatasetExporter
from devsynth_generator.generator import CoverageMatrix, LLMScenarioGenerationError, ScenarioGenerator
from devsynth_generator.models import Conversation, Message
from devsynth_generator.pipeline import BatchGenerationPipeline
from devsynth_generator.taxonomy import default_taxonomy


class FakeBatchGenerator:
    """Controllable fake generator for testing the batch pipeline."""

    def __init__(self, fail_at: int | None = None, fail_until: dict[int, int] | None = None):
        """
        Args:
            fail_at: Index to always fail at (raises RuntimeError).
            fail_until: Mapping of index → number of calls that must fail before
                        succeeding.  E.g. ``{2: 3}`` means index 2 fails on the
                        first 3 calls but succeeds on the 4th.
        """
        self.taxonomy = default_taxonomy()
        self.coverage_matrix = CoverageMatrix(self.taxonomy, seed=3)
        self.fail_at = fail_at
        self.fail_until = fail_until or {}
        self.calls: list[int] = []
        self._call_counts: dict[int, int] = {}

    def generate_one(self, index, spec):
        self.calls.append(index)
        self._call_counts[index] = self._call_counts.get(index, 0) + 1

        if self.fail_at == index:
            raise RuntimeError("generation failed")

        if index in self.fail_until and self._call_counts[index] <= self.fail_until[index]:
            raise LLMScenarioGenerationError(f"attempt {self._call_counts[index]} for index {index}")

        turn_count = {"short": 2, "medium": 4, "long": 6}[spec.conversation_length]
        messages = [
            Message(role="user" if turn_index % 2 == 0 else "assistant", content=f"Turn {turn_index}.")
            for turn_index in range(turn_count)
        ]
        return Conversation(
            id=f"sample-{index}",
            task_type=spec.category,
            category=spec.category,
            subcategory=spec.subcategory,
            intent=spec.intent,
            difficulty=spec.difficulty,
            learning_stage=spec.learning_stage,
            conversation_length=spec.conversation_length,
            language="python",
            messages=messages,
        )


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ------------------------------------------------------------------
# Original tests (updated for max_retries parameter)
# ------------------------------------------------------------------


def test_batch_pipeline_appends_one_valid_conversation_at_a_time(tmp_path):
    generator = FakeBatchGenerator()
    exporter = DatasetExporter(tmp_path)

    result = BatchGenerationPipeline(generator=generator, exporter=exporter).run(count=3, filename="out.jsonl")

    records = read_jsonl(result.path)
    assert result.generated_count == 3
    assert [record["id"] for record in records] == ["sample-0", "sample-1", "sample-2"]
    assert generator.calls == [0, 1, 2]


def test_batch_pipeline_resumes_from_last_successful_sample(tmp_path):
    exporter = DatasetExporter(tmp_path)
    first_generator = FakeBatchGenerator(fail_at=2)

    # First run: fails at index 2 after max retries.
    result1 = BatchGenerationPipeline(generator=first_generator, exporter=exporter, max_retries=1).run(
        count=4, filename="out.jsonl"
    )
    # With max_retries=1, index 2 is skipped (no retry).
    # Indices 0, 1 succeed, index 2 is skipped, index 3 succeeds.
    assert result1.generated_count == 3

    second_generator = FakeBatchGenerator()
    result = BatchGenerationPipeline(generator=second_generator, exporter=exporter).run(count=4, filename="out.jsonl")

    records = read_jsonl(result.path)
    assert result.existing_count == 3
    assert result.generated_count == 1


def test_batch_pipeline_skips_when_requested_count_already_exists(tmp_path):
    exporter = DatasetExporter(tmp_path)
    generator = FakeBatchGenerator()

    BatchGenerationPipeline(generator=generator, exporter=exporter).run(count=2, filename="out.jsonl")
    second_generator = FakeBatchGenerator()
    result = BatchGenerationPipeline(generator=second_generator, exporter=exporter).run(count=2, filename="out.jsonl")

    assert result.generated_count == 0
    assert second_generator.calls == []


def test_batch_pipeline_stops_resume_scan_at_invalid_line(tmp_path):
    exporter = DatasetExporter(tmp_path)
    baseline = ScenarioGenerator(seed=1).generate(1)[0]
    path = exporter.append_jsonl(baseline, "out.jsonl")
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{not json}\n")

    generator = FakeBatchGenerator()
    result = BatchGenerationPipeline(generator=generator, exporter=exporter).run(count=2, filename="out.jsonl")

    assert result.existing_count == 1
    assert generator.calls == [1]


# ------------------------------------------------------------------
# New tests: retry behaviour
# ------------------------------------------------------------------


def test_retries_on_generation_error_then_succeeds(tmp_path):
    """Generator fails twice for index 1 then succeeds on the 3rd attempt."""
    generator = FakeBatchGenerator(fail_until={1: 2})
    exporter = DatasetExporter(tmp_path)

    result = BatchGenerationPipeline(generator=generator, exporter=exporter, max_retries=5).run(
        count=3, filename="out.jsonl"
    )

    assert result.generated_count == 3
    assert result.total_retries == 2
    records = read_jsonl(result.path)
    assert len(records) == 3


def test_skips_sample_after_max_retries_exhausted(tmp_path):
    """Generator always fails for index 1 — should be skipped, batch continues."""
    generator = FakeBatchGenerator(fail_at=1)
    exporter = DatasetExporter(tmp_path)

    result = BatchGenerationPipeline(generator=generator, exporter=exporter, max_retries=3).run(
        count=3, filename="out.jsonl"
    )

    # Index 0 succeeds, index 1 skipped after 3 retries, index 2 succeeds.
    assert result.generated_count == 2
    assert result.failed_count == 1
    records = read_jsonl(result.path)
    assert [r["id"] for r in records] == ["sample-0", "sample-2"]


def test_failed_generations_saved_to_file(tmp_path):
    """Failed attempts should be written to the failures file."""
    generator = FakeBatchGenerator(fail_at=0)
    exporter = DatasetExporter(tmp_path)

    result = BatchGenerationPipeline(generator=generator, exporter=exporter, max_retries=2).run(
        count=2, filename="out.jsonl"
    )

    assert result.failures_path is not None
    assert result.failures_path.exists()
    failures = read_jsonl(result.failures_path)
    assert len(failures) >= 2  # at least 2 attempts for index 0
    assert all("reason" in f for f in failures)
    assert all("sample_index" in f for f in failures)


def test_metrics_tracking(tmp_path):
    """Metrics should accurately reflect successes, failures, and retries."""
    generator = FakeBatchGenerator(fail_until={0: 1, 2: 2})
    exporter = DatasetExporter(tmp_path)

    result = BatchGenerationPipeline(generator=generator, exporter=exporter, max_retries=5).run(
        count=3, filename="out.jsonl"
    )

    assert result.generated_count == 3
    assert result.total_retries == 3  # 1 retry for index 0 + 2 retries for index 2


def test_batch_never_crashes_on_single_failure(tmp_path):
    """Even if one sample always fails, the pipeline must not raise."""
    generator = FakeBatchGenerator(fail_at=2)
    exporter = DatasetExporter(tmp_path)

    # This should NOT raise.
    result = BatchGenerationPipeline(generator=generator, exporter=exporter, max_retries=2).run(
        count=5, filename="out.jsonl"
    )

    assert result.generated_count == 4
    assert result.failed_count == 1
    records = read_jsonl(result.path)
    assert len(records) == 4
