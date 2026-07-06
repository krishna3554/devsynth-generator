import json

import pytest

from devsynth_generator.exporter import DatasetExporter
from devsynth_generator.generator import CoverageMatrix, ScenarioGenerator
from devsynth_generator.models import Conversation, Message
from devsynth_generator.pipeline import BatchGenerationPipeline
from devsynth_generator.taxonomy import default_taxonomy


class FakeBatchGenerator:
    def __init__(self, fail_at: int | None = None):
        self.taxonomy = default_taxonomy()
        self.coverage_matrix = CoverageMatrix(self.taxonomy, seed=3)
        self.fail_at = fail_at
        self.calls = []

    def generate_one(self, index, spec):
        self.calls.append(index)
        if self.fail_at == index:
            raise RuntimeError("generation failed")
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

    with pytest.raises(RuntimeError):
        BatchGenerationPipeline(generator=first_generator, exporter=exporter).run(count=4, filename="out.jsonl")

    assert [record["id"] for record in read_jsonl(tmp_path / "out.jsonl")] == ["sample-0", "sample-1"]

    second_generator = FakeBatchGenerator()
    result = BatchGenerationPipeline(generator=second_generator, exporter=exporter).run(count=4, filename="out.jsonl")

    records = read_jsonl(result.path)
    assert result.existing_count == 2
    assert result.generated_count == 2
    assert [record["id"] for record in records] == ["sample-0", "sample-1", "sample-2", "sample-3"]
    assert second_generator.calls == [2, 3]


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
