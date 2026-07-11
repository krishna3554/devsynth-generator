import pytest

from devsynth_generator.deduplication import SemanticDeduplicator
from devsynth_generator.exporter import DatasetExporter
from devsynth_generator.generator import CoverageMatrix
from devsynth_generator.models import Conversation, Message
from devsynth_generator.pipeline import BatchGenerationPipeline
from devsynth_generator.taxonomy import default_taxonomy


class FakeEmbedder:
    def encode(self, texts, normalize_embeddings=False):
        vectors = []
        for text in texts:
            if "same-topic" in text:
                vectors.append([1.0, 0.0])
            elif "near-topic" in text:
                vectors.append([0.95, 0.05])
            else:
                vectors.append([0.0, 1.0])
        return vectors


def conversation(conversation_id, content):
    return Conversation(
        id=conversation_id,
        task_type="bug_fix",
        difficulty="easy",
        language="python",
        messages=[
            Message(role="user", content=content),
            Message(role="assistant", content="Here is the answer."),
        ],
    )


class DuplicateBatchGenerator:
    def __init__(self):
        self.taxonomy = default_taxonomy()
        self.coverage_matrix = CoverageMatrix(self.taxonomy, seed=1)

    def generate_one(self, index, spec):
        return Conversation(
            id=f"generated-{index}",
            task_type=spec.category,
            category=spec.category,
            subcategory=spec.subcategory,
            intent=spec.intent,
            difficulty=spec.difficulty,
            learning_stage=spec.learning_stage,
            conversation_length="short",
            language="python",
            messages=[
                Message(role="user", content="same-topic generated request"),
                Message(role="assistant", content="same-topic generated answer"),
            ],
        )


def test_semantic_deduplicator_filters_near_duplicates():
    deduplicator = SemanticDeduplicator(threshold=0.9, embedder=FakeEmbedder())
    first = conversation("first", "same-topic failing endpoint")
    duplicate = conversation("duplicate", "same-topic broken endpoint")
    unique = conversation("unique", "different failure mode")

    result = deduplicator.deduplicate([first, duplicate, unique])

    assert [item.id for item in result.unique_conversations] == ["first", "unique"]
    assert result.duplicates[0].duplicate_id == "duplicate"
    assert result.duplicates[0].canonical_id == "first"


def test_semantic_deduplicator_respects_threshold():
    deduplicator = SemanticDeduplicator(threshold=0.999, embedder=FakeEmbedder())
    first = conversation("first", "same-topic failing endpoint")
    near = conversation("near", "near-topic broken endpoint")

    result = deduplicator.deduplicate([first, near])

    assert [item.id for item in result.unique_conversations] == ["first", "near"]
    assert result.duplicates == []


def test_semantic_deduplicator_validates_threshold():
    with pytest.raises(ValueError):
        SemanticDeduplicator(threshold=1.5, embedder=FakeEmbedder())


def test_batch_pipeline_skips_semantic_duplicates(tmp_path):
    exporter = DatasetExporter(tmp_path)
    deduplicator = SemanticDeduplicator(threshold=0.9, embedder=FakeEmbedder())
    existing = conversation("existing", "same-topic already generated")
    exporter.append_jsonl(existing, "out.jsonl")

    generator = DuplicateBatchGenerator()

    # With retry logic, duplicates are retried then skipped — not raised.
    result = BatchGenerationPipeline(
        generator=generator,
        exporter=exporter,
        deduplicator=deduplicator,
        max_retries=2,
    ).run(count=2, filename="out.jsonl")

    assert result.generated_count == 0
    assert result.failed_count == 1

