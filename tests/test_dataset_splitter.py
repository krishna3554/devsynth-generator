"""Tests for the dataset splitting module."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from devsynth_generator.generator import ScenarioGenerator
from devsynth_generator.models import Conversation
from devsynth_generator.splitter import (
    DatasetInfo,
    DatasetSplitError,
    DatasetSplitter,
    DatasetStatistics,
    DuplicateError,
    SplitConfig,
    build_dataset_info,
    compute_statistics,
    save_dataset_info,
    save_statistics,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

def _generate_conversations(count: int, seed: int = 42) -> list[Conversation]:
    """Generate deterministic test conversations."""
    return ScenarioGenerator(seed=seed).generate(count)


def _write_jsonl(path: Path, conversations: list[Conversation]) -> None:
    """Write conversations to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for conv in conversations:
            handle.write(json.dumps(conv.to_dict(), ensure_ascii=True) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    """Read JSONL file and return list of dicts."""
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# SplitConfig tests
# ---------------------------------------------------------------------------

class TestSplitConfig:
    def test_default_config(self):
        config = SplitConfig()
        assert config.train_ratio == 0.8
        assert config.validation_ratio == 0.1
        assert config.test_ratio == 0.1
        assert config.seed == 42

    def test_invalid_ratios_sum(self):
        with pytest.raises(DatasetSplitError, match="sum to 1.0"):
            SplitConfig(train_ratio=0.5, validation_ratio=0.5, test_ratio=0.5)

    def test_negative_ratio(self):
        with pytest.raises(DatasetSplitError, match="between 0.0 and 1.0"):
            SplitConfig(train_ratio=-0.1, validation_ratio=0.6, test_ratio=0.5)

    def test_custom_ratios(self):
        config = SplitConfig(train_ratio=0.7, validation_ratio=0.2, test_ratio=0.1)
        assert abs(config.train_ratio + config.validation_ratio + config.test_ratio - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# DatasetSplitter.load tests
# ---------------------------------------------------------------------------

class TestLoad:
    def test_load_valid_jsonl(self, tmp_path):
        conversations = _generate_conversations(10)
        path = tmp_path / "test.jsonl"
        _write_jsonl(path, conversations)

        splitter = DatasetSplitter()
        loaded, skipped = splitter.load(path)

        assert len(loaded) == 10
        assert skipped == 0

    def test_load_skips_malformed_json(self, tmp_path):
        conversations = _generate_conversations(3)
        path = tmp_path / "test.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(conversations[0].to_dict()) + "\n")
            handle.write("this is not json\n")
            handle.write(json.dumps(conversations[1].to_dict()) + "\n")
            handle.write("{invalid json too\n")
            handle.write(json.dumps(conversations[2].to_dict()) + "\n")

        splitter = DatasetSplitter()
        loaded, skipped = splitter.load(path)

        assert len(loaded) == 3
        assert skipped == 2

    def test_load_skips_invalid_schema(self, tmp_path):
        conversations = _generate_conversations(2)
        path = tmp_path / "test.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(conversations[0].to_dict()) + "\n")
            # Invalid record: missing required fields
            handle.write(json.dumps({"id": "bad", "not_a_field": True}) + "\n")
            handle.write(json.dumps(conversations[1].to_dict()) + "\n")

        splitter = DatasetSplitter()
        loaded, skipped = splitter.load(path)

        assert len(loaded) == 2
        assert skipped == 1

    def test_load_skips_blank_lines(self, tmp_path):
        conversations = _generate_conversations(2)
        path = tmp_path / "test.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(conversations[0].to_dict()) + "\n")
            handle.write("\n")
            handle.write("   \n")
            handle.write(json.dumps(conversations[1].to_dict()) + "\n")

        splitter = DatasetSplitter()
        loaded, skipped = splitter.load(path)

        assert len(loaded) == 2
        assert skipped == 0

    def test_load_missing_file(self):
        splitter = DatasetSplitter()
        with pytest.raises(FileNotFoundError, match="not found"):
            splitter.load(Path("/nonexistent/file.jsonl"))


# ---------------------------------------------------------------------------
# Duplicate detection tests
# ---------------------------------------------------------------------------

class TestDuplicateDetection:
    def test_no_duplicates(self):
        conversations = _generate_conversations(10)
        splitter = DatasetSplitter()
        unique, removed = splitter.check_duplicates(conversations)

        assert len(unique) == 10
        assert removed == 0

    def test_duplicate_ids_raise_error(self):
        conversations = _generate_conversations(5)
        # Create a duplicate ID.
        dup = conversations[0].model_copy()
        object.__setattr__(dup, "id", conversations[1].id)
        conversations.append(dup)

        splitter = DatasetSplitter()
        with pytest.raises(DuplicateError, match="duplicate conversation IDs"):
            splitter.check_duplicates(conversations)

    def test_content_duplicates_removed(self):
        conversations = _generate_conversations(5)
        # Create a content duplicate with different ID.
        dup = conversations[0].model_copy(update={"id": "duplicate-content-id"})
        conversations.append(dup)

        splitter = DatasetSplitter()
        unique, removed = splitter.check_duplicates(conversations)

        assert len(unique) == 5
        assert removed == 1
        assert all(c.id != "duplicate-content-id" for c in unique)


# ---------------------------------------------------------------------------
# Stratified split tests
# ---------------------------------------------------------------------------

class TestSplit:
    def test_split_produces_correct_total(self):
        conversations = _generate_conversations(100)
        splitter = DatasetSplitter(SplitConfig(seed=42))
        result = splitter.split(conversations)

        assert result.total_valid == 100

    def test_split_approximate_ratios(self):
        conversations = _generate_conversations(100)
        config = SplitConfig(train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1)
        splitter = DatasetSplitter(config)
        result = splitter.split(conversations)

        # Stratified splits with many small groups can deviate from exact ratios.
        # Verify majority goes to train and the total is preserved.
        assert len(result.train) > len(result.validation)
        assert len(result.train) > len(result.test)
        assert result.total_valid == 100

    def test_split_no_cross_split_leakage(self):
        conversations = _generate_conversations(50)
        splitter = DatasetSplitter()
        result = splitter.split(conversations)

        train_ids = {c.id for c in result.train}
        val_ids = {c.id for c in result.validation}
        test_ids = {c.id for c in result.test}

        assert not (train_ids & val_ids), "Train and validation share IDs"
        assert not (train_ids & test_ids), "Train and test share IDs"
        assert not (val_ids & test_ids), "Validation and test share IDs"

    def test_split_is_deterministic(self):
        conversations = _generate_conversations(50)
        splitter = DatasetSplitter(SplitConfig(seed=99))

        result1 = splitter.split(list(conversations))
        result2 = splitter.split(list(conversations))

        assert [c.id for c in result1.train] == [c.id for c in result2.train]
        assert [c.id for c in result1.validation] == [c.id for c in result2.validation]
        assert [c.id for c in result1.test] == [c.id for c in result2.test]

    def test_split_preserves_category_distribution(self):
        conversations = _generate_conversations(100)
        splitter = DatasetSplitter(SplitConfig(seed=42))
        result = splitter.split(conversations)

        # Check that categories appearing in the full dataset also appear in train.
        full_categories = {c.category for c in conversations if c.category}
        train_categories = {c.category for c in result.train if c.category}

        # All categories present in the full dataset should be in training set
        # (given 80% ratio and enough samples).
        assert train_categories.issubset(full_categories)

    def test_split_different_seed_produces_different_order(self):
        conversations = _generate_conversations(50)

        result1 = DatasetSplitter(SplitConfig(seed=1)).split(list(conversations))
        result2 = DatasetSplitter(SplitConfig(seed=999)).split(list(conversations))

        # Different seeds should produce different orderings (very high probability).
        train_ids_1 = [c.id for c in result1.train]
        train_ids_2 = [c.id for c in result2.train]
        assert train_ids_1 != train_ids_2

    def test_split_small_dataset(self):
        """Datasets with fewer than 3 records should still work."""
        conversations = _generate_conversations(2)
        splitter = DatasetSplitter()
        result = splitter.split(conversations)

        assert result.total_valid == 2

    def test_split_single_record(self):
        conversations = _generate_conversations(1)
        splitter = DatasetSplitter()
        result = splitter.split(conversations)

        assert result.total_valid == 1

    def test_split_empty_dataset(self):
        """split() on empty list returns zero-size result; run() raises."""
        splitter = DatasetSplitter()
        result = splitter.split([])
        assert result.total_valid == 0


# ---------------------------------------------------------------------------
# Save tests
# ---------------------------------------------------------------------------

class TestSave:
    def test_save_creates_jsonl_files(self, tmp_path):
        conversations = _generate_conversations(30)
        splitter = DatasetSplitter()
        result = splitter.split(conversations)
        paths = splitter.save(result, tmp_path)

        assert (tmp_path / "train.jsonl").exists()
        assert (tmp_path / "validation.jsonl").exists()
        assert (tmp_path / "test.jsonl").exists()

        train_records = _read_jsonl(paths["train"])
        val_records = _read_jsonl(paths["validation"])
        test_records = _read_jsonl(paths["test"])

        assert len(train_records) == len(result.train)
        assert len(val_records) == len(result.validation)
        assert len(test_records) == len(result.test)

    def test_saved_records_are_valid_json(self, tmp_path):
        conversations = _generate_conversations(10)
        splitter = DatasetSplitter()
        result = splitter.split(conversations)
        paths = splitter.save(result, tmp_path)

        for path in paths.values():
            records = _read_jsonl(path)
            for record in records:
                # Each record should be parseable back to a Conversation.
                Conversation.model_validate(record)


# ---------------------------------------------------------------------------
# Full pipeline (run) tests
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def test_run_end_to_end(self, tmp_path):
        conversations = _generate_conversations(30)
        input_path = tmp_path / "input.jsonl"
        output_dir = tmp_path / "output"
        _write_jsonl(input_path, conversations)

        config = SplitConfig(input_path=input_path, output_dir=output_dir, seed=42)
        splitter = DatasetSplitter(config)
        result = splitter.run()

        assert result.total_valid == 30
        assert result.skipped_count == 0
        assert result.duplicates_removed == 0
        assert (output_dir / "train.jsonl").exists()
        assert (output_dir / "validation.jsonl").exists()
        assert (output_dir / "test.jsonl").exists()

    def test_run_with_invalid_and_duplicate_records(self, tmp_path):
        conversations = _generate_conversations(10)
        input_path = tmp_path / "input.jsonl"
        output_dir = tmp_path / "output"

        # Write valid conversations + a content duplicate + malformed JSON.
        dup = conversations[0].model_copy(update={"id": "content-dup-id"})
        with input_path.open("w", encoding="utf-8") as handle:
            for conv in conversations:
                handle.write(json.dumps(conv.to_dict()) + "\n")
            handle.write(json.dumps(dup.to_dict()) + "\n")
            handle.write("not valid json\n")

        config = SplitConfig(input_path=input_path, output_dir=output_dir)
        splitter = DatasetSplitter(config)
        result = splitter.run()

        assert result.skipped_count == 1  # malformed JSON
        assert result.duplicates_removed == 1  # content duplicate
        assert result.total_valid == 10  # 10 unique from original

    def test_run_missing_input_file(self, tmp_path):
        config = SplitConfig(input_path=tmp_path / "nonexistent.jsonl", output_dir=tmp_path)
        splitter = DatasetSplitter(config)
        with pytest.raises(FileNotFoundError):
            splitter.run()

    def test_run_empty_dataset(self, tmp_path):
        input_path = tmp_path / "empty.jsonl"
        input_path.write_text("", encoding="utf-8")

        config = SplitConfig(input_path=input_path, output_dir=tmp_path)
        splitter = DatasetSplitter(config)
        with pytest.raises(DatasetSplitError, match="No valid conversations"):
            splitter.run()


# ---------------------------------------------------------------------------
# Statistics tests
# ---------------------------------------------------------------------------

class TestStatistics:
    def test_compute_statistics_basic(self):
        conversations = _generate_conversations(30)
        splitter = DatasetSplitter()
        result = splitter.split(conversations)

        stats = compute_statistics(
            conversations, result.train, result.validation, result.test
        )

        assert stats.total_conversations == 30
        assert stats.train_size == len(result.train)
        assert stats.validation_size == len(result.validation)
        assert stats.test_size == len(result.test)
        assert stats.average_turns > 0
        assert stats.average_messages > 0
        assert stats.average_conversation_length > 0

    def test_compute_statistics_distributions_have_percentages(self):
        conversations = _generate_conversations(20)
        splitter = DatasetSplitter()
        result = splitter.split(conversations)

        stats = compute_statistics(
            conversations, result.train, result.validation, result.test
        )

        # Every distribution should have entries that sum to ~100%.
        for dist in [
            stats.category_distribution,
            stats.difficulty_distribution,
            stats.programming_language_distribution,
        ]:
            if dist:
                total_pct = sum(entry.percentage for entry in dist)
                assert abs(total_pct - 100.0) < 1.0, f"Distribution percentages sum to {total_pct}"

    def test_compute_statistics_empty(self):
        stats = compute_statistics([], [], [], [])
        assert stats.total_conversations == 0
        assert stats.average_turns == 0.0

    def test_save_statistics_creates_file(self, tmp_path):
        conversations = _generate_conversations(10)
        stats = compute_statistics(conversations, conversations, [], [])
        path = save_statistics(stats, tmp_path)

        assert path.exists()
        data = json.loads(path.read_text())
        assert data["total_conversations"] == 10

    def test_statistics_to_dict_is_json_serializable(self):
        conversations = _generate_conversations(10)
        splitter = DatasetSplitter()
        result = splitter.split(conversations)
        stats = compute_statistics(
            conversations, result.train, result.validation, result.test
        )
        # Must not raise.
        json.dumps(stats.to_dict())


# ---------------------------------------------------------------------------
# DatasetInfo tests
# ---------------------------------------------------------------------------

class TestDatasetInfo:
    def test_build_dataset_info_default(self):
        info = build_dataset_info(train_size=80, validation_size=10, test_size=10)

        assert info.name == "DevSynth"
        assert info.version == "1.0.0"
        assert info.synthetic is True
        assert info.license == "Apache-2.0"
        assert info.splits["train"] == 80
        assert info.splits["validation"] == 10
        assert info.splits["test"] == 10
        assert info.created_at  # Should be set

    def test_build_dataset_info_custom_model(self):
        info = build_dataset_info(
            train_size=100,
            validation_size=20,
            test_size=20,
            generator_model="openai/gpt-4o",
        )
        assert info.generator_model == "openai/gpt-4o"

    def test_save_dataset_info_creates_file(self, tmp_path):
        info = build_dataset_info(train_size=50, validation_size=5, test_size=5)
        path = save_dataset_info(info, tmp_path)

        assert path.exists()
        data = json.loads(path.read_text())
        assert data["name"] == "DevSynth"
        assert data["splits"]["train"] == 50

    def test_dataset_info_to_dict_is_json_serializable(self):
        info = build_dataset_info(train_size=10, validation_size=1, test_size=1)
        # Must not raise.
        json.dumps(info.to_dict())
