"""Tests for the documentation generator module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from devsynth_generator.docs_generator import (
    DocContext,
    DocumentationGenerator,
    generate_changelog,
    generate_citation,
    generate_dataset_card,
    generate_examples,
    generate_pipeline,
    generate_readme,
    generate_schema,
    generate_taxonomy,
    load_doc_context,
)
from devsynth_generator.generator import ScenarioGenerator
from devsynth_generator.taxonomy import default_taxonomy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_context() -> DocContext:
    """Build a DocContext with realistic data for testing."""
    examples = ScenarioGenerator(seed=42).generate(5)
    return DocContext(
        dataset_name="DevSynth",
        version="1.0.0",
        description="Synthetic multi-turn developer conversation dataset.",
        license="Apache-2.0",
        generator_model="test-model",
        created_at="2025-01-15T12:00:00+00:00",
        synthetic=True,
        splits={"train": 8000, "validation": 1000, "test": 1000},
        statistics={
            "total_conversations": 10000,
            "train_size": 8000,
            "validation_size": 1000,
            "test_size": 1000,
            "average_turns": 4.2,
            "average_messages": 4.2,
            "average_code_snippet_percentage": 35.0,
            "average_conversation_length": 250.5,
            "category_distribution": [
                {"value": "bug_fix", "count": 2000, "percentage": 20.0},
                {"value": "feature_request", "count": 1500, "percentage": 15.0},
            ],
            "difficulty_distribution": [
                {"value": "easy", "count": 3000, "percentage": 30.0},
                {"value": "medium", "count": 4000, "percentage": 40.0},
                {"value": "hard", "count": 3000, "percentage": 30.0},
            ],
            "intent_distribution": [
                {"value": "ask_for_implementation", "count": 2000, "percentage": 20.0},
            ],
            "programming_language_distribution": [
                {"value": "python", "count": 3000, "percentage": 30.0},
                {"value": "typescript", "count": 2000, "percentage": 20.0},
            ],
        },
        taxonomy=default_taxonomy(),
        examples=examples,
    )


@pytest.fixture
def input_dir(tmp_path, sample_context) -> Path:
    """Create a temporary input directory with dataset files."""
    # dataset_info.json
    info = {
        "name": sample_context.dataset_name,
        "version": sample_context.version,
        "description": sample_context.description,
        "license": sample_context.license,
        "generator_model": sample_context.generator_model,
        "created_at": sample_context.created_at,
        "synthetic": sample_context.synthetic,
        "splits": sample_context.splits,
    }
    (tmp_path / "dataset_info.json").write_text(json.dumps(info), encoding="utf-8")

    # statistics.json
    (tmp_path / "statistics.json").write_text(
        json.dumps(sample_context.statistics), encoding="utf-8"
    )

    # train.jsonl
    with (tmp_path / "train.jsonl").open("w", encoding="utf-8") as f:
        for conv in sample_context.examples:
            f.write(json.dumps(conv.to_dict(), ensure_ascii=True) + "\n")

    return tmp_path


# ---------------------------------------------------------------------------
# DocContext tests
# ---------------------------------------------------------------------------

class TestDocContext:
    def test_load_from_directory(self, input_dir):
        ctx = load_doc_context(input_dir)
        assert ctx.dataset_name == "DevSynth"
        assert ctx.version == "1.0.0"
        assert ctx.total_conversations == 10000
        assert len(ctx.examples) == 5

    def test_load_missing_files_uses_defaults(self, tmp_path):
        ctx = load_doc_context(tmp_path)
        assert ctx.dataset_name == "DevSynth"
        assert ctx.statistics == {}
        assert ctx.examples == []

    def test_total_conversations_from_splits(self):
        ctx = DocContext(splits={"train": 100, "validation": 10, "test": 10})
        assert ctx.total_conversations == 120

    def test_num_categories(self, sample_context):
        assert sample_context.num_categories == len(default_taxonomy().categories)


# ---------------------------------------------------------------------------
# README generator tests
# ---------------------------------------------------------------------------

class TestReadmeGenerator:
    def test_generates_valid_markdown(self, sample_context):
        readme = generate_readme(sample_context)
        assert "# DevSynth" in readme

    def test_contains_required_sections(self, sample_context):
        readme = generate_readme(sample_context)
        required = [
            "Features", "Dataset Overview", "Folder Structure",
            "Installation", "Dataset Schema", "Categories",
            "Statistics", "Example Conversation", "Generated",
            "Validation Pipeline", "Roadmap", "Version",
            "License", "Citation", "Contributing",
        ]
        for section in required:
            assert section in readme, f"Missing section: {section}"

    def test_statistics_populated(self, sample_context):
        readme = generate_readme(sample_context)
        assert "10,000" in readme  # total conversations
        assert "4.2" in readme  # average turns

    def test_version_appears(self, sample_context):
        readme = generate_readme(sample_context)
        assert "1.0.0" in readme

    def test_no_statistics_fallback(self):
        ctx = DocContext()
        readme = generate_readme(ctx)
        assert "not available" in readme.lower() or "Statistics" in readme


# ---------------------------------------------------------------------------
# Dataset Card generator tests
# ---------------------------------------------------------------------------

class TestDatasetCardGenerator:
    def test_generates_yaml_header(self, sample_context):
        card = generate_dataset_card(sample_context)
        assert card.startswith("---")
        assert "license:" in card

    def test_contains_hf_sections(self, sample_context):
        card = generate_dataset_card(sample_context)
        required = [
            "Motivation", "Intended Use", "Out-of-Scope",
            "Dataset Structure", "Schema", "Supported Tasks",
            "Languages", "Categories", "Statistics",
            "Generation Methodology", "Validation Methodology",
            "Known Limitations", "Bias Discussion",
            "Privacy Statement", "Ethical Considerations", "Citation",
        ]
        for section in required:
            assert section in card, f"Missing section: {section}"

    def test_statistics_tables(self, sample_context):
        card = generate_dataset_card(sample_context)
        assert "bug_fix" in card
        assert "20.0%" in card


# ---------------------------------------------------------------------------
# Changelog generator tests
# ---------------------------------------------------------------------------

class TestChangelogGenerator:
    def test_contains_version(self, sample_context):
        changelog = generate_changelog(sample_context)
        assert "v1.0.0" in changelog

    def test_contains_date(self, sample_context):
        changelog = generate_changelog(sample_context)
        assert "2025-01-15" in changelog

    def test_contains_counts(self, sample_context):
        changelog = generate_changelog(sample_context)
        assert "10,000" in changelog


# ---------------------------------------------------------------------------
# Citation generator tests
# ---------------------------------------------------------------------------

class TestCitationGenerator:
    def test_cff_format(self, sample_context):
        citation = generate_citation(sample_context)
        assert "cff-version: 1.2.0" in citation
        assert 'version: "1.0.0"' in citation

    def test_contains_metadata(self, sample_context):
        citation = generate_citation(sample_context)
        assert "DevSynth" in citation
        assert "Apache-2.0" in citation
        assert "2025-01-15" in citation


# ---------------------------------------------------------------------------
# Schema generator tests
# ---------------------------------------------------------------------------

class TestSchemaGenerator:
    def test_documents_conversation_fields(self, sample_context):
        schema = generate_schema(sample_context)
        assert "## Conversation" in schema
        assert "`id`" in schema
        assert "`messages`" in schema
        assert "`difficulty`" in schema

    def test_documents_message_fields(self, sample_context):
        schema = generate_schema(sample_context)
        assert "## Message" in schema
        assert "`role`" in schema
        assert "`content`" in schema

    def test_documents_code_snippet_fields(self, sample_context):
        schema = generate_schema(sample_context)
        assert "## CodeSnippet" in schema

    def test_contains_required_indicators(self, sample_context):
        schema = generate_schema(sample_context)
        assert "✅" in schema  # required
        assert "❌" in schema  # optional


# ---------------------------------------------------------------------------
# Taxonomy generator tests
# ---------------------------------------------------------------------------

class TestTaxonomyGenerator:
    def test_contains_all_sections(self, sample_context):
        taxonomy = generate_taxonomy(sample_context)
        sections = [
            "Categories", "Subcategories", "Intents",
            "Difficulty Levels", "Learning Stages",
            "Programming Languages", "Roles", "Tools",
            "Interaction Patterns",
        ]
        for section in sections:
            assert section in taxonomy, f"Missing section: {section}"

    def test_lists_taxonomy_values(self, sample_context):
        taxonomy = generate_taxonomy(sample_context)
        for cat in default_taxonomy().categories:
            assert f"`{cat}`" in taxonomy
        for lang in default_taxonomy().languages:
            assert f"`{lang}`" in taxonomy


# ---------------------------------------------------------------------------
# Pipeline generator tests
# ---------------------------------------------------------------------------

class TestPipelineGenerator:
    def test_contains_all_stages(self, sample_context):
        pipeline = generate_pipeline(sample_context)
        stages = [
            "Scenario Generation", "Prompt Construction",
            "OpenRouter", "Schema Validation",
            "PII Detection", "Deduplication",
            "Quality Scoring", "Dataset Split",
        ]
        for stage in stages:
            assert stage in pipeline, f"Missing stage: {stage}"

    def test_contains_generator_model(self, sample_context):
        pipeline = generate_pipeline(sample_context)
        assert sample_context.generator_model in pipeline


# ---------------------------------------------------------------------------
# Examples generator tests
# ---------------------------------------------------------------------------

class TestExamplesGenerator:
    def test_contains_curated_examples(self, sample_context):
        examples = generate_examples(sample_context)
        titles = [
            "Beginner", "Intermediate", "Advanced",
            "Debugging", "Code Review", "Optimization",
            "System Design",
        ]
        for title in titles:
            assert title in examples, f"Missing example: {title}"

    def test_includes_dataset_examples_when_available(self, sample_context):
        examples = generate_examples(sample_context)
        assert "From the Dataset" in examples

    def test_no_dataset_examples_graceful(self):
        ctx = DocContext()
        examples = generate_examples(ctx)
        assert "Curated Examples" in examples
        assert "From the Dataset" not in examples

    def test_all_examples_marked_synthetic(self, sample_context):
        examples = generate_examples(sample_context)
        assert "synthetically generated" in examples.lower()


# ---------------------------------------------------------------------------
# DocumentationGenerator (orchestrator) tests
# ---------------------------------------------------------------------------

class TestDocumentationGenerator:
    def test_generates_all_files(self, input_dir, tmp_path):
        output_dir = tmp_path / "output"
        gen = DocumentationGenerator(input_dir=input_dir, output_dir=output_dir)
        files = gen.generate_all()

        expected = [
            "README.md", "DATASET_CARD.md", "CHANGELOG.md",
            "CITATION.cff", "LICENSE",
            "docs/schema.md", "docs/taxonomy.md",
            "docs/generation_pipeline.md", "docs/examples.md",
        ]
        for name in expected:
            assert name in files, f"Missing file: {name}"
            assert files[name].exists(), f"File not created: {name}"

    def test_files_are_nonempty(self, input_dir, tmp_path):
        output_dir = tmp_path / "output"
        gen = DocumentationGenerator(input_dir=input_dir, output_dir=output_dir)
        files = gen.generate_all()

        for name, path in files.items():
            content = path.read_text(encoding="utf-8")
            assert len(content) > 0, f"Empty file: {name}"

    def test_does_not_overwrite_existing_license(self, input_dir, tmp_path):
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        license_path = output_dir / "LICENSE"
        license_path.write_text("Custom License Text", encoding="utf-8")

        gen = DocumentationGenerator(input_dir=input_dir, output_dir=output_dir)
        files = gen.generate_all()

        assert "LICENSE" not in files  # Should not be in generated dict
        assert license_path.read_text() == "Custom License Text"

    def test_creates_docs_subdirectory(self, input_dir, tmp_path):
        output_dir = tmp_path / "output"
        gen = DocumentationGenerator(input_dir=input_dir, output_dir=output_dir)
        gen.generate_all()

        assert (output_dir / "docs").is_dir()
        assert (output_dir / "docs" / "schema.md").exists()

    def test_with_custom_context(self, tmp_path):
        ctx = DocContext(dataset_name="CustomDS", version="2.0.0")
        gen = DocumentationGenerator(
            input_dir=tmp_path,
            output_dir=tmp_path / "out",
            context=ctx,
        )
        files = gen.generate_all()

        readme = files["README.md"].read_text()
        assert "# CustomDS" in readme
        assert "2.0.0" in readme

    def test_readme_version_awareness(self, input_dir, tmp_path):
        """Version from dataset_info.json should appear in all relevant docs."""
        output_dir = tmp_path / "output"
        gen = DocumentationGenerator(input_dir=input_dir, output_dir=output_dir)
        files = gen.generate_all()

        for name in ("README.md", "DATASET_CARD.md", "CHANGELOG.md", "CITATION.cff"):
            content = files[name].read_text()
            assert "1.0.0" in content, f"Version missing from {name}"
