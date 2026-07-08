"""Generate docs/generation_pipeline.md documenting the full workflow."""

from __future__ import annotations

from .doc_context import DocContext


def generate_pipeline(ctx: DocContext) -> str:
    """Produce docs/generation_pipeline.md from the documentation context."""
    return f"""# Generation Pipeline

Complete documentation of the **{ctx.dataset_name}** data generation workflow (v{ctx.version}).

## Overview

The pipeline transforms a taxonomy of developer scenarios into validated, quality-scored conversations through eight sequential stages.

```text
Taxonomy → Scenario → Prompt → LLM → Validation → Dedup → Quality → Split
```

## 1. Scenario Generation

**Module**: `devsynth_generator/generator/scenario_generator.py`

The `CoverageMatrix` builds a balanced list of scenario specifications from the taxonomy, ensuring even distribution across categories, difficulties, intents, learning stages, and conversation lengths.

Each specification (`ScenarioSpec`) defines:
- Category and subcategory
- Intent
- Difficulty level
- Learning stage
- Conversation length
- Programming language

The `ScenarioGenerator` produces deterministic seed conversations from these specs, which serve as prompts for LLM generation.

## 2. Prompt Construction

**Module**: `devsynth_generator/prompts/prompt_builder.py`

The `PromptBuilder` loads a template (`conversation_seed.txt`) and injects:
- Scenario metadata (category, difficulty, intent, etc.)
- The full JSON output schema (`conversation.schema.json`)
- Additional context variables

The resulting prompt instructs the LLM to produce a realistic developer-assistant conversation matching the specified parameters.

## 3. OpenRouter LLM Generation

**Module**: `devsynth_generator/clients/openrouter_client.py`

The `OpenRouterClient` sends the constructed prompt to an OpenAI-compatible API via OpenRouter with:
- Configurable model selection (default: `{ctx.generator_model}`)
- Temperature control for output diversity
- JSON response format enforcement
- Automatic retry with exponential backoff
- Cumulative token usage tracking

## 4. Schema Validation

**Module**: `devsynth_generator/validator/`

Every generated conversation passes through:

### Pydantic Validation
- Field type checking (string, array, object)
- Required field enforcement (`id`, `task_type`, `difficulty`, `language`, `messages`)
- Value constraints (non-empty strings, minimum array lengths)
- Extra field rejection (`extra="forbid"`)

### Taxonomy Validation
- `task_type` against allowed task types
- `category`, `subcategory`, `intent` against taxonomy values
- `difficulty`, `learning_stage`, `conversation_length` against allowed values
- `language` for conversations and all code snippets
- `role` for every message turn
- `tools` against allowed tool names
- `interaction_pattern` against allowed patterns

## 5. PII Detection

**Module**: `devsynth_generator/validator/validation_pipeline.py`

Regex-based scanning for common PII patterns in all message content and code snippets:

| Pattern | Regex | Example Match |
|---------|-------|---------------|
| Email | `[A-Z0-9._%+-]+@[A-Z0-9.-]+\\.[A-Z]{{2,}}` | `dev@example.com` |
| Phone | `(?:\\+?1[-.\\s]?)?(?:\\(?\\d{{3}}\\)?[-.\\s]?)\\d{{3}}[-.\\s]?\\d{{4}}` | `555-123-4567` |
| SSN | `\\d{{3}}-\\d{{2}}-\\d{{4}}` | `123-45-6789` |
| Credit Card | `(?:\\d[ -]*?){{13,16}}` | `4111-1111-1111-1111` |
| IPv4 | `(?:\\d{{1,3}}\\.){{3}}\\d{{1,3}}` | `192.168.1.1` |

Records containing detected PII are flagged with validation errors.

## 6. Semantic Deduplication

**Module**: `devsynth_generator/deduplication/semantic_deduplicator.py`

Near-duplicate conversations are detected using:
- **Sentence-transformers** embeddings (default model: `all-MiniLM-L6-v2`)
- **Cosine similarity** with a configurable threshold (default: 0.92)
- Conversations above the threshold are rejected as duplicates

## 7. Quality Scoring

**Module**: `devsynth_generator/quality/quality_evaluator.py`

An LLM-as-judge evaluates each conversation on four dimensions:

| Dimension | Weight | Description |
|-----------|--------|-------------|
| Technical Accuracy | 35% | Factual correctness of code and explanations |
| Helpfulness | 25% | Whether the response solves the user's problem |
| Clarity | 20% | Structure, conciseness, and readability |
| Realism | 20% | How natural and plausible the conversation feels |

Each dimension receives a 1–10 score, normalized to 0.0–1.0. The weighted average must meet or exceed the configured threshold (default: 0.7) for the conversation to be accepted.

## 8. Dataset Split

**Module**: `devsynth_generator/splitter/split_dataset.py`

The validated, deduplicated, quality-scored conversations are split into:
- **Train** (80%) — for model training
- **Validation** (10%) — for hyperparameter tuning
- **Test** (10%) — for held-out evaluation

The split is **stratified** by `(category, difficulty, intent)` to preserve the distribution of key fields across all splits. A seeded RNG ensures reproducibility.

## Batch Pipeline

**Module**: `devsynth_generator/pipeline/batch_generation.py`

The `BatchGenerationPipeline` orchestrates the full flow:
1. Generate one conversation per request
2. Validate immediately
3. Deduplicate against existing samples
4. Quality-score and reject below threshold
5. Append to JSONL
6. Resume from last valid sample on restart

This enables resumable, fault-tolerant generation of large datasets.
"""
