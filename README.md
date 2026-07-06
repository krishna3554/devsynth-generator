# devsynth-generator

A high-quality synthetic multi-turn developer conversation dataset generator.

## Layout

- `devsynth_generator/config.py` loads environment-aware settings with `pathlib`, `logging`, and `dotenv`.
- `generator/` creates deterministic, scenario-based, and LLM-backed conversations.
- `pipeline/` runs resumable batch generation.
- `clients/` contains the OpenRouter client.
- `parser/` extracts and validates model JSON responses.
- `validator/` validates generated records.
- `exporter/` writes datasets as JSONL or JSON.
- `taxonomy/` defines categories, intents, tools, roles, and difficulty values.
- `prompts/` stores reusable prompt templates.
- `schemas/` stores dataset schemas.
- `datasets/` stores generated sample data.
- `scripts/` contains CLI entry points.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
python -m devsynth_generator.scripts.generate_dataset --count 5
python -m devsynth_generator.scripts.validate_dataset devsynth_generator/datasets/sample_conversations.jsonl
```

## Resumable LLM Generation

Set `OPENROUTER_API_KEY` in `.env`, then run:

```bash
python -m devsynth_generator.scripts.generate_llm_dataset --count 100 --filename llm_conversations.jsonl
```

The batch pipeline generates one conversation per request, validates it, appends it to JSONL immediately, and resumes from the last valid sample if the run is restarted.

Enable semantic duplicate detection with sentence-transformers:

```bash
pip install -e '.[semantic]'
python -m devsynth_generator.scripts.generate_llm_dataset --count 100 --dedupe --dedupe-threshold 0.92
```

## Validation

`devsynth-validate` runs the full validation pipeline: Pydantic schema checks, taxonomy checks, common PII pattern detection, conversation length constraints, and metadata consistency checks.
