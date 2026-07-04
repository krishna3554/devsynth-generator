# devsynth-generator
<<<<<<< HEAD
A High-Quality Synthetic Multi-turn Developer Conversation Dataset
=======

Synthetic developer conversation generator with a modular Python architecture.

## Layout

- `devsynth_generator/config.py` loads environment-aware settings with `pathlib`, `logging`, and `dotenv`.
- `generator/` creates structured synthetic conversations.
- `validator/` validates generated records.
- `exporter/` writes datasets as JSONL or JSON.
- `taxonomy/` defines tasks, roles, tools, and difficulty levels.
- `prompts/` stores reusable prompt templates.
- `schemas/` stores dataset schemas.
- `datasets/` stores generated sample data.
- `scripts/` contains CLI entry points.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
python -m devsynth_generator.scripts.generate_dataset --count 5
python -m devsynth_generator.scripts.validate_dataset devsynth_generator/datasets/sample_conversations.jsonl
```
>>>>>>> e75d2db (generator, validator, exporter, taxonomy, prompts, schemas, datasets, and scripts directories)
