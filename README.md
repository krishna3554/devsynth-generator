# devsynth-generator

A high-quality synthetic multi-turn developer conversation dataset generator.
Automatically generates dataset and makes a custom filename output with quality validor.

## Layout

- `devsynth_generator/config.py` loads environment-aware settings with `pathlib`, `logging`, and `dotenv`.
- `generator/` creates deterministic, scenario-based, and LLM-backed conversations.
- `pipeline/` runs resumable batch generation.
- `clients/` contains the OpenRouter client.
- `parser/` extracts and validates model JSON responses.
- `validator/` validates generated records.
- `quality/` LLM-based quality evaluation (accuracy, helpfulness, clarity, realism).
- `splitter/` stratified dataset splitting with statistics.
- `docs_generator/` publication-ready documentation generation.
- `deduplication/` semantic deduplication via sentence-transformers.
- `exporter/` writes datasets as JSONL or JSON.
- `taxonomy/` defines categories, intents, tools, roles, and difficulty values.
- `prompts/` stores reusable prompt templates.
- `schemas/` stores dataset schemas.
- `datasets/` stores generated sample data.
- `scripts/` contains CLI entry points.

## Setup

```bash
# Clone and install
git clone https://github.com/your-org/devsynth-generator.git
cd devsynth-generator
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

# Configure environment
cp .env.example .env
# Edit .env and set OPENROUTER_API_KEY for LLM features
```

## Quick Start

```bash
# Generate 5 deterministic seed conversations (no API key needed)
python -m devsynth_generator.scripts.generate_dataset --count 5

# Validate an existing dataset
python -m devsynth_generator.scripts.validate_dataset devsynth_generator/datasets/sample_conversations.jsonl
```

## CLI Tools

### 1. Generate Seed Conversations

Generate deterministic conversations from the taxonomy (no LLM required):

```bash
python -m devsynth_generator.scripts.generate_dataset --count 50
```

### 2. LLM-Backed Generation

Set `OPENROUTER_API_KEY` in `.env`, then run:

```bash
# Basic generation
python -m devsynth_generator.scripts.generate_llm_dataset --count 100

# With custom output filename
python -m devsynth_generator.scripts.generate_llm_dataset --count 100 --filename my_dataset.jsonl

# With semantic deduplication (requires sentence-transformers)
pip install -e '.[semantic]'
python -m devsynth_generator.scripts.generate_llm_dataset --count 100 --dedupe --dedupe-threshold 0.92

# With quality evaluation gate
python -m devsynth_generator.scripts.generate_llm_dataset --count 100 --quality-eval --quality-threshold 0.7
```

The batch pipeline generates one conversation per request, validates it, appends it to JSONL immediately, and **resumes from the last valid sample** if the run is restarted.

### 3. Validate Dataset

Run the full validation pipeline (schema, taxonomy, PII detection, metadata consistency):

```bash
python -m devsynth_generator.scripts.validate_dataset datasets/conversations.jsonl
```

### 4. Quality Evaluation

Score an existing dataset on technical accuracy, helpfulness, clarity, and realism:

```bash
python -m devsynth_generator.scripts.evaluate_dataset datasets/conversations.jsonl --threshold 0.7
```

### 5. Split Dataset

Split a cleaned dataset into stratified train/validation/test splits:

```bash
# Default 80/10/10 split
python -m devsynth_generator.scripts.split_dataset datasets/cleaned/conversations.jsonl

# Custom ratios and output directory
python -m devsynth_generator.scripts.split_dataset datasets/cleaned/conversations.jsonl \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --seed 42 \
  --output-dir datasets/

# Output files:
#   datasets/train.jsonl
#   datasets/validation.jsonl
#   datasets/test.jsonl
#   datasets/statistics.json
#   datasets/dataset_info.json
```

### 6. Generate Documentation

Generate publication-ready documentation (README, Dataset Card, Changelog, Citation, Schema, Taxonomy, etc.):

```bash
python -m devsynth_generator.scripts.generate_docs \
  --input-dir datasets/ \
  --output-dir .

# Output files:
#   README.md, DATASET_CARD.md, CHANGELOG.md, CITATION.cff, LICENSE
#   docs/schema.md, docs/taxonomy.md, docs/generation_pipeline.md, docs/examples.md
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | — | API key for LLM generation (required for LLM features) |
| `OPENROUTER_MODEL` | `openai/gpt-4o-mini` | Model to use for generation |
| `DEVSYNTH_OUTPUT_DIR` | `devsynth_generator/datasets` | Default output directory |
| `DEVSYNTH_RANDOM_SEED` | `42` | Random seed for reproducibility |
| `DEVSYNTH_LOG_LEVEL` | `INFO` | Logging verbosity |
| `DEVSYNTH_DEDUP_THRESHOLD` | `0.92` | Cosine similarity threshold for dedup |
| `DEVSYNTH_QUALITY_THRESHOLD` | `0.7` | Minimum quality score to accept |
| `DEVSYNTH_SPLIT_TRAIN_RATIO` | `0.8` | Training split ratio |
| `DEVSYNTH_SPLIT_VALIDATION_RATIO` | `0.1` | Validation split ratio |
| `DEVSYNTH_SPLIT_TEST_RATIO` | `0.1` | Test split ratio |

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_quality_evaluator.py -v
pytest tests/test_dataset_splitter.py -v
pytest tests/test_docs_generator.py -v
```

## Full Pipeline Example

```bash
# 1. Generate conversations with LLM + quality gate
python -m devsynth_generator.scripts.generate_llm_dataset \
  --count 1000 --quality-eval --dedupe

# 2. Split into train/val/test
python -m devsynth_generator.scripts.split_dataset \
  devsynth_generator/datasets/conversations.jsonl \
  --output-dir datasets/
<<<<<<< HEAD

=======
>>>>>>> fec7fc8 (Sample-output)
# 3. Generate documentation
python -m devsynth_generator.scripts.generate_docs \
  --input-dir datasets/ --output-dir .
```

## License

Apache-2.0 — see [LICENSE](LICENSE).

