# devsynth-generator

A high-quality synthetic multi-turn developer conversation dataset generator.

Automatically generates realistic developer conversations with an end-to-end pipeline that includes LLM-backed generation, schema validation, quality evaluation, semantic deduplication, stratified splitting, and publication-ready documentation — all with resumable batch processing and custom output filenames.

## Features

- **Deterministic seed generation** — taxonomy-driven conversations with no API key required
- **LLM-backed generation** — realistic multi-turn conversations via OpenRouter
- **Resumable batch pipeline** — picks up from the last valid sample on restart
- **Comprehensive validation** — schema, taxonomy, PII detection, and metadata checks
- **Quality evaluation** — LLM-scored accuracy, helpfulness, clarity, and realism
- **Semantic deduplication** — cosine-similarity filtering via sentence-transformers
- **Stratified splitting** — train/validation/test splits with per-split statistics
- **Documentation generation** — README, Dataset Card, Changelog, Citation, Schema, and Taxonomy docs

## Project Layout

```
devsynth_generator/
├── config.py               # Environment-aware settings (pathlib + dotenv)
├── models.py               # Pydantic data models
├── clients/                # OpenRouter API client
├── generator/              # Deterministic, scenario-based, and LLM-backed generators
├── pipeline/               # Resumable batch generation pipeline
├── parser/                 # JSON response extraction and validation
├── validator/              # Schema, taxonomy, PII, and metadata validation
├── quality/                # LLM-based quality evaluation
├── deduplication/          # Semantic deduplication (sentence-transformers)
├── splitter/               # Stratified dataset splitting with statistics
├── docs_generator/         # Publication-ready documentation generation
├── exporter/               # JSONL / JSON export
├── taxonomy/               # Categories, intents, tools, roles, difficulty values
├── prompts/                # Reusable prompt templates
├── schemas/                # Dataset schemas
├── datasets/               # Generated sample data
└── scripts/                # CLI entry points
```

## Setup

```bash
# Clone and install
git clone https://github.com/krishna3554/devsynth-generator.git
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

After installing with `pip install -e .`, the following commands are available:

| Command | Module | Description |
|---------|--------|-------------|
| `devsynth-generate` | `scripts.generate_dataset` | Deterministic seed generation |
| `devsynth-generate-llm` | `scripts.generate_llm_dataset` | LLM-backed generation |
| `devsynth-validate` | `scripts.validate_dataset` | Dataset validation |
| `devsynth-evaluate` | `scripts.evaluate_dataset` | Quality evaluation |
| `devsynth-split` | `scripts.split_dataset` | Stratified splitting |
| `devsynth-docs` | `scripts.generate_docs` | Documentation generation |

You can also run each tool as a module with `python -m devsynth_generator.scripts.<name>`.

---

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
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter API base URL |
| `OPENROUTER_MODEL` | `openai/gpt-4o-mini` | Model to use for generation |
| `OPENROUTER_TIMEOUT_SECONDS` | `60` | Request timeout in seconds |
| `OPENROUTER_MAX_RETRIES` | `3` | Max retries on API failure |
| `OPENROUTER_BACKOFF_SECONDS` | `1.0` | Backoff delay between retries |
| `DEVSYNTH_OUTPUT_DIR` | `devsynth_generator/datasets` | Default output directory |
| `DEVSYNTH_DEFAULT_COUNT` | `10` | Default number of conversations to generate |
| `DEVSYNTH_RANDOM_SEED` | `42` | Random seed for reproducibility |
| `DEVSYNTH_LOG_LEVEL` | `INFO` | Logging verbosity |
| `DEVSYNTH_DEDUP_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model for deduplication |
| `DEVSYNTH_DEDUP_THRESHOLD` | `0.92` | Cosine similarity threshold for dedup |
| `DEVSYNTH_QUALITY_MODEL` | *(uses generation model)* | Override model for quality evaluation |
| `DEVSYNTH_QUALITY_THRESHOLD` | `0.7` | Minimum quality score to accept |
| `DEVSYNTH_QUALITY_TEMPERATURE` | `0.2` | Temperature for quality evaluation calls |
| `DEVSYNTH_MAX_GENERATION_RETRIES` | `5` | Max retries per conversation generation |
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
pytest tests/test_batch_generation.py -v
pytest tests/test_semantic_deduplicator.py -v
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

# 3. Generate documentation
python -m devsynth_generator.scripts.generate_docs \
  --input-dir datasets/ --output-dir .
```

## Contributing

```bash
# Install dev dependencies
pip install -e '.[dev]'

# Run the test suite
pytest tests/ -v
```

## License

Apache-2.0 — see [LICENSE](LICENSE).
