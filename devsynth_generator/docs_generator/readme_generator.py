"""Generate a professional GitHub README.md."""

from __future__ import annotations

import json
from typing import Any

from .doc_context import DocContext


def generate_readme(ctx: DocContext) -> str:
    """Produce a complete README.md from the documentation context."""
    sections = [
        _header(ctx),
        _features(ctx),
        _dataset_overview(ctx),
        _folder_structure(),
        _installation(),
        _dataset_schema(),
        _categories(ctx),
        _statistics_summary(ctx),
        _example_conversation(ctx),
        _generation_methodology(),
        _validation_pipeline(),
        _roadmap(),
        _version_info(ctx),
        _license_section(ctx),
        _citation(ctx),
        _contributing(),
    ]
    return "\n\n---\n\n".join(sections) + "\n"


# ------------------------------------------------------------------
# Section builders
# ------------------------------------------------------------------

def _header(ctx: DocContext) -> str:
    return f"""# {ctx.dataset_name}

> {ctx.description}

[![License: {ctx.license}](https://img.shields.io/badge/License-{ctx.license.replace('-', '--')}-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-{ctx.version}-green.svg)]()
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)]()"""


def _features(ctx: DocContext) -> str:
    total = ctx.total_conversations
    return f"""## ✨ Features

- **{total:,}** synthetic multi-turn developer conversations
- **{ctx.num_categories}** technical categories covering real-world scenarios
- **{len(ctx.taxonomy.languages)}** programming languages: {', '.join(ctx.taxonomy.languages)}
- **{len(ctx.taxonomy.difficulties)}** difficulty levels: {', '.join(ctx.taxonomy.difficulties)}
- Stratified train / validation / test splits
- Comprehensive schema validation and PII detection
- LLM-based quality scoring on 4 dimensions
- Semantic deduplication
- Publication-ready statistics and metadata"""


def _dataset_overview(ctx: DocContext) -> str:
    splits = ctx.splits
    lines = ["## 📊 Dataset Overview", ""]
    lines.append("| Split | Conversations |")
    lines.append("|-------|--------------|")
    for split_name in ("train", "validation", "test"):
        count = splits.get(split_name, 0)
        lines.append(f"| {split_name.capitalize()} | {count:,} |")
    lines.append(f"| **Total** | **{ctx.total_conversations:,}** |")
    return "\n".join(lines)


def _folder_structure() -> str:
    return """## 📁 Folder Structure

```text
devsynth-generator/
├── devsynth_generator/
│   ├── clients/          # OpenRouter API client
│   ├── generator/        # Scenario & LLM generation
│   ├── validator/        # Schema, taxonomy, PII validation
│   ├── quality/          # LLM quality evaluation
│   ├── splitter/         # Stratified dataset splitting
│   ├── docs_generator/   # Documentation generation
│   ├── exporter/         # JSONL / JSON export
│   ├── deduplication/    # Semantic deduplication
│   ├── parser/           # Model response parsing
│   ├── prompts/          # Prompt templates
│   ├── schemas/          # JSON schemas
│   ├── taxonomy/         # Category definitions
│   ├── pipeline/         # Batch generation pipeline
│   └── scripts/          # CLI entry points
├── datasets/             # Generated datasets
├── docs/                 # Generated documentation
└── tests/                # Unit tests
```"""


def _installation() -> str:
    return """## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/your-org/devsynth-generator.git
cd devsynth-generator

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -e '.[dev]'

# Copy environment configuration
cp .env.example .env
```"""


def _dataset_schema() -> str:
    return """## 📋 Dataset Schema

Each conversation record contains:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | ✅ | Unique conversation identifier |
| `task_type` | string | ✅ | Type of development task |
| `category` | string | ❌ | Technical category |
| `subcategory` | string | ❌ | Specific subcategory |
| `intent` | string | ❌ | User's intent |
| `difficulty` | string | ✅ | easy, medium, or hard |
| `learning_stage` | string | ❌ | Developer experience level |
| `conversation_length` | string | ❌ | short, medium, or long |
| `language` | string | ✅ | Programming language |
| `messages` | array | ✅ | Conversation turns |
| `code_snippets` | array | ❌ | Attached code samples |
| `tools` | array | ❌ | Tools referenced |
| `interaction_pattern` | string | ❌ | Conversation style |
| `metadata` | object | ❌ | Quality scores, provenance |

See [docs/schema.md](docs/schema.md) for detailed field documentation."""


def _categories(ctx: DocContext) -> str:
    cats = "\n".join(f"- `{c}`" for c in ctx.taxonomy.categories)
    return f"""## 🏷️ Categories

{cats}

See [docs/taxonomy.md](docs/taxonomy.md) for the full taxonomy."""


def _statistics_summary(ctx: DocContext) -> str:
    stats = ctx.statistics
    if not stats:
        return "## 📈 Statistics\n\n*Statistics not available. Run the split pipeline to generate.*"

    lines = ["## 📈 Statistics", ""]
    lines.append(f"- **Total conversations**: {stats.get('total_conversations', 'N/A'):,}")
    lines.append(f"- **Average turns**: {stats.get('average_turns', 'N/A')}")
    lines.append(f"- **Average messages**: {stats.get('average_messages', 'N/A')}")
    lines.append(f"- **Code snippet coverage**: {stats.get('average_code_snippet_percentage', 'N/A')}%")
    lines.append(f"- **Average conversation length**: {stats.get('average_conversation_length', 'N/A'):,.0f} characters")
    lines.append("")

    # Difficulty distribution.
    diff_dist = stats.get("difficulty_distribution", [])
    if diff_dist:
        lines.append("### Difficulty Distribution")
        lines.append("")
        lines.append("| Difficulty | Count | Percentage |")
        lines.append("|-----------|-------|------------|")
        for entry in diff_dist:
            lines.append(f"| {entry['value']} | {entry['count']:,} | {entry['percentage']}% |")
        lines.append("")

    # Category distribution.
    cat_dist = stats.get("category_distribution", [])
    if cat_dist:
        lines.append("### Category Distribution")
        lines.append("")
        lines.append("| Category | Count | Percentage |")
        lines.append("|----------|-------|------------|")
        for entry in cat_dist:
            lines.append(f"| {entry['value']} | {entry['count']:,} | {entry['percentage']}% |")

    return "\n".join(lines)


def _example_conversation(ctx: DocContext) -> str:
    if not ctx.examples:
        return "## 💬 Example Conversation\n\n*No examples available.*"

    conv = ctx.examples[0]
    msgs = "\n".join(f"  **{m.role.capitalize()}**: {m.content}" for m in conv.messages)
    return f"""## 💬 Example Conversation

> **Category**: `{conv.category}` · **Difficulty**: `{conv.difficulty}` · **Language**: `{conv.language}`

{msgs}

See [docs/examples.md](docs/examples.md) for more examples."""


def _generation_methodology() -> str:
    return """## ⚙️ How the Dataset Was Generated

1. **Scenario Generation** — Coverage-balanced scenarios from the taxonomy
2. **Prompt Construction** — Structured prompts with JSON output schemas
3. **LLM Generation** — Multi-turn conversations via OpenRouter API
4. **Schema Validation** — Pydantic model + taxonomy checks
5. **PII Detection** — Regex patterns for emails, phones, SSNs, credit cards
6. **Deduplication** — Semantic similarity via sentence-transformers
7. **Quality Scoring** — LLM-as-judge on accuracy, helpfulness, clarity, realism
8. **Dataset Split** — Stratified 80/10/10 split preserving distributions

See [docs/generation_pipeline.md](docs/generation_pipeline.md) for details."""


def _validation_pipeline() -> str:
    return """## ✅ Validation Pipeline

Every conversation passes through:

1. **Pydantic schema validation** — field types, constraints, required fields
2. **Taxonomy validation** — categories, intents, languages against allowed values
3. **PII detection** — emails, phone numbers, SSNs, credit cards, IP addresses
4. **Conversation length constraints** — turn counts match declared length
5. **Metadata consistency** — turn counts, coverage matrix alignment
6. **Quality evaluation** — 4-dimension LLM scoring with configurable threshold"""


def _roadmap() -> str:
    return """## 🗺️ Roadmap

- [ ] Scale to 1M+ conversations
- [ ] Add more programming languages
- [ ] Multi-language (natural language) support
- [ ] Fine-tuning benchmarks
- [ ] Hugging Face Hub integration
- [ ] Interactive dataset explorer"""


def _version_info(ctx: DocContext) -> str:
    return f"""## 📌 Version

**Current version**: `{ctx.version}`

Created: {ctx.created_at}"""


def _license_section(ctx: DocContext) -> str:
    return f"""## 📄 License

This project is licensed under the **{ctx.license}** license. See [LICENSE](LICENSE) for details."""


def _citation(ctx: DocContext) -> str:
    return f"""## 📚 Citation

```bibtex
@dataset{{devsynth_{ctx.version.replace('.', '_')},
  title   = {{{ctx.dataset_name}: Synthetic Developer Conversation Dataset}},
  version = {{{ctx.version}}},
  year    = {{{ctx.created_at[:4]}}},
  license = {{{ctx.license}}},
  url     = {{https://github.com/your-org/devsynth-generator}}
}}
```

See [CITATION.cff](CITATION.cff) for the machine-readable citation."""


def _contributing() -> str:
    return """## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes with tests
4. Run the test suite (`pytest tests/ -v`)
5. Submit a pull request

Please ensure all tests pass and code follows existing patterns."""
