"""Orchestrate generation of all documentation files."""

from __future__ import annotations

import logging
from pathlib import Path

from .changelog_generator import generate_changelog
from .citation_generator import generate_citation
from .dataset_card_generator import generate_dataset_card
from .doc_context import DocContext, load_doc_context
from .examples_generator import generate_examples
from .pipeline_generator import generate_pipeline
from .readme_generator import generate_readme
from .schema_generator import generate_schema
from .taxonomy_generator import generate_taxonomy

LOGGER = logging.getLogger(__name__)

APACHE_2_HEADER = """                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
   ...

   See https://www.apache.org/licenses/LICENSE-2.0 for the full license text.
"""


class DocumentationGenerator:
    """Generate all publication-ready documentation from dataset metadata.

    Reads ``dataset_info.json``, ``statistics.json``, and sample
    conversations from the input directory, then writes:

    - ``README.md``
    - ``DATASET_CARD.md``
    - ``CHANGELOG.md``
    - ``CITATION.cff``
    - ``LICENSE`` (only if missing)
    - ``docs/schema.md``
    - ``docs/taxonomy.md``
    - ``docs/generation_pipeline.md``
    - ``docs/examples.md``
    """

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        *,
        context: DocContext | None = None,
    ) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.context = context or load_doc_context(input_dir)

    def generate_all(self) -> dict[str, Path]:
        """Generate all documentation files and return a mapping of name → path."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        docs_dir = self.output_dir / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)

        generated: dict[str, Path] = {}

        # Top-level files.
        generated["README.md"] = self._write(
            self.output_dir / "README.md",
            generate_readme(self.context),
        )
        generated["DATASET_CARD.md"] = self._write(
            self.output_dir / "DATASET_CARD.md",
            generate_dataset_card(self.context),
        )
        generated["CHANGELOG.md"] = self._write(
            self.output_dir / "CHANGELOG.md",
            generate_changelog(self.context),
        )
        generated["CITATION.cff"] = self._write(
            self.output_dir / "CITATION.cff",
            generate_citation(self.context),
        )

        # LICENSE — only create if missing.
        license_path = self.output_dir / "LICENSE"
        if not license_path.exists():
            generated["LICENSE"] = self._write(license_path, APACHE_2_HEADER)
            LOGGER.info("Created LICENSE file (was missing)")
        else:
            LOGGER.info("LICENSE already exists, skipping")

        # docs/ subdirectory.
        generated["docs/schema.md"] = self._write(
            docs_dir / "schema.md",
            generate_schema(self.context),
        )
        generated["docs/taxonomy.md"] = self._write(
            docs_dir / "taxonomy.md",
            generate_taxonomy(self.context),
        )
        generated["docs/generation_pipeline.md"] = self._write(
            docs_dir / "generation_pipeline.md",
            generate_pipeline(self.context),
        )
        generated["docs/examples.md"] = self._write(
            docs_dir / "examples.md",
            generate_examples(self.context),
        )

        LOGGER.info("Generated %s documentation files in %s", len(generated), self.output_dir)
        return generated

    def _write(self, path: Path, content: str) -> Path:
        """Write content to a file and log the action."""
        path.write_text(content, encoding="utf-8")
        LOGGER.info("Wrote %s (%s bytes)", path, len(content))
        return path
