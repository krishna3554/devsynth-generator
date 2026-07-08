"""Generate publication-ready documentation from dataset metadata."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from devsynth_generator.config import configure_logging, load_settings
from devsynth_generator.docs_generator import DocumentationGenerator

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Directory containing dataset_info.json, statistics.json, and train.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for generated documentation (default: project root).",
    )
    return parser


def main() -> None:
    settings = load_settings()
    configure_logging(settings.log_level)
    args = build_parser().parse_args()

    input_dir = Path(args.input_dir) if args.input_dir else settings.output_dir
    output_dir = Path(args.output_dir) if args.output_dir else Path(".")

    LOGGER.info("Generating documentation from %s → %s", input_dir, output_dir)

    generator = DocumentationGenerator(input_dir=input_dir, output_dir=output_dir)
    generated = generator.generate_all()

    print(f"\nDocumentation generated successfully!")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Files:  {len(generated)}")
    for name, path in sorted(generated.items()):
        print(f"    {name}: {path}")


if __name__ == "__main__":
    main()
