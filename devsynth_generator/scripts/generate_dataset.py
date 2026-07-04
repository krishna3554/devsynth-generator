"""Generate a synthetic developer conversation dataset."""

from __future__ import annotations

import argparse
import logging

from devsynth_generator.config import configure_logging, load_settings
from devsynth_generator.exporter import DatasetExporter
from devsynth_generator.generator import ConversationGenerator
from devsynth_generator.validator import ConversationValidator

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=None, help="Number of conversations to generate.")
    parser.add_argument("--filename", default="sample_conversations.jsonl", help="Output dataset filename.")
    parser.add_argument("--format", choices=("jsonl", "json"), default="jsonl", help="Export format.")
    return parser


def main() -> None:
    settings = load_settings()
    configure_logging(settings.log_level)
    args = build_parser().parse_args()

    count = args.count or settings.default_count
    generator = ConversationGenerator(seed=settings.random_seed)
    conversations = generator.generate(count)

    records = [conversation.to_dict() for conversation in conversations]
    errors = ConversationValidator().validate_many(records)
    if errors:
        for error in errors:
            LOGGER.error("%s %s: %s", error.record_id, error.field, error.message)
        raise SystemExit(1)

    exporter = DatasetExporter(settings.output_dir)
    if args.format == "jsonl":
        path = exporter.to_jsonl(conversations, args.filename)
    else:
        path = exporter.to_json(conversations, args.filename)
    print(path)


if __name__ == "__main__":
    main()
