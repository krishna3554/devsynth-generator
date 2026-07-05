"""Generate a synthetic developer conversation dataset with OpenRouter."""

from __future__ import annotations

import argparse
import logging

from devsynth_generator.clients import OpenRouterClient
from devsynth_generator.config import configure_logging, load_settings
from devsynth_generator.exporter import DatasetExporter
from devsynth_generator.generator import LLMScenarioGenerator
from devsynth_generator.validator import ConversationValidator

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=None, help="Number of conversations to generate.")
    parser.add_argument("--filename", default="llm_conversations.jsonl", help="Output dataset filename.")
    parser.add_argument("--format", choices=("jsonl", "json"), default="jsonl", help="Export format.")
    parser.add_argument("--model", default=None, help="OpenRouter model id.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Maximum completion tokens.")
    return parser


def main() -> None:
    settings = load_settings()
    configure_logging(settings.log_level)
    args = build_parser().parse_args()

    client = OpenRouterClient(model=args.model, settings=settings)
    generator = LLMScenarioGenerator(
        client=client,
        seed=settings.random_seed,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    conversations = generator.generate(args.count or settings.default_count)

    errors = ConversationValidator().validate_many([conversation.to_dict() for conversation in conversations])
    if errors:
        for error in errors:
            LOGGER.error("%s %s: %s", error.record_id, error.field, error.message)
        raise SystemExit(1)

    exporter = DatasetExporter(settings.output_dir)
    if args.format == "jsonl":
        path = exporter.to_jsonl(conversations, args.filename)
    else:
        path = exporter.to_json(conversations, args.filename)
    LOGGER.info("OpenRouter token usage: %s", client.usage.to_dict())
    print(path)


if __name__ == "__main__":
    main()
