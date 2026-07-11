"""Generate a synthetic developer conversation dataset with OpenRouter."""

from __future__ import annotations

import argparse
import logging

from devsynth_generator.clients import OpenRouterClient
from devsynth_generator.config import configure_logging, load_settings
from devsynth_generator.deduplication import SemanticDeduplicator
from devsynth_generator.exporter import DatasetExporter
from devsynth_generator.generator import LLMScenarioGenerator
from devsynth_generator.pipeline import BatchGenerationPipeline

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=None, help="Number of conversations to generate.")
    parser.add_argument("--filename", default="llm_conversations.jsonl", help="Output dataset filename.")
    parser.add_argument("--format", choices=("jsonl", "json"), default="jsonl", help="Export format.")
    parser.add_argument("--model", default=None, help="OpenRouter model id.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Maximum completion tokens.")
    parser.add_argument("--max-parse-retries", type=int, default=2, help="Retries for invalid model JSON.")
    parser.add_argument("--dedupe", action="store_true", help="Enable semantic duplicate detection before append.")
    parser.add_argument("--dedupe-threshold", type=float, default=None, help="Cosine similarity duplicate threshold.")
    parser.add_argument("--quality-eval", action="store_true", help="Enable LLM quality evaluation gate.")
    parser.add_argument("--quality-threshold", type=float, default=None, help="Minimum overall quality score (0.0-1.0).")
    parser.add_argument("--quality-model", default=None, help="Model to use for quality evaluation.")
    parser.add_argument("--max-retries", type=int, default=None, help="Max retries per sample on validation failure (default: 5).")
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
        max_parse_retries=args.max_parse_retries,
    )

    exporter = DatasetExporter(settings.output_dir)
    if args.format != "jsonl":
        raise SystemExit("Resumable LLM generation currently supports JSONL output only")
    deduplicator = SemanticDeduplicator(threshold=args.dedupe_threshold) if args.dedupe else None

    quality_evaluator = None
    if args.quality_eval:
        from devsynth_generator.quality import QualityConfig, QualityEvaluator

        quality_config = QualityConfig(
            threshold=args.quality_threshold if args.quality_threshold is not None else settings.quality_threshold,
            temperature=settings.quality_temperature,
            model=args.quality_model or settings.quality_model,
        )
        quality_evaluator = QualityEvaluator(client=client, config=quality_config)
        LOGGER.info(
            "Quality evaluation enabled: threshold=%.2f model=%s",
            quality_config.threshold,
            quality_config.model or client.model,
        )

    max_retries = args.max_retries if args.max_retries is not None else settings.generation_max_retries
    result = BatchGenerationPipeline(
        generator=generator,
        exporter=exporter,
        deduplicator=deduplicator,
        quality_evaluator=quality_evaluator,
        max_retries=max_retries,
    ).run(
        count=args.count or settings.default_count,
        filename=args.filename,
    )
    path = result.path
    LOGGER.info(
        "Batch complete path=%s existing=%s generated=%s total=%s requested=%s",
        path,
        result.existing_count,
        result.generated_count,
        result.total_count,
        result.requested_count,
    )
    LOGGER.info("OpenRouter token usage: %s", client.usage.to_dict())
    print(path)


if __name__ == "__main__":
    main()

