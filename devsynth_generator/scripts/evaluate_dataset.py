"""Evaluate quality of an existing conversation dataset."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from devsynth_generator.clients import OpenRouterClient
from devsynth_generator.config import configure_logging, load_settings
from devsynth_generator.models import Conversation
from devsynth_generator.quality import QualityConfig, QualityEvaluator

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Path to a JSONL dataset file to evaluate.")
    parser.add_argument("--threshold", type=float, default=None, help="Minimum overall quality score (0.0-1.0).")
    parser.add_argument("--model", default=None, help="Model to use for quality evaluation.")
    return parser


def main() -> None:
    settings = load_settings()
    configure_logging(settings.log_level)
    args = build_parser().parse_args()

    client = OpenRouterClient(model=args.model, settings=settings)
    config = QualityConfig(
        threshold=args.threshold if args.threshold is not None else settings.quality_threshold,
        temperature=settings.quality_temperature,
        model=args.model or settings.quality_model,
    )
    evaluator = QualityEvaluator(client=client, config=config)

    conversations: list[Conversation] = []
    with open(args.dataset, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                conversations.append(Conversation.model_validate(record))
            except (json.JSONDecodeError, Exception) as error:
                LOGGER.error("Failed to parse line %s: %s", line_number, error)
                continue

    if not conversations:
        print("No valid conversations found in dataset.")
        raise SystemExit(1)

    print(f"\nEvaluating {len(conversations)} conversations (threshold={config.threshold})...\n")
    print(f"{'ID':<40} {'Accuracy':>8} {'Helpful':>8} {'Clarity':>8} {'Realism':>8} {'Overall':>8} {'Status':>8}")
    print("-" * 100)

    failed_count = 0
    for conversation in conversations:
        result = evaluator.evaluate(conversation)
        dim_scores = {dim.name: dim.score for dim in result.dimensions}
        status = "PASS" if result.passed else "FAIL"
        if not result.passed:
            failed_count += 1
        print(
            f"{conversation.id:<40} "
            f"{dim_scores.get('technical_accuracy', '-'):>8} "
            f"{dim_scores.get('helpfulness', '-'):>8} "
            f"{dim_scores.get('clarity', '-'):>8} "
            f"{dim_scores.get('realism', '-'):>8} "
            f"{result.overall_score:>8.4f} "
            f"{status:>8}"
        )

    print("-" * 100)
    passed_count = len(conversations) - failed_count
    print(f"\nResults: {passed_count}/{len(conversations)} passed, {failed_count}/{len(conversations)} failed")
    LOGGER.info("OpenRouter token usage: %s", client.usage.to_dict())

    if failed_count > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
