"""Split a conversation dataset into train/validation/test splits."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from devsynth_generator.config import configure_logging, load_settings
from devsynth_generator.splitter import (
    DatasetSplitter,
    SplitConfig,
    build_dataset_info,
    compute_statistics,
    save_dataset_info,
    save_statistics,
)

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Path to the input JSONL dataset file.")
    parser.add_argument("--output-dir", default=None, help="Output directory for split files.")
    parser.add_argument("--train-ratio", type=float, default=None, help="Training split ratio (default: 0.8).")
    parser.add_argument("--val-ratio", type=float, default=None, help="Validation split ratio (default: 0.1).")
    parser.add_argument("--test-ratio", type=float, default=None, help="Test split ratio (default: 0.1).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    return parser


def main() -> None:
    settings = load_settings()
    configure_logging(settings.log_level)
    args = build_parser().parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else settings.output_dir

    config = SplitConfig(
        train_ratio=args.train_ratio if args.train_ratio is not None else settings.split_train_ratio,
        validation_ratio=args.val_ratio if args.val_ratio is not None else settings.split_validation_ratio,
        test_ratio=args.test_ratio if args.test_ratio is not None else settings.split_test_ratio,
        seed=args.seed if args.seed is not None else settings.random_seed,
        input_path=input_path,
        output_dir=output_dir,
    )

    LOGGER.info(
        "Splitting dataset: input=%s, output=%s, ratios=%.2f/%.2f/%.2f, seed=%s",
        config.input_path,
        config.output_dir,
        config.train_ratio,
        config.validation_ratio,
        config.test_ratio,
        config.seed,
    )

    splitter = DatasetSplitter(config)
    result = splitter.run()

    # Compute and save statistics.
    all_conversations = result.train + result.validation + result.test
    stats = compute_statistics(all_conversations, result.train, result.validation, result.test)
    stats_path = save_statistics(stats, config.output_dir)

    # Build and save dataset info.
    info = build_dataset_info(
        train_size=len(result.train),
        validation_size=len(result.validation),
        test_size=len(result.test),
        generator_model=settings.openrouter_model,
    )
    info_path = save_dataset_info(info, config.output_dir)

    # Summary.
    print(f"\nDataset split complete!")
    print(f"  Input:        {config.input_path}")
    print(f"  Output:       {config.output_dir}")
    print(f"  Train:        {len(result.train)} conversations")
    print(f"  Validation:   {len(result.validation)} conversations")
    print(f"  Test:         {len(result.test)} conversations")
    print(f"  Skipped:      {result.skipped_count} invalid records")
    print(f"  Deduped:      {result.duplicates_removed} duplicates removed")
    print(f"  Statistics:   {stats_path}")
    print(f"  Dataset info: {info_path}")


if __name__ == "__main__":
    main()
