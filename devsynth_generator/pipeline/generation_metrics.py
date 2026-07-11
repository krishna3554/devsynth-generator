"""Metrics tracking for batch generation runs."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class GenerationMetrics:
    """Track successes, failures, retries, and timing across a generation run."""

    requested: int = 0
    generated: int = 0
    skipped: int = 0
    total_retries: int = 0
    api_errors: int = 0
    validation_failures: Counter = field(default_factory=Counter)
    generation_times: list[float] = field(default_factory=list)

    # Internal: start time for the current sample.
    _sample_start: float = field(default=0.0, repr=False)

    def start_sample(self) -> None:
        """Mark the start of a sample generation attempt."""
        self._sample_start = time.monotonic()

    def record_success(self, attempts: int) -> None:
        """Record a successful generation with the number of attempts it took."""
        self.generated += 1
        self.total_retries += max(0, attempts - 1)
        if self._sample_start > 0:
            self.generation_times.append(time.monotonic() - self._sample_start)

    def record_retry(self, reason: str) -> None:
        """Record a failed attempt that will be retried."""
        self.validation_failures[reason] += 1

    def record_api_error(self) -> None:
        """Record an API-level error (timeout, rate limit, etc.)."""
        self.api_errors += 1

    def record_skip(self) -> None:
        """Record a sample skipped after exhausting all retries."""
        self.skipped += 1

    @property
    def average_generation_time(self) -> float:
        """Average generation time per successful sample in seconds."""
        if not self.generation_times:
            return 0.0
        return sum(self.generation_times) / len(self.generation_times)

    @property
    def total_validation_failures(self) -> int:
        return sum(self.validation_failures.values())

    def summary(self) -> str:
        """Return a formatted multi-line summary of the generation run."""
        lines = [
            "",
            "=" * 50,
            "  Generation Summary",
            "=" * 50,
            f"  Requested:              {self.requested}",
            f"  Generated:              {self.generated}",
            f"  Skipped (max retries):  {self.skipped}",
            f"  Total Retries:          {self.total_retries}",
            f"  Validation Failures:    {self.total_validation_failures}",
            f"  API Errors:             {self.api_errors}",
            f"  Avg Generation Time:    {self.average_generation_time:.1f}s",
        ]
        if self.validation_failures:
            lines.append("")
            lines.append("  Failure Reasons:")
            for reason, count in self.validation_failures.most_common():
                lines.append(f"    {reason}: {count}")
        lines.append("=" * 50)
        lines.append("")
        return "\n".join(lines)
