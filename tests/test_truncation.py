"""Tests for utilities that truncate repetitive text sequences."""

from __future__ import annotations

import re
from re import Match

TRUNCATION_MARKER = "... [continues]"
DEFAULT_MAX_REPEAT = 50
REPETITION_PATTERN = re.compile(r"(.{2,}?)\1{3,}")


def truncate_repetitive_sequences(
    text: str,
    max_repeat: int = DEFAULT_MAX_REPEAT,
) -> str:
    """Truncate repeatedly duplicated sequences of characters."""

    def replace_repetition(match: Match[str]) -> str:
        full_match = match.group(0)
        pattern_unit = match.group(1)
        if not pattern_unit:
            return full_match

        repetitions = len(full_match) // len(pattern_unit)
        if repetitions <= max_repeat:
            return full_match

        truncated = pattern_unit * max_repeat
        return f"{truncated}{TRUNCATION_MARKER}"

    return REPETITION_PATTERN.sub(replace_repetition, text)


def test_truncate_repetitive_sequences_expected_behavior() -> None:
    """Ensure canonical examples either remain untouched or are truncated."""
    cases = [
        ("HAHAHAHAHAHAHAHAHAHAHAHAHA", False),
        ("HA" * 100, True),
        ("LALALA" * 200, True),
        ("hehe" * 150, True),
        ("Normal text here", False),
        ("Some text HAHAHAHA more text", False),
        ("A" * 1000, True),
    ]

    for text, should_truncate in cases:
        truncated = truncate_repetitive_sequences(text)
        if should_truncate:
            assert truncated != text
            assert TRUNCATION_MARKER in truncated
        else:
            assert truncated == text


def test_truncate_repetitive_sequences_respects_custom_threshold() -> None:
    """When the threshold is lowered, shorter patterns should be truncated."""
    text = "HA" * 10
    truncated = truncate_repetitive_sequences(text, max_repeat=5)
    assert truncated.endswith(TRUNCATION_MARKER)
    assert truncated.startswith("HA" * 5)


def test_truncate_repetitive_sequences_preserves_non_repetitive_text() -> None:
    """Ensure mixed content without long repeats is unchanged."""
    text = "This text has some variety without excessive repetition."
    assert truncate_repetitive_sequences(text) == text


if __name__ == "__main__":
    test_truncate_repetitive_sequences_expected_behavior()
    test_truncate_repetitive_sequences_respects_custom_threshold()
    test_truncate_repetitive_sequences_preserves_non_repetitive_text()
