#!/usr/bin/env python3
"""Unified transcript comparison tool supporting multiple formats and analysis modes.

This tool compares two transcript files and can operate in different modes:
- analysis: Detailed categorization of changes (speaker, capitalization,
  punctuation, etc.)
- regression: Similarity-based comparison for detecting regressions
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

# Constants
SPEAKER_PATTERN = re.compile(r"\[([\w-]+)\]\s*(.*)")
PUNCTUATION_PATTERN = re.compile(r"[.,!?;:\-\'\"()]")
MIN_WORD_LENGTH_FOR_SPELLING = 4
MAX_LENGTH_DIFF_FOR_SPELLING = 2
MAX_EXAMPLES_PER_CATEGORY = 3

LOGGER = logging.getLogger(__name__)


class ComparisonMode(Enum):
    """Available comparison modes."""

    ANALYSIS = "analysis"
    REGRESSION = "regression"


class TranscriptFormat(Enum):
    """Supported transcript formats."""

    SRT = "srt"
    JSON = "json"


@dataclass(frozen=True)
class SegmentExample:
    """Example of how a transcript segment changed."""

    segment: int
    before: str
    after: str


@dataclass(frozen=True)
class TranscriptAnalysis:
    """Summary of transcript differences."""

    before_count: int
    after_count: int
    identical_segments: int
    changed_segments: int
    change_categories: dict[str, int]
    examples: dict[str, list[SegmentExample]]
    removed_segments: dict[int, str]
    added_segments: dict[int, str]


@dataclass(frozen=True)
class RegressionAnalysis:
    """Regression-focused comparison results."""

    before_count: int
    after_count: int
    identical_count: int
    differences_count: int
    low_similarity_count: int
    low_similarity_examples: list[tuple[int, float, float, str, str]]


# ============================================================================
# Format-specific loaders
# ============================================================================


def load_srt_transcript(srt_path: Path) -> list[tuple[int, str]]:
    """Extract dialogue segments from an SRT file."""
    with srt_path.open(encoding="utf-8") as file_obj:
        lines = [line.strip() for line in file_obj]

    dialogues: list[tuple[int, str]] = []
    index = 0
    total_lines = len(lines)

    while index < total_lines:
        line = lines[index]
        if not line.isdigit():
            index += 1
            continue

        segment_num = int(line)
        index += 2  # Skip timestamp line

        dialogue_parts: list[str] = []
        while index < total_lines and lines[index]:
            dialogue_parts.append(lines[index])
            index += 1

        dialogues.append((segment_num, " ".join(dialogue_parts).strip()))

        # Skip blank separator line, if present
        while index < total_lines and not lines[index]:
            index += 1

    return dialogues


def load_json_transcript(json_path: Path) -> list[tuple[int, str]]:
    """Load transcript from JSON file and convert to (index, text) format."""
    with json_path.open(encoding="utf-8") as file_obj:
        data: list[dict[str, Any]] = json.load(file_obj)

    # Convert to (index, text) format, including speaker if present
    dialogues: list[tuple[int, str]] = []
    for idx, entry in enumerate(data):
        speaker = entry.get("speaker", "")
        text = entry.get("text", "")
        formatted_text = f"[{speaker}] {text}" if speaker else text
        dialogues.append((idx + 1, formatted_text))

    return dialogues


def detect_format(file_path: Path) -> TranscriptFormat:
    """Detect transcript format based on file extension."""
    suffix = file_path.suffix.lower()
    if suffix == ".srt":
        return TranscriptFormat.SRT
    if suffix == ".json":
        return TranscriptFormat.JSON
    msg = f"Unsupported file format: {suffix}"
    raise ValueError(msg)


def load_transcript(file_path: Path) -> list[tuple[int, str]]:
    """Load transcript file in any supported format."""
    fmt = detect_format(file_path)
    if fmt == TranscriptFormat.SRT:
        return load_srt_transcript(file_path)
    if fmt == TranscriptFormat.JSON:
        return load_json_transcript(file_path)
    msg = f"Unsupported format: {fmt}"
    raise ValueError(msg)


# ============================================================================
# Text comparison utilities
# ============================================================================


def calculate_jaccard_similarity(str1: str, str2: str) -> float:
    """Calculate Jaccard similarity between two strings (word-level)."""
    words1 = set(str1.lower().split())
    words2 = set(str2.lower().split())

    if not words1 and not words2:
        return 1.0
    if not words1 or not words2:
        return 0.0

    intersection = words1.intersection(words2)
    union = words1.union(words2)

    return len(intersection) / len(union)


def calculate_levenshtein_ratio(str1: str, str2: str) -> float:
    """Calculate similarity ratio using SequenceMatcher."""
    return SequenceMatcher(None, str1, str2).ratio()


def parse_segment(text: str) -> tuple[str, str] | None:
    """Parse speaker and content from formatted segment."""
    match = SPEAKER_PATTERN.match(text)
    if not match:
        return None
    speaker, content = match.groups()
    return speaker, content.strip()


def categorize_changes(before_text: str, after_text: str) -> list[str]:
    """Categorize how a segment changed between transcripts."""
    before_segment = parse_segment(before_text)
    after_segment = parse_segment(after_text)
    if not before_segment or not after_segment:
        return ["unknown"]

    categories: set[str] = set()
    before_speaker, before_content = before_segment
    after_speaker, after_content = after_segment

    if before_speaker != after_speaker:
        categories.add("speaker_change")

    categories.update(detect_textual_changes(before_content, after_content))

    return sorted(categories) if categories else ["no_change"]


def detect_textual_changes(before_content: str, after_content: str) -> set[str]:
    """Detect specific types of textual changes."""
    if before_content == after_content:
        return set()

    categories: set[str] = set()

    capitalization_change = detect_capitalization_change(before_content, after_content)
    if capitalization_change:
        categories.add(capitalization_change)

    if punctuation_changed(before_content, after_content):
        categories.add("punctuation")

    categories.update(word_change_categories(before_content, after_content))
    return categories


def detect_capitalization_change(before_content: str, after_content: str) -> str | None:
    """Detect if only capitalization changed."""
    if before_content.lower() != after_content.lower():
        return None

    changes = sum(
        1
        for before_char, after_char in zip(before_content, after_content, strict=False)
        if before_char != after_char and before_char.lower() == after_char.lower()
    )
    if changes > 0:
        return f"capitalization_{changes}"
    return None


def punctuation_changed(before_content: str, after_content: str) -> bool:
    """Check if punctuation changed between two texts."""
    before_punct = set(PUNCTUATION_PATTERN.findall(before_content))
    after_punct = set(PUNCTUATION_PATTERN.findall(after_content))
    return before_punct != after_punct


def word_change_categories(before_content: str, after_content: str) -> set[str]:
    """Categorize word-level changes."""
    categories: set[str] = set()

    before_words = before_content.lower().split()
    after_words = after_content.lower().split()

    if before_words == after_words:
        return categories

    diff = abs(len(before_words) - len(after_words))
    if diff > 0:
        categories.add(f"word_count_diff_{diff}")

    for before_word, after_word in zip(before_words, after_words, strict=False):
        if before_word == after_word:
            continue
        if looks_like_spelling_update(before_word, after_word):
            categories.add("spelling_correction")

    return categories


def looks_like_spelling_update(before_word: str, after_word: str) -> bool:
    """Check if word change appears to be a spelling correction."""
    long_enough = (
        len(before_word) >= MIN_WORD_LENGTH_FOR_SPELLING
        and len(after_word) >= MIN_WORD_LENGTH_FOR_SPELLING
    )
    length_diff_ok = (
        abs(len(before_word) - len(after_word)) <= MAX_LENGTH_DIFF_FOR_SPELLING
    )
    return long_enough and length_diff_ok


# ============================================================================
# Analysis mode
# ============================================================================


def analyze_transcripts_detailed(
    before_dialogues: Sequence[tuple[int, str]],
    after_dialogues: Sequence[tuple[int, str]],
    *,
    max_examples: int = MAX_EXAMPLES_PER_CATEGORY,
) -> TranscriptAnalysis:
    """Perform detailed analysis with change categorization."""
    before_mapping = dict(before_dialogues)
    after_mapping = dict(after_dialogues)

    before_segments = set(before_mapping)
    after_segments = set(after_mapping)
    overlapping_segments = sorted(before_segments & after_segments)

    change_counts: defaultdict[str, int] = defaultdict(int)
    example_map: defaultdict[str, list[SegmentExample]] = defaultdict(list)

    identical_segments = 0
    changed_segments = 0

    for segment in overlapping_segments:
        before_text = before_mapping[segment]
        after_text = after_mapping[segment]

        if before_text == after_text:
            identical_segments += 1
            continue

        changed_segments += 1
        categories = categorize_changes(before_text, after_text)
        for category in categories:
            change_counts[category] += 1
            if len(example_map[category]) < max_examples:
                example_map[category].append(
                    SegmentExample(
                        segment=segment,
                        before=before_text,
                        after=after_text,
                    )
                )

    removed_segments = {
        seg: before_mapping[seg] for seg in sorted(before_segments - after_segments)
    }
    added_segments = {
        seg: after_mapping[seg] for seg in sorted(after_segments - before_segments)
    }

    return TranscriptAnalysis(
        before_count=len(before_segments),
        after_count=len(after_segments),
        identical_segments=identical_segments,
        changed_segments=changed_segments,
        change_categories=dict(change_counts),
        examples=dict(example_map),
        removed_segments=removed_segments,
        added_segments=added_segments,
    )


def format_analysis_report(analysis: TranscriptAnalysis) -> Iterator[str]:
    """Generate human-readable analysis report."""
    overlap = analysis.changed_segments + analysis.identical_segments
    change_rate = (analysis.changed_segments / overlap * 100) if overlap else 0

    yield f"Before segments: {analysis.before_count}"
    yield f"After segments:  {analysis.after_count}"
    yield f"Overlap segments: {overlap}"
    yield f"Identical segments: {analysis.identical_segments}"
    yield f"Changed segments: {analysis.changed_segments}"
    yield f"Change rate: {change_rate:.1f}%"

    if analysis.change_categories:
        yield ""
        yield "Change categories:"
        for category, count in sorted(
            analysis.change_categories.items(),
            key=lambda item: -item[1],
        ):
            yield f"  {category:30s} {count:5d}"

    if analysis.examples:
        yield ""
        yield "Example changes:"
        for category in sorted(analysis.examples):
            yield f"- {category.upper()}"
            for example in analysis.examples[category]:
                yield f"  Segment {example.segment}:"
                yield f"    BEFORE: {example.before}"
                yield f"    AFTER:  {example.after}"

    if analysis.removed_segments:
        yield ""
        yield "Removed segments:"
        for segment, text in analysis.removed_segments.items():
            yield f"  Segment {segment}: {text}"

    if analysis.added_segments:
        yield ""
        yield "Added segments:"
        for segment, text in analysis.added_segments.items():
            yield f"  Segment {segment}: {text}"


# ============================================================================
# Regression mode
# ============================================================================


def analyze_transcripts_regression(
    before_dialogues: Sequence[tuple[int, str]],
    after_dialogues: Sequence[tuple[int, str]],
    *,
    jaccard_threshold: float = 0.8,
    levenshtein_threshold: float = 0.8,
) -> RegressionAnalysis:
    """Perform regression-focused comparison using similarity metrics."""
    before_mapping = dict(before_dialogues)
    after_mapping = dict(after_dialogues)

    identical_count = 0
    differences_count = 0
    low_similarity_count = 0
    low_similarity_examples: list[tuple[int, float, float, str, str]] = []

    max_entries = max(len(before_mapping), len(after_mapping))

    for i in range(1, max_entries + 1):
        before_text = before_mapping.get(i)
        after_text = after_mapping.get(i)

        # Handle missing entries
        if before_text is None or after_text is None:
            differences_count += 1
            if before_text is None:
                low_similarity_examples.append((i, 0.0, 0.0, "", after_text or ""))
            else:
                low_similarity_examples.append((i, 0.0, 0.0, before_text, ""))
            low_similarity_count += 1
            continue

        # Check if identical
        if before_text == after_text:
            identical_count += 1
            continue

        # Calculate similarity
        jaccard = calculate_jaccard_similarity(before_text, after_text)
        levenshtein = calculate_levenshtein_ratio(before_text, after_text)

        differences_count += 1

        if jaccard < jaccard_threshold or levenshtein < levenshtein_threshold:
            low_similarity_count += 1
            low_similarity_examples.append(
                (i, jaccard, levenshtein, before_text, after_text)
            )

    return RegressionAnalysis(
        before_count=len(before_mapping),
        after_count=len(after_mapping),
        identical_count=identical_count,
        differences_count=differences_count,
        low_similarity_count=low_similarity_count,
        low_similarity_examples=low_similarity_examples,
    )


def format_regression_report(analysis: RegressionAnalysis) -> Iterator[str]:
    """Generate human-readable regression report."""
    yield "=" * 80
    yield "TRANSCRIPT REGRESSION REPORT"
    yield "=" * 80
    yield ""
    yield "Entry counts:"
    yield f"  Before:  {analysis.before_count:,} entries"
    yield f"  After:   {analysis.after_count:,} entries"
    yield f"  Difference: {analysis.after_count - analysis.before_count:+,} entries"

    if analysis.before_count != analysis.after_count:
        yield ""
        yield "⚠️  WARNING: Entry counts differ!"

    yield ""
    yield "=" * 80
    yield "COMPARISON SUMMARY"
    yield "=" * 80
    yield ""
    yield f"Identical entries: {analysis.identical_count:,}"
    yield f"Entries with differences: {analysis.differences_count:,}"
    yield f"Entries with low similarity: {analysis.low_similarity_count:,}"

    if analysis.low_similarity_examples:
        yield ""
        yield "=" * 80
        yield "LOW SIMILARITY ENTRIES"
        yield "=" * 80
        yield ""

        for (
            idx,
            jaccard,
            levenshtein,
            before_text,
            after_text,
        ) in analysis.low_similarity_examples:
            if not before_text:
                yield f"Entry {idx}: MISSING IN BEFORE (NEW ENTRY)"
                yield f"  After text: {after_text[:150]}"
            elif not after_text:
                yield f"Entry {idx}: MISSING IN AFTER (REMOVED)"
                yield f"  Before text: {before_text[:150]}"
            else:
                yield f"Entry {idx}: LOW SIMILARITY"
                yield f"  Jaccard: {jaccard:.3f} | Levenshtein: {levenshtein:.3f}"
                yield f"  Before: {before_text[:150]}"
                yield f"  After:  {after_text[:150]}"
            yield ""

    yield "=" * 80
    yield "FINAL RESULT"
    yield "=" * 80
    yield ""

    if (
        analysis.differences_count == 0
        and analysis.before_count == analysis.after_count
    ):
        yield "✅ SUCCESS: Transcripts are identical!"
    elif analysis.low_similarity_count == 0:
        yield "⚠️  WARNING: Minor differences found, but no significant regressions."
    else:
        yield "❌ FAILURE: Significant differences detected!"
        yield (
            f"   {analysis.low_similarity_count} entries have similarity "
            "below threshold."
        )


# ============================================================================
# CLI interface
# ============================================================================


def configure_logging() -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "before",
        type=Path,
        help="Path to the first transcript file (SRT or JSON)",
    )

    parser.add_argument(
        "after",
        type=Path,
        help="Path to the second transcript file (SRT or JSON)",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["analysis", "regression"],
        default="regression",
        help="Comparison mode: 'analysis' for detailed categorization, "
        "'regression' for similarity-based comparison (default: regression)",
    )

    parser.add_argument(
        "--jaccard-threshold",
        type=float,
        default=0.8,
        help="Jaccard similarity threshold for regression mode (default: 0.8)",
    )

    parser.add_argument(
        "--levenshtein-threshold",
        type=float,
        default=0.8,
        help="Levenshtein similarity threshold for regression mode (default: 0.8)",
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=MAX_EXAMPLES_PER_CATEGORY,
        help="Maximum examples per category in analysis mode (default: 3)",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    configure_logging()
    args = parse_args()

    # Validate file paths
    if not args.before.exists():
        LOGGER.error("Error: First transcript not found: %s", args.before)
        return 1

    if not args.after.exists():
        LOGGER.error("Error: Second transcript not found: %s", args.after)
        return 1

    # Load transcripts
    LOGGER.info("Loading first transcript: %s", args.before)
    before_dialogues = load_transcript(args.before)

    LOGGER.info("Loading second transcript: %s", args.after)
    after_dialogues = load_transcript(args.after)

    # Run comparison based on mode
    if args.mode == "analysis":
        LOGGER.info("\nRunning detailed analysis mode...\n")
        analysis = analyze_transcripts_detailed(
            before_dialogues,
            after_dialogues,
            max_examples=args.max_examples,
        )
        for line in format_analysis_report(analysis):
            LOGGER.info(line)
        return 0

    if args.mode == "regression":
        LOGGER.info("\nRunning regression comparison mode...\n")
        analysis = analyze_transcripts_regression(
            before_dialogues,
            after_dialogues,
            jaccard_threshold=args.jaccard_threshold,
            levenshtein_threshold=args.levenshtein_threshold,
        )
        for line in format_regression_report(analysis):
            LOGGER.info(line)

        # Return appropriate exit code for CI/CD
        if (
            analysis.differences_count == 0
            and analysis.before_count == analysis.after_count
        ) or analysis.low_similarity_count == 0:
            return 0
        return 1

    LOGGER.error("Unknown mode: %s", args.mode)
    return 1


if __name__ == "__main__":
    sys.exit(main())
