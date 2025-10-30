"""Analyze differences between pre- and post-cleanup transcripts."""

from __future__ import annotations

import argparse
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

LOGGER = logging.getLogger(__name__)

SPEAKER_PATTERN = re.compile(r"\[([\w-]+)\]\s*(.*)")
PUNCTUATION_PATTERN = re.compile(r"[.,!?;:\-\'\"()]")
MIN_WORD_LENGTH_FOR_SPELLING = 4
MAX_LENGTH_DIFF_FOR_SPELLING = 2
MAX_EXAMPLES_PER_CATEGORY = 3

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence


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


def extract_dialogue_from_srt(srt_path: Path) -> list[tuple[int, str]]:
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


def categorize_changes(before_text: str, after_text: str) -> list[str]:
    """Categorize how a segment changed between transcripts."""
    before_segment = _parse_segment(before_text)
    after_segment = _parse_segment(after_text)
    if not before_segment or not after_segment:
        return ["unknown"]

    categories: set[str] = set()
    before_speaker, before_content = before_segment
    after_speaker, after_content = after_segment

    if before_speaker != after_speaker:
        categories.add("speaker_change")

    categories.update(
        _detect_textual_changes(before_content, after_content),
    )

    return sorted(categories) if categories else ["no_change"]


def analyze_transcripts(
    before_dialogues: Sequence[tuple[int, str]],
    after_dialogues: Sequence[tuple[int, str]],
    *,
    max_examples: int = MAX_EXAMPLES_PER_CATEGORY,
) -> TranscriptAnalysis:
    """Summarize differences between two dialogue sequences."""
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


def analyze_transcript_files(
    before_path: Path,
    after_path: Path,
    *,
    max_examples: int = MAX_EXAMPLES_PER_CATEGORY,
) -> TranscriptAnalysis:
    """Convenience wrapper that loads SRT files and analyzes their differences."""
    before_dialogues = extract_dialogue_from_srt(before_path)
    after_dialogues = extract_dialogue_from_srt(after_path)
    return analyze_transcripts(
        before_dialogues,
        after_dialogues,
        max_examples=max_examples,
    )


def format_analysis(analysis: TranscriptAnalysis) -> Iterator[str]:
    """Yield human-readable lines describing the analysis."""
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


def _parse_segment(text: str) -> tuple[str, str] | None:
    match = SPEAKER_PATTERN.match(text)
    if not match:
        return None
    speaker, content = match.groups()
    return speaker, content.strip()


def _detect_textual_changes(before_content: str, after_content: str) -> set[str]:
    if before_content == after_content:
        return set()

    categories: set[str] = set()

    capitalization_change = _detect_capitalization_change(before_content, after_content)
    if capitalization_change:
        categories.add(capitalization_change)

    if _punctuation_changed(before_content, after_content):
        categories.add("punctuation")

    categories.update(_word_change_categories(before_content, after_content))
    return categories


def _detect_capitalization_change(
    before_content: str,
    after_content: str,
) -> str | None:
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


def _punctuation_changed(before_content: str, after_content: str) -> bool:
    before_punct = set(PUNCTUATION_PATTERN.findall(before_content))
    after_punct = set(PUNCTUATION_PATTERN.findall(after_content))
    return before_punct != after_punct


def _word_change_categories(before_content: str, after_content: str) -> set[str]:
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
        if _looks_like_spelling_update(before_word, after_word):
            categories.add("spelling_correction")

    return categories


def _looks_like_spelling_update(before_word: str, after_word: str) -> bool:
    long_enough = (
        len(before_word) >= MIN_WORD_LENGTH_FOR_SPELLING
        and len(after_word) >= MIN_WORD_LENGTH_FOR_SPELLING
    )
    length_diff_ok = (
        abs(len(before_word) - len(after_word)) <= MAX_LENGTH_DIFF_FOR_SPELLING
    )
    return long_enough and length_diff_ok


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def _parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path, help="Path to the pre-cleanup SRT file.")
    parser.add_argument("after", type=Path, help="Path to the post-cleanup SRT file.")
    return parser.parse_args(args=args)


def cli(argv: Iterable[str] | None = None) -> None:
    """Entrypoint when running the module as a script."""
    parsed = _parse_args(argv)
    _configure_logging()

    analysis = analyze_transcript_files(parsed.before, parsed.after)
    for line in format_analysis(analysis):
        LOGGER.info(line)


if __name__ == "__main__":
    cli()
