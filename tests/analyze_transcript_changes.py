#!/usr/bin/env python3
"""Analyze differences between before and after GPT cleanup transcripts."""

import re
from collections import defaultdict
from pathlib import Path


def extract_dialogue_from_srt(srt_path):
    """Extract just the dialogue lines from an SRT file."""
    dialogues = []
    with open(srt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Check if this is a segment number
        if line.isdigit():
            segment_num = int(line)
            # Skip timestamp line
            i += 2
            # Collect dialogue until empty line
            dialogue_parts = []
            while i < len(lines) and lines[i].strip():
                dialogue_parts.append(lines[i].strip())
                i += 1
            dialogue = ' '.join(dialogue_parts)
            dialogues.append((segment_num, dialogue))
        i += 1

    return dialogues


def categorize_changes(before_text, after_text):
    """Categorize the type of change between two texts."""
    changes = []

    # Extract speaker and content
    before_match = re.match(r'\[([\w-]+)\]\s*(.*)', before_text)
    after_match = re.match(r'\[([\w-]+)\]\s*(.*)', after_text)

    if not before_match or not after_match:
        return ['unknown']

    before_speaker, before_content = before_match.groups()
    after_speaker, after_content = after_match.groups()

    if before_speaker != after_speaker:
        changes.append('speaker_change')

    # Capitalization changes
    if before_content != after_content:
        if before_content.lower() == after_content.lower():
            # Count capitalization changes
            cap_changes = sum(1 for b, a in zip(before_content, after_content) if b != a and b.lower() == a.lower())
            if cap_changes > 0:
                changes.append(f'capitalization_{cap_changes}')

        # Punctuation changes
        before_punct = set(re.findall(r'[.,!?;:\-\'"()]', before_content))
        after_punct = set(re.findall(r'[.,!?;:\-\'"()]', after_content))
        if before_punct != after_punct:
            changes.append('punctuation')

        # Word changes (spelling corrections, etc.)
        before_words = before_content.lower().split()
        after_words = after_content.lower().split()
        if before_words != after_words:
            word_diff = abs(len(before_words) - len(after_words))
            if word_diff > 0:
                changes.append(f'word_count_diff_{word_diff}')

            # Find specific word changes
            for bw, aw in zip(before_words, after_words):
                if bw != aw:
                    # Check if it's a typo correction
                    if len(bw) > 3 and len(aw) > 3:
                        # Simple edit distance check
                        if abs(len(bw) - len(aw)) <= 2:
                            changes.append('spelling_correction')

    return changes if changes else ['no_change']


def main():
    before_path = Path('/Users/naveednadjmabadi/code/audio-transcriber/outputs/gpt-cleanup/final_transcript_before_parallelization.srt')
    after_path = Path('/Users/naveednadjmabadi/code/audio-transcriber/outputs/gpt-cleanup/final_transcript.srt')

    print("Extracting dialogues from SRT files...")
    before_dialogues = extract_dialogue_from_srt(before_path)
    after_dialogues = extract_dialogue_from_srt(after_path)

    print(f"\nBefore: {len(before_dialogues)} segments")
    print(f"After: {len(after_dialogues)} segments")
    print(f"Difference: {len(before_dialogues) - len(after_dialogues)} segments")

    # Find missing segments
    before_nums = {seg[0] for seg in before_dialogues}
    after_nums = {seg[0] for seg in after_dialogues}

    # Create dict for easy lookup
    before_dict = dict(before_dialogues)
    after_dict = dict(after_dialogues)

    # Analyze changes segment by segment
    change_categories = defaultdict(int)
    examples = defaultdict(list)

    # Track which segments changed
    total_changes = 0
    identical_segments = 0

    print("\n" + "="*80)
    print("CHANGE ANALYSIS")
    print("="*80)

    # Compare overlapping segments
    for seg_num in sorted(before_nums):
        if seg_num in after_dict:
            before_text = before_dict[seg_num]
            after_text = after_dict[seg_num]

            if before_text != after_text:
                total_changes += 1
                categories = categorize_changes(before_text, after_text)
                for cat in categories:
                    change_categories[cat] += 1
                    if len(examples[cat]) < 3:  # Keep up to 3 examples
                        examples[cat].append((seg_num, before_text, after_text))
            else:
                identical_segments += 1

    print(f"\nTotal segments analyzed: {len(before_nums & after_nums)}")
    print(f"Identical segments: {identical_segments}")
    print(f"Changed segments: {total_changes}")
    print(f"Change rate: {100 * total_changes / len(before_nums & after_nums):.1f}%")

    print("\n" + "="*80)
    print("CHANGE CATEGORIES")
    print("="*80)
    for category, count in sorted(change_categories.items(), key=lambda x: -x[1]):
        print(f"{category:30s}: {count:5d} occurrences")

    print("\n" + "="*80)
    print("EXAMPLE CHANGES")
    print("="*80)
    for category in sorted(examples.keys()):
        print(f"\n{category.upper()}:")
        for seg_num, before, after in examples[category][:2]:
            print(f"  Segment {seg_num}:")
            print(f"    BEFORE: {before[:100]}...")
            print(f"    AFTER:  {after[:100]}...")

    # Find segments only in before (removed)
    removed_segments = before_nums - after_nums
    if removed_segments:
        print("\n" + "="*80)
        print("REMOVED SEGMENTS")
        print("="*80)
        for seg_num in sorted(removed_segments):
            print(f"Segment {seg_num}: {before_dict[seg_num][:100]}...")

    # Find segments only in after (added - shouldn't happen)
    added_segments = after_nums - before_nums
    if added_segments:
        print("\n" + "="*80)
        print("ADDED SEGMENTS")
        print("="*80)
        for seg_num in sorted(added_segments):
            print(f"Segment {seg_num}: {after_dict[seg_num][:100]}...")


if __name__ == '__main__':
    main()
