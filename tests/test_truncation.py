#!/usr/bin/env python3
"""Test the truncation function with real examples from the error batches."""

import re


def truncate_repetitive_sequences(text: str, max_repeat: int = 50) -> str:
    """Truncate excessively repeated character sequences to prevent GPT token overflow.

    Args:
        text: Input text that may contain long repeated sequences
        max_repeat: Maximum number of times to allow a pattern to repeat

    Returns:
        Text with excessive repetitions truncated and marked
    """
    # Pattern to detect 2+ character sequences repeated many times
    # Example: "HAHAHAHA" or "LALALA" or "hehehehe"
    pattern = r"(.{2,}?)\1{3,}"  # Find patterns repeated 4+ times

    def replace_repetition(match):
        """Replace long repetitions with truncated version."""
        full_match = match.group(0)
        pattern_unit = match.group(1)

        # Count how many times the pattern repeats
        repetitions = len(full_match) // len(pattern_unit)

        # If it repeats more than max_repeat times, truncate it
        if repetitions > max_repeat:
            # Keep first max_repeat repetitions and add indicator
            truncated = pattern_unit * max_repeat
            # Add ellipsis and note about continuation
            return f"{truncated}... [continues]"
        return full_match

    result = re.sub(pattern, replace_repetition, text)
    return result


# Test cases
test_cases = [
    ("HAHAHAHAHAHAHAHAHAHAHAHAHA", "Short laugh (12 HA's) - should NOT truncate"),
    ("HA" * 100, "Long laugh (100 HA's) - SHOULD truncate"),
    ("LALALA" * 200, "Long LALALA (200 times) - SHOULD truncate"),
    ("hehe" * 150, "Long hehe (150 times) - SHOULD truncate"),
    ("Normal text here", "Normal text - should NOT change"),
    ("Some text HAHAHAHA more text", "Mixed content - should NOT truncate short sequences"),
    ("A" * 1000, "Single char repeated 1000 times - SHOULD truncate"),
]

print("Testing truncation function:\n")
for text, description in test_cases:
    original_len = len(text)
    truncated = truncate_repetitive_sequences(text)
    new_len = len(truncated)
    reduction = ((original_len - new_len) / original_len * 100) if original_len > 0 else 0

    print(f"Test: {description}")
    print(f"  Original length: {original_len}")
    print(f"  Truncated length: {new_len}")
    print(f"  Reduction: {reduction:.1f}%")
    if new_len < 200:
        print(f"  Result: {truncated}")
    else:
        print(f"  Result (first 100 chars): {truncated[:100]}...")
    print()
