"""Focused tests for validating production-facing pattern behaviors."""

import pytest

from flyrigloader.discovery.patterns import PatternMatcher


def test_patternmatcher_rejects_positional_groups():
    """PatternMatcher should fail loudly when patterns use positional groups."""

    positional_pattern = r".*_(\\d{8})_.*\\.csv"

    matcher = PatternMatcher([positional_pattern])

    import re

    match_obj = re.search(r".*_(\d{8})_.*\.csv", "mouse_20240101_control.csv")
    assert match_obj is not None, "Sanity check: positional pattern should still match"

    with pytest.raises(ValueError):
        matcher._extract_groups_from_match(match_obj, 0)
