"""Unit tests for get_extraction_patterns helper.

Focus: patterns defined under `folder_parsing.extract_patterns` in various formats.
"""

from flyrigloader.config.yaml_config import get_extraction_patterns


def test_get_extraction_patterns_from_folder_parsing_dict():
    """Ensure dict-valued extract_patterns are flattened in insertion order."""
    config = {
        "folder_parsing": {
            "extract_patterns": {
                "vial": r"v(?P<vial>\\d+)",
                "replicate": r"r(?P<replicate>[A-Z])",
            }
        }
    }
    expected = [r"v(?P<vial>\\d+)", r"r(?P<replicate>[A-Z])"]
    assert get_extraction_patterns(config) == expected


def test_get_extraction_patterns_from_folder_parsing_list():
    """Ensure list-valued extract_patterns pass through unchanged and deduplicated."""
    config = {
        "folder_parsing": {
            "extract_patterns": [
                r"v(?P<vial>\\d+)",
                r"v(?P<vial>\\d+)",  # duplicate should be removed
                r"r(?P<replicate>[A-Z])",
            ]
        }
    }
    expected = [r"v(?P<vial>\\d+)", r"r(?P<replicate>[A-Z])"]
    assert get_extraction_patterns(config) == expected


def test_get_extraction_patterns_from_folder_parsing_str():
    """Ensure single string extract_patterns are converted to a list."""
    pattern = r"v(?P<vial>\\d+)"
    config = {
        "folder_parsing": {
            "extract_patterns": pattern
        }
    }
    assert get_extraction_patterns(config) == [pattern]
