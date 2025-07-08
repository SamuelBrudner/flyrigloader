"""Unit tests for get_extraction_patterns helper.

Focus: patterns defined under `folder_parsing.extract_patterns` in various formats.
Updated to work with Pydantic model configuration objects while maintaining
backward compatibility with dictionary-based configurations.
"""

import pytest
from flyrigloader.config.yaml_config import get_extraction_patterns
from flyrigloader.config.models import LegacyConfigAdapter
from pydantic import ValidationError


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
    
    # Test with dictionary configuration
    assert get_extraction_patterns(config) == expected
    
    # Test with Pydantic model configuration (LegacyConfigAdapter)
    adapter_config = LegacyConfigAdapter(config)
    assert get_extraction_patterns(adapter_config) == expected


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
    
    # Test with dictionary configuration
    assert get_extraction_patterns(config) == expected
    
    # Test with Pydantic model configuration (LegacyConfigAdapter)
    adapter_config = LegacyConfigAdapter(config)
    assert get_extraction_patterns(adapter_config) == expected


def test_get_extraction_patterns_from_folder_parsing_str():
    """Ensure single string extract_patterns are converted to a list."""
    pattern = r"v(?P<vial>\\d+)"
    config = {
        "folder_parsing": {
            "extract_patterns": pattern
        }
    }
    expected = [pattern]
    
    # Test with dictionary configuration
    assert get_extraction_patterns(config) == expected
    
    # Test with Pydantic model configuration (LegacyConfigAdapter)
    adapter_config = LegacyConfigAdapter(config)
    assert get_extraction_patterns(adapter_config) == expected


def test_get_extraction_patterns_with_invalid_regex():
    """Test that invalid regex patterns are handled gracefully."""
    config = {
        "folder_parsing": {
            "extract_patterns": [
                r"v(?P<vial>\\d+)",  # valid pattern
                r"[invalid_regex",   # invalid pattern - unclosed bracket
                r"r(?P<replicate>[A-Z])",  # valid pattern
            ]
        }
    }
    
    # Should return only the valid patterns, ignoring invalid ones
    expected = [r"v(?P<vial>\\d+)", r"r(?P<replicate>[A-Z])"]
    
    # Test with dictionary configuration
    result = get_extraction_patterns(config)
    assert result == expected
    
    # Test with Pydantic model configuration (LegacyConfigAdapter)
    adapter_config = LegacyConfigAdapter(config)
    result_adapter = get_extraction_patterns(adapter_config)
    assert result_adapter == expected


def test_get_extraction_patterns_with_empty_patterns():
    """Test behavior when patterns are empty or None."""
    # Test with empty list
    config_empty_list = {
        "folder_parsing": {
            "extract_patterns": []
        }
    }
    assert get_extraction_patterns(config_empty_list) is None
    
    adapter_config = LegacyConfigAdapter(config_empty_list)
    assert get_extraction_patterns(adapter_config) is None
    
    # Test with empty dict
    config_empty_dict = {
        "folder_parsing": {
            "extract_patterns": {}
        }
    }
    assert get_extraction_patterns(config_empty_dict) is None
    
    adapter_config = LegacyConfigAdapter(config_empty_dict)
    assert get_extraction_patterns(adapter_config) is None


def test_get_extraction_patterns_with_project_level_patterns():
    """Test extraction patterns from project-level configuration."""
    config = {
        "project": {
            "extraction_patterns": [
                r"(?P<date>\\d{4}-\\d{2}-\\d{2})",
                r"(?P<subject>\\w+)"
            ]
        }
    }
    expected = [r"(?P<date>\\d{4}-\\d{2}-\\d{2})", r"(?P<subject>\\w+)"]
    
    # Test with dictionary configuration
    assert get_extraction_patterns(config) == expected
    
    # Test with Pydantic model configuration (LegacyConfigAdapter)
    adapter_config = LegacyConfigAdapter(config)
    assert get_extraction_patterns(adapter_config) == expected


def test_get_extraction_patterns_combined_sources():
    """Test extraction patterns from multiple sources (folder_parsing + project)."""
    config = {
        "folder_parsing": {
            "extract_patterns": [r"v(?P<vial>\\d+)"]
        },
        "project": {
            "extraction_patterns": [r"(?P<date>\\d{4}-\\d{2}-\\d{2})"]
        }
    }
    expected = [r"v(?P<vial>\\d+)", r"(?P<date>\\d{4}-\\d{2}-\\d{2})"]
    
    # Test with dictionary configuration
    assert get_extraction_patterns(config) == expected
    
    # Test with Pydantic model configuration (LegacyConfigAdapter)
    adapter_config = LegacyConfigAdapter(config)
    assert get_extraction_patterns(adapter_config) == expected


def test_get_extraction_patterns_with_experiment_patterns():
    """Test extraction patterns including experiment-specific patterns."""
    config = {
        "folder_parsing": {
            "extract_patterns": [r"v(?P<vial>\\d+)"]
        },
        "experiments": {
            "test_experiment": {
                "datasets": ["test_dataset"],
                "metadata": {
                    "extraction_patterns": [r"(?P<trial>\\d+)"]
                }
            }
        }
    }
    expected = [r"v(?P<vial>\\d+)", r"(?P<trial>\\d+)"]
    
    # Test with dictionary configuration
    assert get_extraction_patterns(config, experiment="test_experiment") == expected
    
    # Test with Pydantic model configuration (LegacyConfigAdapter)
    adapter_config = LegacyConfigAdapter(config)
    assert get_extraction_patterns(adapter_config, experiment="test_experiment") == expected


def test_get_extraction_patterns_with_dataset_patterns():
    """Test extraction patterns including dataset-specific patterns."""
    config = {
        "folder_parsing": {
            "extract_patterns": [r"v(?P<vial>\\d+)"]
        },
        "datasets": {
            "test_dataset": {
                "rig": "rig1",
                "dates_vials": {"2023-05-01": [1, 2, 3]},
                "metadata": {
                    "extraction_patterns": [r"(?P<temperature>\\d+)C"]
                }
            }
        }
    }
    expected = [r"v(?P<vial>\\d+)", r"(?P<temperature>\\d+)C"]
    
    # Test with dictionary configuration
    assert get_extraction_patterns(config, dataset_name="test_dataset") == expected
    
    # Test with Pydantic model configuration (LegacyConfigAdapter)
    adapter_config = LegacyConfigAdapter(config)
    assert get_extraction_patterns(adapter_config, dataset_name="test_dataset") == expected


def test_legacy_config_adapter_dictionary_access():
    """Test that LegacyConfigAdapter provides proper dictionary-style access."""
    config = {
        "folder_parsing": {
            "extract_patterns": {
                "vial": r"v(?P<vial>\\d+)",
                "replicate": r"r(?P<replicate>[A-Z])",
            }
        }
    }
    
    adapter = LegacyConfigAdapter(config)
    
    # Test dictionary-style access methods
    assert "folder_parsing" in adapter
    assert adapter["folder_parsing"]["extract_patterns"]["vial"] == r"v(?P<vial>\\d+)"
    assert adapter.get("folder_parsing") is not None
    assert adapter.get("nonexistent") is None
    
    # Test that get_extraction_patterns works with the adapter
    expected = [r"v(?P<vial>\\d+)", r"r(?P<replicate>[A-Z])"]
    assert get_extraction_patterns(adapter) == expected


def test_legacy_config_adapter_with_invalid_patterns():
    """Test LegacyConfigAdapter handles invalid patterns gracefully."""
    config = {
        "folder_parsing": {
            "extract_patterns": [
                r"v(?P<vial>\\d+)",  # valid
                r"[invalid_regex",   # invalid
                r"r(?P<replicate>[A-Z])",  # valid
            ]
        }
    }
    
    # Should not raise ValidationError during adapter creation
    adapter = LegacyConfigAdapter(config)
    
    # Should return only valid patterns
    expected = [r"v(?P<vial>\\d+)", r"r(?P<replicate>[A-Z])"]
    result = get_extraction_patterns(adapter)
    assert result == expected


def test_backward_compatibility_with_existing_config_format():
    """Test that both old and new configuration formats work identically."""
    # Test configuration that should work with both formats
    config_dict = {
        "folder_parsing": {
            "extract_patterns": {
                "vial": r"v(?P<vial>\\d+)",
                "replicate": r"r(?P<replicate>[A-Z])",
            }
        },
        "project": {
            "extraction_patterns": [r"(?P<date>\\d{4}-\\d{2}-\\d{2})"]
        }
    }
    
    # Test with dictionary (legacy format)
    dict_result = get_extraction_patterns(config_dict)
    
    # Test with LegacyConfigAdapter (new format)
    adapter_config = LegacyConfigAdapter(config_dict)
    adapter_result = get_extraction_patterns(adapter_config)
    
    # Results should be identical
    assert dict_result == adapter_result
    expected = [r"v(?P<vial>\\d+)", r"r(?P<replicate>[A-Z])", r"(?P<date>\\d{4}-\\d{2}-\\d{2})"]
    assert dict_result == expected
    assert adapter_result == expected
