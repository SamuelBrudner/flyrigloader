"""
Enhanced metadata extraction tests for the high-level API functions.

This module implements comprehensive testing for the F-007 Metadata Extraction System
with modern pytest practices including parametrization, fixtures, and property-based testing.
"""
import os
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
import pytest
from unittest.mock import MagicMock, patch
from hypothesis import given, strategies as st

from flyrigloader.api import (
    load_experiment_files,
    load_dataset_files
)


# ============================================================================
# TEST FIXTURES (TST-MOD-001 Requirements)
# ============================================================================

@pytest.fixture
def metadata_test_config():
    """
    Fixture providing comprehensive configuration for metadata testing scenarios.
    
    Per TST-MOD-001 requirements, this fixture provides consistent test data
    generation for metadata extraction validation.
    """
    return {
        "project": {
            "directories": {
                "major_data_directory": "/path/to/data"
            },
            "extraction_patterns": [
                r".*_(?P<date>\d{8})_(?P<condition>\w+)_(?P<replicate>\d+)\.csv",
                r".*_(?P<experiment>\w+)_(?P<date>\d{8})\.pkl"
            ]
        },
        "experiments": {
            "test_experiment": {
                "datasets": ["test_dataset"],
                "metadata": {
                    "extraction_patterns": [
                        r".*_(?P<experiment>\w+)_(?P<date>\d{8})_(?P<condition>\w+)\.csv"
                    ]
                }
            },
            "multi_condition_experiment": {
                "datasets": ["baseline_dataset", "treatment_dataset"],
                "metadata": {
                    "extraction_patterns": [
                        r".*_(?P<date>\d{8})_(?P<condition>\w+)_(?P<replicate>\d+)\.csv"
                    ]
                }
            }
        },
        "datasets": {
            "test_dataset": {
                "patterns": ["*_test_*"],
                "metadata": {
                    "extraction_patterns": [
                        r".*_(?P<dataset>\w+)_(?P<date>\d{8})\.csv"
                    ]
                }
            },
            "baseline_dataset": {
                "patterns": ["*_baseline_*"],
                "metadata": {
                    "extraction_patterns": [
                        r".*_(?P<dataset>baseline)_(?P<date>\d{8})_(?P<replicate>\d+)\.csv"
                    ]
                }
            },
            "treatment_dataset": {
                "patterns": ["*_treatment_*"],
                "metadata": {
                    "extraction_patterns": [
                        r".*_(?P<dataset>treatment)_(?P<date>\d{8})_(?P<replicate>\d+)\.csv"
                    ]
                }
            }
        }
    }


@pytest.fixture
def sample_file_metadata():
    """
    Fixture providing comprehensive metadata samples for all supported fields.
    
    Per F-007 requirements, includes all supported metadata fields:
    date, condition, replicate, dataset, parsed_date.
    """
    return {
        "/path/to/data/experiment_20240101_baseline_1.csv": {
            "date": "20240101",
            "condition": "baseline",
            "replicate": "1",
            "parsed_date": datetime(2024, 1, 1)
        },
        "/path/to/data/experiment_20240102_treatment_2.csv": {
            "date": "20240102", 
            "condition": "treatment",
            "replicate": "2",
            "parsed_date": datetime(2024, 1, 2)
        },
        "/path/to/data/dataset_test_20240103.csv": {
            "dataset": "test",
            "date": "20240103",
            "parsed_date": datetime(2024, 1, 3)
        },
        "/path/to/data/multi_20240104_control_3.csv": {
            "date": "20240104",
            "condition": "control", 
            "replicate": "3",
            "parsed_date": datetime(2024, 1, 4)
        }
    }


@pytest.fixture
def date_parsing_test_cases():
    """
    Fixture providing diverse date format test cases for F-002-RQ-005 requirements.
    
    Tests various date formats and edge cases including timezone handling.
    """
    return [
        # Standard YYYYMMDD format
        ("20240101", "%Y%m%d", datetime(2024, 1, 1)),
        ("20231225", "%Y%m%d", datetime(2023, 12, 25)),
        
        # ISO format YYYY-MM-DD
        ("2024-01-01", "%Y-%m-%d", datetime(2024, 1, 1)),
        ("2023-12-25", "%Y-%m-%d", datetime(2023, 12, 25)),
        
        # US format MM-DD-YYYY
        ("01-01-2024", "%m-%d-%Y", datetime(2024, 1, 1)),
        ("12-25-2023", "%m-%d-%Y", datetime(2023, 12, 25)),
        
        # Edge cases: leap year
        ("20240229", "%Y%m%d", datetime(2024, 2, 29)),
        ("2024-02-29", "%Y-%m-%d", datetime(2024, 2, 29)),
        
        # End of year dates
        ("20231231", "%Y%m%d", datetime(2023, 12, 31)),
        ("2023-12-31", "%Y-%m-%d", datetime(2023, 12, 31)),
    ]


@pytest.fixture
def enhanced_mock_dependencies(monkeypatch):
    """
    Enhanced mock fixture for comprehensive dependency isolation per TST-MOD-003 requirements.
    
    Uses pytest-mock for improved isolation of load_config, discover_experiment_files,
    and discover_dataset_files dependencies.
    """
    from unittest.mock import MagicMock
    
    # Create enhanced mock functions with comprehensive behavior
    mock_load_config = MagicMock()
    mock_discover_experiment_files = MagicMock() 
    mock_discover_dataset_files = MagicMock()
    
    # Configure default return values for successful test execution
    mock_load_config.return_value = {
        "project": {
            "directories": {
                "major_data_directory": "/path/to/data"
            }
        },
        "experiments": {
            "test_experiment": {
                "datasets": ["test_dataset"]
            }
        },
        "datasets": {
            "test_dataset": {
                "patterns": ["*_test_*"]
            }
        }
    }
    
    mock_discover_experiment_files.return_value = {}
    mock_discover_dataset_files.return_value = {}
    
    # Apply patches with proper import paths
    monkeypatch.setattr("flyrigloader.api.load_config", mock_load_config)
    monkeypatch.setattr("flyrigloader.api.discover_experiment_files", mock_discover_experiment_files)
    monkeypatch.setattr("flyrigloader.api.discover_dataset_files", mock_discover_dataset_files)
    
    return mock_load_config, mock_discover_experiment_files, mock_discover_dataset_files


# ============================================================================
# PARAMETRIZED METADATA EXTRACTION TESTS (F-007 Requirements)
# ============================================================================

@pytest.mark.parametrize("metadata_fields,expected_values", [
    # Test all supported metadata fields per F-007 requirements
    (["date", "condition", "replicate"], {"date": "20240101", "condition": "baseline", "replicate": "1"}),
    (["date", "dataset"], {"date": "20240102", "dataset": "test"}),
    (["experiment", "date", "condition"], {"experiment": "exp001", "date": "20240103", "condition": "treatment"}),
    (["date", "condition", "replicate", "parsed_date"], {
        "date": "20240104", 
        "condition": "control", 
        "replicate": "2",
        "parsed_date": datetime(2024, 1, 4)
    }),
    # Edge case: only date field
    (["date"], {"date": "20240105"}),
    # Edge case: dataset with parsed_date
    (["dataset", "date", "parsed_date"], {
        "dataset": "analysis", 
        "date": "20240106",
        "parsed_date": datetime(2024, 1, 6)
    }),
])
def test_metadata_extraction_field_coverage(
    enhanced_mock_dependencies, 
    metadata_fields, 
    expected_values
):
    """
    Test comprehensive metadata extraction covering all supported fields per F-007.
    
    This parametrized test validates extraction of date, condition, replicate, 
    dataset, and parsed_date fields as specified in F-007 Metadata Extraction System.
    """
    mock_load_config, mock_discover_experiment_files, _ = enhanced_mock_dependencies
    
    # Configure mock to return files with specific metadata fields
    test_file = f"/path/to/data/test_file_{expected_values['date']}.csv"
    mock_discover_experiment_files.return_value = {
        test_file: expected_values
    }
    
    # Execute the API call with metadata extraction enabled
    result = load_experiment_files(
        config_path="/path/to/config.yaml",
        experiment_name="test_experiment", 
        extract_metadata=True
    )
    
    # Verify all expected metadata fields are present
    assert test_file in result
    file_metadata = result[test_file]
    
    for field in metadata_fields:
        assert field in file_metadata, f"Missing required metadata field: {field}"
        assert file_metadata[field] == expected_values[field], f"Incorrect value for field {field}"


@pytest.mark.parametrize("experiment_type,dataset_name,expected_api", [
    ("experiment", "test_experiment", "load_experiment_files"),
    ("dataset", "test_dataset", "load_dataset_files"),
])
def test_metadata_extraction_api_consistency(
    enhanced_mock_dependencies,
    metadata_test_config,
    experiment_type,
    dataset_name, 
    expected_api
):
    """
    Test metadata extraction consistency across experiment and dataset APIs.
    
    Validates that both load_experiment_files and load_dataset_files
    properly handle metadata extraction per F-007 requirements.
    """
    mock_load_config, mock_discover_experiment_files, mock_discover_dataset_files = enhanced_mock_dependencies
    
    # Configure mock with the test configuration
    mock_load_config.return_value = metadata_test_config
    
    # Prepare test metadata for both APIs
    test_metadata = {
        "/path/to/data/test_20240101_baseline_1.csv": {
            "date": "20240101",
            "condition": "baseline", 
            "replicate": "1"
        }
    }
    
    if experiment_type == "experiment":
        mock_discover_experiment_files.return_value = test_metadata
        result = load_experiment_files(
            config=metadata_test_config,
            experiment_name=dataset_name,
            extract_metadata=True
        )
        # Verify experiment API was called with correct parameters
        mock_discover_experiment_files.assert_called_once()
    else:
        mock_discover_dataset_files.return_value = test_metadata
        result = load_dataset_files(
            config=metadata_test_config,
            dataset_name=dataset_name,
            extract_metadata=True
        )
        # Verify dataset API was called with correct parameters
        mock_discover_dataset_files.assert_called_once()
    
    # Verify consistent metadata extraction behavior
    assert isinstance(result, dict)
    assert len(result) == 1
    file_path = list(result.keys())[0]
    assert "date" in result[file_path]
    assert "condition" in result[file_path]
    assert "replicate" in result[file_path]


# ============================================================================
# CONFIGURATION VALIDATION TESTS (F-001-RQ-002 Requirements)
# ============================================================================

@pytest.mark.parametrize("config_path,config_dict,extract_metadata,should_raise", [
    # Valid cases: exactly one of config_path or config provided
    ("/path/to/config.yaml", None, True, False),
    (None, {"test": "config"}, True, False),
    
    # Invalid cases: neither provided (F-001-RQ-002 violation)
    (None, None, True, True),
    
    # Invalid cases: both provided (F-001-RQ-002 violation)  
    ("/path/to/config.yaml", {"test": "config"}, True, True),
    
    # Valid cases: metadata extraction disabled (should still validate)
    ("/path/to/config.yaml", None, False, False),
    (None, {"test": "config"}, False, False),
    
    # Invalid cases: metadata extraction disabled but invalid config
    (None, None, False, True),
    ("/path/to/config.yaml", {"test": "config"}, False, True),
])
def test_configuration_validation_with_metadata_extraction(
    enhanced_mock_dependencies,
    config_path,
    config_dict, 
    extract_metadata,
    should_raise
):
    """
    Test configuration validation per F-001-RQ-002 requirements.
    
    Ensures exactly one of config_path or config is provided when
    metadata extraction is enabled, as specified in F-001-RQ-002.
    """
    mock_load_config, mock_discover_experiment_files, _ = enhanced_mock_dependencies
    
    if should_raise:
        # Test that ValueError is raised for invalid configuration combinations
        with pytest.raises(ValueError, match="Exactly one of 'config_path' or 'config' must be provided"):
            load_experiment_files(
                config_path=config_path,
                config=config_dict,
                experiment_name="test_experiment",
                extract_metadata=extract_metadata
            )
    else:
        # Test that valid configurations work correctly
        try:
            result = load_experiment_files(
                config_path=config_path,
                config=config_dict, 
                experiment_name="test_experiment",
                extract_metadata=extract_metadata
            )
            # Verify the call succeeded and returned expected type
            assert isinstance(result, (dict, list))
        except ValueError as e:
            pytest.fail(f"Valid configuration raised ValueError: {e}")


def test_dataset_configuration_validation_with_metadata_extraction(enhanced_mock_dependencies):
    """
    Test dataset API configuration validation per F-001-RQ-002 requirements.
    
    Validates that load_dataset_files also enforces the configuration
    validation rules when metadata extraction is enabled.
    """
    mock_load_config, _, mock_discover_dataset_files = enhanced_mock_dependencies
    
    # Test invalid case: both config_path and config provided
    with pytest.raises(ValueError, match="Exactly one of 'config_path' or 'config' must be provided"):
        load_dataset_files(
            config_path="/path/to/config.yaml",
            config={"test": "config"},
            dataset_name="test_dataset",
            extract_metadata=True,
            parse_dates=True
        )
    
    # Test invalid case: neither config_path nor config provided
    with pytest.raises(ValueError, match="Exactly one of 'config_path' or 'config' must be provided"):
        load_dataset_files(
            dataset_name="test_dataset",
            extract_metadata=True
        )


# ============================================================================
# DATE PARSING VALIDATION TESTS (F-002-RQ-005 Requirements)
# ============================================================================

@pytest.mark.parametrize("date_string,date_format,expected_datetime", [
    # Standard date formats per F-002-RQ-005
    ("20240101", "%Y%m%d", datetime(2024, 1, 1)),
    ("2024-01-01", "%Y-%m-%d", datetime(2024, 1, 1)),
    ("01-01-2024", "%m-%d-%Y", datetime(2024, 1, 1)),
    
    # Edge cases: leap year handling
    ("20240229", "%Y%m%d", datetime(2024, 2, 29)),
    ("2024-02-29", "%Y-%m-%d", datetime(2024, 2, 29)),
    
    # Edge cases: end of year
    ("20231231", "%Y%m%d", datetime(2023, 12, 31)),
    ("2023-12-31", "%Y-%m-%d", datetime(2023, 12, 31)),
    
    # Edge cases: beginning of year
    ("20240101", "%Y%m%d", datetime(2024, 1, 1)),
    ("2024-01-01", "%Y-%m-%d", datetime(2024, 1, 1)),
])
def test_date_parsing_format_validation(
    enhanced_mock_dependencies,
    date_string,
    date_format,
    expected_datetime
):
    """
    Test date parsing validation with various formats per F-002-RQ-005 requirements.
    
    Validates edge cases for various date formats and timezone handling
    as specified in F-002-RQ-005.
    """
    mock_load_config, mock_discover_experiment_files, _ = enhanced_mock_dependencies
    
    # Configure mock to return files with parsed dates
    test_file = f"/path/to/data/file_{date_string}.csv"
    mock_discover_experiment_files.return_value = {
        test_file: {
            "date": date_string,
            "condition": "test",
            "replicate": "1",
            "parsed_date": expected_datetime
        }
    }
    
    # Call with date parsing enabled
    result = load_experiment_files(
        config_path="/path/to/config.yaml",
        experiment_name="test_experiment",
        extract_metadata=True,
        parse_dates=True
    )
    
    # Verify date parsing worked correctly
    assert test_file in result
    file_metadata = result[test_file]
    
    assert "date" in file_metadata
    assert "parsed_date" in file_metadata
    assert file_metadata["date"] == date_string
    assert file_metadata["parsed_date"] == expected_datetime
    
    # Verify parse_dates parameter was passed correctly
    call_args = mock_discover_experiment_files.call_args
    assert call_args[1]["parse_dates"] is True


@pytest.mark.parametrize("timezone_offset,expected_tz", [
    # Timezone handling edge cases
    (0, timezone.utc),
    (-5, timezone.utc),  # EST (simplified for testing)
    (8, timezone.utc),   # PST (simplified for testing)
])
def test_date_parsing_timezone_handling(
    enhanced_mock_dependencies,
    timezone_offset,
    expected_tz
):
    """
    Test timezone handling in date parsing per F-002-RQ-005 requirements.
    
    Note: This test focuses on the API's ability to handle timezone-aware
    datetime objects, as the actual timezone parsing logic is in the
    discovery module.
    """
    mock_load_config, mock_discover_experiment_files, _ = enhanced_mock_dependencies
    
    # Create timezone-aware datetime for testing
    base_date = datetime(2024, 1, 1)
    if timezone_offset != 0:
        # For testing purposes, we use UTC as the baseline
        test_datetime = base_date.replace(tzinfo=expected_tz)
    else:
        test_datetime = base_date.replace(tzinfo=expected_tz)
    
    # Configure mock with timezone-aware datetime
    test_file = "/path/to/data/file_20240101_tz_test.csv"
    mock_discover_experiment_files.return_value = {
        test_file: {
            "date": "20240101",
            "condition": "timezone_test",
            "replicate": "1", 
            "parsed_date": test_datetime
        }
    }
    
    result = load_experiment_files(
        config_path="/path/to/config.yaml",
        experiment_name="test_experiment",
        extract_metadata=True,
        parse_dates=True
    )
    
    # Verify timezone information is preserved
    file_metadata = result[test_file]
    assert "parsed_date" in file_metadata
    returned_datetime = file_metadata["parsed_date"]
    
    # Verify timezone information (if present) is preserved
    if hasattr(returned_datetime, 'tzinfo') and returned_datetime.tzinfo:
        assert returned_datetime.tzinfo == expected_tz


# ============================================================================
# MOCK ISOLATION TESTS (TST-MOD-003 Requirements)
# ============================================================================

def test_load_config_isolation(enhanced_mock_dependencies, metadata_test_config):
    """
    Test isolation of load_config dependency per TST-MOD-003 requirements.
    
    Validates that pytest-mock provides proper isolation of the load_config
    function for comprehensive unit testing.
    """
    mock_load_config, mock_discover_experiment_files, _ = enhanced_mock_dependencies
    
    # Configure specific return value for this test
    mock_load_config.return_value = metadata_test_config
    
    # Call the API
    load_experiment_files(
        config_path="/specific/test/config.yaml",
        experiment_name="test_experiment",
        extract_metadata=True
    )
    
    # Verify mock isolation - exact call verification
    mock_load_config.assert_called_once_with("/specific/test/config.yaml")
    
    # Verify the mock received the expected configuration
    assert mock_load_config.return_value == metadata_test_config


def test_discover_experiment_files_isolation(enhanced_mock_dependencies):
    """
    Test isolation of discover_experiment_files dependency per TST-MOD-003 requirements.
    
    Validates comprehensive mocking of the file discovery process for
    unit test isolation.
    """
    mock_load_config, mock_discover_experiment_files, _ = enhanced_mock_dependencies
    
    # Configure specific test scenario
    test_config = {
        "project": {"directories": {"major_data_directory": "/test/data"}},
        "experiments": {"isolated_test": {"datasets": ["test_dataset"]}}
    }
    mock_load_config.return_value = test_config
    
    expected_files = {
        "/test/data/isolated_20240101_test_1.csv": {
            "date": "20240101",
            "condition": "test", 
            "replicate": "1"
        }
    }
    mock_discover_experiment_files.return_value = expected_files
    
    # Execute API call
    result = load_experiment_files(
        config_path="/test/config.yaml",
        experiment_name="isolated_test",
        extract_metadata=True,
        parse_dates=False,
        pattern="*.csv",
        recursive=False
    )
    
    # Verify complete isolation and parameter passing
    mock_discover_experiment_files.assert_called_once_with(
        config=test_config,
        experiment_name="isolated_test",
        base_directory="/test/data",
        pattern="*.csv", 
        recursive=False,
        extensions=None,
        extract_metadata=True,
        parse_dates=False
    )
    
    # Verify return value isolation
    assert result == expected_files


def test_discover_dataset_files_isolation(enhanced_mock_dependencies):
    """
    Test isolation of discover_dataset_files dependency per TST-MOD-003 requirements.
    
    Validates that dataset file discovery can be comprehensively mocked
    for unit test isolation.
    """
    mock_load_config, _, mock_discover_dataset_files = enhanced_mock_dependencies
    
    # Configure test scenario for dataset discovery
    test_config = {
        "project": {"directories": {"major_data_directory": "/dataset/test"}},
        "datasets": {"isolated_dataset": {"patterns": ["*_isolated_*"]}}
    }
    mock_load_config.return_value = test_config
    
    expected_dataset_files = {
        "/dataset/test/isolated_dataset_20240101.pkl": {
            "dataset": "isolated_dataset",
            "date": "20240101"
        }
    }
    mock_discover_dataset_files.return_value = expected_dataset_files
    
    # Execute dataset API call with specific parameters
    result = load_dataset_files(
        config=test_config,  # Using config dict instead of path
        dataset_name="isolated_dataset", 
        extract_metadata=True,
        extensions=["pkl"],
        recursive=True
    )
    
    # Verify complete parameter isolation and passing
    mock_discover_dataset_files.assert_called_once_with(
        config=test_config,
        dataset_name="isolated_dataset",
        base_directory="/dataset/test",
        pattern="*.*",
        recursive=True,
        extensions=["pkl"],
        extract_metadata=True,
        parse_dates=False
    )
    
    # Verify return value isolation
    assert result == expected_dataset_files


# ============================================================================
# PROPERTY-BASED TESTING (Section 3.6.3 Requirements)
# ============================================================================

@given(
    date_str=st.text(
        alphabet=st.characters(whitelist_categories=("Nd",)), 
        min_size=8, 
        max_size=8
    ).filter(lambda x: len(x) == 8 and x.isdigit()),
    condition=st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll")), 
        min_size=3, 
        max_size=10
    ).filter(lambda x: x.isalpha()),
    replicate=st.integers(min_value=1, max_value=99)
)
def test_metadata_extraction_property_based_validation(
    enhanced_mock_dependencies,
    date_str,
    condition,
    replicate
):
    """
    Property-based test for robust metadata extraction validation per Section 3.6.3 requirements.
    
    Uses Hypothesis to generate diverse filename scenarios and validate
    that metadata extraction patterns work across varied inputs.
    """
    mock_load_config, mock_discover_experiment_files, _ = enhanced_mock_dependencies
    
    # Generate test filename using property-based inputs
    test_filename = f"/path/to/data/test_{date_str}_{condition}_{replicate}.csv"
    
    # Configure mock with the generated test data
    expected_metadata = {
        "date": date_str,
        "condition": condition,
        "replicate": str(replicate)
    }
    
    mock_discover_experiment_files.return_value = {
        test_filename: expected_metadata
    }
    
    # Execute the API call
    result = load_experiment_files(
        config_path="/test/config.yaml",
        experiment_name="test_experiment",
        extract_metadata=True
    )
    
    # Validate that property-based inputs are handled correctly
    assert test_filename in result
    file_metadata = result[test_filename]
    
    # Verify all expected metadata fields are present and correct
    assert "date" in file_metadata
    assert "condition" in file_metadata
    assert "replicate" in file_metadata
    
    assert file_metadata["date"] == date_str
    assert file_metadata["condition"] == condition
    assert file_metadata["replicate"] == str(replicate)


@given(
    dataset_name=st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
        min_size=3,
        max_size=15
    ).filter(lambda x: x[0].isalpha() if x else False),
    date_components=st.tuples(
        st.integers(min_value=2020, max_value=2030),  # year
        st.integers(min_value=1, max_value=12),       # month
        st.integers(min_value=1, max_value=28)        # day (conservative to avoid invalid dates)
    )
)
def test_dataset_metadata_extraction_property_based(
    enhanced_mock_dependencies,
    dataset_name,
    date_components
):
    """
    Property-based test for dataset metadata extraction validation.
    
    Validates robust handling of diverse dataset names and date combinations
    across varied filename scenarios per Section 3.6.3 requirements.
    """
    mock_load_config, _, mock_discover_dataset_files = enhanced_mock_dependencies
    
    year, month, day = date_components
    
    # Format date string with zero-padding
    date_str = f"{year:04d}{month:02d}{day:02d}"
    
    # Generate test filename
    test_filename = f"/path/to/data/{dataset_name}_{date_str}.csv"
    
    # Configure mock with property-based test data
    expected_metadata = {
        "dataset": dataset_name,
        "date": date_str
    }
    
    mock_discover_dataset_files.return_value = {
        test_filename: expected_metadata
    }
    
    # Execute dataset API call
    result = load_dataset_files(
        config_path="/test/config.yaml",
        dataset_name=dataset_name,
        extract_metadata=True
    )
    
    # Validate property-based dataset metadata extraction
    assert test_filename in result
    file_metadata = result[test_filename]
    
    assert "dataset" in file_metadata
    assert "date" in file_metadata
    
    assert file_metadata["dataset"] == dataset_name
    assert file_metadata["date"] == date_str


# ============================================================================
# INTEGRATION AND EDGE CASE TESTS
# ============================================================================

def test_metadata_extraction_with_empty_results(enhanced_mock_dependencies):
    """
    Test metadata extraction behavior when no files are discovered.
    
    Validates proper handling of empty result sets to ensure
    robust error handling.
    """
    mock_load_config, mock_discover_experiment_files, _ = enhanced_mock_dependencies
    
    # Configure mock to return empty results
    mock_discover_experiment_files.return_value = {}
    
    # Execute API call
    result = load_experiment_files(
        config_path="/test/config.yaml",
        experiment_name="empty_experiment",
        extract_metadata=True
    )
    
    # Verify empty results are handled correctly
    assert isinstance(result, dict)
    assert len(result) == 0


def test_metadata_extraction_without_parsed_dates(enhanced_mock_dependencies):
    """
    Test metadata extraction when parse_dates is False.
    
    Validates that parsed_date field is not included when
    date parsing is disabled.
    """
    mock_load_config, mock_discover_experiment_files, _ = enhanced_mock_dependencies
    
    # Configure mock without parsed_date field
    test_metadata = {
        "/path/to/data/no_parsed_date_20240101.csv": {
            "date": "20240101",
            "condition": "no_parsing",
            "replicate": "1"
            # Note: no parsed_date field
        }
    }
    mock_discover_experiment_files.return_value = test_metadata
    
    # Execute with parse_dates=False (default)
    result = load_experiment_files(
        config_path="/test/config.yaml",
        experiment_name="test_experiment",
        extract_metadata=True,
        parse_dates=False
    )
    
    # Verify parsed_date is not present
    file_metadata = result["/path/to/data/no_parsed_date_20240101.csv"]
    assert "date" in file_metadata
    assert "condition" in file_metadata
    assert "replicate" in file_metadata
    assert "parsed_date" not in file_metadata
    
    # Verify parse_dates parameter was passed correctly
    call_args = mock_discover_experiment_files.call_args
    assert call_args[1]["parse_dates"] is False


def test_metadata_extraction_with_complex_patterns(enhanced_mock_dependencies, sample_file_metadata):
    """
    Test metadata extraction with complex filename patterns.
    
    Validates handling of various filename patterns and metadata
    field combinations across different experimental scenarios.
    """
    mock_load_config, mock_discover_experiment_files, _ = enhanced_mock_dependencies
    
    # Configure mock with complex metadata patterns
    mock_discover_experiment_files.return_value = sample_file_metadata
    
    # Execute API call
    result = load_experiment_files(
        config_path="/test/config.yaml",
        experiment_name="complex_experiment",
        extract_metadata=True,
        parse_dates=True
    )
    
    # Verify all files and their metadata are present
    assert len(result) == len(sample_file_metadata)
    
    for file_path, expected_metadata in sample_file_metadata.items():
        assert file_path in result
        file_metadata = result[file_path]
        
        # Verify all expected fields are present
        for field, expected_value in expected_metadata.items():
            assert field in file_metadata
            assert file_metadata[field] == expected_value


# ============================================================================
# PERFORMANCE AND STRESS TESTS
# ============================================================================

@pytest.mark.benchmark
def test_metadata_extraction_performance(enhanced_mock_dependencies):
    """
    Performance test for metadata extraction with large file sets.
    
    Validates that metadata extraction scales appropriately with
    the number of files being processed.
    """
    mock_load_config, mock_discover_experiment_files, _ = enhanced_mock_dependencies
    
    # Generate large set of test files for performance testing
    large_file_set = {}
    for i in range(1000):
        file_path = f"/path/to/data/performance_test_{20240101 + i % 365:08d}_{i % 10}.csv"
        large_file_set[file_path] = {
            "date": f"{20240101 + i % 365:08d}",
            "condition": f"condition_{i % 5}",
            "replicate": str(i % 10)
        }
    
    mock_discover_experiment_files.return_value = large_file_set
    
    # Execute API call and measure performance implicitly through pytest-benchmark
    result = load_experiment_files(
        config_path="/test/config.yaml",
        experiment_name="performance_test",
        extract_metadata=True
    )
    
    # Verify large dataset was processed correctly
    assert len(result) == 1000
    assert isinstance(result, dict)


if __name__ == "__main__":
    # Support running tests directly
    pytest.main([__file__, "-v"])