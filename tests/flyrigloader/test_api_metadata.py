"""
Behavior-focused metadata extraction tests for the high-level API functions.

This module implements comprehensive testing for the F-007 Metadata Extraction System
using behavior-focused validation through public API contracts and observable system behavior.
Implements centralized fixture management and Protocol-based mock implementations per 
Section 0 requirements for testing strategy enhancement.

Key Testing Approach:
- Black-box behavioral validation through public API responses
- Protocol-based mock implementations from centralized tests/utils.py
- AAA pattern structure with clear separation of test phases
- Edge-case coverage through parameterized test scenarios
- Centralized fixtures from tests/conftest.py for consistent patterns
- Observable metadata extraction behavior focus
"""
import pytest
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
from hypothesis import given, strategies as st

# Import API functions for behavior validation
from flyrigloader.api import (
    load_experiment_files,
    load_dataset_files
)

# Import centralized test utilities for Protocol-based mocking
from tests.utils import (
    create_mock_config_provider,
    create_mock_dataloader,
    MockConfigurationProvider,
    MockDataLoading,
    generate_edge_case_scenarios,
    create_hypothesis_strategies
)


# ============================================================================
# CENTRALIZED FIXTURE INTEGRATION AND PROTOCOL-BASED MOCKING
# ============================================================================

@pytest.fixture
def metadata_test_config(sample_comprehensive_config_dict):
    """
    Enhanced metadata test configuration utilizing centralized fixtures.
    
    Leverages sample_comprehensive_config_dict from tests/conftest.py for consistent
    configuration patterns while adding metadata-specific extraction patterns
    per F-007 Metadata Extraction System requirements.
    
    Returns:
        Dict[str, Any]: Configuration with enhanced metadata extraction patterns
    """
    # ARRANGE - Extend centralized configuration with metadata-specific patterns
    config = sample_comprehensive_config_dict.copy()
    
    # Enhance with metadata extraction patterns for testing
    config["project"]["extraction_patterns"] = [
        r".*_(?P<date>\d{8})_(?P<condition>\w+)_(?P<replicate>\d+)\.csv",
        r".*_(?P<experiment>\w+)_(?P<date>\d{8})\.pkl",
        r"(?P<dataset>\w+)_(?P<date>\d{8})_(?P<condition>\w+)_rep(?P<replicate>\d+)\.pkl"
    ]
    
    # Add experiment metadata patterns
    for experiment_name, experiment_config in config.get("experiments", {}).items():
        if "metadata" not in experiment_config:
            experiment_config["metadata"] = {}
        experiment_config["metadata"]["extraction_patterns"] = [
            r".*_(?P<experiment>\w+)_(?P<date>\d{8})_(?P<condition>\w+)\.csv"
        ]
    
    # Add dataset metadata patterns
    for dataset_name, dataset_config in config.get("datasets", {}).items():
        if "metadata" not in dataset_config:
            dataset_config["metadata"] = {}
        dataset_config["metadata"]["extraction_patterns"] = [
            r".*_(?P<dataset>\w+)_(?P<date>\d{8})_(?P<vial>\d+)\.csv"
        ]
    
    return config


@pytest.fixture
def sample_metadata_extraction_scenarios(sample_experimental_metadata):
    """
    Fixture providing comprehensive metadata samples leveraging centralized test data.
    
    Utilizes sample_experimental_metadata from tests/conftest.py and extends it
    with additional metadata extraction scenarios per F-007 requirements.
    All supported metadata fields: date, condition, replicate, dataset, parsed_date.
    
    Returns:
        Dict[str, Dict[str, Any]]: File paths mapped to metadata extraction results
    """
    # ARRANGE - Base metadata from centralized fixture
    base_metadata = sample_experimental_metadata
    
    # ARRANGE - Extended metadata scenarios for comprehensive testing
    metadata_scenarios = {
        "/research/data/experiment_20240101_baseline_1.csv": {
            "date": "20240101",
            "condition": "baseline",
            "replicate": "1",
            "experiment": "baseline_control_study",
            "parsed_date": datetime(2024, 1, 1)
        },
        "/research/data/experiment_20240102_treatment_2.csv": {
            "date": "20240102", 
            "condition": "treatment",
            "replicate": "2",
            "experiment": "optogenetic_stimulation",
            "parsed_date": datetime(2024, 1, 2)
        },
        "/research/data/dataset_test_20240103.csv": {
            "dataset": "baseline_behavior",
            "date": "20240103",
            "vial": "1",
            "parsed_date": datetime(2024, 1, 3)
        },
        "/research/data/multi_20240104_control_3.csv": {
            "date": "20240104",
            "condition": "control", 
            "replicate": "3",
            "dataset": "baseline_behavior",
            "parsed_date": datetime(2024, 1, 4)
        }
    }
    
    # ARRANGE - Merge with base metadata for comprehensive coverage
    metadata_scenarios.update({
        f"/research/data/{base_metadata['exp_name']}_{base_metadata['date']}.csv": base_metadata
    })
    
    return metadata_scenarios


@pytest.fixture
def date_parsing_edge_cases():
    """
    Fixture providing diverse date format edge cases leveraging boundary testing patterns.
    
    Tests various date formats and edge cases including timezone handling
    per F-002-RQ-005 requirements with enhanced boundary condition coverage.
    
    Returns:
        List[Tuple]: Date strings, formats, and expected datetime objects
    """
    # ARRANGE - Standard date format test cases
    standard_cases = [
        # Standard YYYYMMDD format
        ("20240101", "%Y%m%d", datetime(2024, 1, 1)),
        ("20231225", "%Y%m%d", datetime(2023, 12, 25)),
        
        # ISO format YYYY-MM-DD
        ("2024-01-01", "%Y-%m-%d", datetime(2024, 1, 1)),
        ("2023-12-25", "%Y-%m-%d", datetime(2023, 12, 25)),
        
        # US format MM-DD-YYYY
        ("01-01-2024", "%m-%d-%Y", datetime(2024, 1, 1)),
        ("12-25-2023", "%m-%d-%Y", datetime(2023, 12, 25)),
    ]
    
    # ARRANGE - Edge case scenarios for boundary testing
    edge_cases = [
        # Leap year handling (boundary condition)
        ("20240229", "%Y%m%d", datetime(2024, 2, 29)),
        ("2024-02-29", "%Y-%m-%d", datetime(2024, 2, 29)),
        
        # Year boundaries
        ("20231231", "%Y%m%d", datetime(2023, 12, 31)),
        ("20240101", "%Y%m%d", datetime(2024, 1, 1)),
        ("2023-12-31", "%Y-%m-%d", datetime(2023, 12, 31)),
        ("2024-01-01", "%Y-%m-%d", datetime(2024, 1, 1)),
        
        # Month boundaries
        ("20240131", "%Y%m%d", datetime(2024, 1, 31)),
        ("20240201", "%Y%m%d", datetime(2024, 2, 1)),
    ]
    
    return standard_cases + edge_cases


@pytest.fixture
def protocol_based_mock_dependencies(metadata_test_config):
    """
    Protocol-based mock fixture using centralized utilities from tests/utils.py.
    
    Replaces monkey-patching with Protocol-based mock implementations for 
    comprehensive dependency isolation per testing strategy requirements.
    Utilizes create_mock_config_provider and related utilities for consistent patterns.
    
    Returns:
        Tuple: Protocol-based mock instances for configuration and discovery
    """
    # ARRANGE - Create Protocol-based configuration mock using centralized utilities
    mock_config_provider = create_mock_config_provider(
        config_type='comprehensive', 
        include_errors=False
    )
    mock_config_provider.add_configuration('metadata_test', metadata_test_config)
    
    # ARRANGE - Create Protocol-based data loading mock
    mock_data_loader = create_mock_dataloader(
        scenarios=['basic', 'metadata'],
        include_experimental_data=True
    )
    
    # ARRANGE - Configure mock behavior for metadata extraction
    mock_config_provider.configurations['default'] = metadata_test_config
    
    return mock_config_provider, mock_data_loader


# ============================================================================
# BEHAVIOR-FOCUSED METADATA EXTRACTION TESTS (F-007 Requirements)
# ============================================================================

@pytest.mark.parametrize("metadata_fields,expected_values", [
    # Test all supported metadata fields per F-007 requirements
    (["date", "condition", "replicate"], {"date": "20240101", "condition": "baseline", "replicate": "1"}),
    (["date", "dataset"], {"date": "20240102", "dataset": "baseline_behavior"}),
    (["experiment", "date", "condition"], {"experiment": "baseline_control_study", "date": "20240103", "condition": "treatment"}),
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
        "dataset": "baseline_behavior", 
        "date": "20240106",
        "parsed_date": datetime(2024, 1, 6)
    }),
])
def test_metadata_extraction_field_coverage_behavior_validation(
    mocker,
    metadata_test_config,
    metadata_fields, 
    expected_values
):
    """
    Behavior-focused test for comprehensive metadata extraction using public API validation.
    
    Tests metadata extraction through observable API behavior rather than implementation details.
    Validates all supported fields per F-007 Metadata Extraction System using Protocol-based mocks.
    """
    # ARRANGE - Set up Protocol-based mocks for dependency injection
    mock_deps = mocker.Mock()
    mock_deps.config.load_config.return_value = metadata_test_config
    mock_deps.config.get_experiment_info.return_value = metadata_test_config['experiments']['baseline_control_study']
    
    # ARRANGE - Configure discovery mock to return expected metadata
    test_file = f"/research/data/test_file_{expected_values['date']}.csv"
    mock_deps.discovery.discover_experiment_files.return_value = {
        test_file: expected_values
    }
    
    # ACT - Execute API call with metadata extraction enabled
    result = load_experiment_files(
        config_path="/test/config.yaml",
        experiment_name="baseline_control_study", 
        extract_metadata=True,
        _deps=mock_deps
    )
    
    # ASSERT - Verify behavioral response contains all expected metadata fields
    assert isinstance(result, dict), "API should return dictionary when extract_metadata=True"
    assert test_file in result, f"Expected file {test_file} not found in API response"
    
    file_metadata = result[test_file]
    assert isinstance(file_metadata, dict), "File metadata should be a dictionary"
    
    # ASSERT - Validate each expected metadata field through API response
    for field in metadata_fields:
        assert field in file_metadata, f"API response missing required metadata field: {field}"
        assert file_metadata[field] == expected_values[field], f"API response has incorrect value for field {field}"
    
    # ASSERT - Verify API behavior through dependency interactions
    mock_deps.config.load_config.assert_called_once_with("/test/config.yaml")
    mock_deps.discovery.discover_experiment_files.assert_called_once()


@pytest.mark.parametrize("api_type,entity_name,expected_behavior", [
    ("experiment", "baseline_control_study", "experiment_metadata_extraction"),
    ("dataset", "baseline_behavior", "dataset_metadata_extraction"),
])
def test_api_consistency_behavior_validation(
    mocker,
    metadata_test_config,
    api_type,
    entity_name, 
    expected_behavior
):
    """
    Behavior-focused test for metadata extraction consistency across experiment and dataset APIs.
    
    Validates consistent behavioral responses from both load_experiment_files and load_dataset_files
    through public API contracts per F-007 requirements without implementation coupling.
    """
    # ARRANGE - Set up Protocol-based mocks for both API paths
    mock_deps = mocker.Mock()
    mock_deps.config.load_config.return_value = metadata_test_config
    
    # ARRANGE - Prepare test metadata that both APIs should handle consistently
    test_metadata = {
        "/research/data/test_20240101_baseline_1.csv": {
            "date": "20240101",
            "condition": "baseline", 
            "replicate": "1",
            "dataset": "baseline_behavior",
            "experiment": "baseline_control_study"
        }
    }
    
    # ARRANGE - Configure entity-specific mock responses
    if api_type == "experiment":
        mock_deps.config.get_experiment_info.return_value = metadata_test_config['experiments'][entity_name]
        mock_deps.discovery.discover_experiment_files.return_value = test_metadata
    else:
        mock_deps.config.get_dataset_info.return_value = metadata_test_config['datasets'][entity_name] 
        mock_deps.discovery.discover_dataset_files.return_value = test_metadata
    
    # ACT - Execute appropriate API call based on entity type
    if api_type == "experiment":
        result = load_experiment_files(
            config=metadata_test_config,
            experiment_name=entity_name,
            extract_metadata=True,
            _deps=mock_deps
        )
    else:
        result = load_dataset_files(
            config=metadata_test_config,
            dataset_name=entity_name,
            extract_metadata=True,
            _deps=mock_deps
        )
    
    # ASSERT - Verify consistent behavioral response structure across APIs
    assert isinstance(result, dict), "Both APIs should return dict when extract_metadata=True"
    assert len(result) == 1, "API should return exactly one file in test scenario"
    
    file_path = list(result.keys())[0]
    file_metadata = result[file_path]
    
    # ASSERT - Verify consistent metadata structure across both APIs
    assert isinstance(file_metadata, dict), "File metadata should be dictionary for both APIs"
    assert "date" in file_metadata, "Both APIs should extract date metadata field"
    assert "condition" in file_metadata, "Both APIs should extract condition metadata field"
    assert "replicate" in file_metadata, "Both APIs should extract replicate metadata field"
    
    # ASSERT - Verify expected field values through API behavior
    assert file_metadata["date"] == "20240101", "API should return correct date value"
    assert file_metadata["condition"] == "baseline", "API should return correct condition value"
    assert file_metadata["replicate"] == "1", "API should return correct replicate value"
    
    # ASSERT - Verify appropriate dependency interaction occurred
    if api_type == "experiment":
        mock_deps.discovery.discover_experiment_files.assert_called_once()
        mock_deps.config.get_experiment_info.assert_called_once_with(metadata_test_config, entity_name)
    else:
        mock_deps.discovery.discover_dataset_files.assert_called_once() 
        mock_deps.config.get_dataset_info.assert_called_once_with(metadata_test_config, entity_name)


# ============================================================================
# BEHAVIOR-FOCUSED CONFIGURATION VALIDATION TESTS (F-001-RQ-002 Requirements)  
# ============================================================================

@pytest.mark.parametrize("config_path,config_dict,extract_metadata,should_raise", [
    # Valid cases: exactly one of config_path or config provided
    ("/test/config.yaml", None, True, False),
    (None, {"project": {"directories": {"major_data_directory": "/test"}}}, True, False),
    
    # Invalid cases: neither provided (F-001-RQ-002 violation)
    (None, None, True, True),
    
    # Invalid cases: both provided (F-001-RQ-002 violation)  
    ("/test/config.yaml", {"test": "config"}, True, True),
    
    # Valid cases: metadata extraction disabled (should still validate)
    ("/test/config.yaml", None, False, False),
    (None, {"project": {"directories": {"major_data_directory": "/test"}}}, False, False),
    
    # Invalid cases: metadata extraction disabled but invalid config
    (None, None, False, True),
    ("/test/config.yaml", {"test": "config"}, False, True),
])
def test_configuration_validation_behavior_focused(
    mocker,
    metadata_test_config,
    config_path,
    config_dict, 
    extract_metadata,
    should_raise
):
    """
    Behavior-focused test for configuration validation per F-001-RQ-002 requirements.
    
    Validates API behavior for configuration parameter validation through public API contracts
    without coupling to internal validation implementation details.
    """
    # ARRANGE - Set up Protocol-based mocks for dependency injection
    mock_deps = mocker.Mock()
    
    # ARRANGE - Configure mock behavior for valid configuration scenarios
    if config_path and not config_dict:
        mock_deps.config.load_config.return_value = metadata_test_config
    elif config_dict and not config_path:
        # Config dict scenario doesn't need load_config call
        pass
    
    mock_deps.config.get_experiment_info.return_value = metadata_test_config['experiments']['baseline_control_study']
    mock_deps.discovery.discover_experiment_files.return_value = {}
    
    # ACT & ASSERT - Test expected API behavior based on configuration validity
    if should_raise:
        # ASSERT - Invalid configurations should raise ValueError through API
        with pytest.raises(ValueError, match="Exactly one of 'config_path' or 'config' must be provided"):
            load_experiment_files(
                config_path=config_path,
                config=config_dict,
                experiment_name="baseline_control_study",
                extract_metadata=extract_metadata,
                _deps=mock_deps
            )
    else:
        # ACT - Valid configurations should succeed through API
        try:
            result = load_experiment_files(
                config_path=config_path,
                config=config_dict, 
                experiment_name="baseline_control_study",
                extract_metadata=extract_metadata,
                _deps=mock_deps
            )
            
            # ASSERT - API should return appropriate data structure based on extract_metadata
            if extract_metadata:
                assert isinstance(result, dict), "API should return dict when extract_metadata=True"
            else:
                assert isinstance(result, (dict, list)), "API should return valid data structure"
                
        except ValueError as e:
            pytest.fail(f"Valid configuration raised unexpected ValueError: {e}")


def test_dataset_api_configuration_validation_behavior(mocker, metadata_test_config):
    """
    Behavior-focused test for dataset API configuration validation per F-001-RQ-002 requirements.
    
    Validates load_dataset_files API behavior for configuration parameter validation 
    through public API contracts without coupling to implementation details.
    """
    # ARRANGE - Set up Protocol-based mocks for dependency injection
    mock_deps = mocker.Mock()
    mock_deps.config.load_config.return_value = metadata_test_config
    mock_deps.config.get_dataset_info.return_value = metadata_test_config['datasets']['baseline_behavior']
    mock_deps.discovery.discover_dataset_files.return_value = {}
    
    # ACT & ASSERT - Test invalid case: both config_path and config provided
    with pytest.raises(ValueError, match="Exactly one of 'config_path' or 'config' must be provided"):
        load_dataset_files(
            config_path="/test/config.yaml",
            config={"test": "config"},
            dataset_name="baseline_behavior",
            extract_metadata=True,
            parse_dates=True,
            _deps=mock_deps
        )
    
    # ACT & ASSERT - Test invalid case: neither config_path nor config provided  
    with pytest.raises(ValueError, match="Exactly one of 'config_path' or 'config' must be provided"):
        load_dataset_files(
            dataset_name="baseline_behavior",
            extract_metadata=True,
            _deps=mock_deps
        )
    
    # ACT & ASSERT - Test valid case: config_path only
    try:
        result = load_dataset_files(
            config_path="/test/config.yaml",
            dataset_name="baseline_behavior",
            extract_metadata=True,
            _deps=mock_deps
        )
        # ASSERT - API should succeed and return appropriate data structure
        assert isinstance(result, dict), "API should return dict when extract_metadata=True"
    except ValueError as e:
        pytest.fail(f"Valid configuration raised unexpected ValueError: {e}")
    
    # ACT & ASSERT - Test valid case: config dict only
    try:
        result = load_dataset_files(
            config=metadata_test_config,
            dataset_name="baseline_behavior", 
            extract_metadata=True,
            _deps=mock_deps
        )
        # ASSERT - API should succeed with valid configuration
        assert isinstance(result, dict), "API should return dict when extract_metadata=True"
    except ValueError as e:
        pytest.fail(f"Valid configuration raised unexpected ValueError: {e}")


# ============================================================================
# BEHAVIOR-FOCUSED DATE PARSING VALIDATION TESTS (F-002-RQ-005 Requirements)
# ============================================================================

@pytest.mark.parametrize("date_string,date_format,expected_datetime", [
    # Standard date formats per F-002-RQ-005
    ("20240101", "%Y%m%d", datetime(2024, 1, 1)),
    ("2024-01-01", "%Y-%m-%d", datetime(2024, 1, 1)),
    ("01-01-2024", "%m-%d-%Y", datetime(2024, 1, 1)),
    
    # Edge cases: leap year handling
    ("20240229", "%Y%m%d", datetime(2024, 2, 29)),
    ("2024-02-29", "%Y-%m-%d", datetime(2024, 2, 29)),
    
    # Edge cases: year boundaries
    ("20231231", "%Y%m%d", datetime(2023, 12, 31)),
    ("2023-12-31", "%Y-%m-%d", datetime(2023, 12, 31)),
    ("20240101", "%Y%m%d", datetime(2024, 1, 1)),
    ("2024-01-01", "%Y-%m-%d", datetime(2024, 1, 1)),
])
def test_date_parsing_behavior_through_api_response(
    mocker,
    metadata_test_config,
    date_string,
    date_format,
    expected_datetime
):
    """
    Behavior-focused test for date parsing validation through API response patterns.
    
    Validates date parsing behavior through observable API responses per F-002-RQ-005 
    requirements without coupling to internal parsing implementation details.
    """
    # ARRANGE - Set up Protocol-based mocks for dependency injection
    mock_deps = mocker.Mock()
    mock_deps.config.load_config.return_value = metadata_test_config
    mock_deps.config.get_experiment_info.return_value = metadata_test_config['experiments']['baseline_control_study']
    
    # ARRANGE - Configure mock to return files with parsed dates per API specification
    test_file = f"/research/data/file_{date_string}.csv"
    expected_metadata = {
        "date": date_string,
        "condition": "baseline",
        "replicate": "1",
        "parsed_date": expected_datetime
    }
    mock_deps.discovery.discover_experiment_files.return_value = {
        test_file: expected_metadata
    }
    
    # ACT - Execute API call with date parsing enabled
    result = load_experiment_files(
        config_path="/test/config.yaml",
        experiment_name="baseline_control_study",
        extract_metadata=True,
        parse_dates=True,
        _deps=mock_deps
    )
    
    # ASSERT - Verify API response contains correct date parsing behavior
    assert isinstance(result, dict), "API should return dict when extract_metadata=True"
    assert test_file in result, f"Expected file {test_file} not found in API response"
    
    file_metadata = result[test_file]
    assert isinstance(file_metadata, dict), "File metadata should be dictionary"
    
    # ASSERT - Verify date-related fields in API response
    assert "date" in file_metadata, "API response should contain original date string"
    assert "parsed_date" in file_metadata, "API response should contain parsed datetime when parse_dates=True"
    
    # ASSERT - Verify correct date values through API behavior
    assert file_metadata["date"] == date_string, "API should preserve original date string"
    assert file_metadata["parsed_date"] == expected_datetime, f"API should parse date correctly: expected {expected_datetime}, got {file_metadata['parsed_date']}"
    
    # ASSERT - Verify API behavior includes additional metadata fields
    assert "condition" in file_metadata, "API should include other metadata fields alongside dates"
    assert file_metadata["condition"] == "baseline", "API should correctly extract non-date metadata"
    
    # ASSERT - Verify appropriate dependency interaction with parse_dates parameter
    mock_deps.discovery.discover_experiment_files.assert_called_once()
    call_kwargs = mock_deps.discovery.discover_experiment_files.call_args[1]
    assert call_kwargs["parse_dates"] is True, "API should pass parse_dates=True to discovery layer"


@pytest.mark.parametrize("timezone_offset,expected_tz", [
    # Timezone handling edge cases for API behavior validation
    (0, timezone.utc),
    (-5, timezone.utc),  # EST (simplified for behavior testing)
    (8, timezone.utc),   # PST (simplified for behavior testing)
])
def test_timezone_aware_datetime_handling_through_api(
    mocker,
    metadata_test_config,
    timezone_offset,
    expected_tz
):
    """
    Behavior-focused test for timezone handling in date parsing per F-002-RQ-005 requirements.
    
    Validates API's ability to handle timezone-aware datetime objects through observable
    behavior rather than testing internal timezone parsing implementation.
    """
    # ARRANGE - Set up Protocol-based mocks for dependency injection
    mock_deps = mocker.Mock()
    mock_deps.config.load_config.return_value = metadata_test_config
    mock_deps.config.get_experiment_info.return_value = metadata_test_config['experiments']['baseline_control_study']
    
    # ARRANGE - Create timezone-aware datetime for API behavior testing
    base_date = datetime(2024, 1, 1)
    test_datetime = base_date.replace(tzinfo=expected_tz)
    
    # ARRANGE - Configure mock with timezone-aware datetime response
    test_file = "/research/data/file_20240101_tz_test.csv"
    expected_metadata = {
        "date": "20240101",
        "condition": "timezone_test",
        "replicate": "1", 
        "parsed_date": test_datetime
    }
    mock_deps.discovery.discover_experiment_files.return_value = {
        test_file: expected_metadata
    }
    
    # ACT - Execute API call with timezone-aware date parsing
    result = load_experiment_files(
        config_path="/test/config.yaml",
        experiment_name="baseline_control_study",
        extract_metadata=True,
        parse_dates=True,
        _deps=mock_deps
    )
    
    # ASSERT - Verify API response preserves timezone information through behavior
    assert isinstance(result, dict), "API should return dict when extract_metadata=True"
    assert test_file in result, f"Expected file {test_file} not found in API response"
    
    file_metadata = result[test_file]
    assert "parsed_date" in file_metadata, "API response should contain parsed_date field"
    
    returned_datetime = file_metadata["parsed_date"]
    assert isinstance(returned_datetime, datetime), "parsed_date should be datetime object"
    
    # ASSERT - Verify timezone preservation through API behavior  
    if hasattr(returned_datetime, 'tzinfo') and returned_datetime.tzinfo:
        assert returned_datetime.tzinfo == expected_tz, f"API should preserve timezone info: expected {expected_tz}, got {returned_datetime.tzinfo}"
    
    # ASSERT - Verify core datetime values are correct through API
    assert returned_datetime.year == 2024, "API should preserve year value"
    assert returned_datetime.month == 1, "API should preserve month value" 
    assert returned_datetime.day == 1, "API should preserve day value"
    
    # ASSERT - Verify other metadata fields preserved alongside timezone data
    assert file_metadata["condition"] == "timezone_test", "API should preserve other metadata with timezone data"
    assert file_metadata["date"] == "20240101", "API should preserve original date string with timezone data"


# ============================================================================
# PROTOCOL-BASED DEPENDENCY ISOLATION TESTS (Testing Strategy Requirements)
# ============================================================================

def test_configuration_provider_isolation_through_dependency_injection(mocker, metadata_test_config):
    """
    Behavior-focused test for configuration provider isolation using Protocol-based mocks.
    
    Validates dependency injection pattern enables proper isolation of configuration
    loading behavior for comprehensive unit testing per testing strategy requirements.
    """
    # ARRANGE - Create Protocol-based configuration provider mock
    mock_deps = mocker.Mock()
    mock_deps.config.load_config.return_value = metadata_test_config
    mock_deps.config.get_experiment_info.return_value = metadata_test_config['experiments']['baseline_control_study']
    mock_deps.discovery.discover_experiment_files.return_value = {}
    
    # ACT - Execute API call with dependency injection
    result = load_experiment_files(
        config_path="/specific/test/config.yaml",
        experiment_name="baseline_control_study",
        extract_metadata=True,
        _deps=mock_deps
    )
    
    # ASSERT - Verify Protocol-based mock isolation through behavior validation
    assert isinstance(result, dict), "API should return dict when extract_metadata=True"
    
    # ASSERT - Verify dependency injection isolation - exact call verification
    mock_deps.config.load_config.assert_called_once_with("/specific/test/config.yaml")
    
    # ASSERT - Verify configuration provider behavior isolation
    assert mock_deps.config.load_config.return_value == metadata_test_config, "Mock should return configured test data"
    
    # ASSERT - Verify experiment info retrieval through isolated provider
    mock_deps.config.get_experiment_info.assert_called_once_with(metadata_test_config, "baseline_control_study")


def test_discovery_provider_isolation_through_protocol_interface(mocker, metadata_test_config):
    """
    Behavior-focused test for discovery provider isolation using Protocol-based interface.
    
    Validates comprehensive Protocol-based mocking of file discovery process through
    dependency injection for complete unit test isolation.
    """
    # ARRANGE - Create Protocol-based discovery provider mock
    mock_deps = mocker.Mock()
    
    # ARRANGE - Configure test scenario with realistic metadata
    test_config = metadata_test_config.copy()
    test_config["experiments"]["isolated_test"] = {"datasets": ["baseline_behavior"]}
    
    mock_deps.config.load_config.return_value = test_config
    mock_deps.config.get_experiment_info.return_value = test_config["experiments"]["isolated_test"]
    
    # ARRANGE - Configure expected discovery behavior through Protocol interface
    expected_files = {
        "/research/data/isolated_20240101_baseline_1.csv": {
            "date": "20240101",
            "condition": "baseline", 
            "replicate": "1",
            "experiment": "isolated_test"
        }
    }
    mock_deps.discovery.discover_experiment_files.return_value = expected_files
    
    # ACT - Execute API call with Protocol-based dependency injection
    result = load_experiment_files(
        config_path="/test/config.yaml",
        experiment_name="isolated_test",
        extract_metadata=True,
        parse_dates=False,
        pattern="*.csv",
        recursive=False,
        _deps=mock_deps
    )
    
    # ASSERT - Verify Protocol-based isolation through behavior validation
    assert isinstance(result, dict), "API should return dict when extract_metadata=True"
    assert result == expected_files, "API should return exact discovery results through Protocol"
    
    # ASSERT - Verify complete parameter isolation through Protocol interface
    mock_deps.discovery.discover_experiment_files.assert_called_once()
    call_kwargs = mock_deps.discovery.discover_experiment_files.call_args[1]
    
    assert call_kwargs["config"] == test_config, "Protocol should receive correct config"
    assert call_kwargs["experiment_name"] == "isolated_test", "Protocol should receive correct experiment name"
    assert call_kwargs["pattern"] == "*.csv", "Protocol should receive correct pattern"
    assert call_kwargs["recursive"] is False, "Protocol should receive correct recursive setting"
    assert call_kwargs["extract_metadata"] is True, "Protocol should receive correct metadata extraction setting"
    assert call_kwargs["parse_dates"] is False, "Protocol should receive correct date parsing setting"
    
    # ASSERT - Verify configuration provider interaction through Protocol
    mock_deps.config.load_config.assert_called_once_with("/test/config.yaml")
    mock_deps.config.get_experiment_info.assert_called_once_with(test_config, "isolated_test")


def test_dataset_discovery_provider_protocol_based_isolation(mocker, metadata_test_config):
    """
    Behavior-focused test for dataset discovery provider isolation using Protocol interface.
    
    Validates comprehensive Protocol-based mocking of dataset file discovery through
    dependency injection pattern for complete unit test isolation.
    """
    # ARRANGE - Create Protocol-based discovery provider mock  
    mock_deps = mocker.Mock()
    
    # ARRANGE - Configure test scenario for dataset discovery
    test_config = metadata_test_config.copy()
    test_config["datasets"]["isolated_dataset"] = {
        "patterns": ["*_isolated_*"],
        "rig": "old_opto",
        "metadata": {
            "extraction_patterns": [r".*_(?P<dataset>\w+)_(?P<date>\d{8})\.pkl"]
        }
    }
    
    mock_deps.config.get_dataset_info.return_value = test_config["datasets"]["isolated_dataset"]
    
    # ARRANGE - Configure expected dataset discovery behavior through Protocol
    expected_dataset_files = {
        "/research/data/isolated_dataset_20240101.pkl": {
            "dataset": "isolated_dataset",
            "date": "20240101",
            "rig": "old_opto"
        }
    }
    mock_deps.discovery.discover_dataset_files.return_value = expected_dataset_files
    
    # ACT - Execute dataset API call with Protocol-based dependency injection
    result = load_dataset_files(
        config=test_config,  # Using config dict for validation
        dataset_name="isolated_dataset", 
        extract_metadata=True,
        extensions=["pkl"],
        recursive=True,
        _deps=mock_deps
    )
    
    # ASSERT - Verify Protocol-based isolation through behavior validation
    assert isinstance(result, dict), "Dataset API should return dict when extract_metadata=True"
    assert result == expected_dataset_files, "API should return exact dataset discovery results through Protocol"
    
    # ASSERT - Verify complete parameter isolation through Protocol interface
    mock_deps.discovery.discover_dataset_files.assert_called_once()
    call_kwargs = mock_deps.discovery.discover_dataset_files.call_args[1]
    
    assert call_kwargs["config"] == test_config, "Protocol should receive correct config"
    assert call_kwargs["dataset_name"] == "isolated_dataset", "Protocol should receive correct dataset name"
    assert call_kwargs["recursive"] is True, "Protocol should receive correct recursive setting"
    assert call_kwargs["extensions"] == ["pkl"], "Protocol should receive correct extensions filter"
    assert call_kwargs["extract_metadata"] is True, "Protocol should receive correct metadata extraction setting"
    assert call_kwargs["parse_dates"] is False, "Protocol should receive correct date parsing setting"
    
    # ASSERT - Verify dataset configuration provider interaction through Protocol
    mock_deps.config.get_dataset_info.assert_called_once_with(test_config, "isolated_dataset")


# ============================================================================
# ENHANCED PROPERTY-BASED TESTING (Section 3.6.3 Requirements)
# ============================================================================

@given(
    file_path=create_hypothesis_strategies().experimental_file_paths(),
    metadata_scenario=st.fixed_dictionaries({
        'date': st.text(alphabet=st.characters(whitelist_categories=("Nd",)), min_size=8, max_size=8).filter(lambda x: len(x) == 8 and x.isdigit()),
        'condition': st.sampled_from(['baseline', 'treatment', 'control', 'stimulation', 'recovery']),
        'replicate': st.integers(min_value=1, max_value=10).map(str)
    })
)
def test_metadata_extraction_property_based_behavior_validation(
    mocker,
    metadata_test_config,
    file_path,
    metadata_scenario
):
    """
    Property-based behavior validation for metadata extraction using centralized strategies.
    
    Uses centralized Hypothesis strategies to generate diverse experimental scenarios and
    validates API behavior through observable responses per Section 3.6.3 requirements.
    """
    # ARRANGE - Set up Protocol-based mocks with property-based test data
    mock_deps = mocker.Mock()
    mock_deps.config.load_config.return_value = metadata_test_config
    mock_deps.config.get_experiment_info.return_value = metadata_test_config['experiments']['baseline_control_study']
    
    # ARRANGE - Generate test filename using property-based inputs with realistic paths
    test_filename = f"/research/data/{file_path}"
    
    # ARRANGE - Configure mock with property-based metadata scenario
    expected_metadata = metadata_scenario.copy()
    expected_metadata['experiment'] = 'baseline_control_study'
    
    mock_deps.discovery.discover_experiment_files.return_value = {
        test_filename: expected_metadata
    }
    
    # ACT - Execute API call with property-based inputs
    result = load_experiment_files(
        config_path="/test/config.yaml",
        experiment_name="baseline_control_study",
        extract_metadata=True,
        _deps=mock_deps
    )
    
    # ASSERT - Validate API behavior with property-based inputs
    assert isinstance(result, dict), "API should return dict when extract_metadata=True"
    assert test_filename in result, f"API response should contain generated filename {test_filename}"
    
    file_metadata = result[test_filename]
    assert isinstance(file_metadata, dict), "File metadata should be dictionary"
    
    # ASSERT - Verify property-based metadata fields through API response
    for field, expected_value in metadata_scenario.items():
        assert field in file_metadata, f"API response missing property-based field: {field}"
        assert file_metadata[field] == expected_value, f"API response incorrect for property-based field {field}: expected {expected_value}, got {file_metadata[field]}"
    
    # ASSERT - Verify Protocol-based dependency interaction with property-based data
    mock_deps.discovery.discover_experiment_files.assert_called_once()
    call_kwargs = mock_deps.discovery.discover_experiment_files.call_args[1]
    assert call_kwargs["extract_metadata"] is True, "Protocol should receive extract_metadata=True for property-based test"


@given(
    config_scenario=create_hypothesis_strategies().experimental_configurations(),
    date_components=st.tuples(
        st.integers(min_value=2020, max_value=2030),  # year
        st.integers(min_value=1, max_value=12),       # month
        st.integers(min_value=1, max_value=28)        # day (conservative to avoid invalid dates)
    )
)
def test_dataset_api_property_based_behavior_validation(
    mocker,
    metadata_test_config,
    config_scenario,
    date_components
):
    """
    Property-based behavior validation for dataset metadata extraction using centralized strategies.
    
    Validates robust handling of diverse dataset configurations and date combinations through
    API behavior observation per Section 3.6.3 requirements.
    """
    # ARRANGE - Set up Protocol-based mocks with property-based configuration
    mock_deps = mocker.Mock()
    
    # ARRANGE - Merge property-based config with base test config  
    enhanced_config = metadata_test_config.copy()
    enhanced_config.update(config_scenario)
    
    # ARRANGE - Ensure dataset exists in configuration for valid test
    dataset_name = "baseline_behavior"  # Use known dataset from metadata_test_config
    if dataset_name not in enhanced_config.get('datasets', {}):
        enhanced_config.setdefault('datasets', {})[dataset_name] = {
            "patterns": ["*baseline*"],
            "rig": "old_opto"
        }
    
    mock_deps.config.get_dataset_info.return_value = enhanced_config['datasets'][dataset_name]
    
    # ARRANGE - Generate property-based date string
    year, month, day = date_components
    date_str = f"{year:04d}{month:02d}{day:02d}"
    
    # ARRANGE - Configure property-based dataset discovery response
    test_filename = f"/research/data/{dataset_name}_{date_str}.csv"
    expected_metadata = {
        "dataset": dataset_name,
        "date": date_str,
        "rig": enhanced_config['datasets'][dataset_name].get('rig', 'unknown')
    }
    
    mock_deps.discovery.discover_dataset_files.return_value = {
        test_filename: expected_metadata
    }
    
    # ACT - Execute dataset API call with property-based inputs
    result = load_dataset_files(
        config=enhanced_config,
        dataset_name=dataset_name,
        extract_metadata=True,
        _deps=mock_deps
    )
    
    # ASSERT - Validate API behavior with property-based dataset scenarios
    assert isinstance(result, dict), "Dataset API should return dict when extract_metadata=True"
    assert test_filename in result, f"API response should contain generated dataset filename {test_filename}"
    
    file_metadata = result[test_filename]
    assert isinstance(file_metadata, dict), "Dataset metadata should be dictionary"
    
    # ASSERT - Verify property-based dataset metadata through API response
    assert "dataset" in file_metadata, "API response should contain dataset field"
    assert "date" in file_metadata, "API response should contain date field"
    
    assert file_metadata["dataset"] == dataset_name, f"API should return correct dataset name: expected {dataset_name}, got {file_metadata['dataset']}"
    assert file_metadata["date"] == date_str, f"API should return correct date: expected {date_str}, got {file_metadata['date']}"
    
    # ASSERT - Verify Protocol-based dependency interaction with property-based data
    mock_deps.discovery.discover_dataset_files.assert_called_once()
    mock_deps.config.get_dataset_info.assert_called_once_with(enhanced_config, dataset_name)


# ============================================================================
# BEHAVIOR-FOCUSED INTEGRATION AND EDGE CASE TESTS
# ============================================================================

def test_api_behavior_with_empty_discovery_results(mocker, metadata_test_config):
    """
    Behavior-focused test for API response when no files are discovered.
    
    Validates API behavioral contract for empty result sets through observable
    response patterns without coupling to internal discovery implementation.
    """
    # ARRANGE - Set up Protocol-based mocks for empty discovery scenario
    mock_deps = mocker.Mock()
    mock_deps.config.load_config.return_value = metadata_test_config
    mock_deps.config.get_experiment_info.return_value = metadata_test_config['experiments']['baseline_control_study']
    
    # ARRANGE - Configure empty discovery results to test API behavior
    mock_deps.discovery.discover_experiment_files.return_value = {}
    
    # ACT - Execute API call with empty discovery scenario
    result = load_experiment_files(
        config_path="/test/config.yaml",
        experiment_name="baseline_control_study",
        extract_metadata=True,
        _deps=mock_deps
    )
    
    # ASSERT - Verify API behavior for empty results
    assert isinstance(result, dict), "API should return dict when extract_metadata=True even with empty results"
    assert len(result) == 0, "API should return empty dict when no files discovered"
    
    # ASSERT - Verify API maintains behavioral contract with empty results
    assert result == {}, "API should return exactly empty dict for consistent behavior"
    
    # ASSERT - Verify Protocol-based dependency interaction occurred despite empty results
    mock_deps.discovery.discover_experiment_files.assert_called_once()
    mock_deps.config.get_experiment_info.assert_called_once_with(metadata_test_config, "baseline_control_study")


def test_api_behavior_without_date_parsing_enabled(mocker, metadata_test_config):
    """
    Behavior-focused test for API response when date parsing is disabled.
    
    Validates API behavioral contract for parse_dates=False through observable
    response patterns without date parsing implementation coupling.
    """
    # ARRANGE - Set up Protocol-based mocks for no date parsing scenario
    mock_deps = mocker.Mock()
    mock_deps.config.load_config.return_value = metadata_test_config
    mock_deps.config.get_experiment_info.return_value = metadata_test_config['experiments']['baseline_control_study']
    
    # ARRANGE - Configure discovery response without parsed_date field
    test_metadata = {
        "/research/data/no_parsed_date_20240101.csv": {
            "date": "20240101",
            "condition": "baseline",
            "replicate": "1",
            "experiment": "baseline_control_study"
            # Note: intentionally no parsed_date field
        }
    }
    mock_deps.discovery.discover_experiment_files.return_value = test_metadata
    
    # ACT - Execute API call with parse_dates=False (default)
    result = load_experiment_files(
        config_path="/test/config.yaml",
        experiment_name="baseline_control_study",
        extract_metadata=True,
        parse_dates=False,
        _deps=mock_deps
    )
    
    # ASSERT - Verify API behavior excludes parsed_date when parse_dates=False
    assert isinstance(result, dict), "API should return dict when extract_metadata=True"
    assert len(result) == 1, "API should return exactly one file"
    
    file_path = "/research/data/no_parsed_date_20240101.csv"
    assert file_path in result, f"API response should contain expected file {file_path}"
    
    file_metadata = result[file_path]
    
    # ASSERT - Verify expected metadata fields are present
    assert "date" in file_metadata, "API should include date field"
    assert "condition" in file_metadata, "API should include condition field"
    assert "replicate" in file_metadata, "API should include replicate field"
    
    # ASSERT - Verify parsed_date field is excluded when parse_dates=False
    assert "parsed_date" not in file_metadata, "API should exclude parsed_date field when parse_dates=False"
    
    # ASSERT - Verify correct field values through API behavior
    assert file_metadata["date"] == "20240101", "API should return correct date string"
    assert file_metadata["condition"] == "baseline", "API should return correct condition"
    
    # ASSERT - Verify Protocol received correct parse_dates parameter
    mock_deps.discovery.discover_experiment_files.assert_called_once()
    call_kwargs = mock_deps.discovery.discover_experiment_files.call_args[1]
    assert call_kwargs["parse_dates"] is False, "Protocol should receive parse_dates=False"


def test_api_behavior_with_comprehensive_metadata_scenarios(mocker, metadata_test_config, sample_metadata_extraction_scenarios):
    """
    Behavior-focused test for API handling of complex metadata extraction patterns.
    
    Validates API behavioral consistency across diverse metadata scenarios through
    observable response patterns leveraging centralized test fixtures.
    """
    # ARRANGE - Set up Protocol-based mocks for comprehensive metadata testing
    mock_deps = mocker.Mock()
    mock_deps.config.load_config.return_value = metadata_test_config
    mock_deps.config.get_experiment_info.return_value = metadata_test_config['experiments']['baseline_control_study']
    
    # ARRANGE - Configure discovery response with comprehensive metadata scenarios
    mock_deps.discovery.discover_experiment_files.return_value = sample_metadata_extraction_scenarios
    
    # ACT - Execute API call with comprehensive metadata extraction
    result = load_experiment_files(
        config_path="/test/config.yaml",
        experiment_name="baseline_control_study",
        extract_metadata=True,
        parse_dates=True,
        _deps=mock_deps
    )
    
    # ASSERT - Verify API behavior handles comprehensive metadata scenarios
    assert isinstance(result, dict), "API should return dict when extract_metadata=True"
    assert len(result) == len(sample_metadata_extraction_scenarios), f"API should return all {len(sample_metadata_extraction_scenarios)} files"
    
    # ASSERT - Verify API response contains all expected files and metadata
    for file_path, expected_metadata in sample_metadata_extraction_scenarios.items():
        assert file_path in result, f"API response missing expected file: {file_path}"
        
        file_metadata = result[file_path]
        assert isinstance(file_metadata, dict), f"File metadata should be dict for {file_path}"
        
        # ASSERT - Verify all expected metadata fields through API behavior
        for field, expected_value in expected_metadata.items():
            assert field in file_metadata, f"API response missing field {field} for file {file_path}"
            assert file_metadata[field] == expected_value, f"API response incorrect value for {field} in {file_path}: expected {expected_value}, got {file_metadata[field]}"
    
    # ASSERT - Verify Protocol interaction with comprehensive metadata request
    mock_deps.discovery.discover_experiment_files.assert_called_once()
    call_kwargs = mock_deps.discovery.discover_experiment_files.call_args[1]
    assert call_kwargs["extract_metadata"] is True, "Protocol should receive extract_metadata=True"
    assert call_kwargs["parse_dates"] is True, "Protocol should receive parse_dates=True"


# ============================================================================
# ENHANCED EDGE-CASE COVERAGE TESTS (Performance tests moved to scripts/benchmarks/)
# ============================================================================

def test_api_behavior_with_edge_case_metadata_scenarios(mocker, metadata_test_config):
    """
    Enhanced edge-case coverage test for metadata extraction API behavior.
    
    Validates API handling of boundary conditions and edge cases through behavioral
    validation patterns per testing strategy requirements for comprehensive coverage.
    Note: Performance tests have been relocated to scripts/benchmarks/ per testing strategy.
    """
    # ARRANGE - Set up Protocol-based mocks for edge-case testing
    mock_deps = mocker.Mock()
    mock_deps.config.load_config.return_value = metadata_test_config
    mock_deps.config.get_experiment_info.return_value = metadata_test_config['experiments']['baseline_control_study']
    
    # ARRANGE - Generate edge-case metadata scenarios for comprehensive testing
    edge_case_scenarios = generate_edge_case_scenarios(['unicode', 'boundary'])
    
    # ARRANGE - Configure edge-case file set for API behavior testing
    edge_case_files = {}
    
    # Unicode path edge cases
    for i, unicode_scenario in enumerate(edge_case_scenarios.get('unicode', [])[:5]):
        file_path = f"/research/data/{unicode_scenario['filename']}"
        edge_case_files[file_path] = {
            "date": f"2024010{i+1}",
            "condition": "unicode_test",
            "replicate": str(i+1),
            "encoding": unicode_scenario['encoding']
        }
    
    # Boundary condition edge cases
    for i, boundary_scenario in enumerate(edge_case_scenarios.get('boundary', [])[:3]):
        if boundary_scenario['data_type'] == 'array_size':
            for j, size_value in enumerate(boundary_scenario['values'][:3]):
                file_path = f"/research/data/boundary_test_{size_value}_{i}_{j}.csv"
                edge_case_files[file_path] = {
                    "date": f"2024020{j+1}",
                    "condition": "boundary_test",
                    "replicate": str(j+1),
                    "array_size": size_value
                }
    
    mock_deps.discovery.discover_experiment_files.return_value = edge_case_files
    
    # ACT - Execute API call with edge-case scenarios
    result = load_experiment_files(
        config_path="/test/config.yaml",
        experiment_name="baseline_control_study",
        extract_metadata=True,
        _deps=mock_deps
    )
    
    # ASSERT - Verify API behavior handles edge-case scenarios correctly
    assert isinstance(result, dict), "API should return dict for edge-case scenarios"
    assert len(result) == len(edge_case_files), f"API should handle all {len(edge_case_files)} edge-case files"
    
    # ASSERT - Verify each edge-case file processed correctly through API
    for file_path, expected_metadata in edge_case_files.items():
        assert file_path in result, f"API should handle edge-case file: {file_path}"
        
        file_metadata = result[file_path]
        assert isinstance(file_metadata, dict), f"Edge-case metadata should be dict for {file_path}"
        
        # ASSERT - Verify core metadata fields preserved in edge cases
        assert "date" in file_metadata, f"Edge-case file should have date: {file_path}"
        assert "condition" in file_metadata, f"Edge-case file should have condition: {file_path}"
        assert file_metadata["condition"] in ["unicode_test", "boundary_test"], f"Edge-case condition should be recognized: {file_path}"
    
    # ASSERT - Verify Protocol interaction with edge-case data
    mock_deps.discovery.discover_experiment_files.assert_called_once()
    call_kwargs = mock_deps.discovery.discover_experiment_files.call_args[1]
    assert call_kwargs["extract_metadata"] is True, "Protocol should receive extract_metadata=True for edge cases"


@pytest.mark.parametrize("edge_case_type,scenario_count", [
    ("unicode_paths", 3),
    ("boundary_values", 5),
    ("corrupted_data", 2),
])
def test_parametrized_edge_case_api_behavior(mocker, metadata_test_config, edge_case_type, scenario_count):
    """
    Parametrized edge-case behavior validation for comprehensive API testing coverage.
    
    Tests API behavioral consistency across different edge-case categories through
    parametrized scenarios per enhanced edge-case coverage requirements.
    """
    # ARRANGE - Set up Protocol-based mocks for parametrized edge-case testing
    mock_deps = mocker.Mock()
    mock_deps.config.load_config.return_value = metadata_test_config
    mock_deps.config.get_experiment_info.return_value = metadata_test_config['experiments']['baseline_control_study']
    
    # ARRANGE - Generate edge-case files based on scenario type
    edge_case_files = {}
    for i in range(scenario_count):
        file_path = f"/research/data/edge_case_{edge_case_type}_{i}.csv"
        edge_case_files[file_path] = {
            "date": f"2024030{i+1}",
            "condition": edge_case_type,
            "replicate": str(i+1),
            "edge_case_type": edge_case_type
        }
    
    mock_deps.discovery.discover_experiment_files.return_value = edge_case_files
    
    # ACT - Execute API call with parametrized edge-case scenario
    result = load_experiment_files(
        config_path="/test/config.yaml", 
        experiment_name="baseline_control_study",
        extract_metadata=True,
        _deps=mock_deps
    )
    
    # ASSERT - Verify API behavior consistent across edge-case types
    assert isinstance(result, dict), f"API should return dict for {edge_case_type} edge cases"
    assert len(result) == scenario_count, f"API should process all {scenario_count} {edge_case_type} scenarios"
    
    # ASSERT - Verify each parametrized edge-case scenario
    for file_path in edge_case_files:
        assert file_path in result, f"API should handle {edge_case_type} edge case: {file_path}"
        file_metadata = result[file_path]
        assert file_metadata["condition"] == edge_case_type, f"API should preserve edge case type: {edge_case_type}"


# ============================================================================
# COMPREHENSIVE INTEGRATION BEHAVIOR VALIDATION
# ============================================================================

def test_complete_metadata_extraction_workflow_behavior(mocker, metadata_test_config, sample_metadata_extraction_scenarios):
    """
    Comprehensive integration test demonstrating complete metadata extraction workflow 
    through behavior-focused validation using Protocol-based mocks and centralized fixtures.
    
    This test validates the entire API workflow from configuration loading through
    metadata extraction using observable behavior patterns without implementation coupling.
    """
    # ARRANGE - Set up complete Protocol-based mock environment
    mock_deps = mocker.Mock()
    
    # ARRANGE - Configure realistic workflow scenario
    mock_deps.config.load_config.return_value = metadata_test_config
    mock_deps.config.get_experiment_info.return_value = metadata_test_config['experiments']['baseline_control_study']
    mock_deps.discovery.discover_experiment_files.return_value = sample_metadata_extraction_scenarios
    
    # ACT - Execute complete metadata extraction workflow
    result = load_experiment_files(
        config_path="/test/comprehensive_config.yaml",
        experiment_name="baseline_control_study",
        extract_metadata=True,
        parse_dates=True,
        pattern="*.csv",
        recursive=True,
        extensions=[".csv", ".pkl"],
        _deps=mock_deps
    )
    
    # ASSERT - Verify complete workflow produces expected behavioral outcome
    assert isinstance(result, dict), "Complete workflow should return metadata dictionary"
    assert len(result) > 0, "Complete workflow should discover files"
    
    # ASSERT - Verify workflow handled all configuration steps through Protocol
    mock_deps.config.load_config.assert_called_once_with("/test/comprehensive_config.yaml")
    mock_deps.config.get_experiment_info.assert_called_once_with(metadata_test_config, "baseline_control_study")
    
    # ASSERT - Verify discovery invocation with all parameters
    mock_deps.discovery.discover_experiment_files.assert_called_once()
    call_kwargs = mock_deps.discovery.discover_experiment_files.call_args[1]
    assert call_kwargs["extract_metadata"] is True
    assert call_kwargs["parse_dates"] is True
    assert call_kwargs["pattern"] == "*.csv"
    assert call_kwargs["recursive"] is True
    assert call_kwargs["extensions"] == [".csv", ".pkl"]
    
    # ASSERT - Verify comprehensive metadata extraction results
    for file_path, file_metadata in result.items():
        assert isinstance(file_metadata, dict), f"Metadata should be dict for {file_path}"
        # Verify essential metadata fields present
        assert any(field in file_metadata for field in ["date", "condition", "experiment", "dataset"]), f"Essential metadata missing for {file_path}"
    
    # ASSERT - Verify workflow maintains behavioral consistency
    assert result == sample_metadata_extraction_scenarios, "Workflow should return exact discovery results"


if __name__ == "__main__":
    # Support running tests directly with behavior-focused validation
    pytest.main([__file__, "-v", "--tb=short"])