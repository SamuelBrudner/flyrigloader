"""
Behavior-focused test suite for config-aware file discovery functionality.

This module implements modern pytest practices with parametrization, property-based testing,
comprehensive behavioral validation, and centralized fixture utilization following the
refactoring requirements for behavior-focused testing per Section 0.

Converted from implementation-coupled testing to black-box behavioral validation focusing on:
- Observable discovery results and API behavior rather than internal implementation details
- Protocol-based mock implementations for consistent dependency isolation
- Centralized fixture utilization from tests/conftest.py and tests/utils.py
- AAA pattern structure with clear separation of test phases
- Enhanced edge-case coverage through parameterized boundary condition testing
- Public API behavioral contracts instead of whitebox assertions

Test Categories:
- CONFIG-DISC-001: Configuration-driven file discovery behavioral validation
- CONFIG-DISC-002: Experiment-specific discovery result verification
- CONFIG-DISC-003: Dataset-specific discovery behavior testing
- CONFIG-DISC-004: Edge-case and error scenario behavioral validation
"""
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from unittest.mock import MagicMock

import pytest
import yaml
from hypothesis import given, strategies as st, assume, settings

# Import the functionality under test - public API only
from flyrigloader.config.discovery import (
    discover_files_with_config,
    discover_experiment_files,
    discover_dataset_files
)
from flyrigloader.config.yaml_config import (
    get_ignore_patterns,
    get_mandatory_substrings,
    get_dataset_info,
    get_experiment_info,
    get_extraction_patterns
)

# Import centralized testing utilities per refactoring requirements
from tests.utils import (
    create_mock_filesystem,
    create_mock_config_provider,
    create_mock_dataloader,
    generate_edge_case_scenarios,
    create_hypothesis_strategies,
    create_integration_test_environment
)


# ============================================================================
# HYPOTHESIS STRATEGIES FOR BEHAVIORAL PROPERTY-BASED TESTING
# ============================================================================

@pytest.fixture(scope="function")
def discovery_strategies():
    """
    Centralized hypothesis strategies for config-aware discovery testing.
    Focuses on input-output behavioral validation rather than internal state.
    """
    strategies = create_hypothesis_strategies()
    
    # Additional discovery-specific strategies
    class DiscoveryStrategies:
        @staticmethod
        def valid_discovery_patterns():
            """Generate valid file patterns for discovery testing."""
            return st.sampled_from([
                "**/*.csv",
                "**/*.pkl", 
                "**/*.pickle",
                "**/experiment_*.csv",
                "**/data_*.pkl",
                "**/*_processed.*"
            ])
        
        @staticmethod 
        def invalid_discovery_patterns():
            """Generate invalid patterns for error boundary testing."""
            return st.sampled_from([
                "",           # Empty pattern
                None,         # None pattern
                "//invalid",  # Invalid glob
                "\\invalid",  # Windows path issues
            ])
        
        @staticmethod
        def discovery_configurations():
            """Generate realistic discovery configurations."""
            return st.fixed_dictionaries({
                "project": st.fixed_dictionaries({
                    "directories": st.fixed_dictionaries({
                        "major_data_directory": st.just("/test/data")
                    }),
                    "ignore_substrings": st.lists(
                        st.sampled_from(["._", "temp_", "backup", ".DS_Store"]),
                        min_size=0, max_size=4
                    ),
                    "extraction_patterns": st.lists(
                        st.just(r".*_(?P<date>\d{4}-\d{2}-\d{2})_(?P<condition>\w+)\.csv"),
                        min_size=0, max_size=2
                    )
                }),
                "datasets": st.dictionaries(
                    keys=st.text(min_size=3, max_size=20),
                    values=st.fixed_dictionaries({
                        "rig": st.just("test_rig"),
                        "dates_vials": st.dictionaries(
                            keys=st.sampled_from(["2023-01-01", "2023-01-02", "2023_01_03"]),
                            values=st.lists(st.integers(min_value=1, max_value=6), min_size=1, max_size=3)
                        )
                    }),
                    min_size=1, max_size=3
                ),
                "experiments": st.dictionaries(
                    keys=st.text(min_size=3, max_size=20),
                    values=st.fixed_dictionaries({
                        "datasets": st.lists(st.text(min_size=3, max_size=20), min_size=1, max_size=2),
                        "filters": st.just({
                            "ignore_substrings": ["exclude_me"],
                            "mandatory_experiment_strings": ["include_me"]
                        })
                    }),
                    min_size=1, max_size=2
                )
            })
    
    # Combine with base strategies
    strategies.discovery = DiscoveryStrategies()
    return strategies


# ============================================================================
# BEHAVIORAL VALIDATION TEST CASES
# ============================================================================

@pytest.mark.parametrize("config_scenario,expected_behavior", [
    # Test project-level filtering behavior
    (
        {"experiment": None, "additional_ignore": []},
        {"should_include_files": True, "min_files_found": 1, "filters_applied": True}
    ),
    
    # Test experiment-specific filtering behavior
    (
        {"experiment": "test_experiment", "additional_ignore": ["exclude_me"]},
        {"should_include_files": True, "min_files_found": 1, "filters_applied": True}
    ),
    
    # Test multi-dataset experiment behavior
    (
        {"experiment": "multi_dataset_experiment", "additional_ignore": ["noise", "artifact"]},
        {"should_include_files": True, "min_files_found": 1, "filters_applied": True}
    ),
    
    # Test basic experiment behavior (no additional filters)
    (
        {"experiment": "basic_experiment", "additional_ignore": []},
        {"should_include_files": True, "min_files_found": 1, "filters_applied": True}
    ),
])
def test_discover_files_with_config_behavioral_validation(
    temp_experiment_directory, 
    config_scenario,
    expected_behavior
):
    """
    Behavioral validation for discover_files_with_config function.
    
    Tests observable discovery behavior and results rather than implementation details.
    Validates filtering effectiveness through result analysis instead of internal inspection.
    
    Args:
        temp_experiment_directory: Centralized experiment directory fixture
        config_scenario: Configuration scenario to test
        expected_behavior: Expected behavioral outcomes
    """
    # ARRANGE - Set up test configuration and behavioral expectations
    base_config = {
        "project": {
            "directories": {
                "major_data_directory": str(temp_experiment_directory["directory"])
            },
            "ignore_substrings": [
                "backup",
                "temp",
                "._"
            ] + config_scenario["additional_ignore"],
            "extraction_patterns": [
                r".*_(?P<date>\d{4}-\d{2}-\d{2})_(?P<condition>\w+)\.csv"
            ]
        },
        "experiments": {
            "test_experiment": {
                "datasets": ["neural_data"],
                "filters": {
                    "ignore_substrings": ["exclude_me"],
                    "mandatory_experiment_strings": ["mouse"]
                }
            },
            "basic_experiment": {
                "datasets": ["neural_data"]
            },
            "multi_dataset_experiment": {
                "datasets": ["neural_data"],
                "filters": {
                    "ignore_substrings": ["noise", "artifact"],
                    "mandatory_experiment_strings": ["valid", "processed"]
                }
            }
        },
        "datasets": {
            "neural_data": {
                "dates_vials": {
                    "20240101": ["mouse_001", "mouse_002"],
                    "20240102": ["mouse_003", "mouse_004"]
                }
            }
        }
    }
    
    experiment = config_scenario["experiment"]
    search_directory = str(temp_experiment_directory["directory"])
    
    # ACT - Execute discovery with configuration-driven filtering
    discovered_files = discover_files_with_config(
        config=base_config,
        directory=search_directory,
        pattern="**/*.pkl",
        recursive=True,
        experiment=experiment
    )
    
    # ASSERT - Verify behavioral outcomes through observable results
    assert isinstance(discovered_files, list), "Discovery should return list of file paths"
    
    if expected_behavior["should_include_files"]:
        assert len(discovered_files) >= expected_behavior["min_files_found"], \
            f"Should find at least {expected_behavior['min_files_found']} files"
    
    # Verify filtering behavior through result analysis
    if expected_behavior["filters_applied"]:
        # Verify ignore patterns are effectively applied
        for file_path in discovered_files:
            for ignore_pattern in base_config["project"]["ignore_substrings"]:
                clean_pattern = ignore_pattern.replace('*', '')
                assert clean_pattern not in file_path, \
                    f"File {file_path} should be filtered out by pattern {ignore_pattern}"
        
        # Verify experiment-specific filtering if applicable
        if experiment and experiment in base_config["experiments"]:
            exp_config = base_config["experiments"][experiment]
            if "filters" in exp_config:
                ignore_substrings = exp_config["filters"].get("ignore_substrings", [])
                mandatory_substrings = exp_config["filters"].get("mandatory_experiment_strings", [])
                
                # Check ignore patterns are applied
                for file_path in discovered_files:
                    for ignore_pattern in ignore_substrings:
                        assert ignore_pattern not in file_path, \
                            f"Experiment filter should exclude files containing {ignore_pattern}"
                
                # Check mandatory patterns (if files found, they should match requirements)
                if mandatory_substrings and discovered_files:
                    has_valid_files = any(
                        any(substring in file_path for substring in mandatory_substrings)
                        for file_path in discovered_files
                    )
                    assert has_valid_files, \
                        f"At least one file should contain mandatory substrings: {mandatory_substrings}"


@pytest.mark.parametrize("date_format,should_discover", [
    ("2023-01-01", True),   # Standard dash format
    ("2023_01_02", True),   # Underscore format
    ("2023/01/03", False),  # Invalid slash format
    ("23-01-01", False),    # Two-digit year
    ("2023-13-01", False),  # Invalid month
    ("not-a-date", False),  # Non-date string
    ("", False),            # Empty string
])
def test_date_based_discovery_behavioral_validation(
    temp_experiment_directory,
    date_format,
    should_discover
):
    """
    Behavioral validation for date-based directory discovery.
    
    Tests observable date format handling through discovery results
    rather than internal date parsing implementation.
    
    Args:
        temp_experiment_directory: Centralized experiment directory fixture
        date_format: Date format to test behavioral validation for
        should_discover: Whether files should be discovered for this format
    """
    # ARRANGE - Set up date-specific test configuration
    base_dir = temp_experiment_directory["directory"]
    date_dir = base_dir / date_format
    date_dir.mkdir(exist_ok=True)
    
    test_file = date_dir / "test_data.pkl"
    test_file.write_text("test data content")
    
    config = {
        "project": {
            "directories": {
                "major_data_directory": str(base_dir)
            },
            "ignore_substrings": []
        },
        "datasets": {
            "test_dataset": {
                "rig": "test_rig",
                "dates_vials": {date_format: [1]}
            }
        }
    }
    
    # ACT - Execute dataset discovery with date format
    try:
        discovered_files = discover_dataset_files(
            config=config,
            dataset_name="test_dataset",
            base_directory=str(base_dir)
        )
        discovery_succeeded = True
        
    except (KeyError, ValueError, OSError) as e:
        discovered_files = []
        discovery_succeeded = False
    
    # ASSERT - Verify behavioral outcomes for date handling
    if should_discover:
        assert discovery_succeeded, \
            f"Valid date format {date_format} should allow successful discovery"
        assert len(discovered_files) >= 1, \
            f"Should discover files for valid date format {date_format}"
        assert any(date_format in file_path for file_path in discovered_files), \
            f"Discovered files should include date directory {date_format}"
    else:
        # Invalid dates should result in no files found or controlled failure
        if discovery_succeeded:
            assert len(discovered_files) == 0, \
                f"Invalid date format {date_format} should not discover files"
        # Note: Some invalid formats may raise exceptions, which is acceptable behavior


@pytest.mark.parametrize("extension_config,pattern,expected_outcome", [
    # Test extension filtering behavior
    (["pkl"], "**/*.*", {"extensions_found": [".pkl"], "other_extensions_excluded": True}),
    (["pkl", "csv"], "**/*.*", {"extensions_found": [".pkl", ".csv"], "other_extensions_excluded": True}),
    (["csv"], "**/*.pkl", {"extensions_found": [".pkl"], "pattern_overrides": True}),
    (None, "**/*.pkl", {"extensions_found": [".pkl"], "pattern_controls": True}),
    ([], "**/*.*", {"extensions_found": [], "empty_list_excludes_all": True}),
])
def test_extension_filtering_behavioral_validation(
    temp_experiment_directory,
    extension_config,
    pattern,
    expected_outcome
):
    """
    Behavioral validation for extension-based filtering.
    
    Tests observable extension filtering behavior through result analysis
    rather than internal filtering mechanism inspection.
    
    Args:
        temp_experiment_directory: Centralized experiment directory fixture
        extension_config: Extension configuration to test
        pattern: File pattern for discovery
        expected_outcome: Expected behavioral outcomes
    """
    # ARRANGE - Set up multi-extension test environment
    test_config = {
        "project": {
            "directories": {
                "major_data_directory": str(temp_experiment_directory["directory"])
            },
            "ignore_substrings": []
        }
    }
    
    # Create files with different extensions in test directory
    test_files = {
        "data.pkl": "pickle data",
        "data.csv": "csv data", 
        "data.json": "json data",
        "data.txt": "text data"
    }
    
    for filename, content in test_files.items():
        file_path = temp_experiment_directory["directory"] / filename
        file_path.write_text(content)
    
    # ACT - Execute discovery with extension filtering
    discovered_files = discover_files_with_config(
        config=test_config,
        directory=str(temp_experiment_directory["directory"]),
        pattern=pattern,
        recursive=True,
        extensions=extension_config
    )
    
    # ASSERT - Verify extension filtering behavioral outcomes
    assert isinstance(discovered_files, list), "Should return list of file paths"
    
    if expected_outcome.get("empty_list_excludes_all", False):
        assert len(discovered_files) == 0, \
            "Empty extension list should exclude all files"
        return
    
    # Analyze discovered file extensions
    discovered_extensions = set()
    for file_path in discovered_files:
        extension = Path(file_path).suffix
        discovered_extensions.add(extension)
    
    expected_extensions = set(expected_outcome["extensions_found"])
    
    # Verify expected extensions are present
    for expected_ext in expected_extensions:
        assert expected_ext in discovered_extensions, \
            f"Expected extension {expected_ext} should be found in results"
    
    # Verify exclusion behavior when specified
    if expected_outcome.get("other_extensions_excluded", False):
        all_possible_extensions = {".pkl", ".csv", ".json", ".txt"}
        excluded_extensions = all_possible_extensions - expected_extensions
        
        for excluded_ext in excluded_extensions:
            assert excluded_ext not in discovered_extensions, \
                f"Extension {excluded_ext} should be excluded from results"


# ============================================================================
# PROPERTY-BASED BEHAVIORAL TESTING WITH HYPOTHESIS
# ============================================================================

@given(
    config=st.fixed_dictionaries({
        "project": st.fixed_dictionaries({
            "ignore_substrings": st.lists(
                st.sampled_from(["backup", "temp", "._", "old", "test"]),
                min_size=1, max_size=3
            )
        })
    }),
    pattern=st.sampled_from(["**/*.csv", "**/*.pkl", "**/*.*"])
)
@settings(max_examples=30, deadline=2000)
def test_pattern_filtering_behavioral_properties(
    temp_experiment_directory,
    config,
    pattern
):
    """
    Property-based behavioral validation for pattern filtering.
    
    Tests that filtering behavior is consistent and observable across
    various input combinations without inspecting internal state.
    
    Args:
        temp_experiment_directory: Centralized experiment directory fixture
        config: Hypothesis-generated configuration
        pattern: Hypothesis-generated file pattern
    """
    # ARRANGE - Set up property-based test environment
    assume(len(config["project"]["ignore_substrings"]) > 0)
    
    # Create test files that should and shouldn't be filtered
    test_dir = temp_experiment_directory["directory"]
    created_files = []
    
    for i, ignore_pattern in enumerate(config["project"]["ignore_substrings"]):
        # Files that should be filtered out
        filtered_file = test_dir / f"file_{ignore_pattern}_{i}.csv"
        filtered_file.write_text("filtered content")
        created_files.append(str(filtered_file))
        
        # Files that should not be filtered
        valid_file = test_dir / f"valid_file_{i}.csv"
        valid_file.write_text("valid content")
        created_files.append(str(valid_file))
    
    config["project"]["directories"] = {"major_data_directory": str(test_dir)}
    
    # ACT - Execute discovery with filtering
    discovered_files = discover_files_with_config(
        config=config,
        directory=str(test_dir),
        pattern=pattern,
        recursive=True
    )
    
    # ASSERT - Verify filtering behavioral properties
    assert isinstance(discovered_files, list), "Should return list of file paths"
    
    # Property: No discovered file should contain ignore patterns
    for file_path in discovered_files:
        for ignore_pattern in config["project"]["ignore_substrings"]:
            clean_pattern = ignore_pattern.replace('*', '')
            assert clean_pattern not in file_path, \
                f"Filtering property violated: {file_path} contains ignored pattern {ignore_pattern}"
    
    # Property: Valid files should be discoverable (if pattern matches)
    pattern_extension = None
    if "*.csv" in pattern:
        pattern_extension = ".csv"
    elif "*.pkl" in pattern:
        pattern_extension = ".pkl"
    
    if pattern_extension:
        expected_valid_files = [
            f for f in created_files 
            if f.endswith(pattern_extension) and 
            not any(ignore in f for ignore in config["project"]["ignore_substrings"])
        ]
        
        if expected_valid_files:
            assert len(discovered_files) > 0, \
                "Should discover at least some valid files matching pattern"


@given(
    dataset_config=st.fixed_dictionaries({
        "dates_vials": st.dictionaries(
            keys=st.sampled_from(["2023-01-01", "2023-01-02", "2023_01_03"]),
            values=st.lists(st.integers(min_value=1, max_value=5), min_size=1, max_size=3),
            min_size=1, max_size=3
        ),
        "rig": st.just("test_rig")
    })
)
@settings(max_examples=20, deadline=3000)
def test_dataset_discovery_behavioral_properties(
    temp_experiment_directory,
    dataset_config
):
    """
    Property-based behavioral validation for dataset file discovery.
    
    Tests that dataset discovery behavior is consistent across various
    date and vial configurations through result analysis.
    
    Args:
        temp_experiment_directory: Centralized experiment directory fixture  
        dataset_config: Hypothesis-generated dataset configuration
    """
    # ARRANGE - Set up property-based dataset test
    base_dir = temp_experiment_directory["directory"]
    created_files = []
    
    # Create files for each date/vial combination
    for date, vials in dataset_config["dates_vials"].items():
        date_dir = base_dir / date
        date_dir.mkdir(exist_ok=True)
        
        for vial in vials:
            filename = f"data_{date}_vial_{vial}.pkl"
            file_path = date_dir / filename
            file_path.write_text(f"date,vial,data\n{date},{vial},test_data")
            created_files.append(str(file_path))
    
    config = {
        "project": {
            "directories": {"major_data_directory": str(base_dir)},
            "ignore_substrings": []
        },
        "datasets": {
            "test_dataset": dataset_config
        }
    }
    
    # ACT - Execute dataset discovery
    discovered_files = discover_dataset_files(
        config=config,
        dataset_name="test_dataset",
        base_directory=str(base_dir)
    )
    
    # ASSERT - Verify dataset discovery behavioral properties
    assert isinstance(discovered_files, list), "Should return list of file paths"
    
    # Property: All specified dates should be represented in results
    specified_dates = list(dataset_config["dates_vials"].keys())
    for date in specified_dates:
        date_represented = any(date in file_path for file_path in discovered_files)
        assert date_represented, \
            f"Dataset discovery should include files from date {date}"
    
    # Property: Discovery should respect directory structure
    for file_path in discovered_files:
        file_exists_in_created = any(created_file in file_path for created_file in created_files)
        assert file_exists_in_created or Path(file_path).exists(), \
            f"Discovered file should correspond to actual file: {file_path}"


# ============================================================================
# ERROR SCENARIO BEHAVIORAL VALIDATION
# ============================================================================

@pytest.mark.parametrize("invalid_config,expected_error_type", [
    # Missing required sections
    ({}, KeyError),
    ({"project": {}}, KeyError),
    ({"experiments": {"test": {}}}, KeyError),
    
    # Invalid data types
    ({"datasets": {"test": {"dates_vials": "not_a_dict"}}}, (ValueError, TypeError)),
    ({"datasets": {"test": {"dates_vials": {123: [1, 2]}}}}, (ValueError, TypeError)),
    ({"datasets": {"test": {"dates_vials": {"2023-01-01": "not_a_list"}}}}, (ValueError, TypeError)),
    
    # Malformed configuration structures
    ({"project": {"ignore_substrings": "should_be_list"}}, (ValueError, TypeError)),
    ({"experiments": {"test": {"filters": {"mandatory_experiment_strings": 123}}}}, (ValueError, TypeError)),
])
def test_invalid_configuration_error_behavior(invalid_config, expected_error_type):
    """
    Behavioral validation for error handling with invalid configurations.
    
    Tests observable error behavior and proper exception handling
    without inspecting internal validation mechanisms.
    
    Args:
        invalid_config: Invalid configuration to test error behavior
        expected_error_type: Expected exception type for behavioral validation
    """
    # ARRANGE - Set up invalid configuration scenario
    test_directory = "/test/directory"
    
    # ACT & ASSERT - Verify error behavioral outcomes
    with pytest.raises(expected_error_type):
        if "datasets" in invalid_config:
            # Test dataset access behavior
            get_dataset_info(invalid_config, "test")
        elif "experiments" in invalid_config:
            # Test experiment access behavior
            get_experiment_info(invalid_config, "test")
        else:
            # Test pattern extraction behavior
            get_ignore_patterns(invalid_config)


@pytest.mark.parametrize("missing_entity,entity_type", [
    ("nonexistent_dataset", "dataset"),
    ("nonexistent_experiment", "experiment"),
])
def test_missing_entity_error_behavior(missing_entity, entity_type):
    """
    Behavioral validation for missing entity error handling.
    
    Tests observable error behavior when accessing non-existent entities
    through public API rather than internal state inspection.
    
    Args:
        missing_entity: Name of missing entity to test
        entity_type: Type of entity for behavioral validation
    """
    # ARRANGE - Set up configuration without target entity
    valid_config = {
        "project": {"ignore_substrings": []},
        "datasets": {"existing_dataset": {"rig": "test_rig"}},
        "experiments": {"existing_experiment": {"datasets": ["existing_dataset"]}}
    }
    
    # ACT & ASSERT - Verify missing entity error behavior
    with pytest.raises(KeyError) as exc_info:
        if entity_type == "dataset":
            get_dataset_info(valid_config, missing_entity)
        else:
            get_experiment_info(valid_config, missing_entity)
    
    # Verify error message contains entity name for debuggability
    assert missing_entity in str(exc_info.value), \
        "Error message should contain missing entity name"


# ============================================================================
# CROSS-PLATFORM BEHAVIORAL VALIDATION
# ============================================================================

@pytest.mark.parametrize("path_scenario,expected_behavior", [
    ("unix_style", {"should_normalize": True, "should_discover": True}),
    ("windows_style", {"should_normalize": True, "should_discover": True}),
])
def test_cross_platform_path_behavioral_validation(
    temp_experiment_directory,
    path_scenario,
    expected_behavior
):
    """
    Behavioral validation for cross-platform path handling.
    
    Tests observable path normalization and discovery behavior
    rather than internal path manipulation implementation.
    
    Args:
        temp_experiment_directory: Centralized experiment directory fixture
        path_scenario: Path scenario to test behavioral validation
        expected_behavior: Expected cross-platform behavioral outcomes
    """
    # ARRANGE - Set up cross-platform test environment
    test_dir = temp_experiment_directory["directory"]
    subdir = test_dir / "subdir"
    subdir.mkdir(exist_ok=True)
    
    test_file = subdir / "test_file.pkl"
    test_file.write_text("cross-platform test data")
    
    config = {
        "project": {
            "directories": {"major_data_directory": str(test_dir)},
            "ignore_substrings": []
        }
    }
    
    # Handle path scenario variations
    if path_scenario == "windows_style" and os.name == 'posix':
        pytest.skip("Windows path test skipped on Unix system")
    
    # ACT - Execute discovery with cross-platform paths
    discovered_files = discover_files_with_config(
        config=config,
        directory=str(test_dir),
        pattern="**/*.pkl",
        recursive=True
    )
    
    # ASSERT - Verify cross-platform behavioral outcomes
    assert isinstance(discovered_files, list), "Should return list of file paths"
    
    if expected_behavior["should_discover"]:
        assert len(discovered_files) >= 1, "Should discover files across platforms"
        assert any("test_file.pkl" in file_path for file_path in discovered_files), \
            "Should find target test file"
    
    if expected_behavior["should_normalize"]:
        # Verify paths are usable and normalized
        for file_path in discovered_files:
            assert os.path.exists(file_path), \
                f"Normalized path should exist: {file_path}"
            # Verify path uses appropriate separators for platform
            assert os.sep in file_path or len(discovered_files) == 0, \
                "Path should use platform-appropriate separators"


# ============================================================================
# INTEGRATION BEHAVIORAL TESTING
# ============================================================================

def test_end_to_end_experiment_discovery_behavioral_workflow(temp_experiment_directory):
    """
    Comprehensive behavioral validation for experiment discovery workflow.
    
    Tests complete experiment discovery behavior from configuration to results
    without inspecting internal processing steps or implementation details.
    
    Args:
        temp_experiment_directory: Centralized experiment directory fixture
    """
    # ARRANGE - Set up comprehensive experiment discovery scenario
    base_dir = temp_experiment_directory["directory"]
    
    # Create realistic file structure
    test_files = [
        "include_me_experiment_data.pkl",
        "include_me_processed_data.pkl", 
        "exclude_me_backup_data.pkl",
        "regular_data_file.pkl",
        "temp_processing_file.pkl"
    ]
    
    for filename in test_files:
        file_path = base_dir / filename
        file_path.write_text(f"test content for {filename}")
    
    config = {
        "project": {
            "directories": {"major_data_directory": str(base_dir)},
            "ignore_substrings": ["backup", "temp"]
        },
        "experiments": {
            "test_experiment": {
                "datasets": ["neural_data"],
                "filters": {
                    "ignore_substrings": ["exclude_me"],
                    "mandatory_experiment_strings": ["include_me"]
                }
            },
            "basic_experiment": {
                "datasets": ["neural_data"]
            }
        },
        "datasets": {
            "neural_data": {
                "dates_vials": {"20240101": ["mouse_001", "mouse_002"]}
            }
        }
    }
    
    # ACT - Execute comprehensive experiment discovery
    experiment_files = discover_experiment_files(
        config=config,
        experiment_name="test_experiment",
        base_directory=str(base_dir)
    )
    
    basic_files = discover_experiment_files(
        config=config,
        experiment_name="basic_experiment", 
        base_directory=str(base_dir)
    )
    
    # ASSERT - Verify end-to-end behavioral outcomes
    assert isinstance(experiment_files, list), "Should return list of experiment files"
    assert isinstance(basic_files, list), "Should return list of basic experiment files"
    
    # Verify experiment-specific filtering behavior
    for file_path in experiment_files:
        assert "include_me" in file_path, \
            "Experiment files should contain mandatory strings"
        assert "exclude_me" not in file_path, \
            "Experiment files should not contain excluded patterns"
        assert "backup" not in file_path, \
            "Should apply project-level filtering"
        assert "temp" not in file_path, \
            "Should apply project-level filtering"
    
    # Verify basic experiment has different filtering behavior
    assert len(basic_files) >= len(experiment_files), \
        "Basic experiment should find more files (fewer filters)"
    
    # Verify configuration hierarchy is properly applied
    project_filtered_files = [f for f in basic_files if "backup" not in f and "temp" not in f]
    assert len(basic_files) == len(project_filtered_files), \
        "Basic experiment should still apply project-level filters"


def test_end_to_end_dataset_discovery_behavioral_workflow(temp_experiment_directory):
    """
    Comprehensive behavioral validation for dataset discovery workflow.
    
    Tests complete dataset discovery behavior focusing on observable results
    and date-based directory handling without implementation inspection.
    
    Args:
        temp_experiment_directory: Centralized experiment directory fixture
    """
    # ARRANGE - Set up dataset discovery scenario
    base_dir = temp_experiment_directory["directory"]
    
    # Create date-based directory structure
    date_dirs = ["2023-01-01", "2023-01-02", "2023_01_03"]
    for date_dir in date_dirs:
        dir_path = base_dir / date_dir
        dir_path.mkdir(exist_ok=True)
        
        # Create files in each date directory
        for i in range(2):
            file_path = dir_path / f"dataset_file_{i}.pkl"
            file_path.write_text(f"dataset content for {date_dir}")
    
    config = {
        "project": {
            "directories": {"major_data_directory": str(base_dir)},
            "ignore_substrings": ["backup", "temp"]
        },
        "datasets": {
            "test_dataset": {
                "rig": "test_rig",
                "dates_vials": {
                    "2023-01-01": [1, 2],
                    "2023-01-02": [3, 4],
                    "2023_01_03": [5, 6]  # Alternative date format
                }
            }
        }
    }
    
    # ACT - Execute dataset discovery
    dataset_files = discover_dataset_files(
        config=config,
        dataset_name="test_dataset",
        base_directory=str(base_dir)
    )
    
    # ASSERT - Verify dataset discovery behavioral outcomes
    assert isinstance(dataset_files, list), "Should return list of dataset files"
    assert len(dataset_files) > 0, "Should discover dataset files"
    
    # Verify date-based discovery behavior
    for expected_date in date_dirs:
        date_files_found = any(expected_date in file_path for file_path in dataset_files)
        assert date_files_found, \
            f"Should find files for date directory {expected_date}"
    
    # Verify project-level filtering is applied
    for file_path in dataset_files:
        assert "backup" not in file_path, \
            "Should apply project-level ignore patterns"
        assert "temp" not in file_path, \
            "Should apply project-level ignore patterns"
    
    # Verify file discovery respects directory structure
    for file_path in dataset_files:
        path_obj = Path(file_path)
        assert any(date in str(path_obj.parent) for date in date_dirs), \
            f"File should be in expected date directory: {file_path}"


# ============================================================================
# METADATA EXTRACTION BEHAVIORAL TESTING
# ============================================================================

@pytest.mark.parametrize("metadata_scenario,expected_result_structure", [
    ({"extract_metadata": False, "parse_dates": False}, {"result_type": list, "has_metadata": False}),
    ({"extract_metadata": True, "parse_dates": False}, {"result_type": dict, "has_metadata": True}),
    ({"extract_metadata": False, "parse_dates": True}, {"result_type": dict, "has_metadata": True}),
    ({"extract_metadata": True, "parse_dates": True}, {"result_type": dict, "has_metadata": True}),
])
def test_metadata_extraction_behavioral_modes(
    temp_experiment_directory,
    metadata_scenario,
    expected_result_structure
):
    """
    Behavioral validation for metadata extraction modes.
    
    Tests observable metadata extraction behavior through result structure analysis
    rather than internal metadata processing implementation inspection.
    
    Args:
        temp_experiment_directory: Centralized experiment directory fixture
        metadata_scenario: Metadata extraction scenario configuration
        expected_result_structure: Expected result structure and behavior
    """
    # ARRANGE - Set up metadata extraction test scenario
    test_dir = temp_experiment_directory["directory"]
    
    # Create files with extractable patterns
    test_files = [
        "experiment_20240101_control_rep1.csv",
        "experiment_20240102_treatment_rep2.csv",
        "baseline_data_20240103.csv"
    ]
    
    for filename in test_files:
        file_path = test_dir / filename
        file_path.write_text("timestamp,signal\n0.0,1.5\n0.1,1.8")
    
    config = {
        "project": {
            "directories": {"major_data_directory": str(test_dir)},
            "ignore_substrings": [],
            "extraction_patterns": [
                r".*_(?P<date>\d{8})_(?P<condition>\w+)_rep(?P<replicate>\d+)\.csv",
                r".*_(?P<date>\d{8})\.csv"
            ]
        }
    }
    
    # ACT - Execute discovery with metadata extraction options
    result = discover_files_with_config(
        config=config,
        directory=str(test_dir),
        pattern="**/*.csv",
        recursive=True,
        extract_metadata=metadata_scenario["extract_metadata"],
        parse_dates=metadata_scenario["parse_dates"]
    )
    
    # ASSERT - Verify metadata extraction behavioral outcomes
    expected_type = expected_result_structure["result_type"]
    assert isinstance(result, expected_type), \
        f"Result should be {expected_type.__name__} for given metadata options"
    
    if expected_result_structure["has_metadata"]:
        assert isinstance(result, dict), "Metadata extraction should return dictionary"
        assert len(result) > 0, "Should have metadata entries"
        
        # Verify structure of metadata results
        for file_path, metadata in result.items():
            assert isinstance(file_path, str), "Keys should be file path strings"
            assert isinstance(metadata, dict), "Values should be metadata dictionaries"
            
            if metadata_scenario["extract_metadata"]:
                # Should have extracted metadata fields
                assert len(metadata) > 0, "Should extract metadata fields"
            
            if metadata_scenario["parse_dates"]:
                # Should have date parsing information
                assert len(metadata) > 0, "Should have date parsing metadata"
    else:
        assert isinstance(result, list), "Non-metadata mode should return file list"
        assert all(isinstance(file_path, str) for file_path in result), \
            "File list should contain string paths"


if __name__ == "__main__":
    # Enable running tests directly with behavioral focus
    pytest.main([__file__, "-v", "--tb=short"])