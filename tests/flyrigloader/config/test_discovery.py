"""
Comprehensive test suite for config-aware file discovery functionality.

This module implements modern pytest practices with parametrization, property-based testing,
comprehensive mocking scenarios, and performance validation against SLA requirements.
Follows TST-MOD-001 through TST-MOD-004 requirements for test modernization.
"""
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from unittest.mock import Mock, patch, MagicMock

import pytest
import yaml
from hypothesis import given, strategies as st, assume, settings
from hypothesis.strategies import composite

# Import the functionality under test
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


# ============================================================================
# HYPOTHESIS STRATEGIES FOR PROPERTY-BASED TESTING
# ============================================================================

@composite
def valid_date_string(draw):
    """Generate valid date strings in various formats."""
    year = draw(st.integers(min_value=2020, max_value=2030))
    month = draw(st.integers(min_value=1, max_value=12))
    day = draw(st.integers(min_value=1, max_value=28))  # Safe day range
    
    # Choose format
    format_choice = draw(st.sampled_from(['dash', 'underscore']))
    if format_choice == 'dash':
        return f"{year:04d}-{month:02d}-{day:02d}"
    else:
        return f"{year:04d}_{month:02d}_{day:02d}"


@composite
def invalid_date_string(draw):
    """Generate invalid date strings for edge case testing."""
    invalid_formats = [
        "not-a-date",
        "2023/01/01",  # Wrong separator
        "23-01-01",    # Two-digit year
        "2023-13-01",  # Invalid month
        "2023-01-32",  # Invalid day
        "2023-1-1",    # Single digit month/day
        "",            # Empty string
    ]
    return draw(st.sampled_from(invalid_formats))


@composite
def file_extension_strategy(draw):
    """Generate various file extensions for testing."""
    extensions = ['.csv', '.pkl', '.pickle', '.gz', '.txt', '.json', '.yaml']
    return draw(st.sampled_from(extensions))


@composite
def ignore_pattern_strategy(draw):
    """Generate various ignore patterns for testing."""
    patterns = [
        "._",           # Hidden files
        "temp_",        # Temporary files
        "backup",       # Backup files
        "*.log",        # Log files with glob
        "*_old.*",      # Old files with glob
        "test_*",       # Test files
        ".DS_Store",    # System files
    ]
    return draw(st.sampled_from(patterns))


@composite
def mandatory_substring_strategy(draw):
    """Generate mandatory substring patterns."""
    substrings = [
        "experiment",
        "data",
        "valid",
        "processed",
        "final",
        "analysis",
    ]
    return draw(st.sampled_from(substrings))


# ============================================================================
# FIXTURE DEFINITIONS
# ============================================================================

@pytest.fixture
def sample_config_base():
    """
    Base configuration fixture with comprehensive structure.
    Follows F-001-RQ-003 validation requirements.
    """
    return {
        "project": {
            "directories": {
                "major_data_directory": "/path/to/data"
            },
            "ignore_substrings": [
                "._",
                "temp_",
                "backup"
            ],
            "extraction_patterns": [
                r".*_(?P<date>\d{4}-\d{2}-\d{2})_(?P<condition>\w+)\.csv",
                r".*_(?P<experiment>\w+)_(?P<replicate>\d+)\.pkl"
            ]
        },
        "datasets": {
            "test_dataset": {
                "rig": "test_rig",
                "dates_vials": {
                    "2023-01-01": [1, 2],
                    "2023-01-02": [3, 4],
                    "2023_01_03": [5, 6]  # Alternative date format
                },
                "metadata": {
                    "extraction_patterns": [
                        r".*_(?P<dataset>\w+)_(?P<date>\d{8})\.csv"
                    ]
                }
            },
            "multi_date_dataset": {
                "rig": "old_opto",
                "dates_vials": {
                    "2023-02-01": [1, 2, 3],
                    "2023-02-15": [4, 5],
                    "2023-03-01": [6]
                }
            }
        },
        "experiments": {
            "test_experiment": {
                "datasets": ["test_dataset"],
                "filters": {
                    "ignore_substrings": ["exclude_me"],
                    "mandatory_experiment_strings": ["include_me"]
                },
                "metadata": {
                    "extraction_patterns": [
                        r".*_(?P<experiment>\w+)_(?P<date>\d{8})\.csv"
                    ]
                }
            },
            "basic_experiment": {
                "datasets": ["test_dataset"]
            },
            "multi_dataset_experiment": {
                "datasets": ["test_dataset", "multi_date_dataset"],
                "filters": {
                    "ignore_substrings": ["noise", "artifact"],
                    "mandatory_experiment_strings": ["valid", "processed"]
                }
            }
        }
    }


@pytest.fixture
def complex_directory_structure(tmp_path):
    """
    Create a complex temporary directory structure for comprehensive testing.
    Tests F-002-RQ-001 recursive traversal and F-002-RQ-005 date-based resolution.
    """
    base_dir = tmp_path / "test_data"
    base_dir.mkdir()
    
    # Create date directories with various formats
    date_dirs = [
        "2023-01-01",
        "2023-01-02",
        "2023_01_03",  # Alternative format
        "2023-02-01",
        "2023-02-15",
        "invalid_date",
    ]
    
    files_created = []
    
    for date_dir in date_dirs:
        date_path = base_dir / date_dir
        date_path.mkdir()
        
        # Create various types of files
        test_files = [
            f"include_me_data_{date_dir}.csv",
            f"include_me_processed_{date_dir}.pkl",
            f"exclude_me_data_{date_dir}.csv",
            f"regular_data_{date_dir}.csv",
            f"._hidden_file_{date_dir}.txt",
            f"temp_backup_{date_dir}.csv",
            f"valid_experiment_{date_dir}.csv",
            f"processed_analysis_{date_dir}.json",
            f"noise_artifact_{date_dir}.csv",
        ]
        
        for filename in test_files:
            file_path = date_path / filename
            file_path.write_text(f"Sample data for {filename}")
            files_created.append(str(file_path))
    
    # Create nested subdirectories for recursive testing
    nested_dir = base_dir / "2023-01-01" / "subdir"
    nested_dir.mkdir()
    nested_file = nested_dir / "nested_include_me_data.csv"
    nested_file.write_text("Nested sample data")
    files_created.append(str(nested_file))
    
    return {
        "base_dir": str(base_dir),
        "files_created": files_created,
        "date_dirs": [str(base_dir / d) for d in date_dirs]
    }


@pytest.fixture
def performance_test_structure(tmp_path):
    """
    Create a large directory structure for performance testing.
    Tests TST-PERF-001 requirements (<5 seconds for 10,000 files).
    """
    base_dir = tmp_path / "perf_test"
    base_dir.mkdir()
    
    files_created = []
    
    # Create 100 date directories with 100 files each (10,000 total)
    for i in range(100):
        date_dir = base_dir / f"2023-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}"
        date_dir.mkdir(exist_ok=True)
        
        for j in range(100):
            filename = f"data_{i:03d}_{j:03d}.csv"
            file_path = date_dir / filename
            file_path.write_text(f"data,{i},{j}\n1,2,3")
            files_created.append(str(file_path))
    
    return {
        "base_dir": str(base_dir),
        "files_created": files_created,
        "total_files": len(files_created)
    }


@pytest.fixture
def mock_filesystem_permissions(monkeypatch):
    """
    Mock filesystem operations for permission scenario testing.
    Tests TST-MOD-003 enhanced filesystem mocking requirements.
    """
    permission_errors = {}
    
    def mock_listdir(path):
        if str(path) in permission_errors:
            raise PermissionError(f"Permission denied: {path}")
        return []
    
    def mock_exists(path):
        return str(path) not in permission_errors
    
    def mock_is_dir(path):
        return str(path) not in permission_errors
    
    monkeypatch.setattr("os.listdir", mock_listdir)
    monkeypatch.setattr("pathlib.Path.exists", mock_exists)
    monkeypatch.setattr("pathlib.Path.is_dir", mock_is_dir)
    
    return permission_errors


# ============================================================================
# PARAMETRIZED TEST CASES
# ============================================================================

@pytest.mark.parametrize("config_modification,expected_ignore_count,expected_mandatory_count", [
    # Test project-level patterns only
    ({}, 3, 0),
    
    # Test experiment with additional filters
    ({"experiment": "test_experiment"}, 4, 1),
    
    # Test experiment with multiple mandatory strings
    ({"experiment": "multi_dataset_experiment"}, 5, 2),
    
    # Test basic experiment (no additional filters)
    ({"experiment": "basic_experiment"}, 3, 0),
])
def test_discover_files_with_config_parametrized(
    sample_config_base, 
    complex_directory_structure, 
    config_modification,
    expected_ignore_count,
    expected_mandatory_count
):
    """
    Parametrized test for discover_files_with_config function.
    Tests TST-MOD-002 pytest.mark.parametrize implementation.
    
    Args:
        sample_config_base: Base configuration fixture
        complex_directory_structure: Complex test directory structure
        config_modification: Modifications to apply to base config
        expected_ignore_count: Expected number of ignore patterns
        expected_mandatory_count: Expected number of mandatory substrings
    """
    # Apply configuration modifications
    config = sample_config_base.copy()
    experiment = config_modification.get("experiment")
    
    # Test the pattern extraction functions
    ignore_patterns = get_ignore_patterns(config, experiment)
    mandatory_substrings = get_mandatory_substrings(config, experiment)
    
    assert len(ignore_patterns) == expected_ignore_count
    assert len(mandatory_substrings) == expected_mandatory_count
    
    # Test file discovery
    files = discover_files_with_config(
        config=config,
        directory=complex_directory_structure["base_dir"],
        pattern="**/*.csv",
        recursive=True,
        experiment=experiment
    )
    
    # Verify basic functionality
    assert isinstance(files, list)
    assert len(files) > 0
    
    # Verify all returned files are CSV files
    assert all(f.endswith('.csv') for f in files)
    
    # Verify ignore patterns are applied
    for pattern in ignore_patterns:
        # Convert pattern to simple substring check
        pattern_clean = pattern.replace('*', '')
        assert not any(pattern_clean in f for f in files)
    
    # Verify mandatory substrings are applied (files must contain at least one of the mandatory substrings)
    if mandatory_substrings:
        # Debug output to understand what's happening
        print("\nDebug: Checking mandatory substrings:")
        print(f"Mandatory substrings: {mandatory_substrings}")
        print("Files after filtering:")
        for f in files:
            matches = [substring for substring in mandatory_substrings if substring in f]
            print(f"- {f} (matches: {matches if matches else 'none'})")
        
        # Check that at least one file matches at least one mandatory substring
        has_matching_file = any(
            any(substring in f for substring in mandatory_substrings)
            for f in files
        )
        assert has_matching_file, "No files matched the mandatory substrings"


@pytest.mark.parametrize("date_format,expected_valid", [
    ("2023-01-01", True),
    ("2023_01_02", True),
    ("2023/01/03", False),
    ("23-01-01", False),
    ("2023-13-01", False),
    ("not-a-date", False),
    ("", False),
])
def test_date_directory_resolution(
    sample_config_base,
    tmp_path,
    date_format,
    expected_valid
):
    """
    Parametrized test for date-based directory resolution.
    Tests F-002-RQ-005 date format validation requirements.
    
    Args:
        sample_config_base: Base configuration fixture
        tmp_path: Pytest temporary path fixture
        date_format: Date format to test
        expected_valid: Whether the date format should be considered valid
    """
    # Create test directory structure
    base_dir = tmp_path / "date_test"
    base_dir.mkdir()
    date_dir = base_dir / date_format
    date_dir.mkdir()
    
    # Create test file
    test_file = date_dir / "test_data.csv"
    test_file.write_text("test,data\n1,2")
    
    # Update config to use our test directory
    config = sample_config_base.copy()
    config["datasets"]["test_dataset"]["dates_vials"] = {date_format: [1]}
    
    # Test dataset file discovery
    try:
        files = discover_dataset_files(
            config=config,
            dataset_name="test_dataset",
            base_directory=str(base_dir)
        )
        
        if expected_valid:
            assert len(files) >= 1
            assert any(date_format in f for f in files)
        else:
            # Invalid dates should result in no files found or empty list
            assert len(files) == 0 or not any(date_format in f for f in files)
            
    except (KeyError, ValueError) as e:
        # Some invalid date formats might raise exceptions
        if expected_valid:
            pytest.fail(f"Valid date format {date_format} raised exception: {e}")


@pytest.mark.parametrize("extension_list,pattern,expected_extensions", [
    (["csv"], "**/*.*", [".csv"]),
    (["pkl", "pickle"], "**/*.*", [".pkl", ".pickle"]),
    (["csv", "json"], "**/*.csv", [".csv"]),  # Pattern overrides extensions
    (None, "**/*.pkl", [".pkl"]),
    ([], "**/*.*", []),  # Empty extension list
])
def test_extension_filtering(
    sample_config_base,
    complex_directory_structure,
    extension_list,
    pattern,
    expected_extensions
):
    """
    Parametrized test for extension-based filtering.
    Tests F-002-RQ-002 extension filtering requirements.
    
    Args:
        sample_config_base: Base configuration fixture
        complex_directory_structure: Test directory structure
        extension_list: List of extensions to filter by
        pattern: File pattern to match
        expected_extensions: Expected file extensions in results
    """
    files = discover_files_with_config(
        config=sample_config_base,
        directory=complex_directory_structure["base_dir"],
        pattern=pattern,
        recursive=True,
        extensions=extension_list
    )
    
    if expected_extensions:
        # Verify all files have expected extensions
        for file_path in files:
            file_ext = Path(file_path).suffix
            assert file_ext in expected_extensions
    else:
        # Empty extension list should return no files
        assert len(files) == 0


# ============================================================================
# PROPERTY-BASED TESTS USING HYPOTHESIS
# ============================================================================

@given(
    ignore_patterns=st.lists(ignore_pattern_strategy(), min_size=1, max_size=5),
    mandatory_substrings=st.lists(mandatory_substring_strategy(), min_size=0, max_size=3)
)
@settings(max_examples=50, deadline=1000)
def test_pattern_matching_robustness(
    tmp_path,
    ignore_patterns,
    mandatory_substrings
):
    """
    Property-based test for pattern matching robustness.
    Tests F-007 Metadata Extraction System requirements with Hypothesis.
    
    Args:
        tmp_path: Pytest temporary path fixture
        ignore_patterns: Hypothesis-generated ignore patterns
        mandatory_substrings: Hypothesis-generated mandatory substrings
    """
    # Create test directory
    test_dir = tmp_path / "pattern_test"
    test_dir.mkdir()
    
    # Create files that should be ignored and included
    test_files = []
    for i, pattern in enumerate(ignore_patterns):
        # Create files containing ignore patterns
        pattern_clean = pattern.replace('*', '').replace('.', '_')
        ignored_file = test_dir / f"file_{pattern_clean}_{i}.csv"
        ignored_file.write_text("ignored,data")
        test_files.append(str(ignored_file))
    
    # Create files with mandatory substrings
    for i, substring in enumerate(mandatory_substrings):
        included_file = test_dir / f"file_{substring}_valid_{i}.csv"
        included_file.write_text("valid,data")
        test_files.append(str(included_file))
    
    # Create a control file that should always be included (if no mandatory substrings)
    if not mandatory_substrings:
        control_file = test_dir / "control_file.csv"
        control_file.write_text("control,data")
        test_files.append(str(control_file))
    
    # Build test configuration
    config = {
        "project": {
            "ignore_substrings": ignore_patterns
        }
    }
    
    # Test with mandatory substrings as experiment filter
    if mandatory_substrings:
        config["experiments"] = {
            "test_exp": {
                "filters": {
                    "mandatory_experiment_strings": mandatory_substrings
                }
            }
        }
        experiment = "test_exp"
    else:
        experiment = None
    
    # Test file discovery
    files = discover_files_with_config(
        config=config,
        directory=str(test_dir),
        pattern="**/*.csv",
        recursive=True,
        experiment=experiment
    )
    
    # Verify ignore patterns work
    for pattern in ignore_patterns:
        pattern_clean = pattern.replace('*', '')
        assert not any(pattern_clean in f for f in files), \
            f"Files containing ignore pattern '{pattern_clean}' were not filtered out"
    
    # Verify mandatory substrings work
    if mandatory_substrings:
        for substring in mandatory_substrings:
            assert all(substring in f for f in files), \
                f"Not all files contain mandatory substring '{substring}'"


@given(
    dates=st.lists(valid_date_string(), min_size=1, max_size=10, unique=True),
    vial_numbers=st.lists(st.integers(min_value=1, max_value=10), min_size=1, max_size=5)
)
@settings(max_examples=30, deadline=2000)
def test_dataset_discovery_property_based(
    tmp_path,
    dates,
    vial_numbers
):
    """
    Property-based test for dataset file discovery.
    Tests comprehensive date and vial number combinations.
    
    Args:
        tmp_path: Pytest temporary path fixture
        dates: Hypothesis-generated list of valid dates
        vial_numbers: Hypothesis-generated list of vial numbers
    """
    assume(len(dates) > 0 and len(vial_numbers) > 0)
    
    # Create test directory structure
    base_dir = tmp_path / "dataset_test"
    base_dir.mkdir()
    
    # Build dates_vials structure
    dates_vials = {}
    files_created = []
    
    for date in dates:
        dates_vials[date] = vial_numbers.copy()
        
        # Create date directory
        date_dir = base_dir / date
        date_dir.mkdir()
        
        # Create files for each vial
        for vial in vial_numbers:
            filename = f"data_{date}_vial_{vial}.csv"
            file_path = date_dir / filename
            file_path.write_text(f"date,vial,data\n{date},{vial},test")
            files_created.append(str(file_path))
    
    # Build test configuration
    config = {
        "project": {
            "directories": {
                "major_data_directory": str(base_dir)
            }
        },
        "datasets": {
            "test_dataset": {
                "rig": "test_rig",
                "dates_vials": dates_vials
            }
        }
    }
    
    # Test dataset discovery
    discovered_files = discover_dataset_files(
        config=config,
        dataset_name="test_dataset",
        base_directory=str(base_dir)
    )
    
    # Verify all expected files are discovered
    assert len(discovered_files) >= len(files_created)
    
    # Verify all date directories are represented
    for date in dates:
        assert any(date in f for f in discovered_files), \
            f"No files found for date {date}"


# ============================================================================
# ERROR SCENARIO TESTING
# ============================================================================

@pytest.mark.parametrize("invalid_config,expected_exception", [
    # Missing required sections
    ({}, KeyError),
    ({"project": {}}, KeyError),
    ({"experiments": {"test": {}}}, KeyError),
    
    # Invalid structure
    ({"datasets": {"test": {"dates_vials": "not_a_dict"}}}, (ValueError, TypeError)),
    ({"datasets": {"test": {"dates_vials": {123: [1, 2]}}}}, (ValueError, TypeError)),
    ({"datasets": {"test": {"dates_vials": {"2023-01-01": "not_a_list"}}}}, (ValueError, TypeError)),
    
    # Malformed patterns
    ({"project": {"ignore_substrings": "should_be_list"}}, (ValueError, TypeError)),
    ({"experiments": {"test": {"filters": {"mandatory_experiment_strings": 123}}}}, (ValueError, TypeError)),
])
def test_invalid_configuration_scenarios(invalid_config, expected_exception):
    """
    Test error handling for invalid configuration structures.
    Tests F-001-RQ-003 validation requirements for error scenarios.
    
    Args:
        invalid_config: Invalid configuration to test
        expected_exception: Expected exception type
    """
    with pytest.raises(expected_exception):
        if "datasets" in invalid_config:
            # Test dataset access
            get_dataset_info(invalid_config, "test")
        elif "experiments" in invalid_config:
            # Test experiment access
            get_experiment_info(invalid_config, "test")
        else:
            # Test pattern extraction
            get_ignore_patterns(invalid_config)


@pytest.mark.parametrize("missing_entity,entity_type", [
    ("nonexistent_dataset", "dataset"),
    ("nonexistent_experiment", "experiment"),
])
def test_missing_entity_errors(sample_config_base, missing_entity, entity_type):
    """
    Test error handling for missing datasets and experiments.
    
    Args:
        sample_config_base: Base configuration fixture
        missing_entity: Name of missing entity
        entity_type: Type of entity (dataset or experiment)
    """
    with pytest.raises(KeyError) as exc_info:
        if entity_type == "dataset":
            get_dataset_info(sample_config_base, missing_entity)
        else:
            get_experiment_info(sample_config_base, missing_entity)
    
    assert missing_entity in str(exc_info.value)


# ============================================================================
# PERFORMANCE TESTING
# ============================================================================

@pytest.mark.benchmark
def test_file_discovery_performance_sla(
    performance_test_structure,
    sample_config_base,
    benchmark
):
    """
    Test file discovery performance against SLA requirements.
    Tests TST-PERF-001 requirement: <5 seconds for 10,000 files.
    
    Args:
        performance_test_structure: Large test directory structure
        sample_config_base: Base configuration fixture
        benchmark: Pytest benchmark fixture
    """
    def discover_large_dataset():
        return discover_files_with_config(
            config=sample_config_base,
            directory=performance_test_structure["base_dir"],
            pattern="**/*.csv",
            recursive=True
        )
    
    # Benchmark the operation
    result = benchmark(discover_large_dataset)
    
    # Verify SLA compliance
    assert benchmark.stats['mean'] < 5.0, \
        f"File discovery took {benchmark.stats['mean']:.2f}s, exceeding 5s SLA"
    
    # Verify correctness
    assert len(result) == performance_test_structure["total_files"]


@pytest.mark.benchmark
def test_experiment_discovery_performance(
    performance_test_structure,
    sample_config_base,
    benchmark
):
    """
    Test experiment file discovery performance.
    
    Args:
        performance_test_structure: Large test directory structure
        sample_config_base: Base configuration fixture
        benchmark: Pytest benchmark fixture
    """
    # Update config for performance test
    config = sample_config_base.copy()
    config["project"]["directories"]["major_data_directory"] = performance_test_structure["base_dir"]
    
    def discover_experiment_files_perf():
        return discover_experiment_files(
            config=config,
            experiment_name="test_experiment",
            base_directory=performance_test_structure["base_dir"]
        )
    
    # Benchmark the operation
    result = benchmark(discover_experiment_files_perf)
    
    # Verify performance is reasonable
    assert benchmark.stats['mean'] < 10.0, \
        f"Experiment discovery took {benchmark.stats['mean']:.2f}s, exceeding reasonable threshold"
    
    # Verify some files were found
    assert len(result) > 0


# ============================================================================
# CROSS-PLATFORM COMPATIBILITY TESTING
# ============================================================================

@pytest.mark.parametrize("path_separator,expected_normalization", [
    ("/", True),   # Unix-style paths
    ("\\", True),  # Windows-style paths
])
def test_cross_platform_path_handling(
    tmp_path,
    sample_config_base,
    path_separator,
    expected_normalization
):
    """
    Test cross-platform path handling compatibility.
    Tests TST-MOD-003 cross-platform compatibility requirements.
    
    Args:
        tmp_path: Pytest temporary path fixture
        sample_config_base: Base configuration fixture
        path_separator: Path separator to test
        expected_normalization: Whether paths should be normalized
    """
    # Create test structure
    test_dir = tmp_path / "cross_platform_test"
    test_dir.mkdir()
    
    subdir = test_dir / "subdir"
    subdir.mkdir()
    
    test_file = subdir / "test_file.csv"
    test_file.write_text("test,data\n1,2")
    
    # Test with different path formats
    if path_separator == "\\":
        # Simulate Windows-style path (if on Unix system)
        import os
        if os.name == 'posix':
            # Skip Windows path test on Unix systems
            pytest.skip("Windows path test skipped on Unix system")
    
    # Test file discovery with cross-platform paths
    files = discover_files_with_config(
        config=sample_config_base,
        directory=str(test_dir),
        pattern="**/*.csv",
        recursive=True
    )
    
    # Verify files are found and paths are normalized
    assert len(files) >= 1
    assert any("test_file.csv" in f for f in files)
    
    # Verify path normalization
    for file_path in files:
        # Ensure paths use the OS-appropriate separator
        assert os.path.exists(file_path), f"Normalized path {file_path} should exist"


# ============================================================================
# INTEGRATION TESTS WITH MOCKING
# ============================================================================

def test_filesystem_permission_handling(
    mock_filesystem_permissions,
    sample_config_base,
    tmp_path
):
    """
    Test handling of filesystem permission errors.
    Tests TST-MOD-003 enhanced filesystem mocking requirements.
    
    Args:
        mock_filesystem_permissions: Mock filesystem permissions fixture
        sample_config_base: Base configuration fixture
        tmp_path: Pytest temporary path fixture
    """
    # Create test directory
    test_dir = tmp_path / "permission_test"
    test_dir.mkdir()
    
    # Simulate permission error on specific directory
    restricted_dir = str(test_dir / "restricted")
    mock_filesystem_permissions[restricted_dir] = True
    
    # Test file discovery with permission errors
    # Should handle gracefully and continue with accessible directories
    files = discover_files_with_config(
        config=sample_config_base,
        directory=str(test_dir),
        pattern="**/*.csv",
        recursive=True
    )
    
    # Should not crash and return empty list or accessible files only
    assert isinstance(files, list)


@patch('flyrigloader.discovery.files.discover_files')
def test_discover_files_with_config_mocking(
    mock_discover_files,
    sample_config_base
):
    """
    Test discover_files_with_config with comprehensive mocking.
    Tests TST-MOD-003 standardized mocking strategies.
    
    Args:
        mock_discover_files: Mock discover_files function
        sample_config_base: Base configuration fixture
    """
    # Configure mock return value
    expected_files = [
        "/path/to/file1.csv",
        "/path/to/file2.csv"
    ]
    mock_discover_files.return_value = expected_files
    
    # Test the function
    result = discover_files_with_config(
        config=sample_config_base,
        directory="/test/directory",
        pattern="**/*.csv",
        recursive=True,
        experiment="test_experiment"
    )
    
    # Verify mock was called with correct parameters
    mock_discover_files.assert_called_once()
    call_args = mock_discover_files.call_args
    
    # Verify ignore patterns were passed correctly
    assert 'ignore_patterns' in call_args.kwargs
    assert 'mandatory_substrings' in call_args.kwargs
    
    # Verify result
    assert result == expected_files


# ============================================================================
# COMPREHENSIVE INTEGRATION TESTS
# ============================================================================

def test_end_to_end_experiment_workflow(
    sample_config_base,
    complex_directory_structure
):
    """
    Comprehensive end-to-end test for experiment file discovery workflow.
    Tests TST-INTEG-001 end-to-end workflow validation.
    
    Args:
        sample_config_base: Base configuration fixture
        complex_directory_structure: Complex test directory structure
    """
    # Update config to use test directory
    config = sample_config_base.copy()
    config["project"]["directories"]["major_data_directory"] = complex_directory_structure["base_dir"]
    
    # Test complete experiment discovery workflow
    experiment_files = discover_experiment_files(
        config=config,
        experiment_name="test_experiment",
        base_directory=complex_directory_structure["base_dir"]
    )
    
    # Verify experiment-specific filtering
    assert isinstance(experiment_files, list)
    assert len(experiment_files) > 0
    
    # All files should contain mandatory experiment strings
    assert all("include_me" in f for f in experiment_files)
    
    # No files should contain ignore patterns
    assert not any("exclude_me" in f for f in experiment_files)
    assert not any("._" in f for f in experiment_files)
    assert not any("temp_" in f for f in experiment_files)
    
    # Test with basic experiment (no additional filters)
    basic_files = discover_experiment_files(
        config=config,
        experiment_name="basic_experiment",
        base_directory=complex_directory_structure["base_dir"]
    )
    
    # Should have more files (only project-level filters applied)
    assert len(basic_files) >= len(experiment_files)


def test_end_to_end_dataset_workflow(
    sample_config_base,
    complex_directory_structure
):
    """
    Comprehensive end-to-end test for dataset file discovery workflow.
    
    Args:
        sample_config_base: Base configuration fixture
        complex_directory_structure: Complex test directory structure
    """
    # Update config to use test directory
    config = sample_config_base.copy()
    
    # Test dataset discovery
    dataset_files = discover_dataset_files(
        config=config,
        dataset_name="test_dataset",
        base_directory=complex_directory_structure["base_dir"]
    )
    
    # Verify dataset-specific behavior
    assert isinstance(dataset_files, list)
    assert len(dataset_files) > 0
    
    # Files should be from correct date directories
    expected_dates = ["2023-01-01", "2023-01-02", "2023_01_03"]
    for date in expected_dates:
        assert any(date in f for f in dataset_files), \
            f"No files found for expected date {date}"
    
    # Should respect project-level ignore patterns
    assert not any("._" in f for f in dataset_files)
    assert not any("temp_" in f for f in dataset_files)


# ============================================================================
# METADATA EXTRACTION TESTING
# ============================================================================

@pytest.mark.parametrize("extract_metadata,parse_dates,expected_result_type", [
    (False, False, list),
    (True, False, dict),
    (False, True, dict),
    (True, True, dict),
])
def test_metadata_extraction_modes(
    sample_config_base,
    complex_directory_structure,
    extract_metadata,
    parse_dates,
    expected_result_type
):
    """
    Test different metadata extraction and date parsing modes.
    Tests F-007 Metadata Extraction System requirements.
    
    Args:
        sample_config_base: Base configuration fixture
        complex_directory_structure: Test directory structure
        extract_metadata: Whether to extract metadata
        parse_dates: Whether to parse dates
        expected_result_type: Expected type of result
    """
    result = discover_files_with_config(
        config=sample_config_base,
        directory=complex_directory_structure["base_dir"],
        pattern="**/*.csv",
        recursive=True,
        extract_metadata=extract_metadata,
        parse_dates=parse_dates
    )
    
    # Verify result type
    assert isinstance(result, expected_result_type)
    
    if expected_result_type == dict:
        # Verify dictionary structure
        assert len(result) > 0
        
        # Each key should be a file path
        for file_path, metadata in result.items():
            assert isinstance(file_path, str)
            assert isinstance(metadata, dict)
            
            if extract_metadata:
                # Should have extracted metadata fields
                assert len(metadata) > 0
            
            if parse_dates:
                # Should have date parsing information
                assert len(metadata) > 0


if __name__ == "__main__":
    # Enable running tests directly
    pytest.main([__file__, "-v"])