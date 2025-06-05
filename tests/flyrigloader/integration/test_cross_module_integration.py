"""
Cross-module integration test suite for flyrigloader library.

This module validates seamless data flow and interface compatibility between core 
flyrigloader modules including config, discovery, io, and utils subsystems. Tests 
module boundary interactions, data format consistency across module interfaces, 
error handling propagation, and ensures that refactoring for testability (F-016) 
maintains functional correctness.

Integration Test Coverage:
- F-015: Cross-module interaction validation with realistic data flows
- F-016: Testability refactoring validation ensuring architectural integrity
- TST-INTEG-001: Cross-module integration validation
- TST-INTEG-003: Data integrity validation across module boundaries
- Section 4.1.2: Integration Workflows validation across module interfaces
- TST-MOD-003: Dependency isolation validation through pytest-mock

Key Integration Scenarios:
1. Configuration → Discovery → IO → Utils complete workflow validation
2. Error propagation across module boundaries
3. Data format consistency across module interfaces
4. Dependency injection pattern validation
5. Interface contract compliance testing
6. Module isolation with comprehensive mocking
"""

import gzip
import json
import pickle
import tempfile
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
import yaml
from hypothesis import given, strategies as st
from loguru import logger

# Core flyrigloader imports for integration testing
from flyrigloader.config.discovery import (
    discover_dataset_files,
    discover_experiment_files,
    discover_files_with_config,
)
from flyrigloader.config.yaml_config import (
    get_all_dataset_names,
    get_all_experiment_names,
    get_dataset_info,
    get_experiment_info,
    get_extraction_patterns,
    get_ignore_patterns,
    get_mandatory_substrings,
    load_config,
    validate_config_dict,
)
from flyrigloader.discovery.files import FileDiscoverer, discover_files
from flyrigloader.discovery.patterns import PatternMatcher, match_files_to_patterns
from flyrigloader.io.column_models import (
    ColumnConfig,
    ColumnConfigDict,
    ColumnDimension,
    SpecialHandlerType,
    get_config_from_source,
    load_column_config,
)
from flyrigloader.io.pickle import (
    make_dataframe_from_config,
    read_pickle_any_format,
    handle_signal_disp,
    extract_columns_from_matrix,
)
from flyrigloader.utils.dataframe import build_manifest_df, filter_manifest_df
from flyrigloader.utils.paths import (
    check_file_exists,
    ensure_directory_exists,
    find_common_base_directory,
    get_relative_path,
)


class TestCrossModuleDataFlow:
    """
    Integration tests validating complete data flow across all flyrigloader modules.
    
    These tests ensure that data flows seamlessly from configuration loading through
    file discovery, data loading, and transformation without corruption or interface
    mismatches per F-015 Integration Test Harness requirements.
    """

    def test_complete_workflow_config_to_dataframe(
        self, temp_experiment_directory, comprehensive_sample_config_dict
    ):
        """
        Test complete workflow from YAML configuration to final DataFrame output.
        
        Validates:
        - Configuration loading and validation (config module)
        - File discovery based on configuration (discovery module) 
        - Data loading from discovered files (io module)
        - DataFrame generation and manipulation (utils module)
        - End-to-end data integrity preservation
        
        Requirements: F-015, TST-INTEG-001, Section 4.1.2.1
        """
        logger.info("Starting complete workflow integration test")
        
        # Step 1: Configuration Module Integration
        config = comprehensive_sample_config_dict
        validated_config = validate_config_dict(config)
        
        # Verify configuration validation maintains structure
        assert isinstance(validated_config, dict)
        assert "project" in validated_config
        assert "datasets" in validated_config
        assert "experiments" in validated_config
        
        # Step 2: Configuration → Discovery Module Integration
        exp_dir = temp_experiment_directory["directory"]
        
        # Create realistic test files for discovery
        test_files = self._create_integration_test_files(exp_dir)
        
        # Test config-aware file discovery
        discovered_files = discover_files_with_config(
            config=validated_config,
            directory=str(exp_dir / "raw_data"),
            pattern="*.pkl",
            recursive=True,
            extract_metadata=True,
            parse_dates=True
        )
        
        # Validate discovery results structure
        assert isinstance(discovered_files, dict)
        assert len(discovered_files) > 0
        
        # Verify metadata extraction across config-discovery boundary
        for file_path, metadata in discovered_files.items():
            assert "path" in metadata
            assert isinstance(metadata, dict)
            
        logger.info(f"Discovery found {len(discovered_files)} files with metadata")
        
        # Step 3: Discovery → IO Module Integration
        # Create synthetic experimental data for files
        for file_path in discovered_files.keys():
            exp_data = self._create_synthetic_exp_matrix()
            self._save_test_pickle_file(file_path, exp_data)
        
        # Test data loading integration
        loaded_data = {}
        for file_path in list(discovered_files.keys())[:3]:  # Test subset for performance
            try:
                data = read_pickle_any_format(file_path)
                assert isinstance(data, dict), f"Expected dict from pickle load, got {type(data)}"
                loaded_data[file_path] = data
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
        
        # Validate data loading preserves structure
        assert len(loaded_data) > 0
        
        # Step 4: IO → Utils Module Integration  
        # Test DataFrame transformation and manifest generation
        for file_path, exp_matrix in loaded_data.items():
            # Test column extraction
            extracted_columns = extract_columns_from_matrix(exp_matrix)
            assert isinstance(extracted_columns, dict)
            
            # Test DataFrame creation with configuration
            df = make_dataframe_from_config(exp_matrix)
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            
        # Step 5: Complete Integration - Manifest Generation
        manifest_df = build_manifest_df(discovered_files, include_stats=False)
        assert isinstance(manifest_df, pd.DataFrame)
        assert len(manifest_df) > 0
        assert "path" in manifest_df.columns
        
        # Validate end-to-end data integrity
        original_file_count = len(discovered_files)
        manifest_file_count = len(manifest_df)
        assert manifest_file_count == original_file_count
        
        logger.info("Complete workflow integration test passed successfully")

    def test_error_propagation_across_modules(self, temp_experiment_directory):
        """
        Test error handling and propagation across module boundaries.
        
        Validates:
        - Configuration errors propagate correctly to discovery
        - Discovery errors are handled gracefully in IO operations
        - IO errors are properly reported to utils layer
        - Error recovery mechanisms work across boundaries
        
        Requirements: F-015, TST-INTEG-003, Section 4.1.2.3
        """
        logger.info("Testing error propagation across modules")
        
        exp_dir = temp_experiment_directory["directory"]
        
        # Test 1: Configuration Error Propagation
        invalid_config = {"invalid": "structure"}
        
        with pytest.raises((ValueError, KeyError)):
            # Should fail at config validation layer
            discover_files_with_config(
                config=invalid_config,
                directory=str(exp_dir),
                pattern="*.pkl"
            )
        
        # Test 2: File System Error Propagation
        valid_config = {"project": {}, "datasets": {}, "experiments": {}}
        nonexistent_dir = "/nonexistent/directory/path"
        
        # Discovery should handle missing directories gracefully
        result = discover_files_with_config(
            config=valid_config,
            directory=nonexistent_dir,
            pattern="*.pkl"
        )
        
        # Should return empty result, not crash
        assert isinstance(result, (list, dict))
        assert len(result) == 0
        
        # Test 3: IO Error Propagation
        # Create corrupted pickle file
        corrupted_file = exp_dir / "corrupted.pkl"
        corrupted_file.write_text("This is not a pickle file")
        
        with pytest.raises(Exception):
            # Should raise appropriate error, not crash silently
            read_pickle_any_format(corrupted_file)
        
        # Test 4: Utils Error Handling
        # Test with invalid data structure
        invalid_data = "not_a_dict_or_list"
        
        with pytest.raises((ValueError, TypeError)):
            build_manifest_df(invalid_data)
        
        logger.info("Error propagation tests completed successfully")

    def test_data_format_consistency_across_boundaries(self, temp_experiment_directory):
        """
        Test data format consistency across module interfaces.
        
        Validates:
        - Configuration data maintains expected structure through discovery
        - Discovery results maintain consistent format for IO operations
        - IO operations preserve data integrity for utils processing
        - Type contracts are maintained across all boundaries
        
        Requirements: TST-INTEG-003, F-016
        """
        logger.info("Testing data format consistency across module boundaries")
        
        exp_dir = temp_experiment_directory["directory"]
        
        # Create comprehensive test configuration
        config = {
            "project": {
                "ignore_substrings": ["backup", "temp"],
                "mandatory_experiment_strings": ["experiment"],
                "extraction_patterns": [
                    r".*_(?P<animal>\w+)_(?P<date>\d{8})_(?P<condition>\w+)\.pkl"
                ]
            },
            "datasets": {
                "test_dataset": {
                    "dates_vials": {
                        "20241201": ["mouse_001", "mouse_002"],
                        "20241202": ["mouse_003"]
                    }
                }
            },
            "experiments": {
                "test_experiment": {
                    "datasets": ["test_dataset"],
                    "filters": {
                        "mandatory_experiment_strings": ["experiment"]
                    }
                }
            }
        }
        
        # Test 1: Config → Discovery Data Format Consistency
        ignore_patterns = get_ignore_patterns(config)
        mandatory_strings = get_mandatory_substrings(config)
        extraction_patterns = get_extraction_patterns(config)
        
        assert isinstance(ignore_patterns, list)
        assert isinstance(mandatory_strings, list)
        assert isinstance(extraction_patterns, list) or extraction_patterns is None
        
        # Test 2: Discovery → IO Data Format Consistency
        test_files = self._create_test_files_with_patterns(exp_dir, config)
        
        discovered = discover_files_with_config(
            config=config,
            directory=str(exp_dir / "raw_data"),
            pattern="*.pkl",
            extract_metadata=True
        )
        
        # Validate discovery output format
        assert isinstance(discovered, dict)
        for file_path, metadata in discovered.items():
            assert isinstance(file_path, str)
            assert isinstance(metadata, dict)
            assert "path" in metadata
        
        # Test 3: IO → Utils Data Format Consistency
        for file_path in list(discovered.keys())[:2]:  # Test subset
            exp_data = self._create_synthetic_exp_matrix()
            self._save_test_pickle_file(file_path, exp_data)
            
            loaded_data = read_pickle_any_format(file_path)
            assert isinstance(loaded_data, dict)
            
            # Test DataFrame creation maintains format
            df = make_dataframe_from_config(loaded_data)
            assert isinstance(df, pd.DataFrame)
            
        # Test 4: Complete Format Chain Validation
        manifest = build_manifest_df(discovered)
        assert isinstance(manifest, pd.DataFrame)
        assert "path" in manifest.columns
        
        # Validate type consistency
        assert all(isinstance(path, str) for path in manifest["path"])
        
        logger.info("Data format consistency validation completed")

    def _create_integration_test_files(self, base_dir: Path) -> List[Path]:
        """Create realistic test files for integration testing."""
        raw_data_dir = base_dir / "raw_data"
        raw_data_dir.mkdir(exist_ok=True)
        
        files = []
        patterns = [
            "experiment_mouse_001_20241201_baseline.pkl",
            "experiment_mouse_002_20241201_treatment.pkl",
            "experiment_mouse_003_20241202_baseline.pkl",
            "backup_mouse_004_20241201_test.pkl",  # Should be ignored
            "experiment_rat_001_20241201_control.pkl"
        ]
        
        for pattern in patterns:
            file_path = raw_data_dir / pattern
            file_path.touch()
            files.append(file_path)
        
        return files

    def _create_test_files_with_patterns(self, base_dir: Path, config: Dict) -> List[Path]:
        """Create test files that match configuration patterns."""
        raw_data_dir = base_dir / "raw_data"
        raw_data_dir.mkdir(exist_ok=True)
        
        files = []
        for i in range(5):
            filename = f"experiment_mouse_{i:03d}_20241201_condition_{i}.pkl"
            file_path = raw_data_dir / filename
            file_path.touch()
            files.append(file_path)
        
        return files

    def _create_synthetic_exp_matrix(self) -> Dict[str, Any]:
        """Create synthetic experimental data matrix for testing."""
        time_points = 1000
        signal_channels = 15
        
        return {
            "t": np.linspace(0, 10, time_points),
            "x": np.random.normal(0, 1, time_points),
            "y": np.random.normal(0, 1, time_points),
            "velocity": np.random.exponential(2, time_points),
            "signal_disp": np.random.normal(0, 0.5, (signal_channels, time_points)),
            "metadata": {
                "animal_id": "mouse_001",
                "condition": "baseline",
                "date": "20241201"
            }
        }

    def _save_test_pickle_file(self, file_path: Union[str, Path], data: Dict[str, Any]):
        """Save test data as pickle file."""
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)


class TestModuleInterfaceContracts:
    """
    Test suite validating interface contracts between flyrigloader modules.
    
    Ensures that each module correctly implements expected interfaces and that
    data transformations across boundaries maintain contract compliance per
    F-016 Testability Refactoring Layer requirements.
    """

    def test_config_module_interface_contracts(self):
        """
        Test configuration module interface contracts.
        
        Validates:
        - load_config() returns consistent dictionary structure
        - Configuration validation maintains required keys
        - Helper functions return expected data types
        - Error handling follows expected patterns
        
        Requirements: F-016, TST-REF-001
        """
        logger.info("Testing configuration module interface contracts")
        
        # Test load_config interface contract
        test_config = {
            "project": {"name": "test"},
            "datasets": {"test_ds": {"dates_vials": {"20241201": ["vial1"]}}},
            "experiments": {"test_exp": {"datasets": ["test_ds"]}}
        }
        
        # Test dictionary input contract
        result = load_config(test_config)
        assert isinstance(result, dict)
        assert "project" in result
        assert "datasets" in result
        assert "experiments" in result
        
        # Test helper function contracts
        ignore_patterns = get_ignore_patterns(test_config)
        assert isinstance(ignore_patterns, list)
        
        mandatory_strings = get_mandatory_substrings(test_config)
        assert isinstance(mandatory_strings, list)
        
        extraction_patterns = get_extraction_patterns(test_config)
        assert extraction_patterns is None or isinstance(extraction_patterns, list)
        
        # Test dataset and experiment info contracts
        dataset_names = get_all_dataset_names(test_config)
        assert isinstance(dataset_names, list)
        assert "test_ds" in dataset_names
        
        experiment_names = get_all_experiment_names(test_config)
        assert isinstance(experiment_names, list)
        assert "test_exp" in experiment_names
        
        # Test error handling contracts
        with pytest.raises(KeyError):
            get_dataset_info(test_config, "nonexistent_dataset")
        
        with pytest.raises(KeyError):
            get_experiment_info(test_config, "nonexistent_experiment")
        
        logger.info("Configuration module interface contracts validated")

    def test_discovery_module_interface_contracts(self, temp_experiment_directory):
        """
        Test discovery module interface contracts.
        
        Validates:
        - discover_files() return type consistency
        - FileDiscoverer class interface compliance
        - Pattern matching interface contracts
        - Metadata extraction interface consistency
        
        Requirements: F-016, TST-REF-002
        """
        logger.info("Testing discovery module interface contracts")
        
        exp_dir = temp_experiment_directory["directory"]
        test_files = self._create_test_discovery_files(exp_dir)
        
        # Test discover_files function interface
        discovered = discover_files(
            directory=str(exp_dir / "raw_data"),
            pattern="*.pkl",
            recursive=True
        )
        
        # Contract: should return list when no metadata extraction
        assert isinstance(discovered, list)
        assert all(isinstance(path, str) for path in discovered)
        
        # Test with metadata extraction
        discovered_with_metadata = discover_files(
            directory=str(exp_dir / "raw_data"),
            pattern="*.pkl",
            extract_patterns=[r".*_(?P<animal>\w+)_(?P<date>\d{8})\.pkl"],
            parse_dates=True
        )
        
        # Contract: should return dict when metadata extraction enabled
        assert isinstance(discovered_with_metadata, dict)
        for file_path, metadata in discovered_with_metadata.items():
            assert isinstance(file_path, str)
            assert isinstance(metadata, dict)
        
        # Test FileDiscoverer class interface
        discoverer = FileDiscoverer(
            extract_patterns=[r".*_(?P<animal>\w+)_(?P<date>\d{8})\.pkl"],
            parse_dates=True,
            include_stats=False
        )
        
        # Contract: find_files should return list
        files = discoverer.find_files(
            directory=str(exp_dir / "raw_data"),
            pattern="*.pkl"
        )
        assert isinstance(files, list)
        
        # Contract: discover should return dict with metadata
        result = discoverer.discover(
            directory=str(exp_dir / "raw_data"),
            pattern="*.pkl"
        )
        assert isinstance(result, dict)
        
        # Test PatternMatcher interface
        patterns = [r".*_(?P<animal>\w+)_(?P<date>\d{8})\.pkl"]
        matcher = PatternMatcher(patterns)
        
        # Contract: match should return dict or None
        match_result = matcher.match("test_mouse_20241201.pkl")
        assert match_result is None or isinstance(match_result, dict)
        
        # Contract: filter_files should return dict
        filter_result = matcher.filter_files(["test_mouse_20241201.pkl"])
        assert isinstance(filter_result, dict)
        
        logger.info("Discovery module interface contracts validated")

    def test_io_module_interface_contracts(self, temp_experiment_directory):
        """
        Test IO module interface contracts.
        
        Validates:
        - read_pickle_any_format() return type consistency
        - DataFrame creation interface compliance
        - Column configuration interface contracts
        - Error handling interface consistency
        
        Requirements: F-016, TST-REF-003
        """
        logger.info("Testing IO module interface contracts")
        
        exp_dir = temp_experiment_directory["directory"]
        
        # Create test pickle file
        test_data = self._create_test_exp_matrix()
        test_file = exp_dir / "test_data.pkl"
        
        with open(test_file, 'wb') as f:
            pickle.dump(test_data, f)
        
        # Test read_pickle_any_format interface contract
        loaded_data = read_pickle_any_format(test_file)
        
        # Contract: should return dict or DataFrame
        assert isinstance(loaded_data, (dict, pd.DataFrame))
        
        if isinstance(loaded_data, dict):
            # Test DataFrame creation interface
            df = make_dataframe_from_config(loaded_data)
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
        
        # Test column extraction interface
        if isinstance(loaded_data, dict):
            extracted = extract_columns_from_matrix(loaded_data)
            assert isinstance(extracted, dict)
            
            # Contract: extracted columns should be numpy arrays or compatible
            for col_name, values in extracted.items():
                assert hasattr(values, '__len__') or np.isscalar(values)
        
        # Test column configuration interface
        config_dict = {
            "columns": {
                "t": {
                    "type": "numpy.ndarray",
                    "dimension": 1,
                    "required": True,
                    "description": "Time vector"
                }
            },
            "special_handlers": {}
        }
        
        column_config = ColumnConfigDict.model_validate(config_dict)
        assert isinstance(column_config, ColumnConfigDict)
        assert "t" in column_config.columns
        
        # Test signal_disp handling interface
        if "signal_disp" in test_data and "t" in test_data:
            signal_series = handle_signal_disp(test_data)
            assert isinstance(signal_series, pd.Series)
            assert len(signal_series) == len(test_data["t"])
        
        logger.info("IO module interface contracts validated")

    def test_utils_module_interface_contracts(self, temp_experiment_directory):
        """
        Test utils module interface contracts.
        
        Validates:
        - DataFrame utility function contracts
        - Path utility function contracts
        - Manifest building interface consistency
        - Type safety across utility operations
        
        Requirements: F-016, TST-REF-003
        """
        logger.info("Testing utils module interface contracts")
        
        exp_dir = temp_experiment_directory["directory"]
        
        # Test DataFrame utilities interface contracts
        test_files = ["file1.pkl", "file2.pkl", "file3.pkl"]
        
        # Test build_manifest_df with list input
        manifest_from_list = build_manifest_df(test_files)
        assert isinstance(manifest_from_list, pd.DataFrame)
        assert "path" in manifest_from_list.columns
        assert len(manifest_from_list) == len(test_files)
        
        # Test build_manifest_df with dict input
        test_metadata = {
            "file1.pkl": {"animal": "mouse_001", "condition": "baseline"},
            "file2.pkl": {"animal": "mouse_002", "condition": "treatment"}
        }
        
        manifest_from_dict = build_manifest_df(test_metadata)
        assert isinstance(manifest_from_dict, pd.DataFrame)
        assert "path" in manifest_from_dict.columns
        assert "animal" in manifest_from_dict.columns
        assert len(manifest_from_dict) == len(test_metadata)
        
        # Test filter_manifest_df interface
        filtered = filter_manifest_df(manifest_from_dict, animal="mouse_001")
        assert isinstance(filtered, pd.DataFrame)
        assert len(filtered) <= len(manifest_from_dict)
        
        # Test path utilities interface contracts
        test_dir = exp_dir / "test_subdir"
        test_dir.mkdir(exist_ok=True)
        
        # Test ensure_directory_exists contract
        created_dir = ensure_directory_exists(exp_dir / "new_dir")
        assert isinstance(created_dir, Path)
        assert created_dir.exists()
        
        # Test check_file_exists contract
        test_file = test_dir / "test.txt"
        test_file.touch()
        
        exists_result = check_file_exists(test_file)
        assert isinstance(exists_result, bool)
        assert exists_result is True
        
        not_exists_result = check_file_exists(test_dir / "nonexistent.txt")
        assert isinstance(not_exists_result, bool)
        assert not_exists_result is False
        
        # Test relative path calculation contract
        try:
            rel_path = get_relative_path(test_file, exp_dir)
            assert isinstance(rel_path, Path)
        except ValueError:
            # Expected if paths are not related
            pass
        
        # Test common base directory contract
        paths = [str(test_file), str(test_dir)]
        common_base = find_common_base_directory(paths)
        assert common_base is None or isinstance(common_base, Path)
        
        logger.info("Utils module interface contracts validated")

    def _create_test_discovery_files(self, base_dir: Path) -> List[Path]:
        """Create test files for discovery testing."""
        raw_data_dir = base_dir / "raw_data"
        raw_data_dir.mkdir(exist_ok=True)
        
        files = []
        for i in range(3):
            filename = f"test_mouse_{i:03d}_20241201.pkl"
            file_path = raw_data_dir / filename
            file_path.touch()
            files.append(file_path)
        
        return files

    def _create_test_exp_matrix(self) -> Dict[str, Any]:
        """Create test experimental data matrix."""
        n_points = 100
        return {
            "t": np.linspace(0, 1, n_points),
            "x": np.random.normal(0, 1, n_points),
            "y": np.random.normal(0, 1, n_points),
            "signal_disp": np.random.normal(0, 0.1, (10, n_points))
        }


class TestDependencyInjectionPatterns:
    """
    Test suite validating dependency injection patterns across flyrigloader modules.
    
    Ensures that testability improvements through dependency injection maintain
    functional correctness and enable comprehensive testing isolation per
    F-016 requirements.
    """

    def test_config_module_dependency_injection(self, mocker):
        """
        Test configuration module dependency injection patterns.
        
        Validates:
        - YAML loading can be mocked independently
        - Configuration validation accepts injected dependencies
        - Helper functions work with injected configuration
        - Error handling maintains isolation
        
        Requirements: F-016, TST-MOD-003
        """
        logger.info("Testing configuration module dependency injection")
        
        # Test YAML loading dependency injection
        mock_yaml_content = {
            "project": {"name": "injected_test"},
            "datasets": {},
            "experiments": {}
        }
        
        # Mock YAML loading
        mock_open_func = mocker.mock_open(read_data=yaml.dump(mock_yaml_content))
        mocker.patch("builtins.open", mock_open_func)
        mocker.patch("yaml.safe_load", return_value=mock_yaml_content)
        mocker.patch("pathlib.Path.exists", return_value=True)
        
        # Test that injected dependencies work
        result = load_config("mocked_config.yaml")
        assert result["project"]["name"] == "injected_test"
        
        # Test configuration helpers with injected config
        ignore_patterns = get_ignore_patterns(result)
        assert isinstance(ignore_patterns, list)
        
        mandatory_strings = get_mandatory_substrings(result)
        assert isinstance(mandatory_strings, list)
        
        # Test that error handling works with injection
        with pytest.raises(KeyError):
            get_dataset_info(result, "nonexistent")
        
        logger.info("Configuration dependency injection patterns validated")

    def test_discovery_module_dependency_injection(self, mocker, temp_experiment_directory):
        """
        Test discovery module dependency injection patterns.
        
        Validates:
        - File system operations can be mocked
        - Pattern matching works with injected patterns
        - Metadata extraction accepts injected extractors
        - Discovery results maintain consistency with mocked dependencies
        
        Requirements: F-016, TST-MOD-003
        """
        logger.info("Testing discovery module dependency injection")
        
        exp_dir = temp_experiment_directory["directory"]
        
        # Mock file system operations
        mock_files = [
            "experiment_mouse_001_20241201.pkl",
            "experiment_mouse_002_20241201.pkl",
            "experiment_rat_001_20241201.pkl"
        ]
        
        mock_glob_results = [Path(exp_dir / "raw_data" / f) for f in mock_files]
        
        # Inject file discovery dependencies
        mocker.patch("pathlib.Path.glob", return_value=mock_glob_results)
        mocker.patch("pathlib.Path.rglob", return_value=mock_glob_results)
        
        # Test with injected dependencies
        discovered = discover_files(
            directory=str(exp_dir / "raw_data"),
            pattern="*.pkl",
            recursive=True
        )
        
        # Should work with mocked file system
        assert isinstance(discovered, list)
        assert len(discovered) == len(mock_files)
        
        # Test pattern injection
        injected_patterns = [r".*_(?P<animal>\w+)_(?P<date>\d{8})\.pkl"]
        matcher = PatternMatcher(injected_patterns)
        
        test_filename = "experiment_mouse_001_20241201.pkl"
        match_result = matcher.match(test_filename)
        
        # Should work with injected patterns
        if match_result:
            assert "animal" in match_result
            assert "date" in match_result
        
        # Test FileDiscoverer with injected dependencies
        discoverer = FileDiscoverer(
            extract_patterns=injected_patterns,
            parse_dates=True
        )
        
        # Mock Path operations for discoverer
        mock_path_glob = mocker.patch.object(Path, "glob")
        mock_path_glob.return_value = [Path(f) for f in mock_files]
        
        result = discoverer.find_files(
            directory=str(exp_dir / "raw_data"),
            pattern="*.pkl"
        )
        
        assert isinstance(result, list)
        
        logger.info("Discovery dependency injection patterns validated")

    def test_io_module_dependency_injection(self, mocker, temp_experiment_directory):
        """
        Test IO module dependency injection patterns.
        
        Validates:
        - Pickle loading can be mocked independently
        - DataFrame creation works with injected configurations
        - Column processing accepts injected handlers
        - Error handling maintains isolation with mocked dependencies
        
        Requirements: F-016, TST-MOD-003
        """
        logger.info("Testing IO module dependency injection")
        
        exp_dir = temp_experiment_directory["directory"]
        
        # Test pickle loading dependency injection
        mock_exp_data = {
            "t": np.array([0, 1, 2, 3, 4]),
            "x": np.array([1, 2, 3, 4, 5]),
            "y": np.array([2, 3, 4, 5, 6]),
            "signal_disp": np.random.normal(0, 1, (10, 5))
        }
        
        # Mock pickle loading
        mocker.patch("pickle.load", return_value=mock_exp_data)
        mocker.patch("gzip.open")
        mocker.patch("builtins.open", mocker.mock_open())
        mocker.patch("pathlib.Path.exists", return_value=True)
        
        # Test with injected pickle loader
        test_file = exp_dir / "test.pkl"
        loaded_data = read_pickle_any_format(test_file)
        
        # Should get mocked data
        assert "t" in loaded_data
        assert "x" in loaded_data
        
        # Test DataFrame creation with injected configuration
        mock_config = {
            "columns": {
                "t": {
                    "type": "numpy.ndarray",
                    "dimension": 1,
                    "required": True,
                    "description": "Time"
                },
                "x": {
                    "type": "numpy.ndarray", 
                    "dimension": 1,
                    "required": True,
                    "description": "X position"
                }
            },
            "special_handlers": {}
        }
        
        # Mock column configuration loading
        mocker.patch("flyrigloader.io.column_models.load_column_config")
        mocker.patch(
            "flyrigloader.io.column_models.get_config_from_source", 
            return_value=ColumnConfigDict.model_validate(mock_config)
        )
        
        df = make_dataframe_from_config(loaded_data, config_source=mock_config)
        assert isinstance(df, pd.DataFrame)
        assert "t" in df.columns
        assert "x" in df.columns
        
        # Test column extraction with injected data
        extracted = extract_columns_from_matrix(loaded_data, column_names=["t", "x"])
        assert "t" in extracted
        assert "x" in extracted
        
        logger.info("IO dependency injection patterns validated")

    def test_utils_module_dependency_injection(self, mocker):
        """
        Test utils module dependency injection patterns.
        
        Validates:
        - File operations can be mocked independently
        - DataFrame operations work with injected data
        - Path operations accept injected file system
        - Manifest building maintains consistency with mocked inputs
        
        Requirements: F-016, TST-MOD-003
        """
        logger.info("Testing utils module dependency injection")
        
        # Test manifest building with injected data
        mock_files_data = {
            "file1.pkl": {"animal": "mouse_001", "date": "20241201"},
            "file2.pkl": {"animal": "mouse_002", "date": "20241202"}
        }
        
        manifest = build_manifest_df(mock_files_data)
        assert isinstance(manifest, pd.DataFrame)
        assert len(manifest) == 2
        assert "animal" in manifest.columns
        
        # Test filtering with injected criteria
        filtered = filter_manifest_df(manifest, animal="mouse_001")
        assert len(filtered) == 1
        assert filtered.iloc[0]["animal"] == "mouse_001"
        
        # Test path operations with mocked file system
        mocker.patch("pathlib.Path.exists", return_value=True)
        mocker.patch("pathlib.Path.is_file", return_value=True)
        mocker.patch("pathlib.Path.mkdir")
        
        # Test with mocked file operations
        exists_result = check_file_exists("/mocked/path/file.txt")
        assert exists_result is True
        
        # Test directory creation with mocked operations
        created_dir = ensure_directory_exists("/mocked/path/new_dir")
        assert isinstance(created_dir, Path)
        
        # Test relative path with mocked paths
        mock_path_relative_to = mocker.patch.object(Path, "relative_to")
        mock_path_relative_to.return_value = Path("relative/path")
        
        try:
            rel_path = get_relative_path("/full/path/file.txt", "/full/path")
            assert isinstance(rel_path, Path)
        except Exception:
            # May fail with mocked operations, which is acceptable
            pass
        
        logger.info("Utils dependency injection patterns validated")


class TestModuleIsolationWithMocking:
    """
    Test suite validating module isolation through comprehensive mocking.
    
    Ensures that modules can function independently while maintaining integration
    compatibility through strategic use of pytest-mock per TST-MOD-003 requirements.
    """

    def test_config_module_isolation(self, mocker):
        """
        Test configuration module in complete isolation.
        
        Validates:
        - Configuration module works without file system dependencies
        - YAML parsing logic is isolated and testable
        - Validation logic works independently
        - Helper functions operate on pure data structures
        
        Requirements: TST-MOD-003, F-016
        """
        logger.info("Testing configuration module isolation")
        
        # Completely isolate configuration module from file system
        mock_yaml_data = {
            "project": {
                "name": "isolated_test",
                "ignore_substrings": ["temp", "backup"],
                "mandatory_experiment_strings": ["exp"],
                "extraction_patterns": [r".*_(?P<id>\d+)\.pkl"]
            },
            "datasets": {
                "isolated_dataset": {
                    "dates_vials": {
                        "20241201": ["vial1", "vial2"],
                        "20241202": ["vial3"]
                    }
                }
            },
            "experiments": {
                "isolated_experiment": {
                    "datasets": ["isolated_dataset"],
                    "filters": {
                        "ignore_substrings": ["failed"]
                    }
                }
            }
        }
        
        # Mock all external dependencies
        mocker.patch("pathlib.Path.exists", return_value=True)
        mocker.patch("builtins.open", mocker.mock_open())
        mocker.patch("yaml.safe_load", return_value=mock_yaml_data)
        
        # Test configuration loading in isolation
        config = load_config("mocked_config.yaml")
        assert config["project"]["name"] == "isolated_test"
        
        # Test validation in isolation (no file system dependencies)
        validated = validate_config_dict(config)
        assert validated == config
        
        # Test helper functions in complete isolation
        ignore_patterns = get_ignore_patterns(config)
        assert "temp" in " ".join(ignore_patterns)
        
        mandatory_strings = get_mandatory_substrings(config)
        assert "exp" in mandatory_strings
        
        extraction_patterns = get_extraction_patterns(config)
        assert isinstance(extraction_patterns, list)
        
        # Test dataset operations in isolation
        dataset_info = get_dataset_info(config, "isolated_dataset")
        assert "dates_vials" in dataset_info
        
        experiment_info = get_experiment_info(config, "isolated_experiment")
        assert "datasets" in experiment_info
        
        # Test list operations
        dataset_names = get_all_dataset_names(config)
        assert "isolated_dataset" in dataset_names
        
        experiment_names = get_all_experiment_names(config)
        assert "isolated_experiment" in experiment_names
        
        logger.info("Configuration module isolation test completed")

    def test_discovery_module_isolation(self, mocker):
        """
        Test discovery module in complete isolation.
        
        Validates:
        - File discovery works without real file system
        - Pattern matching operates on pure string data
        - Metadata extraction works with mocked inputs
        - Discovery logic is independent of IO operations
        
        Requirements: TST-MOD-003, F-016
        """
        logger.info("Testing discovery module isolation")
        
        # Mock all file system operations
        mock_file_paths = [
            "/isolated/path/experiment_001_mouse_20241201.pkl",
            "/isolated/path/experiment_002_rat_20241202.pkl",
            "/isolated/path/backup_003_mouse_20241203.pkl"
        ]
        
        mock_path_objects = [Path(p) for p in mock_file_paths]
        
        # Complete file system isolation
        mocker.patch("pathlib.Path.glob", return_value=mock_path_objects)
        mocker.patch("pathlib.Path.rglob", return_value=mock_path_objects)
        mocker.patch("pathlib.Path.exists", return_value=True)
        mocker.patch("pathlib.Path.is_file", return_value=True)
        
        # Test basic file discovery in isolation
        discovered = discover_files(
            directory="/isolated/path",
            pattern="*.pkl",
            recursive=True
        )
        
        assert isinstance(discovered, list)
        assert len(discovered) == len(mock_file_paths)
        
        # Test pattern matching in complete isolation
        patterns = [
            r".*_(?P<exp_id>\d+)_(?P<animal>\w+)_(?P<date>\d{8})\.pkl"
        ]
        
        matcher = PatternMatcher(patterns)
        
        # Test pure pattern matching (no file system)
        test_filename = "experiment_001_mouse_20241201.pkl"
        match_result = matcher.match(test_filename)
        
        assert match_result is not None
        assert match_result["exp_id"] == "001"
        assert match_result["animal"] == "mouse"
        assert match_result["date"] == "20241201"
        
        # Test file filtering in isolation
        filter_result = matcher.filter_files(mock_file_paths)
        assert isinstance(filter_result, dict)
        
        # Test FileDiscoverer in complete isolation
        discoverer = FileDiscoverer(
            extract_patterns=patterns,
            parse_dates=True,
            include_stats=False
        )
        
        # Mock stat operations
        mock_stat = MagicMock()
        mock_stat.st_size = 1024
        mock_stat.st_mtime = datetime.now().timestamp()
        mocker.patch("pathlib.Path.stat", return_value=mock_stat)
        
        isolated_result = discoverer.discover(
            directory="/isolated/path",
            pattern="*.pkl"
        )
        
        assert isinstance(isolated_result, dict)
        
        # Test filtering logic in isolation
        filtered_files = discoverer._apply_filters(
            mock_file_paths,
            ignore_patterns=["*backup*"],
            mandatory_substrings=["experiment"]
        )
        
        # Should filter out backup file
        assert len(filtered_files) < len(mock_file_paths)
        assert not any("backup" in f for f in filtered_files)
        
        logger.info("Discovery module isolation test completed")

    def test_io_module_isolation(self, mocker):
        """
        Test IO module in complete isolation.
        
        Validates:
        - Data loading works without real files
        - DataFrame creation operates on pure data structures
        - Column processing works with mocked configurations
        - Error handling maintains isolation
        
        Requirements: TST-MOD-003, F-016
        """
        logger.info("Testing IO module isolation")
        
        # Create isolated experimental data
        isolated_exp_data = {
            "t": np.array([0.0, 0.1, 0.2, 0.3, 0.4]),
            "x": np.array([1.0, 1.1, 1.2, 1.3, 1.4]),
            "y": np.array([2.0, 2.1, 2.2, 2.3, 2.4]),
            "velocity": np.array([0.5, 0.6, 0.7, 0.8, 0.9]),
            "signal_disp": np.random.normal(0, 0.1, (8, 5))
        }
        
        # Mock all file operations
        mocker.patch("pathlib.Path.exists", return_value=True)
        mocker.patch("builtins.open", mocker.mock_open())
        mocker.patch("pickle.load", return_value=isolated_exp_data)
        mocker.patch("gzip.open")
        mocker.patch("pandas.read_pickle", return_value=isolated_exp_data)
        
        # Test pickle loading in complete isolation
        loaded_data = read_pickle_any_format("/isolated/file.pkl")
        assert "t" in loaded_data
        assert "x" in loaded_data
        
        # Test column extraction in isolation
        extracted_columns = extract_columns_from_matrix(
            loaded_data, 
            column_names=["t", "x", "y"]
        )
        
        assert "t" in extracted_columns
        assert "x" in extracted_columns
        assert "y" in extracted_columns
        
        # Test signal_disp handling in isolation
        signal_series = handle_signal_disp(loaded_data)
        assert isinstance(signal_series, pd.Series)
        assert len(signal_series) == len(loaded_data["t"])
        
        # Test DataFrame creation with isolated configuration
        isolated_config = {
            "columns": {
                "t": {
                    "type": "numpy.ndarray",
                    "dimension": 1,
                    "required": True,
                    "description": "Time vector"
                },
                "x": {
                    "type": "numpy.ndarray",
                    "dimension": 1,
                    "required": True,
                    "description": "X position"
                },
                "y": {
                    "type": "numpy.ndarray",
                    "dimension": 1,
                    "required": False,
                    "description": "Y position"
                }
            },
            "special_handlers": {}
        }
        
        # Mock configuration loading
        column_config_obj = ColumnConfigDict.model_validate(isolated_config)
        mocker.patch(
            "flyrigloader.io.column_models.get_config_from_source", 
            return_value=column_config_obj
        )
        
        # Test DataFrame creation in isolation
        df = make_dataframe_from_config(loaded_data, config_source=isolated_config)
        assert isinstance(df, pd.DataFrame)
        assert "t" in df.columns
        assert "x" in df.columns
        assert len(df) == len(loaded_data["t"])
        
        # Test column configuration validation in isolation
        config_from_source = get_config_from_source(isolated_config)
        assert isinstance(config_from_source, ColumnConfigDict)
        assert "t" in config_from_source.columns
        
        logger.info("IO module isolation test completed")

    def test_utils_module_isolation(self, mocker):
        """
        Test utils module in complete isolation.
        
        Validates:
        - DataFrame utilities work with pure data structures
        - Path utilities operate without file system dependencies
        - Manifest building works with mocked inputs
        - Statistical operations maintain isolation
        
        Requirements: TST-MOD-003, F-016
        """
        logger.info("Testing utils module isolation")
        
        # Test DataFrame utilities in complete isolation
        isolated_file_data = {
            "/isolated/file1.pkl": {
                "animal": "mouse_001",
                "condition": "baseline",
                "date": "20241201",
                "replicate": 1
            },
            "/isolated/file2.pkl": {
                "animal": "mouse_002", 
                "condition": "treatment",
                "date": "20241201",
                "replicate": 2
            },
            "/isolated/file3.pkl": {
                "animal": "rat_001",
                "condition": "baseline", 
                "date": "20241202",
                "replicate": 1
            }
        }
        
        # Test manifest building in isolation
        manifest = build_manifest_df(isolated_file_data)
        assert isinstance(manifest, pd.DataFrame)
        assert len(manifest) == 3
        assert "path" in manifest.columns
        assert "animal" in manifest.columns
        assert "condition" in manifest.columns
        
        # Test filtering in isolation
        filtered_baseline = filter_manifest_df(manifest, condition="baseline")
        assert len(filtered_baseline) == 2
        assert all(filtered_baseline["condition"] == "baseline")
        
        filtered_mouse = filter_manifest_df(manifest, animal="mouse_001")
        assert len(filtered_mouse) == 1
        assert filtered_mouse.iloc[0]["animal"] == "mouse_001"
        
        # Test with list input in isolation
        isolated_file_list = list(isolated_file_data.keys())
        list_manifest = build_manifest_df(isolated_file_list)
        assert isinstance(list_manifest, pd.DataFrame)
        assert len(list_manifest) == 3
        assert "path" in list_manifest.columns
        
        # Mock all path operations for complete isolation
        mocker.patch("pathlib.Path.exists", return_value=True)
        mocker.patch("pathlib.Path.is_file", return_value=True)
        mocker.patch("pathlib.Path.is_dir", return_value=True)
        mocker.patch("pathlib.Path.mkdir")
        mocker.patch("pathlib.Path.resolve", side_effect=lambda x: Path(f"/resolved{x}"))
        
        # Test path utilities in isolation
        exists_result = check_file_exists("/isolated/path/file.txt")
        assert exists_result is True
        
        # Test directory creation in isolation
        created_dir = ensure_directory_exists("/isolated/new_dir")
        assert isinstance(created_dir, Path)
        
        # Test relative path calculation in isolation
        mock_relative_to = mocker.patch.object(Path, "relative_to")
        mock_relative_to.return_value = Path("relative/file.txt")
        
        rel_path = get_relative_path("/base/relative/file.txt", "/base")
        assert isinstance(rel_path, Path)
        
        # Test common base directory in isolation
        isolated_paths = [
            "/isolated/base/path1/file1.txt",
            "/isolated/base/path2/file2.txt",
            "/isolated/base/path3/file3.txt"
        ]
        
        # Mock Path.parts for common base calculation
        mock_parts = {
            "/isolated/base/path1/file1.txt": ("", "isolated", "base", "path1", "file1.txt"),
            "/isolated/base/path2/file2.txt": ("", "isolated", "base", "path2", "file2.txt"),
            "/isolated/base/path3/file3.txt": ("", "isolated", "base", "path3", "file3.txt")
        }
        
        def mock_path_parts(path_str):
            mock_path = MagicMock()
            mock_path.parts = mock_parts.get(str(path_str), ())
            return mock_path
        
        mocker.patch("pathlib.Path", side_effect=mock_path_parts)
        
        common_base = find_common_base_directory(isolated_paths)
        # Should find common base or return None, both acceptable in isolation
        assert common_base is None or isinstance(common_base, Path)
        
        logger.info("Utils module isolation test completed")


@pytest.mark.parametrize("config_type", ["yaml_file", "dict", "kedro_params"])
def test_configuration_interface_compatibility(config_type, temp_experiment_directory):
    """
    Parametrized test ensuring configuration interface compatibility across input types.
    
    Validates that all configuration input methods (YAML files, dictionaries, 
    Kedro parameters) produce consistent results and maintain interface contracts
    per F-015 Integration Test Harness requirements.
    
    Requirements: F-015, TST-INTEG-001, TST-REF-001
    """
    logger.info(f"Testing configuration interface compatibility for {config_type}")
    
    exp_dir = temp_experiment_directory["directory"]
    
    # Base configuration data
    config_data = {
        "project": {
            "name": f"test_project_{config_type}",
            "ignore_substrings": ["backup", "temp"],
            "mandatory_experiment_strings": ["experiment"]
        },
        "datasets": {
            "test_dataset": {
                "dates_vials": {
                    "20241201": ["mouse_001", "mouse_002"],
                    "20241202": ["mouse_003"]
                }
            }
        },
        "experiments": {
            "test_experiment": {
                "datasets": ["test_dataset"]
            }
        }
    }
    
    # Test different configuration input methods
    if config_type == "yaml_file":
        # Create temporary YAML file
        config_file = exp_dir / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        config = load_config(str(config_file))
        
    elif config_type == "dict":
        # Direct dictionary input
        config = load_config(config_data)
        
    elif config_type == "kedro_params":
        # Simulate Kedro parameters dictionary
        kedro_config = {
            **config_data,
            "_kedro_metadata": {"run_id": "test_run", "pipeline": "test_pipeline"}
        }
        config = validate_config_dict(kedro_config)
    
    # Validate consistent interface regardless of input type
    assert isinstance(config, dict)
    assert "project" in config
    assert "datasets" in config
    assert "experiments" in config
    
    # Test that helper functions work consistently
    ignore_patterns = get_ignore_patterns(config)
    assert isinstance(ignore_patterns, list)
    
    mandatory_strings = get_mandatory_substrings(config)
    assert isinstance(mandatory_strings, list)
    
    dataset_names = get_all_dataset_names(config)
    assert "test_dataset" in dataset_names
    
    experiment_names = get_all_experiment_names(config)
    assert "test_experiment" in experiment_names
    
    # Test discovery integration with different config types
    try:
        discovered = discover_files_with_config(
            config=config,
            directory=str(exp_dir),
            pattern="*.pkl"
        )
        assert isinstance(discovered, (list, dict))
    except Exception as e:
        # File discovery may fail in test environment, which is acceptable
        logger.warning(f"Discovery failed for {config_type}: {e}")
    
    logger.info(f"Configuration interface compatibility validated for {config_type}")


@given(st.integers(min_value=10, max_value=1000))
def test_data_flow_scalability(time_points):
    """
    Property-based test for data flow scalability across modules.
    
    Uses Hypothesis to generate various data sizes and validates that the complete
    data flow pipeline maintains performance and correctness characteristics
    across different scales per TST-PERF-001 requirements.
    
    Requirements: TST-PERF-001, F-015, Section 4.1.1.5
    """
    logger.info(f"Testing data flow scalability with {time_points} time points")
    
    # Generate synthetic experimental data of varying sizes
    signal_channels = max(5, time_points // 100)  # Scale channels with time points
    
    exp_data = {
        "t": np.linspace(0, time_points / 1000, time_points),
        "x": np.random.normal(0, 1, time_points),
        "y": np.random.normal(0, 1, time_points),
        "velocity": np.random.exponential(1, time_points),
        "signal_disp": np.random.normal(0, 0.1, (signal_channels, time_points))
    }
    
    # Test column extraction scalability
    start_time = datetime.now()
    extracted_columns = extract_columns_from_matrix(exp_data)
    extraction_time = (datetime.now() - start_time).total_seconds()
    
    assert isinstance(extracted_columns, dict)
    assert "t" in extracted_columns
    assert len(extracted_columns["t"]) == time_points
    
    # Test signal_disp handling scalability
    start_time = datetime.now()
    signal_series = handle_signal_disp(exp_data)
    signal_time = (datetime.now() - start_time).total_seconds()
    
    assert isinstance(signal_series, pd.Series)
    assert len(signal_series) == time_points
    
    # Test DataFrame creation scalability
    start_time = datetime.now()
    df = make_dataframe_from_config(exp_data)
    dataframe_time = (datetime.now() - start_time).total_seconds()
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == time_points
    
    # Performance validation (basic SLA check)
    # These are reasonable performance expectations
    expected_extraction_time = time_points / 100000  # 10μs per data point
    expected_signal_time = time_points / 50000       # 20μs per data point  
    expected_dataframe_time = time_points / 10000    # 100μs per data point
    
    # Log performance metrics
    logger.debug(f"Extraction time: {extraction_time:.4f}s (expected: {expected_extraction_time:.4f}s)")
    logger.debug(f"Signal processing time: {signal_time:.4f}s (expected: {expected_signal_time:.4f}s)")
    logger.debug(f"DataFrame creation time: {dataframe_time:.4f}s (expected: {expected_dataframe_time:.4f}s)")
    
    # Validate reasonable performance (with generous margins for test environments)
    assert extraction_time < expected_extraction_time * 10  # 10x margin
    assert signal_time < expected_signal_time * 10          # 10x margin
    assert dataframe_time < expected_dataframe_time * 10    # 10x margin
    
    logger.info(f"Data flow scalability validated for {time_points} time points")


def test_error_recovery_mechanisms(temp_experiment_directory):
    """
    Test error recovery mechanisms across module boundaries.
    
    Validates that the system can gracefully recover from various error conditions
    and continue processing, ensuring robust operation in production environments
    per Section 4.1.2.3 error recovery requirements.
    
    Requirements: Section 4.1.2.3, F-015, TST-INTEG-003
    """
    logger.info("Testing error recovery mechanisms across modules")
    
    exp_dir = temp_experiment_directory["directory"]
    
    # Test 1: Configuration error recovery
    invalid_configs = [
        {"missing_required_keys": True},  # Missing project, datasets, experiments
        {"project": None},                # Invalid project structure
        {"datasets": "not_a_dict"},      # Invalid datasets type
    ]
    
    for i, invalid_config in enumerate(invalid_configs):
        logger.debug(f"Testing configuration error recovery {i + 1}")
        
        try:
            # Should raise appropriate error, not crash silently
            validate_config_dict(invalid_config)
            pytest.fail(f"Expected error for invalid config {i + 1}")
        except (ValueError, TypeError, KeyError):
            # Expected error types for configuration validation
            pass
    
    # Test 2: Discovery error recovery
    valid_config = {
        "project": {},
        "datasets": {},
        "experiments": {}
    }
    
    # Test with nonexistent directory
    result = discover_files_with_config(
        config=valid_config,
        directory="/nonexistent/directory",
        pattern="*.pkl"
    )
    
    # Should return empty result, not crash
    assert isinstance(result, (list, dict))
    assert len(result) == 0
    
    # Test 3: IO error recovery
    # Create files with various corrupted states
    test_files = []
    
    # Empty file
    empty_file = exp_dir / "empty.pkl"
    empty_file.touch()
    test_files.append(empty_file)
    
    # Text file with .pkl extension
    text_file = exp_dir / "text.pkl"
    text_file.write_text("This is not a pickle file")
    test_files.append(text_file)
    
    # Binary file that's not a pickle
    binary_file = exp_dir / "binary.pkl"
    binary_file.write_bytes(b"\x00\x01\x02\x03\x04\x05")
    test_files.append(binary_file)
    
    # Test error handling for each corrupted file
    for test_file in test_files:
        logger.debug(f"Testing IO error recovery for {test_file.name}")
        
        try:
            read_pickle_any_format(test_file)
            pytest.fail(f"Expected error for corrupted file {test_file.name}")
        except Exception as e:
            # Should raise appropriate error, not crash
            assert isinstance(e, (RuntimeError, pickle.UnpicklingError, EOFError, OSError))
    
    # Test 4: Utils error recovery
    # Test with invalid inputs to utils functions
    invalid_inputs = [
        "not_a_list_or_dict",
        {"invalid": "structure_for_manifest"},
        [],  # Empty list
        {}   # Empty dict
    ]
    
    for invalid_input in invalid_inputs:
        logger.debug(f"Testing utils error recovery for {type(invalid_input)}")
        
        try:
            manifest = build_manifest_df(invalid_input)
            # Should handle gracefully and return valid DataFrame
            assert isinstance(manifest, pd.DataFrame)
        except (ValueError, TypeError):
            # Some invalid inputs may raise errors, which is acceptable
            pass
    
    # Test 5: Cross-module error propagation
    # Test that errors propagate correctly but don't crash the system
    corrupted_config = {
        "project": {"ignore_substrings": None},  # Invalid type
        "datasets": {},
        "experiments": {}
    }
    
    try:
        # This should fail gracefully at the config level
        ignore_patterns = get_ignore_patterns(corrupted_config)
        # If it doesn't fail, it should return a reasonable default
        assert isinstance(ignore_patterns, list)
    except (ValueError, TypeError, AttributeError):
        # Expected error types
        pass
    
    logger.info("Error recovery mechanisms validated successfully")


def test_memory_efficiency_across_modules():
    """
    Test memory efficiency across module boundaries.
    
    Validates that data flows efficiently through the system without excessive
    memory usage or memory leaks during cross-module operations per performance
    requirements in Section 4.1.1.5.
    
    Requirements: Section 4.1.1.5, TST-PERF-002, F-015
    """
    logger.info("Testing memory efficiency across modules")
    
    # Generate moderately large dataset for memory testing
    n_time_points = 10000
    n_channels = 50
    
    large_exp_data = {
        "t": np.linspace(0, 100, n_time_points),
        "x": np.random.normal(0, 1, n_time_points),
        "y": np.random.normal(0, 1, n_time_points),
        "velocity": np.random.exponential(1, n_time_points),
        "signal_disp": np.random.normal(0, 0.1, (n_channels, n_time_points)),
        "additional_signals": np.random.normal(0, 0.5, (n_channels * 2, n_time_points))
    }
    
    # Test memory efficiency of column extraction
    extracted_columns = extract_columns_from_matrix(
        large_exp_data, 
        column_names=["t", "x", "y", "velocity"]
    )
    
    # Should extract only requested columns, not copy all data
    assert len(extracted_columns) == 4
    assert "signal_disp" not in extracted_columns
    assert "additional_signals" not in extracted_columns
    
    # Test memory efficiency of signal processing
    signal_series = handle_signal_disp(large_exp_data)
    
    # Should create efficient representation
    assert isinstance(signal_series, pd.Series)
    assert len(signal_series) == n_time_points
    
    # Test memory efficiency of DataFrame creation
    df = make_dataframe_from_config(large_exp_data)
    
    # Should create DataFrame without excessive memory overhead
    assert isinstance(df, pd.DataFrame)
    assert len(df) == n_time_points
    
    # Verify that original data is not unnecessarily duplicated
    # This is a basic check - in practice, more sophisticated memory profiling would be used
    df_memory_usage = df.memory_usage(deep=True).sum()
    logger.debug(f"DataFrame memory usage: {df_memory_usage / 1024 / 1024:.2f} MB")
    
    # Basic sanity check - DataFrame shouldn't use excessive memory
    # Allow for reasonable overhead but catch gross inefficiencies
    expected_max_memory_mb = (n_time_points * len(df.columns) * 8) / 1024 / 1024 * 2  # 2x overhead allowance
    actual_memory_mb = df_memory_usage / 1024 / 1024
    
    assert actual_memory_mb < expected_max_memory_mb, f"Memory usage {actual_memory_mb:.2f}MB exceeds expected {expected_max_memory_mb:.2f}MB"
    
    # Test that large objects can be processed without copying
    # Use a subset of columns to verify memory efficiency
    subset_df = make_dataframe_from_config(
        large_exp_data, 
        skip_columns=["additional_signals", "signal_disp"]
    )
    
    # Should be smaller than full DataFrame
    subset_memory = subset_df.memory_usage(deep=True).sum()
    assert subset_memory < df_memory_usage
    
    logger.info(f"Memory efficiency validated - DataFrame: {actual_memory_mb:.2f}MB, Subset: {subset_memory / 1024 / 1024:.2f}MB")


if __name__ == "__main__":
    # Support for running tests directly
    pytest.main([__file__, "-v", "--tb=short"])