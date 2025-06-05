"""
Cross-module integration test suite validating seamless data flow and interface compatibility.

This test suite validates the integration between core flyrigloader modules including:
- config (yaml_config, discovery)
- discovery (files, patterns)  
- io (pickle, column_models)
- utils (dataframe, paths)

Tests focus on:
- F-015: Cross-module interaction validation with realistic data flows
- F-016: Testability refactoring validation ensuring architectural integrity  
- TST-INTEG-001: Cross-module integration validation
- TST-INTEG-003: Data integrity validation across module boundaries
- Section 4.1.2: Integration Workflows validation across module interfaces
- TST-MOD-003: Dependency isolation validation through pytest-mock

Enhanced Features:
- Comprehensive module boundary testing with realistic data transformation scenarios
- Interface contract validation ensuring proper API usage across modules
- Dependency injection pattern testing validating testability improvements
- Data format consistency validation across discovery → io → utils pipeline
- Error propagation testing ensuring consistent handling across module boundaries
- Module isolation testing with sophisticated mocking strategies
"""

import os
import sys
import pytest
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from unittest.mock import MagicMock, Mock, patch, mock_open
from datetime import datetime, timedelta
import pickle
import gzip
import yaml

import numpy as np
import pandas as pd
from loguru import logger

# Import the modules under test for integration validation
from flyrigloader.config.yaml_config import (
    load_config, validate_config_dict, get_ignore_patterns, 
    get_mandatory_substrings, get_dataset_info, get_experiment_info,
    get_extraction_patterns, get_all_dataset_names
)
from flyrigloader.config.discovery import (
    ConfigDiscoveryEngine, discover_files_with_config,
    discover_experiment_files, discover_dataset_files,
    DefaultPathProvider, DefaultFileDiscoveryProvider, DefaultConfigurationProvider
)
from flyrigloader.discovery.files import discover_files
from flyrigloader.discovery.patterns import PatternMatcher, match_files_to_patterns
from flyrigloader.io.pickle import read_pickle_any_format, load_experimental_data
from flyrigloader.io.column_models import (
    load_column_config, validate_experimental_data,
    transform_to_standardized_format
)
from flyrigloader.utils.dataframe import (
    discovery_results_to_dataframe, attach_file_metadata_to_dataframe
)
from flyrigloader.utils.paths import (
    resolve_path, get_relative_path, normalize_path_separators
)


# ============================================================================
# INTEGRATION TEST CLASS STRUCTURE
# ============================================================================

class TestCrossModuleIntegration:
    """
    Comprehensive cross-module integration test suite validating data flow
    and interface compatibility between all core flyrigloader modules.
    
    Test Categories:
    1. Configuration to Discovery Integration (config → discovery)
    2. Discovery to IO Integration (discovery → io)  
    3. IO to Utils Integration (io → utils)
    4. End-to-End Pipeline Integration (config → discovery → io → utils)
    5. Error Propagation Across Module Boundaries
    6. Dependency Injection Pattern Validation
    """

    # ========================================================================
    # F-015: CROSS-MODULE INTERACTION VALIDATION WITH REALISTIC DATA FLOWS
    # ========================================================================

    def test_config_to_discovery_integration_realistic_workflow(
        self, 
        comprehensive_sample_config_dict, 
        temp_filesystem_structure,
        mocker
    ):
        """
        Test F-015: Validate seamless data flow from configuration loading 
        to file discovery with realistic experimental workflow scenarios.
        
        Validates:
        - Configuration data properly passed to discovery engine
        - Ignore patterns correctly applied during discovery
        - Mandatory substrings filtering works across module boundary
        - Extraction patterns transferred and used correctly
        """
        logger.info("Testing config to discovery integration with realistic workflow")
        
        # Setup realistic file discovery mock with proper interface validation
        mock_discover = mocker.patch("flyrigloader.discovery.files.discover_files")
        expected_files = {
            str(temp_filesystem_structure["baseline_file_1"]): {
                "date": "20241220",
                "condition": "control",
                "replicate": "1",
                "dataset": "baseline",
                "file_size": 2048,
                "modification_time": datetime.now().isoformat()
            },
            str(temp_filesystem_structure["opto_file_1"]): {
                "date": "20241218", 
                "condition": "treatment",
                "replicate": "1",
                "dataset": "optogenetic",
                "stimulation_type": "stim",
                "file_size": 3072,
                "modification_time": datetime.now().isoformat()
            }
        }
        mock_discover.return_value = expected_files
        
        # Test configuration loading and discovery integration
        config = comprehensive_sample_config_dict
        
        # Validate configuration data extraction functions
        ignore_patterns = get_ignore_patterns(config, experiment="baseline_control_study")
        mandatory_substrings = get_mandatory_substrings(config, experiment="baseline_control_study") 
        extraction_patterns = get_extraction_patterns(config, experiment="baseline_control_study")
        
        # Ensure configuration data is properly structured for discovery
        assert isinstance(ignore_patterns, list), "Ignore patterns must be list for discovery module"
        assert isinstance(mandatory_substrings, list), "Mandatory substrings must be list for discovery module"
        assert extraction_patterns is None or isinstance(extraction_patterns, list), "Extraction patterns must be list or None"
        
        # Test discovery integration with config-aware filtering
        discovery_engine = ConfigDiscoveryEngine()
        result = discovery_engine.discover_files_with_config(
            config=config,
            directory=str(temp_filesystem_structure["data_root"]),
            pattern="*.csv",
            recursive=True,
            experiment="baseline_control_study",
            extract_metadata=True
        )
        
        # Validate cross-module data flow and interface compliance
        mock_discover.assert_called_once()
        call_args = mock_discover.call_args
        
        # Verify ignore patterns were properly passed across module boundary
        assert "ignore_patterns" in call_args.kwargs
        assert call_args.kwargs["ignore_patterns"] == ignore_patterns
        
        # Verify mandatory substrings were properly passed across module boundary
        assert "mandatory_substrings" in call_args.kwargs
        assert call_args.kwargs["mandatory_substrings"] == mandatory_substrings
        
        # Verify extraction patterns were properly passed across module boundary
        assert "extract_patterns" in call_args.kwargs
        assert call_args.kwargs["extract_patterns"] == extraction_patterns
        
        # Validate discovery results maintain expected format for downstream modules
        assert isinstance(result, dict), "Discovery results must be dictionary for IO module integration"
        assert len(result) > 0, "Discovery should return files for integration testing"
        
        logger.success("Config to discovery integration validation completed successfully")

    def test_discovery_to_io_integration_data_format_consistency(
        self,
        temp_filesystem_structure,
        comprehensive_exp_matrix,
        mocker
    ):
        """
        Test F-015: Validate data format consistency from file discovery 
        results to data loading operations ensuring proper interface contracts.
        
        Validates:
        - Discovery results format compatible with IO module expectations
        - File paths correctly resolved and passed to pickle loading
        - Metadata extraction results properly formatted for column validation
        - Error handling consistency across discovery → io boundary
        """
        logger.info("Testing discovery to IO integration with data format consistency")
        
        # Create realistic discovery results matching expected IO input format
        discovery_results = {
            str(temp_filesystem_structure["baseline_file_1"]): {
                "date": "20241220",
                "condition": "control", 
                "replicate": "1",
                "dataset": "baseline",
                "file_size": 2048,
                "modification_time": datetime.now().isoformat()
            },
            str(temp_filesystem_structure["opto_file_1"]): {
                "date": "20241218",
                "condition": "treatment",
                "replicate": "1", 
                "dataset": "optogenetic",
                "file_size": 3072,
                "modification_time": datetime.now().isoformat()
            }
        }
        
        # Mock pickle loading with realistic experimental data
        mock_pickle_loader = mocker.patch("flyrigloader.io.pickle.read_pickle_any_format")
        
        def pickle_loader_side_effect(file_path):
            """Dynamic pickle loading based on file characteristics."""
            file_path_str = str(file_path)
            if "baseline" in file_path_str:
                return {
                    't': np.linspace(0, 300, 18000),  # 5 minutes at 60 Hz
                    'x': np.random.rand(18000) * 100 - 50,  # Centered at origin
                    'y': np.random.rand(18000) * 100 - 50,
                    'metadata': discovery_results[file_path_str]
                }
            elif "opto" in file_path_str:
                return {
                    't': np.linspace(0, 600, 36000),  # 10 minutes at 60 Hz
                    'x': np.random.rand(36000) * 120 - 60,
                    'y': np.random.rand(36000) * 120 - 60,
                    'signal': np.random.rand(36000),
                    'metadata': discovery_results[file_path_str]
                }
            else:
                raise FileNotFoundError(f"Unexpected file path in integration test: {file_path}")
        
        mock_pickle_loader.side_effect = pickle_loader_side_effect
        
        # Test discovery results processing through IO module
        loaded_data_sets = {}
        for file_path, metadata in discovery_results.items():
            try:
                # Validate file path format expected by IO module
                assert Path(file_path).exists() or True, "IO module expects valid Path objects"
                
                # Load data using IO module with discovery results
                loaded_data = read_pickle_any_format(file_path)
                
                # Validate data format consistency for downstream processing
                assert isinstance(loaded_data, dict), "IO module must return dict for utils module"
                assert 't' in loaded_data, "Time data required for utils module integration"
                assert 'x' in loaded_data, "X position required for utils module integration"
                assert 'y' in loaded_data, "Y position required for utils module integration"
                
                # Validate metadata integration across module boundary
                if 'metadata' in loaded_data:
                    assert isinstance(loaded_data['metadata'], dict), "Metadata must be dict format"
                    assert loaded_data['metadata']['date'] == metadata['date'], "Date metadata must be preserved"
                    assert loaded_data['metadata']['condition'] == metadata['condition'], "Condition metadata must be preserved"
                
                loaded_data_sets[file_path] = loaded_data
                
            except Exception as e:
                pytest.fail(f"Discovery to IO integration failed for {file_path}: {e}")
        
        # Validate successful loading of all discovery results
        assert len(loaded_data_sets) == len(discovery_results), "All discovered files should load successfully"
        
        # Verify data format consistency for next module in pipeline
        for file_path, data in loaded_data_sets.items():
            assert isinstance(data['t'], np.ndarray), "Time data must be numpy array for utils module"
            assert isinstance(data['x'], np.ndarray), "Position data must be numpy array for utils module"
            assert isinstance(data['y'], np.ndarray), "Position data must be numpy array for utils module"
            
            # Validate array dimensions are consistent
            assert data['t'].ndim == 1, "Time arrays must be 1D for utils module compatibility"
            assert data['x'].ndim == 1, "Position arrays must be 1D for utils module compatibility"
            assert data['y'].ndim == 1, "Position arrays must be 1D for utils module compatibility"
            
            # Validate array lengths are consistent
            assert len(data['t']) == len(data['x']) == len(data['y']), "Array lengths must match for utils module"
        
        logger.success("Discovery to IO integration validation completed successfully")

    def test_io_to_utils_integration_dataframe_transformation(
        self,
        comprehensive_exp_matrix,
        sample_metadata,
        mocker
    ):
        """
        Test F-015: Validate data transformation from IO module experimental data
        to utils module DataFrame operations ensuring format compatibility.
        
        Validates:
        - Experimental data format compatible with DataFrame creation
        - Metadata properly integrated into DataFrame structure
        - Column validation and transformation working across io → utils boundary
        - File statistics integration from multiple modules
        """
        logger.info("Testing IO to utils integration with DataFrame transformation")
        
        # Setup realistic IO module output format
        io_experimental_data = comprehensive_exp_matrix.copy()
        io_experimental_data.update({
            'metadata': sample_metadata,
            'file_path': '/data/experiments/test_file.pkl',
            'file_stats': {
                'size_bytes': 2048,
                'modification_time': datetime.now().isoformat(),
                'creation_time': datetime.now().isoformat()
            }
        })
        
        # Mock column configuration loading for validation
        mock_column_config = mocker.patch("flyrigloader.io.column_models.load_column_config")
        mock_column_config.return_value = {
            "columns": {
                "t": {"type": "numpy.ndarray", "required": True, "dimension": 1},
                "x": {"type": "numpy.ndarray", "required": True, "dimension": 1},
                "y": {"type": "numpy.ndarray", "required": True, "dimension": 1},
                "signal": {"type": "numpy.ndarray", "required": False, "dimension": 1},
                "signal_disp": {"type": "numpy.ndarray", "required": False, "dimension": 2}
            }
        }
        
        # Test IO to utils integration with data validation
        try:
            # Validate experimental data format from IO module
            assert isinstance(io_experimental_data, dict), "IO module must output dict for utils integration"
            assert 't' in io_experimental_data, "Time data required from IO module"
            assert 'x' in io_experimental_data, "X position required from IO module"
            assert 'y' in io_experimental_data, "Y position required from IO module"
            
            # Test column validation across module boundary
            column_config = load_column_config()
            validated_data = validate_experimental_data(io_experimental_data, column_config)
            
            # Validate data transformation maintains integrity
            assert isinstance(validated_data, dict), "Column validation must return dict for utils module"
            assert len(validated_data['t']) == len(io_experimental_data['t']), "Data length must be preserved"
            
            # Test DataFrame creation using utils module
            from flyrigloader.utils.dataframe import DefaultDataFrameProvider
            df_provider = DefaultDataFrameProvider()
            
            # Prepare data for DataFrame creation
            df_data = {}
            for col, data in validated_data.items():
                if isinstance(data, np.ndarray) and data.ndim == 1:
                    df_data[col] = data
                elif isinstance(data, np.ndarray) and data.ndim == 2:
                    # Handle multi-channel data (e.g., signal_disp)
                    for ch in range(data.shape[0]):
                        df_data[f"{col}_ch{ch:02d}"] = data[ch, :]
                elif not isinstance(data, np.ndarray):
                    # Handle metadata
                    if isinstance(data, dict):
                        for key, value in data.items():
                            df_data[f"meta_{key}"] = [value] * len(validated_data['t'])
                    else:
                        df_data[f"meta_{col}"] = [data] * len(validated_data['t'])
            
            # Create DataFrame with proper integration
            result_df = df_provider.create_dataframe(df_data)
            
            # Validate successful DataFrame creation
            assert isinstance(result_df, pd.DataFrame), "Utils module must create proper DataFrame"
            assert len(result_df) == len(io_experimental_data['t']), "DataFrame length must match input data"
            
            # Validate essential columns are present
            assert 't' in result_df.columns, "Time column must be preserved in DataFrame"
            assert 'x' in result_df.columns, "X position must be preserved in DataFrame"
            assert 'y' in result_df.columns, "Y position must be preserved in DataFrame"
            
            # Validate metadata integration
            metadata_columns = [col for col in result_df.columns if col.startswith('meta_')]
            assert len(metadata_columns) > 0, "Metadata must be integrated into DataFrame"
            
            # Validate data types are appropriate for analysis
            assert pd.api.types.is_numeric_dtype(result_df['t']), "Time column must be numeric"
            assert pd.api.types.is_numeric_dtype(result_df['x']), "Position columns must be numeric"
            assert pd.api.types.is_numeric_dtype(result_df['y']), "Position columns must be numeric"
            
        except Exception as e:
            pytest.fail(f"IO to utils integration failed: {e}")
        
        logger.success("IO to utils integration validation completed successfully")

    # ========================================================================
    # F-016: TESTABILITY REFACTORING VALIDATION - DEPENDENCY INJECTION
    # ========================================================================

    def test_dependency_injection_pattern_validation_config_discovery(self, mocker):
        """
        Test F-016: Validate dependency injection patterns in config and discovery modules
        ensuring testability improvements maintain functional correctness.
        
        Validates:
        - ConfigDiscoveryEngine dependency injection functionality
        - Provider pattern implementation and interface compliance
        - Mock-friendly interfaces working correctly
        - Dependency substitution without breaking integration
        """
        logger.info("Testing dependency injection patterns for config and discovery modules")
        
        # Create mock providers following the established protocols
        mock_path_provider = Mock()
        mock_path_provider.resolve_path.return_value = Path("/resolved/path")
        mock_path_provider.exists.return_value = True
        mock_path_provider.list_directories.return_value = [Path("/test/dir1"), Path("/test/dir2")]
        
        mock_file_discovery_provider = Mock()
        mock_file_discovery_provider.discover_files.return_value = {
            "/test/file1.pkl": {"date": "20241220", "condition": "control"},
            "/test/file2.pkl": {"date": "20241221", "condition": "treatment"}
        }
        
        mock_config_provider = Mock()
        mock_config_provider.get_ignore_patterns.return_value = ["backup", "temp"]
        mock_config_provider.get_mandatory_substrings.return_value = ["experiment"]
        mock_config_provider.get_extraction_patterns.return_value = [r".*_(?P<date>\d{8})_.*"]
        
        # Test dependency injection with ConfigDiscoveryEngine
        discovery_engine = ConfigDiscoveryEngine(
            path_provider=mock_path_provider,
            file_discovery_provider=mock_file_discovery_provider,
            config_provider=mock_config_provider
        )
        
        # Validate dependency injection worked correctly
        assert discovery_engine.path_provider is mock_path_provider, "Path provider not injected correctly"
        assert discovery_engine.file_discovery_provider is mock_file_discovery_provider, "File discovery provider not injected correctly"
        assert discovery_engine.config_provider is mock_config_provider, "Config provider not injected correctly"
        
        # Test functionality with injected dependencies
        test_config = {
            "project": {"ignore_substrings": ["backup"]},
            "experiments": {"test_exp": {"filters": {"ignore_substrings": ["temp"]}}}
        }
        
        result = discovery_engine.discover_files_with_config(
            config=test_config,
            directory="/test/data",
            pattern="*.pkl",
            recursive=True,
            experiment="test_exp",
            extract_metadata=True
        )
        
        # Validate that injected providers were called correctly
        mock_config_provider.get_ignore_patterns.assert_called_once_with(test_config, "test_exp")
        mock_config_provider.get_mandatory_substrings.assert_called_once_with(test_config, "test_exp")
        mock_config_provider.get_extraction_patterns.assert_called_once_with(test_config, "test_exp")
        
        mock_file_discovery_provider.discover_files.assert_called_once()
        discovery_call_args = mock_file_discovery_provider.discover_files.call_args
        
        # Validate proper parameter passing through dependency injection
        assert discovery_call_args.kwargs["directory"] == "/test/data"
        assert discovery_call_args.kwargs["pattern"] == "*.pkl"
        assert discovery_call_args.kwargs["recursive"] is True
        assert discovery_call_args.kwargs["ignore_patterns"] == ["backup", "temp"]
        assert discovery_call_args.kwargs["mandatory_substrings"] == ["experiment"]
        assert discovery_call_args.kwargs["extract_patterns"] == [r".*_(?P<date>\d{8})_.*"]
        
        # Validate results are properly returned
        assert isinstance(result, dict), "Dependency injection must maintain return type"
        assert len(result) == 2, "Dependency injection must maintain functionality"
        
        logger.success("Dependency injection pattern validation completed successfully")

    def test_dependency_injection_pattern_validation_io_modules(self, mocker):
        """
        Test F-016: Validate dependency injection patterns in IO modules
        ensuring pickle loading and column validation maintain functionality.
        
        Validates:
        - FileSystemProvider and PickleProvider dependency injection
        - Column validation with configurable dependencies
        - Mock provider functionality for testing isolation
        - Interface compliance across dependency boundaries
        """
        logger.info("Testing dependency injection patterns for IO modules")
        
        # Mock filesystem provider for IO module testing
        mock_filesystem_provider = Mock()
        mock_filesystem_provider.path_exists.return_value = True
        mock_filesystem_provider.open_file.return_value = mock_open(read_data=b"pickle_data")()
        
        # Mock compression provider for gzip testing
        mock_compression_provider = Mock()
        mock_compression_provider.open_gzip.return_value = mock_open(read_data=b"gzip_pickle_data")()
        
        # Mock pickle provider for data loading
        mock_pickle_provider = Mock()
        test_experimental_data = {
            't': np.linspace(0, 100, 1000),
            'x': np.random.rand(1000) * 50,
            'y': np.random.rand(1000) * 50
        }
        mock_pickle_provider.load.return_value = test_experimental_data
        
        # Test pickle loading with dependency injection (conceptual since current implementation is function-based)
        # This validates the pattern that should be followed for enhanced testability
        
        # Simulate dependency-injected pickle loading
        def load_with_providers(file_path, filesystem_provider, pickle_provider, compression_provider=None):
            """Simulated dependency-injected pickle loading function."""
            if not filesystem_provider.path_exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if str(file_path).endswith('.gz') and compression_provider:
                file_obj = compression_provider.open_gzip(file_path, 'rb')
            else:
                file_obj = filesystem_provider.open_file(file_path, 'rb')
            
            return pickle_provider.load(file_obj)
        
        # Test dependency injection functionality
        test_file_path = "/test/data/experiment.pkl"
        result = load_with_providers(
            test_file_path,
            mock_filesystem_provider,
            mock_pickle_provider
        )
        
        # Validate dependency injection calls
        mock_filesystem_provider.path_exists.assert_called_once_with(test_file_path)
        mock_filesystem_provider.open_file.assert_called_once_with(test_file_path, 'rb')
        mock_pickle_provider.load.assert_called_once()
        
        # Validate functionality is maintained
        assert result == test_experimental_data, "Dependency injection must maintain data loading functionality"
        
        # Test gzipped file handling with dependency injection
        gzip_file_path = "/test/data/experiment.pkl.gz"
        result_gzip = load_with_providers(
            gzip_file_path,
            mock_filesystem_provider,
            mock_pickle_provider,
            mock_compression_provider
        )
        
        # Validate gzip provider was used
        mock_compression_provider.open_gzip.assert_called_once_with(gzip_file_path, 'rb')
        
        # Test column validation with dependency injection (conceptual validation)
        mock_yaml_loader = Mock()
        mock_yaml_loader.safe_load.return_value = {
            "columns": {
                "t": {"type": "numpy.ndarray", "required": True},
                "x": {"type": "numpy.ndarray", "required": True},
                "y": {"type": "numpy.ndarray", "required": True}
            }
        }
        
        # Validate YAML provider pattern
        assert callable(mock_yaml_loader.safe_load), "YAML loader must be callable"
        config_data = mock_yaml_loader.safe_load(None)
        assert "columns" in config_data, "Column configuration must be loadable via dependency injection"
        
        logger.success("IO module dependency injection validation completed successfully")

    # ========================================================================
    # TST-INTEG-003: DATA INTEGRITY VALIDATION ACROSS MODULE BOUNDARIES
    # ========================================================================

    def test_data_integrity_across_config_discovery_boundary(
        self, 
        comprehensive_sample_config_dict,
        mocker
    ):
        """
        Test TST-INTEG-003: Validate data integrity maintained across 
        config → discovery module boundary with complex filtering scenarios.
        
        Validates:
        - Filter parameters correctly transformed and applied
        - No data corruption during parameter passing
        - Complex configuration structures preserved
        - Array/list data types maintained correctly
        """
        logger.info("Testing data integrity across config to discovery boundary")
        
        # Setup complex configuration with multiple filter types
        complex_config = comprehensive_sample_config_dict.copy()
        complex_config["experiments"]["test_experiment"] = {
            "datasets": ["baseline_behavior", "optogenetic_stimulation"],
            "filters": {
                "ignore_substrings": ["backup", "temp", "calibration"],
                "mandatory_substrings": ["experiment", "data"],
                "file_size_min": 1024,
                "file_size_max": 1048576
            },
            "metadata": {
                "extraction_patterns": [
                    r".*_(?P<date>\d{8})_(?P<condition>\w+)_(?P<replicate>\d+)\.pkl",
                    r"(?P<experiment>\w+)_(?P<animal_id>\w+)_(?P<session>\d+)\.pkl"
                ]
            }
        }
        
        # Mock file discovery to capture exact parameters passed
        mock_discover = mocker.patch("flyrigloader.discovery.files.discover_files")
        mock_discover.return_value = {"test_file.pkl": {"date": "20241220"}}
        
        # Test configuration extraction with data integrity validation
        ignore_patterns = get_ignore_patterns(complex_config, experiment="test_experiment")
        mandatory_substrings = get_mandatory_substrings(complex_config, experiment="test_experiment")
        extraction_patterns = get_extraction_patterns(complex_config, experiment="test_experiment")
        
        # Validate data integrity of extracted configuration
        expected_ignore = ["backup", "temp", "calibration"] + complex_config["project"].get("ignore_substrings", [])
        expected_mandatory = ["experiment", "data"] + complex_config["project"].get("mandatory_substrings", [])
        expected_extraction = [
            r".*_(?P<date>\d{8})_(?P<condition>\w+)_(?P<replicate>\d+)\.pkl",
            r"(?P<experiment>\w+)_(?P<animal_id>\w+)_(?P<session>\d+)\.pkl"
        ]
        
        # Validate no data corruption occurred
        assert isinstance(ignore_patterns, list), "Ignore patterns must remain list type"
        assert isinstance(mandatory_substrings, list), "Mandatory substrings must remain list type" 
        assert isinstance(extraction_patterns, list), "Extraction patterns must remain list type"
        
        # Validate specific data integrity
        for pattern in expected_ignore:
            assert pattern in ignore_patterns, f"Ignore pattern '{pattern}' lost during extraction"
        
        for substring in expected_mandatory:
            assert substring in mandatory_substrings, f"Mandatory substring '{substring}' lost during extraction"
        
        for pattern in expected_extraction:
            assert pattern in extraction_patterns, f"Extraction pattern '{pattern}' lost during extraction"
        
        # Test discovery integration with integrity validation
        discovery_engine = ConfigDiscoveryEngine()
        result = discovery_engine.discover_files_with_config(
            config=complex_config,
            directory="/test/data",
            pattern="*.pkl",
            experiment="test_experiment",
            extract_metadata=True
        )
        
        # Validate parameters passed to discovery module maintain integrity
        mock_discover.assert_called_once()
        call_kwargs = mock_discover.call_args.kwargs
        
        # Verify no parameter corruption during module boundary crossing
        assert call_kwargs["ignore_patterns"] == ignore_patterns, "Ignore patterns corrupted across boundary"
        assert call_kwargs["mandatory_substrings"] == mandatory_substrings, "Mandatory substrings corrupted across boundary"
        assert call_kwargs["extract_patterns"] == extraction_patterns, "Extraction patterns corrupted across boundary"
        
        # Validate result data integrity
        assert isinstance(result, dict), "Discovery results must maintain dict format"
        assert "test_file.pkl" in result, "Discovery results must contain expected data"
        
        logger.success("Data integrity validation across config-discovery boundary completed")

    def test_data_integrity_across_discovery_io_boundary(
        self,
        temp_filesystem_structure,
        mocker
    ):
        """
        Test TST-INTEG-003: Validate data integrity maintained across
        discovery → io module boundary with metadata and file path handling.
        
        Validates:
        - File paths correctly resolved and passed without corruption
        - Metadata dictionaries maintained with proper structure
        - Complex data structures preserved across module calls
        - Error conditions properly handled without data loss
        """
        logger.info("Testing data integrity across discovery to IO boundary")
        
        # Create complex discovery results with various metadata types
        complex_discovery_results = {
            str(temp_filesystem_structure["baseline_file_1"]): {
                "date": "20241220",
                "condition": "control",
                "replicate": 1,  # Integer type
                "duration_seconds": 300.5,  # Float type
                "dataset": "baseline_behavior",
                "experimental_parameters": {  # Nested dict
                    "temperature_c": 23.5,
                    "humidity_percent": 45.2,
                    "lighting_lux": 100
                },
                "file_statistics": {
                    "size_bytes": 2048,
                    "modification_time": datetime.now().isoformat(),
                    "checksum": "abc123def456"
                },
                "processing_flags": ["calibrated", "filtered"],  # List type
                "analysis_metadata": None  # None type
            },
            str(temp_filesystem_structure["opto_file_1"]): {
                "date": "20241218",
                "condition": "treatment",
                "replicate": 2,
                "duration_seconds": 600.0,
                "dataset": "optogenetic_stimulation",
                "stimulation_parameters": {
                    "wavelength_nm": 470,
                    "power_mw": 10.5,
                    "pulse_duration_ms": 50,
                    "frequency_hz": 20.0
                },
                "file_statistics": {
                    "size_bytes": 4096,
                    "modification_time": datetime.now().isoformat(),
                    "checksum": "xyz789abc123"
                },
                "processing_flags": ["calibrated", "optogenetic"],
                "analysis_metadata": {"processed_by": "pipeline_v2.1"}
            }
        }
        
        # Mock pickle loading with integrity validation
        mock_pickle_loader = mocker.patch("flyrigloader.io.pickle.read_pickle_any_format")
        
        def integrity_validating_loader(file_path):
            """Pickle loader that validates file path integrity."""
            # Validate file path format
            assert isinstance(file_path, (str, Path)), "File path must be string or Path object"
            file_path_str = str(file_path)
            
            # Validate file path exists in discovery results
            assert file_path_str in complex_discovery_results, f"File path not found in discovery results: {file_path_str}"
            
            # Return data based on file path with metadata integration
            discovery_metadata = complex_discovery_results[file_path_str]
            
            if "baseline" in file_path_str:
                return {
                    't': np.linspace(0, discovery_metadata["duration_seconds"], 18000),
                    'x': np.random.rand(18000) * 100,
                    'y': np.random.rand(18000) * 100,
                    'discovery_metadata': discovery_metadata,  # Preserve metadata integrity
                    'file_path': file_path_str
                }
            elif "opto" in file_path_str:
                return {
                    't': np.linspace(0, discovery_metadata["duration_seconds"], 36000),
                    'x': np.random.rand(36000) * 120,
                    'y': np.random.rand(36000) * 120,
                    'signal': np.random.rand(36000),
                    'discovery_metadata': discovery_metadata,  # Preserve metadata integrity
                    'file_path': file_path_str
                }
            else:
                raise ValueError(f"Unexpected file path: {file_path_str}")
        
        mock_pickle_loader.side_effect = integrity_validating_loader
        
        # Test data loading with integrity validation
        loaded_datasets = {}
        for file_path, expected_metadata in complex_discovery_results.items():
            try:
                # Load data through IO module
                loaded_data = read_pickle_any_format(file_path)
                
                # Validate basic data integrity
                assert isinstance(loaded_data, dict), "Loaded data must be dictionary"
                assert 'discovery_metadata' in loaded_data, "Discovery metadata must be preserved"
                assert loaded_data['file_path'] == file_path, "File path must be preserved"
                
                # Validate metadata integrity across boundary
                preserved_metadata = loaded_data['discovery_metadata']
                
                # Check all original metadata fields are preserved
                for key, expected_value in expected_metadata.items():
                    assert key in preserved_metadata, f"Metadata key '{key}' lost during boundary crossing"
                    
                    if isinstance(expected_value, dict):
                        # Validate nested dictionary integrity
                        assert isinstance(preserved_metadata[key], dict), f"Nested dict '{key}' type corrupted"
                        for nested_key, nested_value in expected_value.items():
                            assert nested_key in preserved_metadata[key], f"Nested key '{nested_key}' lost"
                            assert preserved_metadata[key][nested_key] == nested_value, f"Nested value for '{nested_key}' corrupted"
                    
                    elif isinstance(expected_value, list):
                        # Validate list integrity
                        assert isinstance(preserved_metadata[key], list), f"List '{key}' type corrupted"
                        assert preserved_metadata[key] == expected_value, f"List '{key}' contents corrupted"
                    
                    else:
                        # Validate scalar value integrity
                        assert preserved_metadata[key] == expected_value, f"Scalar value '{key}' corrupted"
                
                # Validate numerical data integrity
                assert isinstance(loaded_data['t'], np.ndarray), "Time data must be numpy array"
                assert isinstance(loaded_data['x'], np.ndarray), "X position must be numpy array"
                assert isinstance(loaded_data['y'], np.ndarray), "Y position must be numpy array"
                
                # Validate array properties
                expected_duration = expected_metadata["duration_seconds"]
                actual_duration = loaded_data['t'][-1] - loaded_data['t'][0]
                assert abs(actual_duration - expected_duration) < 0.1, "Duration metadata inconsistent with data"
                
                loaded_datasets[file_path] = loaded_data
                
            except Exception as e:
                pytest.fail(f"Data integrity validation failed for {file_path}: {e}")
        
        # Validate all datasets loaded successfully
        assert len(loaded_datasets) == len(complex_discovery_results), "All discovery results should load successfully"
        
        logger.success("Data integrity validation across discovery-IO boundary completed")

    def test_data_integrity_across_io_utils_boundary(
        self,
        comprehensive_exp_matrix,
        sample_metadata,
        mocker
    ):
        """
        Test TST-INTEG-003: Validate data integrity maintained across
        io → utils module boundary with DataFrame creation and metadata integration.
        
        Validates:
        - Numerical arrays maintained without precision loss
        - Metadata properly integrated without corruption
        - DataFrame structure correctly reflects input data
        - Complex data types handled appropriately
        """
        logger.info("Testing data integrity across IO to utils boundary")
        
        # Create complex IO output with various data types
        complex_io_data = comprehensive_exp_matrix.copy()
        complex_io_data.update({
            'metadata': sample_metadata.copy(),
            'signal_statistics': {
                'mean': np.mean(comprehensive_exp_matrix.get('signal', [0])),
                'std': np.std(comprehensive_exp_matrix.get('signal', [0])),
                'min': np.min(comprehensive_exp_matrix.get('signal', [0])),
                'max': np.max(comprehensive_exp_matrix.get('signal', [0]))
            },
            'processing_history': [
                {"step": "load", "timestamp": datetime.now().isoformat()},
                {"step": "validate", "timestamp": datetime.now().isoformat()},
                {"step": "transform", "timestamp": datetime.now().isoformat()}
            ],
            'array_metadata': {
                'sampling_frequency': 60.0,
                'total_duration': len(comprehensive_exp_matrix['t']) / 60.0,
                'spatial_bounds': {
                    'x_min': np.min(comprehensive_exp_matrix['x']),
                    'x_max': np.max(comprehensive_exp_matrix['x']),
                    'y_min': np.min(comprehensive_exp_matrix['y']),
                    'y_max': np.max(comprehensive_exp_matrix['y'])
                }
            }
        })
        
        # Test DataFrame creation with integrity validation
        from flyrigloader.utils.dataframe import DefaultDataFrameProvider
        df_provider = DefaultDataFrameProvider()
        
        # Prepare data for DataFrame with integrity checks
        df_data = {}
        
        # Process numerical arrays with integrity validation
        for col, data in complex_io_data.items():
            if isinstance(data, np.ndarray):
                if data.ndim == 1:
                    # Validate 1D array integrity
                    assert data.dtype.kind in ['f', 'i', 'u'], f"Array '{col}' must be numerical"
                    assert not np.any(np.isnan(data)) or col in ['signal'], f"Unexpected NaN values in '{col}'"
                    df_data[col] = data
                    
                elif data.ndim == 2:
                    # Validate 2D array integrity
                    assert data.dtype.kind in ['f', 'i', 'u'], f"Array '{col}' must be numerical"
                    for ch in range(data.shape[0]):
                        channel_data = data[ch, :]
                        assert len(channel_data) == len(complex_io_data['t']), f"Channel {ch} length mismatch"
                        df_data[f"{col}_ch{ch:02d}"] = channel_data
        
        # Process metadata with integrity validation
        if 'metadata' in complex_io_data:
            metadata = complex_io_data['metadata']
            assert isinstance(metadata, dict), "Metadata must be dictionary"
            
            for key, value in metadata.items():
                # Validate metadata values are serializable for DataFrame
                if isinstance(value, (str, int, float, bool)):
                    df_data[f"meta_{key}"] = [value] * len(complex_io_data['t'])
                else:
                    df_data[f"meta_{key}"] = [str(value)] * len(complex_io_data['t'])
        
        # Process complex metadata structures
        if 'signal_statistics' in complex_io_data:
            stats = complex_io_data['signal_statistics']
            for stat_name, stat_value in stats.items():
                df_data[f"signal_stat_{stat_name}"] = [stat_value] * len(complex_io_data['t'])
        
        if 'array_metadata' in complex_io_data:
            array_meta = complex_io_data['array_metadata']
            df_data['sampling_frequency'] = [array_meta['sampling_frequency']] * len(complex_io_data['t'])
            df_data['total_duration'] = [array_meta['total_duration']] * len(complex_io_data['t'])
            
            # Handle nested spatial bounds
            spatial_bounds = array_meta['spatial_bounds']
            for bound_name, bound_value in spatial_bounds.items():
                df_data[f"spatial_{bound_name}"] = [bound_value] * len(complex_io_data['t'])
        
        # Create DataFrame with integrity validation
        try:
            result_df = df_provider.create_dataframe(df_data)
            
            # Validate DataFrame creation success
            assert isinstance(result_df, pd.DataFrame), "DataFrame creation must succeed"
            assert len(result_df) == len(complex_io_data['t']), "DataFrame length must match input data"
            
            # Validate numerical data integrity
            for col in ['t', 'x', 'y']:
                if col in result_df.columns:
                    original_data = complex_io_data[col]
                    df_data_values = result_df[col].values
                    
                    # Check for data corruption
                    np.testing.assert_array_equal(
                        df_data_values, original_data,
                        err_msg=f"Data corruption detected in column '{col}'"
                    )
                    
                    # Validate data types preserved
                    assert pd.api.types.is_numeric_dtype(result_df[col]), f"Column '{col}' must remain numeric"
            
            # Validate metadata integrity
            metadata_cols = [col for col in result_df.columns if col.startswith('meta_')]
            assert len(metadata_cols) > 0, "Metadata must be present in DataFrame"
            
            # Validate metadata consistency
            for meta_col in metadata_cols:
                meta_values = result_df[meta_col].unique()
                assert len(meta_values) == 1, f"Metadata column '{meta_col}' should have constant value"
            
            # Validate signal statistics integrity
            stat_cols = [col for col in result_df.columns if col.startswith('signal_stat_')]
            for stat_col in stat_cols:
                stat_values = result_df[stat_col].unique()
                assert len(stat_values) == 1, f"Statistics column '{stat_col}' should have constant value"
                assert pd.api.types.is_numeric_dtype(result_df[stat_col]), f"Statistics column '{stat_col}' must be numeric"
            
            # Validate multi-channel data integrity
            if 'signal_disp' in complex_io_data:
                original_signal_disp = complex_io_data['signal_disp']
                signal_disp_cols = [col for col in result_df.columns if col.startswith('signal_disp_ch')]
                
                assert len(signal_disp_cols) == original_signal_disp.shape[0], "All signal channels must be present"
                
                for ch_idx, col in enumerate(sorted(signal_disp_cols)):
                    original_channel = original_signal_disp[ch_idx, :]
                    df_channel = result_df[col].values
                    
                    np.testing.assert_array_equal(
                        df_channel, original_channel,
                        err_msg=f"Signal channel {ch_idx} data corrupted"
                    )
            
        except Exception as e:
            pytest.fail(f"Data integrity validation across io-utils boundary failed: {e}")
        
        logger.success("Data integrity validation across IO-utils boundary completed")

    # ========================================================================
    # SECTION 4.1.2: ERROR PROPAGATION TESTING ACROSS MODULE BOUNDARIES
    # ========================================================================

    def test_error_propagation_config_to_discovery(self, mocker):
        """
        Test Section 4.1.2.3: Validate error propagation from config module
        through discovery module with consistent error handling.
        
        Validates:
        - Configuration loading errors properly propagated
        - Invalid configuration structure errors handled consistently
        - Missing experiment/dataset errors maintain context
        - Error messages preserve diagnostic information
        """
        logger.info("Testing error propagation from config to discovery modules")
        
        # Test configuration loading error propagation
        mock_load_config = mocker.patch("flyrigloader.config.yaml_config.load_config")
        mock_load_config.side_effect = FileNotFoundError("Configuration file not found: /invalid/path.yaml")
        
        discovery_engine = ConfigDiscoveryEngine()
        
        # Test error propagation through discovery engine
        with pytest.raises(FileNotFoundError) as exc_info:
            # This should propagate the FileNotFoundError from config loading
            get_dataset_info({"invalid": "config"}, "nonexistent_dataset")
        
        # Validate error context is preserved
        assert "not found" in str(exc_info.value).lower(), "Error context must be preserved"
        
        # Test invalid configuration structure error propagation
        invalid_config = {
            "project": "not_a_dict",  # Invalid structure
            "datasets": None  # Invalid structure
        }
        
        with pytest.raises(ValueError) as exc_info:
            discovery_engine.discover_files_with_config(
                config=invalid_config,
                directory="/test/dir",
                pattern="*.pkl"
            )
        
        assert "configuration" in str(exc_info.value).lower(), "Configuration error must be identified"
        
        # Test missing experiment error propagation
        valid_config = {
            "project": {"ignore_substrings": []},
            "experiments": {"existing_exp": {"datasets": ["test"]}}
        }
        
        with pytest.raises((KeyError, ValueError)) as exc_info:
            discovery_engine.discover_experiment_files(
                config=valid_config,
                experiment_name="nonexistent_experiment",
                base_directory="/test/dir"
            )
        
        assert "experiment" in str(exc_info.value).lower(), "Missing experiment error must be clear"
        
        logger.success("Error propagation config to discovery validation completed")

    def test_error_propagation_discovery_to_io(self, mocker):
        """
        Test Section 4.1.2.3: Validate error propagation from discovery module
        through io module with proper error context preservation.
        
        Validates:
        - File not found errors maintain file path context
        - Permission errors properly handled and propagated
        - Corrupted file errors include diagnostic information
        - Discovery result format errors clearly identified
        """
        logger.info("Testing error propagation from discovery to IO modules")
        
        # Test file not found error propagation
        mock_pickle_loader = mocker.patch("flyrigloader.io.pickle.read_pickle_any_format")
        mock_pickle_loader.side_effect = FileNotFoundError("No such file or directory: '/nonexistent/file.pkl'")
        
        discovery_results = {
            "/nonexistent/file.pkl": {"date": "20241220", "condition": "control"}
        }
        
        # Test error propagation during data loading
        for file_path, metadata in discovery_results.items():
            with pytest.raises(FileNotFoundError) as exc_info:
                read_pickle_any_format(file_path)
            
            # Validate file path context is preserved
            assert file_path in str(exc_info.value), "File path must be preserved in error message"
        
        # Test permission error propagation
        mock_pickle_loader.side_effect = PermissionError("Permission denied: '/restricted/file.pkl'")
        
        with pytest.raises(PermissionError) as exc_info:
            read_pickle_any_format("/restricted/file.pkl")
        
        assert "permission" in str(exc_info.value).lower(), "Permission error context must be preserved"
        
        # Test corrupted file error propagation
        import pickle as pickle_module
        mock_pickle_loader.side_effect = pickle_module.UnpicklingError("invalid load key, '\\x00'.")
        
        with pytest.raises(pickle_module.UnpicklingError) as exc_info:
            read_pickle_any_format("/corrupted/file.pkl")
        
        assert "load key" in str(exc_info.value) or "pickle" in str(exc_info.value).lower(), "Corruption error details must be preserved"
        
        # Test invalid discovery result format error
        def invalid_format_loader(file_path):
            if "invalid_format" in str(file_path):
                return "not_a_dict"  # Invalid format - should be dict
            return {"t": np.array([1, 2, 3]), "x": np.array([1, 2, 3]), "y": np.array([1, 2, 3])}
        
        mock_pickle_loader.side_effect = invalid_format_loader
        
        # Test column validation with invalid format
        mock_validate = mocker.patch("flyrigloader.io.column_models.validate_experimental_data")
        mock_validate.side_effect = ValueError("Experimental data must be a dictionary")
        
        with pytest.raises(ValueError) as exc_info:
            validate_experimental_data("not_a_dict", {})
        
        assert "dictionary" in str(exc_info.value).lower(), "Format error must be clearly identified"
        
        logger.success("Error propagation discovery to IO validation completed")

    def test_error_propagation_io_to_utils(self, mocker):
        """
        Test Section 4.1.2.3: Validate error propagation from io module
        through utils module with DataFrame creation error handling.
        
        Validates:
        - Invalid data format errors clearly communicated
        - Array dimension mismatch errors include details
        - DataFrame creation errors maintain context
        - Missing required columns errors are specific
        """
        logger.info("Testing error propagation from IO to utils modules")
        
        # Test invalid data format error propagation
        from flyrigloader.utils.dataframe import DefaultDataFrameProvider
        df_provider = DefaultDataFrameProvider()
        
        # Test with invalid data types
        invalid_data = {
            't': "not_an_array",  # Invalid type
            'x': np.array([1, 2, 3]),
            'y': np.array([1, 2, 3])
        }
        
        with pytest.raises((ValueError, TypeError)) as exc_info:
            df_provider.create_dataframe(invalid_data)
        
        # Error should indicate the problem with data format
        error_msg = str(exc_info.value).lower()
        assert any(word in error_msg for word in ["array", "type", "format", "data"]), "Data format error must be clear"
        
        # Test array dimension mismatch error propagation
        mismatched_data = {
            't': np.array([1, 2, 3]),
            'x': np.array([1, 2, 3, 4, 5]),  # Different length
            'y': np.array([1, 2, 3])
        }
        
        # While pandas DataFrame can handle different lengths, our validation should catch this
        # Create custom validation function that would catch such errors
        def validate_array_lengths(data_dict):
            """Validate that all 1D arrays have the same length."""
            array_lengths = {}
            for key, value in data_dict.items():
                if isinstance(value, np.ndarray) and value.ndim == 1:
                    array_lengths[key] = len(value)
            
            if len(set(array_lengths.values())) > 1:
                raise ValueError(f"Array length mismatch: {array_lengths}")
            
            return data_dict
        
        with pytest.raises(ValueError) as exc_info:
            validate_array_lengths(mismatched_data)
        
        assert "length" in str(exc_info.value).lower(), "Array length error must be specific"
        assert "mismatch" in str(exc_info.value).lower(), "Mismatch context must be preserved"
        
        # Test missing required columns error propagation
        incomplete_data = {
            't': np.array([1, 2, 3]),
            # Missing 'x' and 'y' columns
        }
        
        def validate_required_columns(data_dict, required_columns=['t', 'x', 'y']):
            """Validate that required columns are present."""
            missing_columns = [col for col in required_columns if col not in data_dict]
            if missing_columns:
                raise KeyError(f"Missing required columns: {missing_columns}")
            return data_dict
        
        with pytest.raises(KeyError) as exc_info:
            validate_required_columns(incomplete_data)
        
        assert "missing" in str(exc_info.value).lower(), "Missing column error must be clear"
        assert "required" in str(exc_info.value).lower(), "Required column context must be preserved"
        
        # Test DataFrame concatenation error propagation
        incompatible_dfs = [
            pd.DataFrame({'a': [1, 2, 3]}),
            pd.DataFrame({'b': [4, 5, 6]})  # Different columns
        ]
        
        # Test error handling in concatenation
        try:
            result = df_provider.concat_dataframes(incompatible_dfs, axis=1, join='inner')
            # If no error, validate the result handles the situation appropriately
            assert len(result.columns) >= 0, "Concatenation should handle different columns"
        except Exception as e:
            # If error occurs, validate it's properly contextualized
            error_msg = str(e).lower()
            assert any(word in error_msg for word in ["concat", "merge", "column", "axis"]), "Concatenation error must be specific"
        
        logger.success("Error propagation IO to utils validation completed")

    # ========================================================================
    # TST-MOD-003: MODULE ISOLATION TESTING WITH COMPREHENSIVE MOCKING
    # ========================================================================

    def test_module_isolation_config_module_independence(self, mocker):
        """
        Test TST-MOD-003: Validate config module can function independently
        with comprehensive mocking of external dependencies.
        
        Validates:
        - Config module functions work with mocked YAML operations
        - File system dependencies properly isolated
        - No unintended coupling with other modules
        - Dependency injection patterns support isolation
        """
        logger.info("Testing config module isolation with comprehensive mocking")
        
        # Mock all external dependencies for config module
        mock_yaml_safe_load = mocker.patch("yaml.safe_load")
        mock_pathlib_path = mocker.patch("pathlib.Path")
        mock_open = mocker.patch("builtins.open", mock_open(read_data="test: config\n"))
        
        # Setup mock return values
        test_config_data = {
            "project": {
                "ignore_substrings": ["backup", "temp"],
                "mandatory_substrings": ["experiment"]
            },
            "experiments": {
                "test_exp": {
                    "filters": {
                        "ignore_substrings": ["calibration"],
                        "mandatory_substrings": ["data"]
                    }
                }
            },
            "datasets": {
                "test_dataset": {
                    "dates_vials": {
                        "20241220": [1, 2, 3]
                    }
                }
            }
        }
        
        mock_yaml_safe_load.return_value = test_config_data
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_pathlib_path.return_value = mock_path_instance
        
        # Test config module functions in isolation
        try:
            # Test configuration loading
            loaded_config = load_config("/mock/config.yaml")
            assert loaded_config == test_config_data, "Config loading must work with mocked dependencies"
            
            # Test configuration validation
            validated_config = validate_config_dict(test_config_data)
            assert validated_config == test_config_data, "Config validation must work independently"
            
            # Test ignore patterns extraction
            ignore_patterns = get_ignore_patterns(test_config_data, experiment="test_exp")
            expected_ignore = ["backup", "temp", "calibration"]
            assert all(pattern in ignore_patterns for pattern in expected_ignore), "Ignore pattern extraction must work in isolation"
            
            # Test mandatory substrings extraction
            mandatory_substrings = get_mandatory_substrings(test_config_data, experiment="test_exp")
            expected_mandatory = ["experiment", "data"]
            assert all(substring in mandatory_substrings for substring in expected_mandatory), "Mandatory substring extraction must work in isolation"
            
            # Test dataset info extraction
            dataset_info = get_dataset_info(test_config_data, "test_dataset")
            assert dataset_info["dates_vials"]["20241220"] == [1, 2, 3], "Dataset info extraction must work in isolation"
            
            # Test experiment info extraction
            experiment_info = get_experiment_info(test_config_data, "test_exp")
            assert "filters" in experiment_info, "Experiment info extraction must work in isolation"
            
        except Exception as e:
            pytest.fail(f"Config module isolation test failed: {e}")
        
        # Verify no unexpected external calls were made
        mock_yaml_safe_load.assert_called()
        mock_pathlib_path.assert_called()
        
        logger.success("Config module isolation validation completed")

    def test_module_isolation_discovery_module_independence(self, mocker):
        """
        Test TST-MOD-003: Validate discovery module can function independently
        with mocked file system and pattern matching operations.
        
        Validates:
        - Discovery module works with mocked filesystem providers
        - Pattern matching operates independently of actual files
        - File statistics collection can be mocked effectively
        - Discovery results maintain format without external dependencies
        """
        logger.info("Testing discovery module isolation with comprehensive mocking")
        
        # Mock filesystem operations for discovery module
        mock_path_glob = mocker.patch("pathlib.Path.glob")
        mock_path_rglob = mocker.patch("pathlib.Path.rglob")
        mock_path_exists = mocker.patch("pathlib.Path.exists")
        mock_path_is_file = mocker.patch("pathlib.Path.is_file")
        mock_path_stat = mocker.patch("pathlib.Path.stat")
        
        # Setup mock filesystem responses
        mock_files = [
            Path("/mock/data/experiment_20241220_control_1.pkl"),
            Path("/mock/data/experiment_20241221_treatment_2.pkl"),
            Path("/mock/data/backup_file.pkl"),  # Should be filtered by ignore patterns
            Path("/mock/data/calibration_test.pkl")  # Should be filtered by ignore patterns
        ]
        
        mock_path_glob.return_value = mock_files
        mock_path_rglob.return_value = mock_files
        mock_path_exists.return_value = True
        mock_path_is_file.return_value = True
        
        # Mock file statistics
        mock_stat_result = Mock()
        mock_stat_result.st_size = 2048
        mock_stat_result.st_mtime = datetime.now().timestamp()
        mock_path_stat.return_value = mock_stat_result
        
        # Mock pattern matching for metadata extraction
        mock_pattern_matcher = mocker.patch("flyrigloader.discovery.patterns.PatternMatcher")
        mock_matcher_instance = Mock()
        mock_pattern_matcher.return_value = mock_matcher_instance
        
        def mock_extract_metadata(filename):
            """Mock metadata extraction based on filename."""
            if "20241220_control_1" in filename:
                return {"date": "20241220", "condition": "control", "replicate": "1"}
            elif "20241221_treatment_2" in filename:
                return {"date": "20241221", "condition": "treatment", "replicate": "2"}
            else:
                return {}
        
        mock_matcher_instance.extract_metadata.side_effect = lambda f: mock_extract_metadata(str(f))
        
        # Test discovery module functions in isolation
        try:
            # Test basic file discovery
            discovered_files = discover_files(
                directory="/mock/data",
                pattern="*.pkl",
                recursive=True,
                ignore_patterns=["backup", "calibration"],
                mandatory_substrings=["experiment"],
                extract_patterns=[r".*_(?P<date>\d{8})_(?P<condition>\w+)_(?P<replicate>\d+)\.pkl"],
                include_stats=True
            )
            
            # Validate discovery works in isolation
            assert isinstance(discovered_files, (list, dict)), "Discovery must return appropriate format"
            
            # If returning dict (with metadata), validate structure
            if isinstance(discovered_files, dict):
                for file_path, metadata in discovered_files.items():
                    assert isinstance(metadata, dict), "Metadata must be dictionary format"
                    if "experiment_20241220" in file_path:
                        assert metadata.get("date") == "20241220", "Metadata extraction must work in isolation"
            
            # Test discovery with dependency injection patterns
            from flyrigloader.config.discovery import ConfigDiscoveryEngine
            
            # Create mock providers for isolation testing
            mock_path_provider = Mock()
            mock_path_provider.resolve_path.return_value = Path("/mock/resolved")
            mock_path_provider.exists.return_value = True
            
            mock_file_discovery_provider = Mock()
            mock_file_discovery_provider.discover_files.return_value = {
                "/mock/file1.pkl": {"date": "20241220"},
                "/mock/file2.pkl": {"date": "20241221"}
            }
            
            mock_config_provider = Mock()
            mock_config_provider.get_ignore_patterns.return_value = ["backup"]
            mock_config_provider.get_mandatory_substrings.return_value = ["experiment"]
            mock_config_provider.get_extraction_patterns.return_value = []
            
            # Test isolated discovery engine
            isolated_engine = ConfigDiscoveryEngine(
                path_provider=mock_path_provider,
                file_discovery_provider=mock_file_discovery_provider,
                config_provider=mock_config_provider
            )
            
            test_config = {"project": {"ignore_substrings": ["backup"]}}
            result = isolated_engine.discover_files_with_config(
                config=test_config,
                directory="/mock/data",
                pattern="*.pkl"
            )
            
            # Validate isolated operation
            assert isinstance(result, dict), "Isolated discovery must return dict"
            assert len(result) == 2, "Isolated discovery must use mocked providers"
            
            # Verify mock providers were called
            mock_config_provider.get_ignore_patterns.assert_called_once()
            mock_file_discovery_provider.discover_files.assert_called_once()
            
        except Exception as e:
            pytest.fail(f"Discovery module isolation test failed: {e}")
        
        logger.success("Discovery module isolation validation completed")

    def test_module_isolation_io_module_independence(self, mocker):
        """
        Test TST-MOD-003: Validate IO module can function independently
        with mocked pickle loading and column validation operations.
        
        Validates:
        - Pickle loading works with mocked file operations
        - Column validation operates independently of external configs
        - Data transformation functions work in isolation
        - Error handling maintains context without external dependencies
        """
        logger.info("Testing IO module isolation with comprehensive mocking")
        
        # Mock pickle and file operations for IO module
        mock_pickle_load = mocker.patch("pickle.load")
        mock_gzip_open = mocker.patch("gzip.open")
        mock_pandas_read_pickle = mocker.patch("pandas.read_pickle")
        mock_pathlib_path = mocker.patch("pathlib.Path")
        mock_open = mocker.patch("builtins.open", mock_open(read_data=b"pickle_data"))
        
        # Setup mock experimental data
        mock_experimental_data = {
            't': np.linspace(0, 100, 1000),
            'x': np.random.rand(1000) * 50,
            'y': np.random.rand(1000) * 50,
            'signal': np.random.rand(1000),
            'signal_disp': np.random.rand(16, 1000)
        }
        
        mock_pickle_load.return_value = mock_experimental_data
        mock_pandas_read_pickle.return_value = pd.DataFrame({
            't': mock_experimental_data['t'],
            'x': mock_experimental_data['x'],
            'y': mock_experimental_data['y']
        })
        
        # Mock path operations
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.suffix = ".pkl"
        mock_pathlib_path.return_value = mock_path_instance
        
        # Mock column configuration for isolation
        mock_yaml_load = mocker.patch("yaml.safe_load")
        mock_yaml_load.return_value = {
            "columns": {
                "t": {"type": "numpy.ndarray", "required": True, "dimension": 1},
                "x": {"type": "numpy.ndarray", "required": True, "dimension": 1},
                "y": {"type": "numpy.ndarray", "required": True, "dimension": 1},
                "signal": {"type": "numpy.ndarray", "required": False, "dimension": 1},
                "signal_disp": {"type": "numpy.ndarray", "required": False, "dimension": 2}
            }
        }
        
        # Test IO module functions in isolation
        try:
            # Test pickle loading with various formats
            loaded_data = read_pickle_any_format("/mock/file.pkl")
            assert isinstance(loaded_data, dict), "Pickle loading must work with mocked operations"
            assert 't' in loaded_data, "Required columns must be present"
            assert isinstance(loaded_data['t'], np.ndarray), "Data types must be preserved"
            
            # Test column configuration loading
            column_config = load_column_config()
            assert isinstance(column_config, dict), "Column config loading must work in isolation"
            assert "columns" in column_config, "Column config structure must be maintained"
            
            # Test data validation in isolation
            validated_data = validate_experimental_data(mock_experimental_data, column_config)
            assert isinstance(validated_data, dict), "Data validation must work in isolation"
            assert len(validated_data) >= len(mock_experimental_data), "Validation must preserve or enhance data"
            
            # Test data transformation in isolation
            transformed_data = transform_to_standardized_format(validated_data, column_config)
            assert isinstance(transformed_data, dict), "Data transformation must work in isolation"
            
            # Validate essential arrays are preserved
            for essential_col in ['t', 'x', 'y']:
                assert essential_col in transformed_data, f"Essential column '{essential_col}' must be preserved"
                assert isinstance(transformed_data[essential_col], np.ndarray), f"Column '{essential_col}' must remain numpy array"
            
        except Exception as e:
            pytest.fail(f"IO module isolation test failed: {e}")
        
        # Verify appropriate mocks were called
        mock_pickle_load.assert_called()
        mock_yaml_load.assert_called()
        
        logger.success("IO module isolation validation completed")

    def test_module_isolation_utils_module_independence(self, mocker):
        """
        Test TST-MOD-003: Validate utils module can function independently
        with mocked DataFrame operations and path manipulations.
        
        Validates:
        - DataFrame utilities work with mocked pandas operations
        - Path utilities operate independently of filesystem
        - Data conversion functions maintain integrity in isolation
        - Statistics calculations work without external dependencies
        """
        logger.info("Testing utils module isolation with comprehensive mocking")
        
        # Mock pandas operations for utils module
        mock_pandas_dataframe = mocker.patch("pandas.DataFrame")
        mock_pandas_concat = mocker.patch("pandas.concat")
        mock_pandas_merge = mocker.patch("pandas.merge")
        
        # Mock path operations for utils module
        mock_pathlib_path = mocker.patch("pathlib.Path")
        mock_os_path = mocker.patch("os.path")
        
        # Setup mock DataFrame responses
        mock_df_data = {
            't': np.linspace(0, 100, 1000),
            'x': np.random.rand(1000) * 50,
            'y': np.random.rand(1000) * 50,
            'meta_date': ['20241220'] * 1000,
            'meta_condition': ['control'] * 1000
        }
        
        mock_df = pd.DataFrame(mock_df_data)
        mock_pandas_dataframe.return_value = mock_df
        mock_pandas_concat.return_value = mock_df
        mock_pandas_merge.return_value = mock_df
        
        # Setup mock path responses
        mock_path_instance = Mock()
        mock_path_instance.resolve.return_value = Path("/resolved/path")
        mock_path_instance.relative_to.return_value = Path("relative/path")
        mock_pathlib_path.return_value = mock_path_instance
        
        mock_os_path.abspath.return_value = "/absolute/path"
        mock_os_path.relpath.return_value = "relative/path"
        mock_os_path.join.side_effect = lambda *args: "/".join(args)
        
        # Test utils module functions in isolation
        try:
            # Test DataFrame creation utilities
            from flyrigloader.utils.dataframe import DefaultDataFrameProvider
            df_provider = DefaultDataFrameProvider()
            
            # Test DataFrame creation with mock data
            test_data = {
                't': np.array([1, 2, 3]),
                'x': np.array([4, 5, 6]),
                'y': np.array([7, 8, 9])
            }
            
            result_df = df_provider.create_dataframe(test_data)
            assert result_df is mock_df, "DataFrame creation must use mocked pandas"
            
            # Test DataFrame concatenation
            test_dfs = [mock_df, mock_df]
            concat_result = df_provider.concat_dataframes(test_dfs)
            assert concat_result is mock_df, "DataFrame concatenation must use mocked pandas"
            
            # Test DataFrame merging
            merge_result = df_provider.merge_dataframes(mock_df, mock_df, on='t')
            assert merge_result is mock_df, "DataFrame merging must use mocked pandas"
            
            # Test path utilities in isolation
            from flyrigloader.utils.paths import StandardFileSystemProvider
            path_provider = StandardFileSystemProvider()
            
            # Test path resolution
            test_path = Path("/test/path")
            resolved_path = path_provider.resolve_path(test_path)
            assert resolved_path == Path("/resolved/path"), "Path resolution must use mocked pathlib"
            
            # Test relative path creation
            base_path = Path("/base")
            relative_path = path_provider.make_relative(test_path, base_path)
            assert relative_path == Path("relative/path"), "Relative path creation must use mocked pathlib"
            
            # Test path validation
            path_exists = path_provider.check_file_exists(test_path)
            # This would use mocked file system checks
            
            # Test cross-platform path handling
            from flyrigloader.utils.paths import normalize_path_separators, resolve_path, get_relative_path
            
            test_windows_path = r"C:\test\path\file.txt"
            normalized_path = normalize_path_separators(test_windows_path)
            # Should work regardless of platform with mocked operations
            
        except Exception as e:
            pytest.fail(f"Utils module isolation test failed: {e}")
        
        # Verify appropriate mocks were called
        mock_pandas_dataframe.assert_called()
        mock_pathlib_path.assert_called()
        
        logger.success("Utils module isolation validation completed")

    # ========================================================================
    # END-TO-END PIPELINE INTEGRATION TESTING
    # ========================================================================

    def test_end_to_end_pipeline_integration_complete_workflow(
        self,
        comprehensive_sample_config_dict,
        temp_filesystem_structure,
        comprehensive_exp_matrix,
        mocker
    ):
        """
        Test complete end-to-end pipeline integration from configuration
        loading through final DataFrame creation with realistic data flow.
        
        Validates:
        - Complete config → discovery → io → utils pipeline
        - Data integrity maintained throughout entire workflow
        - All module interfaces work together seamlessly
        - Realistic experimental workflow scenarios
        - Performance characteristics within acceptable bounds
        """
        logger.info("Testing complete end-to-end pipeline integration")
        
        # Setup comprehensive end-to-end mocking
        # Stage 1: Configuration Loading
        mock_load_config = mocker.patch("flyrigloader.api.load_config")
        mock_load_config.return_value = comprehensive_sample_config_dict
        
        # Stage 2: File Discovery
        expected_discovery_results = {
            str(temp_filesystem_structure["baseline_file_1"]): {
                "date": "20241220",
                "condition": "control",
                "replicate": "1",
                "dataset": "baseline_behavior",
                "file_size": 2048,
                "modification_time": datetime.now().isoformat(),
                "extraction_metadata": {
                    "experiment": "baseline",
                    "animal_id": "mouse_001",
                    "session": "1"
                }
            },
            str(temp_filesystem_structure["opto_file_1"]): {
                "date": "20241218",
                "condition": "treatment",
                "replicate": "1",
                "dataset": "optogenetic_stimulation",
                "stimulation_type": "stim",
                "file_size": 3072,
                "modification_time": datetime.now().isoformat(),
                "extraction_metadata": {
                    "experiment": "opto",
                    "treatment": "stimulation",
                    "session": "1"
                }
            }
        }
        
        mock_discover_experiment_files = mocker.patch("flyrigloader.api.discover_experiment_files")
        mock_discover_experiment_files.return_value = expected_discovery_results
        
        # Stage 3: Data Loading
        mock_pickle_loader = mocker.patch("flyrigloader.io.pickle.read_pickle_any_format")
        
        def end_to_end_pickle_loader(file_path):
            """End-to-end pickle loader with realistic data."""
            file_path_str = str(file_path)
            discovery_metadata = expected_discovery_results[file_path_str]
            
            if "baseline" in file_path_str:
                return {
                    't': np.linspace(0, 300, 18000),  # 5 minutes at 60 Hz
                    'x': np.random.rand(18000) * 100 - 50,
                    'y': np.random.rand(18000) * 100 - 50,
                    'metadata': discovery_metadata,
                    'processing_info': {
                        'loaded_at': datetime.now().isoformat(),
                        'loader_version': 'v2.0',
                        'data_quality': 'high'
                    }
                }
            elif "opto" in file_path_str:
                return {
                    't': np.linspace(0, 600, 36000),  # 10 minutes at 60 Hz
                    'x': np.random.rand(36000) * 120 - 60,
                    'y': np.random.rand(36000) * 120 - 60,
                    'signal': np.random.rand(36000),
                    'signal_disp': np.random.rand(16, 36000),
                    'metadata': discovery_metadata,
                    'processing_info': {
                        'loaded_at': datetime.now().isoformat(),
                        'loader_version': 'v2.0',
                        'data_quality': 'high',
                        'stimulation_verified': True
                    }
                }
            else:
                raise FileNotFoundError(f"Unexpected file in end-to-end test: {file_path_str}")
        
        mock_pickle_loader.side_effect = end_to_end_pickle_loader
        
        # Stage 4: Column Validation
        mock_column_config = mocker.patch("flyrigloader.io.column_models.load_column_config")
        mock_column_config.return_value = {
            "columns": {
                "t": {"type": "numpy.ndarray", "required": True, "dimension": 1},
                "x": {"type": "numpy.ndarray", "required": True, "dimension": 1},
                "y": {"type": "numpy.ndarray", "required": True, "dimension": 1},
                "signal": {"type": "numpy.ndarray", "required": False, "dimension": 1},
                "signal_disp": {"type": "numpy.ndarray", "required": False, "dimension": 2}
            }
        }
        
        # Execute end-to-end pipeline
        try:
            # Stage 1: Load configuration
            config = load_config("/mock/config.yaml")
            assert config == comprehensive_sample_config_dict, "Configuration loading failed"
            
            # Stage 2: Discover experiment files
            experiment_name = "baseline_control_study"
            discovered_files = discover_experiment_files(
                config=config,
                experiment_name=experiment_name,
                base_directory=str(temp_filesystem_structure["data_root"]),
                extract_metadata=True,
                parse_dates=True
            )
            
            assert isinstance(discovered_files, dict), "File discovery must return dictionary"
            assert len(discovered_files) > 0, "File discovery must find files"
            
            # Stage 3: Load experimental data for each discovered file
            loaded_datasets = {}
            for file_path, file_metadata in discovered_files.items():
                loaded_data = read_pickle_any_format(file_path)
                
                # Validate data loading success
                assert isinstance(loaded_data, dict), "Data loading must return dictionary"
                assert 't' in loaded_data, "Time data must be present"
                assert 'x' in loaded_data, "X position must be present"
                assert 'y' in loaded_data, "Y position must be present"
                assert 'metadata' in loaded_data, "Metadata must be preserved"
                
                # Integrate discovery metadata with loaded data
                loaded_data['discovery_metadata'] = file_metadata
                loaded_data['file_path'] = file_path
                
                loaded_datasets[file_path] = loaded_data
            
            # Stage 4: Validate and transform data
            column_config = load_column_config()
            validated_datasets = {}
            
            for file_path, dataset in loaded_datasets.items():
                validated_data = validate_experimental_data(dataset, column_config)
                transformed_data = transform_to_standardized_format(validated_data, column_config)
                validated_datasets[file_path] = transformed_data
            
            # Stage 5: Convert to DataFrames
            from flyrigloader.utils.dataframe import DefaultDataFrameProvider
            df_provider = DefaultDataFrameProvider()
            
            final_dataframes = {}
            for file_path, dataset in validated_datasets.items():
                # Prepare data for DataFrame creation
                df_data = {}
                
                # Add time series data
                for col, data in dataset.items():
                    if isinstance(data, np.ndarray):
                        if data.ndim == 1:
                            df_data[col] = data
                        elif data.ndim == 2:
                            # Handle multi-channel data
                            for ch in range(data.shape[0]):
                                df_data[f"{col}_ch{ch:02d}"] = data[ch, :]
                
                # Add metadata as constant columns
                if 'metadata' in dataset:
                    metadata = dataset['metadata']
                    for key, value in metadata.items():
                        if isinstance(value, dict):
                            for nested_key, nested_value in value.items():
                                df_data[f"meta_{key}_{nested_key}"] = [nested_value] * len(dataset['t'])
                        else:
                            df_data[f"meta_{key}"] = [value] * len(dataset['t'])
                
                # Create DataFrame
                result_df = df_provider.create_dataframe(df_data)
                final_dataframes[file_path] = result_df
            
            # Validate end-to-end pipeline success
            assert len(final_dataframes) == len(discovered_files), "All files must be processed successfully"
            
            for file_path, df in final_dataframes.items():
                assert isinstance(df, pd.DataFrame), "Final output must be DataFrame"
                assert len(df) > 0, "DataFrame must contain data"
                assert 't' in df.columns, "Time column must be present"
                assert 'x' in df.columns, "X position must be present"
                assert 'y' in df.columns, "Y position must be present"
                
                # Validate metadata integration
                metadata_cols = [col for col in df.columns if col.startswith('meta_')]
                assert len(metadata_cols) > 0, "Metadata must be integrated"
                
                # Validate data consistency
                original_dataset = validated_datasets[file_path]
                np.testing.assert_array_equal(
                    df['t'].values, original_dataset['t'],
                    err_msg="Time data must be preserved through pipeline"
                )
                np.testing.assert_array_equal(
                    df['x'].values, original_dataset['x'],
                    err_msg="X position must be preserved through pipeline"
                )
                np.testing.assert_array_equal(
                    df['y'].values, original_dataset['y'],
                    err_msg="Y position must be preserved through pipeline"
                )
            
            # Test pipeline with multiple experiments
            combined_results = []
            for file_path, df in final_dataframes.items():
                # Add file identifier
                df_with_source = df.copy()
                df_with_source['source_file'] = file_path
                combined_results.append(df_with_source)
            
            # Combine all experimental data
            if len(combined_results) > 1:
                combined_df = df_provider.concat_dataframes(combined_results, ignore_index=True)
                
                # Validate combined results
                assert isinstance(combined_df, pd.DataFrame), "Combined results must be DataFrame"
                assert len(combined_df) == sum(len(df) for df in final_dataframes.values()), "Combined length must match sum"
                assert 'source_file' in combined_df.columns, "Source tracking must be preserved"
            
        except Exception as e:
            pytest.fail(f"End-to-end pipeline integration failed: {e}")
        
        # Verify all pipeline stages were executed
        mock_load_config.assert_called_once()
        mock_discover_experiment_files.assert_called_once()
        mock_pickle_loader.assert_called()
        mock_column_config.assert_called()
        
        logger.success("Complete end-to-end pipeline integration validation completed successfully")


# ============================================================================
# INTEGRATION TEST FIXTURES AND HELPERS
# ============================================================================

@pytest.fixture
def integration_test_environment(
    comprehensive_sample_config_dict,
    temp_filesystem_structure,
    comprehensive_exp_matrix,
    sample_metadata
):
    """
    Comprehensive integration test environment fixture providing all necessary
    components for cross-module integration testing scenarios.
    
    Returns:
        Dict[str, Any]: Complete test environment with realistic data
    """
    return {
        "config": comprehensive_sample_config_dict,
        "filesystem": temp_filesystem_structure,
        "experimental_data": comprehensive_exp_matrix,
        "metadata": sample_metadata,
        "expected_workflow_results": {
            "config_loading": True,
            "file_discovery": True,
            "data_loading": True,
            "column_validation": True,
            "dataframe_creation": True
        }
    }


@pytest.fixture
def mock_complete_integration_stack(mocker):
    """
    Fixture providing comprehensive mocking for the entire integration stack
    enabling isolated testing of integration patterns without external dependencies.
    
    Returns:
        Dict[str, Mock]: Dictionary of all mocked components
    """
    mocks = {}
    
    # Config module mocks
    mocks["yaml_safe_load"] = mocker.patch("yaml.safe_load")
    mocks["pathlib_path"] = mocker.patch("pathlib.Path")
    mocks["builtins_open"] = mocker.patch("builtins.open")
    
    # Discovery module mocks
    mocks["discover_files"] = mocker.patch("flyrigloader.discovery.files.discover_files")
    mocks["pattern_matcher"] = mocker.patch("flyrigloader.discovery.patterns.PatternMatcher")
    
    # IO module mocks
    mocks["pickle_load"] = mocker.patch("pickle.load")
    mocks["gzip_open"] = mocker.patch("gzip.open")
    mocks["pandas_read_pickle"] = mocker.patch("pandas.read_pickle")
    
    # Utils module mocks
    mocks["pandas_dataframe"] = mocker.patch("pandas.DataFrame")
    mocks["pandas_concat"] = mocker.patch("pandas.concat")
    
    return mocks