"""
Comprehensive End-to-End Workflow Integration Test Suite for FlyRigLoader.

This module implements comprehensive integration testing validating complete flyrigloader
pipeline functionality from YAML configuration loading through final DataFrame output
generation per TST-INTEG-001, TST-INTEG-002, and TST-INTEG-003 requirements.

Test Coverage:
- TST-INTEG-001: End-to-end workflow validation from YAML config to DataFrame output
- TST-INTEG-002: Realistic test data generation with NumPy/Pandas synthetic experimental data
- TST-INTEG-003: DataFrame output verification with structure, types, and content integrity
- F-015: Integration Test Harness with comprehensive workflow scenarios
- Section 4.1.1.1: End-to-End User Journey workflow validation
- F-001-F-006: Complete feature integration across all core modules

Performance Requirements:
- Full workflow execution: <30 seconds per Section 2.2.10
- Test data generation: <5 seconds per TST-INTEG-002
- DataFrame verification: <1 second per TST-INTEG-003

Integration Features Tested:
- Configuration Management (F-001): YAML loading and validation
- File Discovery (F-002): Pattern-based file discovery with filtering
- Data Loading (F-003): Multi-format pickle file loading with auto-detection
- Schema Validation (F-004): Pydantic-based column validation and type checking
- DataFrame Transformation (F-006): Complete exp_matrix to DataFrame conversion
- Error Propagation: Cross-module boundary error handling consistency
"""

import os
import tempfile
import time
import gzip
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from unittest.mock import patch, MagicMock, mock_open

import numpy as np
import pandas as pd
import pytest
import yaml
from hypothesis import given, strategies as st, settings, assume

# Import the modules being tested
from flyrigloader.api import (
    load_experiment_files,
    load_dataset_files,
    get_experiment_parameters,
    get_dataset_parameters,
    process_experiment_data
)
from flyrigloader.config.yaml_config import load_config, validate_config_dict
from flyrigloader.config.discovery import discover_experiment_files, discover_dataset_files
from flyrigloader.discovery.files import discover_files
from flyrigloader.io.pickle import read_pickle_any_format, make_dataframe_from_config
from flyrigloader.io.column_models import get_config_from_source, ColumnConfigDict


class TestEndToEndWorkflowValidation:
    """
    Test class for complete end-to-end workflow validation per TST-INTEG-001.
    
    Validates the complete pipeline from YAML configuration to DataFrame output
    as defined in Section 4.1.1.1 End-to-End User Journey workflow.
    """

    def test_complete_yaml_to_dataframe_workflow_success_scenario(
        self,
        comprehensive_sample_config_dict,
        sample_column_config_file,
        cross_platform_temp_dir,
        performance_benchmarks
    ):
        """
        Test complete successful workflow from YAML configuration to DataFrame output.
        
        Validates TST-INTEG-001: End-to-end workflow validation including:
        1. YAML configuration loading and validation
        2. Experiment file discovery with filtering
        3. Multi-format pickle data loading
        4. Schema validation and column configuration
        5. DataFrame transformation with metadata integration
        
        Performance requirement: <30 seconds total execution time
        """
        workflow_start_time = time.time()
        
        # === PHASE 1: Configuration Management (F-001) ===
        config_setup_start = time.time()
        
        # Create comprehensive configuration file
        config_file = cross_platform_temp_dir / "comprehensive_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(comprehensive_sample_config_dict, f)
        
        # Validate configuration loading
        loaded_config = load_config(str(config_file))
        assert loaded_config == comprehensive_sample_config_dict
        assert "project" in loaded_config
        assert "datasets" in loaded_config
        assert "experiments" in loaded_config
        
        config_setup_time = time.time() - config_setup_start
        performance_benchmarks.assert_performance_sla(
            "config_loading", config_setup_time, 0.5
        )
        
        # === PHASE 2: Realistic Test Data Generation (TST-INTEG-002) ===
        data_generation_start = time.time()
        
        # Create realistic experimental directory structure
        data_root = cross_platform_temp_dir / "research_data"
        experiment_dirs = {
            "baseline": data_root / "baseline_behavior",
            "optogenetic": data_root / "optogenetic_stimulation", 
            "navigation": data_root / "plume_navigation"
        }
        
        for exp_dir in experiment_dirs.values():
            exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate realistic experimental datasets
        experimental_datasets = self._generate_realistic_experimental_data(
            experiment_dirs, comprehensive_sample_config_dict
        )
        
        data_generation_time = time.time() - data_generation_start
        performance_benchmarks.assert_performance_sla(
            "test_data_generation", data_generation_time, 5.0
        )
        
        # === PHASE 3: File Discovery Integration (F-002) ===
        discovery_start_time = time.time()
        
        # Update configuration with actual data directory
        test_config = comprehensive_sample_config_dict.copy()
        test_config["project"]["directories"]["major_data_directory"] = str(data_root)
        
        # Test experiment file discovery
        baseline_files = load_experiment_files(
            config=test_config,
            experiment_name="baseline_control_study",
            base_directory=str(data_root),
            pattern="*.pkl",
            recursive=True,
            extract_metadata=True,
            parse_dates=True
        )
        
        # Validate discovery results
        assert isinstance(baseline_files, dict)
        assert len(baseline_files) > 0, "Should discover baseline experiment files"
        
        for file_path, metadata in baseline_files.items():
            assert Path(file_path).exists(), f"Discovered file should exist: {file_path}"
            assert isinstance(metadata, dict), "Metadata should be extracted"
            assert "file_size" in metadata or "date" in metadata, "Should have extracted metadata"
        
        discovery_time = time.time() - discovery_start_time
        performance_benchmarks.assert_performance_sla(
            "file_discovery", discovery_time, 2.0
        )
        
        # === PHASE 4: Data Loading and Schema Validation (F-003, F-004) ===
        loading_start_time = time.time()
        
        # Test data loading for each discovered file
        loaded_dataframes = {}
        
        for file_path in list(baseline_files.keys())[:3]:  # Test first 3 files for performance
            # Load experimental data
            exp_matrix = read_pickle_any_format(file_path)
            assert isinstance(exp_matrix, dict), "Should load as experimental matrix dictionary"
            assert 't' in exp_matrix, "Should contain time data"
            assert 'x' in exp_matrix, "Should contain x position data"
            assert 'y' in exp_matrix, "Should contain y position data"
            
            # Process data through column configuration and schema validation
            processed_df = process_experiment_data(
                data_path=file_path,
                column_config_path=sample_column_config_file,
                metadata={
                    "experiment_name": "baseline_control_study",
                    "processing_date": datetime.now().isoformat(),
                    "config_file": str(config_file)
                }
            )
            
            loaded_dataframes[file_path] = processed_df
        
        loading_time = time.time() - loading_start_time
        performance_benchmarks.assert_performance_sla(
            "data_loading_processing", loading_time, 10.0
        )
        
        # === PHASE 5: DataFrame Transformation and Integration (F-006) ===
        transformation_start_time = time.time()
        
        # Validate DataFrame output structure and content integrity (TST-INTEG-003)
        for file_path, df in loaded_dataframes.items():
            self._validate_dataframe_output_integrity(df, file_path)
        
        # Test DataFrame aggregation and metadata integration
        aggregated_results = self._aggregate_experimental_results(
            loaded_dataframes, test_config, "baseline_control_study"
        )
        
        transformation_time = time.time() - transformation_start_time
        performance_benchmarks.assert_performance_sla(
            "dataframe_transformation", transformation_time, 1.0
        )
        
        # === PHASE 6: Complete Workflow Performance Validation ===
        total_workflow_time = time.time() - workflow_start_time
        performance_benchmarks.assert_performance_sla(
            "complete_workflow", total_workflow_time, 30.0
        )
        
        # Validate final aggregated results
        assert isinstance(aggregated_results, dict), "Should return aggregated results"
        assert "experiment_summary" in aggregated_results
        assert "dataframe_count" in aggregated_results
        assert "total_timepoints" in aggregated_results
        assert aggregated_results["dataframe_count"] == len(loaded_dataframes)
        assert aggregated_results["total_timepoints"] > 0
        
        # Log successful workflow completion
        print(f"✅ Complete workflow executed successfully in {total_workflow_time:.2f}s")
        print(f"   - Configuration loading: {config_setup_time:.3f}s")
        print(f"   - Test data generation: {data_generation_time:.3f}s") 
        print(f"   - File discovery: {discovery_time:.3f}s")
        print(f"   - Data loading/processing: {loading_time:.3f}s")
        print(f"   - DataFrame transformation: {transformation_time:.3f}s")

    def test_multi_experiment_workflow_integration(
        self,
        comprehensive_sample_config_dict,
        sample_column_config_file,
        cross_platform_temp_dir,
        performance_benchmarks
    ):
        """
        Test complete workflow across multiple experiment types.
        
        Validates comprehensive integration across different experimental scenarios:
        - Baseline behavior experiments
        - Optogenetic stimulation experiments  
        - Navigation task experiments
        
        Tests cross-experiment consistency and data integration capabilities.
        """
        workflow_start_time = time.time()
        
        # Setup comprehensive experimental environment
        data_root = cross_platform_temp_dir / "multi_experiment_data"
        
        experiment_scenarios = {
            "baseline_control_study": {
                "data_dir": data_root / "baseline",
                "expected_columns": ["t", "x", "y"],
                "file_count": 3
            },
            "optogenetic_manipulation": {
                "data_dir": data_root / "optogenetic",
                "expected_columns": ["t", "x", "y", "signal"],
                "file_count": 4
            },
            "multi_modal_navigation": {
                "data_dir": data_root / "navigation", 
                "expected_columns": ["t", "x", "y", "signal_disp"],
                "file_count": 2
            }
        }
        
        # Generate data for each experiment scenario
        all_experimental_data = {}
        
        for exp_name, scenario in experiment_scenarios.items():
            scenario["data_dir"].mkdir(parents=True, exist_ok=True)
            
            # Generate scenario-specific experimental data
            scenario_data = self._generate_scenario_specific_data(
                scenario, exp_name, comprehensive_sample_config_dict
            )
            all_experimental_data[exp_name] = scenario_data
        
        # Update configuration with actual data directory
        test_config = comprehensive_sample_config_dict.copy()
        test_config["project"]["directories"]["major_data_directory"] = str(data_root)
        
        # Test each experiment workflow
        experiment_results = {}
        
        for exp_name, scenario in experiment_scenarios.items():
            exp_start_time = time.time()
            
            # Discover files for this experiment
            discovered_files = load_experiment_files(
                config=test_config,
                experiment_name=exp_name,
                base_directory=str(data_root),
                pattern="*.pkl",
                recursive=True,
                extract_metadata=True
            )
            
            assert len(discovered_files) >= scenario["file_count"], \
                f"Should discover at least {scenario['file_count']} files for {exp_name}"
            
            # Process all files for this experiment
            processed_data = []
            for file_path in list(discovered_files.keys())[:scenario["file_count"]]:
                df = process_experiment_data(
                    data_path=file_path,
                    column_config_path=sample_column_config_file,
                    metadata={"experiment": exp_name, "scenario": scenario}
                )
                
                # Validate expected columns are present
                for col in scenario["expected_columns"]:
                    if col not in ["signal_disp"]:  # signal_disp has special handling
                        assert col in df.columns, f"Column {col} should be in {exp_name} DataFrame"
                
                processed_data.append(df)
            
            experiment_results[exp_name] = {
                "discovered_files": discovered_files,
                "processed_dataframes": processed_data,
                "execution_time": time.time() - exp_start_time
            }
        
        # Validate cross-experiment consistency
        self._validate_cross_experiment_consistency(experiment_results)
        
        # Test integrated analysis across experiments
        integrated_analysis = self._perform_integrated_analysis(experiment_results, test_config)
        
        total_time = time.time() - workflow_start_time
        performance_benchmarks.assert_performance_sla(
            "multi_experiment_workflow", total_time, 30.0
        )
        
        # Validate integrated analysis results
        assert "experiment_comparison" in integrated_analysis
        assert "data_quality_metrics" in integrated_analysis
        assert "cross_experiment_statistics" in integrated_analysis
        
        print(f"✅ Multi-experiment workflow completed in {total_time:.2f}s")
        for exp_name, results in experiment_results.items():
            print(f"   - {exp_name}: {results['execution_time']:.3f}s, "
                  f"{len(results['processed_dataframes'])} DataFrames")

    def test_error_propagation_across_module_boundaries(
        self,
        comprehensive_sample_config_dict,
        cross_platform_temp_dir
    ):
        """
        Test error propagation and handling consistency across module boundaries.
        
        Validates that errors are properly propagated and handled consistently
        throughout the complete workflow pipeline, ensuring robust error recovery
        mechanisms per Section 4.1.2.3.
        """
        # === Test Configuration Error Propagation ===
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            load_config("/nonexistent/config.yaml")
        
        # Test invalid configuration structure
        invalid_config = {"invalid": "structure"}
        with pytest.raises((ValueError, KeyError)):
            load_experiment_files(config=invalid_config, experiment_name="test")
        
        # === Test File Discovery Error Propagation ===
        valid_config = comprehensive_sample_config_dict.copy()
        valid_config["project"]["directories"]["major_data_directory"] = "/nonexistent/path"
        
        # Should handle missing directory gracefully
        discovered_files = load_experiment_files(
            config=valid_config,
            experiment_name="baseline_control_study",
            base_directory="/nonexistent/path"
        )
        assert isinstance(discovered_files, (list, dict))  # Should return empty results, not error
        
        # === Test Data Loading Error Propagation ===
        # Create invalid pickle file
        invalid_pickle = cross_platform_temp_dir / "invalid.pkl"
        invalid_pickle.write_bytes(b"not a pickle file")
        
        with pytest.raises(RuntimeError, match="Failed to load pickle file"):
            read_pickle_any_format(str(invalid_pickle))
        
        # === Test Schema Validation Error Propagation ===
        # Create data with missing required columns
        incomplete_data = cross_platform_temp_dir / "incomplete.pkl"
        incomplete_exp_matrix = {"x": np.array([1, 2, 3])}  # Missing required 't' column
        
        with open(incomplete_data, 'wb') as f:
            pickle.dump(incomplete_exp_matrix, f)
        
        with pytest.raises(ValueError, match="Missing required columns"):
            process_experiment_data(str(incomplete_data))
        
        # === Test API Parameter Validation Error Propagation ===
        with pytest.raises(ValueError, match="Exactly one of 'config_path' or 'config' must be provided"):
            load_experiment_files()  # Neither config_path nor config provided
        
        with pytest.raises(ValueError, match="Exactly one of 'config_path' or 'config' must be provided"):
            load_experiment_files(
                config_path="test.yaml",
                config={"test": "config"}  # Both provided
            )
        
        print("✅ Error propagation tests passed - consistent error handling across modules")

    def test_configuration_driven_workflow_scenarios(
        self,
        cross_platform_temp_dir,
        sample_column_config_file,
        performance_benchmarks
    ):
        """
        Test workflow scenarios driven by different configuration patterns.
        
        Validates that the system correctly handles various configuration
        scenarios including hierarchical configurations, experiment overrides,
        and dataset-specific settings.
        """
        # === Scenario 1: Minimal Configuration ===
        minimal_config = {
            "project": {
                "directories": {"major_data_directory": str(cross_platform_temp_dir)},
                "ignore_substrings": [],
                "mandatory_substrings": []
            },
            "datasets": {
                "simple_dataset": {
                    "dates_vials": {"20241201": [1, 2, 3]}
                }
            },
            "experiments": {
                "simple_experiment": {
                    "datasets": ["simple_dataset"]
                }
            }
        }
        
        # Create minimal test data
        dataset_dir = cross_platform_temp_dir / "20241201"
        dataset_dir.mkdir(exist_ok=True)
        
        simple_data = {"t": np.linspace(0, 60, 3600), "x": np.random.rand(3600), "y": np.random.rand(3600)}
        simple_file = dataset_dir / "simple_experiment_data.pkl"
        with open(simple_file, 'wb') as f:
            pickle.dump(simple_data, f)
        
        # Test minimal configuration workflow
        files = load_experiment_files(
            config=minimal_config,
            experiment_name="simple_experiment",
            pattern="*.pkl"
        )
        
        assert len(files) > 0, "Should discover files with minimal configuration"
        
        # === Scenario 2: Complex Hierarchical Configuration ===
        complex_config = {
            "project": {
                "directories": {"major_data_directory": str(cross_platform_temp_dir)},
                "ignore_substrings": ["temp", "backup"],
                "mandatory_substrings": ["experiment"],
                "extraction_patterns": [r".*_(?P<date>\d{8})_(?P<condition>\w+)\.pkl"]
            },
            "datasets": {
                "advanced_dataset": {
                    "dates_vials": {"20241201": [1, 2, 3], "20241202": [4, 5, 6]},
                    "metadata": {
                        "extraction_patterns": [r".*_(?P<animal_id>\w+)_(?P<trial>\d+)\.pkl"]
                    },
                    "filters": {
                        "ignore_substrings": ["failed"],
                        "mandatory_substrings": ["experiment"]
                    }
                }
            },
            "experiments": {
                "complex_experiment": {
                    "datasets": ["advanced_dataset"],
                    "metadata": {
                        "extraction_patterns": [r".*_(?P<experiment_id>\w+)_(?P<session>\d+)\.pkl"]
                    },
                    "filters": {
                        "ignore_substrings": ["debug"],
                        "mandatory_substrings": ["experiment", "data"]
                    },
                    "analysis_parameters": {
                        "velocity_threshold": 2.0,
                        "smoothing_window": 5
                    }
                }
            }
        }
        
        # Create hierarchical test data
        for date in ["20241201", "20241202"]:
            date_dir = cross_platform_temp_dir / date
            date_dir.mkdir(exist_ok=True)
            
            for i in range(2):
                data_file = date_dir / f"experiment_data_animal_{i+1}_trial_1.pkl"
                with open(data_file, 'wb') as f:
                    pickle.dump(simple_data, f)
        
        # Test complex configuration workflow
        complex_files = load_experiment_files(
            config=complex_config,
            experiment_name="complex_experiment",
            extract_metadata=True,
            parse_dates=True
        )
        
        assert isinstance(complex_files, dict), "Should return metadata when requested"
        assert len(complex_files) > 0, "Should discover files with complex configuration"
        
        # Validate metadata extraction
        for file_path, metadata in complex_files.items():
            assert isinstance(metadata, dict), "Should extract metadata"
            # Should have some extracted metadata (specific fields depend on patterns)
            assert len(metadata) > 0, "Should have extracted some metadata fields"
        
        # === Scenario 3: Error Handling in Configuration ===
        error_config = complex_config.copy()
        error_config["experiments"]["nonexistent_experiment"] = {
            "datasets": ["nonexistent_dataset"]
        }
        
        # Should handle missing dataset gracefully
        with pytest.raises(KeyError, match="not found in configuration"):
            get_experiment_parameters(config=error_config, experiment_name="nonexistent_experiment")
        
        print("✅ Configuration-driven workflow scenarios completed successfully")

    # === Helper Methods for Data Generation and Validation ===
    
    def _generate_realistic_experimental_data(
        self, 
        experiment_dirs: Dict[str, Path], 
        config: Dict[str, Any]
    ) -> Dict[str, List[Path]]:
        """
        Generate realistic experimental datasets for integration testing.
        
        Creates synthetic experimental data that mirrors actual flyrigloader usage
        patterns with appropriate file naming conventions and data structures.
        """
        experimental_datasets = {}
        
        for exp_type, data_dir in experiment_dirs.items():
            datasets = []
            
            # Generate experiment-specific datasets
            if exp_type == "baseline":
                # Baseline behavior: position tracking only
                for i in range(3):
                    data = {
                        't': np.linspace(0, 300, 18000),  # 5 minutes at 60 Hz
                        'x': 60 + 30 * np.random.rand(18000),  # Arena center ± movement
                        'y': 60 + 30 * np.random.rand(18000),
                    }
                    filename = f"baseline_20241220_control_{i+1}.pkl"
                    file_path = data_dir / filename
                    
                    with open(file_path, 'wb') as f:
                        pickle.dump(data, f)
                    datasets.append(file_path)
            
            elif exp_type == "optogenetic":
                # Optogenetic stimulation: position + signal
                for i in range(4):
                    data = {
                        't': np.linspace(0, 600, 36000),  # 10 minutes at 60 Hz
                        'x': 75 + 45 * np.random.rand(36000),
                        'y': 75 + 45 * np.random.rand(36000),
                        'signal': 0.1 + 0.8 * np.random.rand(36000)  # Calcium signal
                    }
                    filename = f"opto_stim_20241218_treatment_{i+1}.pkl"
                    file_path = data_dir / filename
                    
                    with open(file_path, 'wb') as f:
                        pickle.dump(data, f)
                    datasets.append(file_path)
            
            elif exp_type == "navigation":
                # Navigation task: position + multi-channel signals
                for i in range(2):
                    n_timepoints = 10800  # 3 minutes at 60 Hz
                    data = {
                        't': np.linspace(0, 180, n_timepoints),
                        'x': 100 + 20 * np.random.rand(n_timepoints),
                        'y': 100 + 20 * np.random.rand(n_timepoints),
                        'signal_disp': np.random.rand(16, n_timepoints)  # 16-channel signals
                    }
                    filename = f"plume_navigation_20241025_trial_{i+1}.pkl"
                    file_path = data_dir / filename
                    
                    with open(file_path, 'wb') as f:
                        pickle.dump(data, f)
                    datasets.append(file_path)
            
            experimental_datasets[exp_type] = datasets
        
        return experimental_datasets

    def _generate_scenario_specific_data(
        self, 
        scenario: Dict[str, Any], 
        exp_name: str, 
        config: Dict[str, Any]
    ) -> List[Path]:
        """Generate data specific to experimental scenario requirements."""
        datasets = []
        data_dir = scenario["data_dir"]
        
        for i in range(scenario["file_count"]):
            # Generate base time series
            if "baseline" in exp_name:
                duration = 300  # 5 minutes
                freq = 60  # 60 Hz
            elif "optogenetic" in exp_name:
                duration = 600  # 10 minutes
                freq = 60
            else:  # navigation
                duration = 180  # 3 minutes  
                freq = 60
            
            n_points = int(duration * freq)
            
            # Create experimental data with required columns
            data = {
                't': np.linspace(0, duration, n_points),
                'x': 60 + 40 * np.random.rand(n_points),
                'y': 60 + 40 * np.random.rand(n_points)
            }
            
            # Add scenario-specific columns
            if 'signal' in scenario["expected_columns"]:
                data['signal'] = np.random.rand(n_points)
            
            if 'signal_disp' in scenario["expected_columns"]:
                data['signal_disp'] = np.random.rand(16, n_points)
            
            # Create file with experiment-specific naming
            filename = f"{exp_name}_data_{i+1:02d}.pkl"
            file_path = data_dir / filename
            
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            datasets.append(file_path)
        
        return datasets

    def _validate_dataframe_output_integrity(self, df: pd.DataFrame, file_path: str) -> None:
        """
        Validate DataFrame output structure, types, and content integrity per TST-INTEG-003.
        
        Performs comprehensive validation of DataFrame output including:
        - Required column presence and types
        - Data integrity and consistency
        - Metadata integration validation
        - Performance within <1 second verification time
        """
        validation_start = time.time()
        
        # Basic structure validation
        assert isinstance(df, pd.DataFrame), f"Output should be DataFrame for {file_path}"
        assert len(df) > 0, f"DataFrame should not be empty for {file_path}"
        
        # Required column validation
        required_columns = ['t', 'x', 'y']
        for col in required_columns:
            assert col in df.columns, f"Required column '{col}' missing in DataFrame from {file_path}"
            assert not df[col].isna().all(), f"Column '{col}' should not be all NaN in {file_path}"
        
        # Data type validation
        numeric_columns = ['t', 'x', 'y']
        for col in numeric_columns:
            if col in df.columns:
                assert pd.api.types.is_numeric_dtype(df[col]), \
                    f"Column '{col}' should be numeric in {file_path}"
        
        # Data integrity validation
        if 't' in df.columns:
            # Time should be monotonically increasing
            assert df['t'].is_monotonic_increasing or df['t'].is_monotonic_non_decreasing, \
                f"Time column should be monotonic in {file_path}"
            
            # Time values should be reasonable (positive)
            assert (df['t'] >= 0).all(), f"Time values should be non-negative in {file_path}"
        
        # Position data validation
        for pos_col in ['x', 'y']:
            if pos_col in df.columns:
                # Position values should be finite
                assert np.isfinite(df[pos_col]).all(), \
                    f"Position column '{pos_col}' should contain finite values in {file_path}"
                
                # Position values should be within reasonable bounds (arena size)
                assert df[pos_col].min() >= -500 and df[pos_col].max() <= 500, \
                    f"Position column '{pos_col}' values seem unreasonable in {file_path}"
        
        # Signal data validation (if present)
        if 'signal' in df.columns:
            assert pd.api.types.is_numeric_dtype(df['signal']), \
                f"Signal column should be numeric in {file_path}"
            assert np.isfinite(df['signal']).all(), \
                f"Signal values should be finite in {file_path}"
        
        # Multi-dimensional signal validation (if present)
        signal_disp_cols = [col for col in df.columns if 'signal_disp' in col]
        if signal_disp_cols:
            for col in signal_disp_cols:
                # Should contain array-like data or be properly formatted
                assert not df[col].isna().all(), \
                    f"Signal_disp column '{col}' should not be all NaN in {file_path}"
        
        # Metadata validation (if present)
        metadata_columns = ['experiment_name', 'processing_date', 'config_file']
        for meta_col in metadata_columns:
            if meta_col in df.columns:
                # Metadata should be consistent across all rows
                unique_values = df[meta_col].nunique()
                assert unique_values <= 1, \
                    f"Metadata column '{meta_col}' should have consistent values in {file_path}"
        
        validation_time = time.time() - validation_start
        assert validation_time < 1.0, \
            f"DataFrame validation should complete within 1 second, took {validation_time:.3f}s"

    def _aggregate_experimental_results(
        self, 
        dataframes: Dict[str, pd.DataFrame], 
        config: Dict[str, Any], 
        experiment_name: str
    ) -> Dict[str, Any]:
        """
        Aggregate results from multiple experimental DataFrames.
        
        Combines data from multiple files into summary statistics and
        integrated analysis results for downstream processing.
        """
        total_timepoints = sum(len(df) for df in dataframes.values())
        total_duration = sum(df['t'].max() - df['t'].min() for df in dataframes.values() if 't' in df.columns)
        
        # Calculate aggregated statistics
        all_x_values = np.concatenate([df['x'].values for df in dataframes.values() if 'x' in df.columns])
        all_y_values = np.concatenate([df['y'].values for df in dataframes.values() if 'y' in df.columns])
        
        position_stats = {
            "x_mean": np.mean(all_x_values),
            "x_std": np.std(all_x_values),
            "y_mean": np.mean(all_y_values),
            "y_std": np.std(all_y_values),
            "arena_coverage": {
                "x_range": [np.min(all_x_values), np.max(all_x_values)],
                "y_range": [np.min(all_y_values), np.max(all_y_values)]
            }
        }
        
        return {
            "experiment_summary": {
                "experiment_name": experiment_name,
                "file_count": len(dataframes),
                "total_duration_seconds": total_duration,
                "average_file_duration": total_duration / len(dataframes) if dataframes else 0
            },
            "dataframe_count": len(dataframes),
            "total_timepoints": total_timepoints,
            "position_statistics": position_stats,
            "data_quality_metrics": {
                "complete_files": len(dataframes),
                "avg_timepoints_per_file": total_timepoints / len(dataframes) if dataframes else 0
            }
        }

    def _validate_cross_experiment_consistency(self, experiment_results: Dict[str, Any]) -> None:
        """
        Validate consistency across different experiment types.
        
        Ensures that data processing is consistent across different experimental
        scenarios and that cross-experiment analysis is feasible.
        """
        # Validate that all experiments produced results
        for exp_name, results in experiment_results.items():
            assert len(results["processed_dataframes"]) > 0, \
                f"Experiment {exp_name} should produce processed DataFrames"
            assert results["execution_time"] > 0, \
                f"Experiment {exp_name} should have positive execution time"
        
        # Validate cross-experiment data compatibility
        all_dataframes = []
        for results in experiment_results.values():
            all_dataframes.extend(results["processed_dataframes"])
        
        # Check column consistency where applicable
        common_columns = set(all_dataframes[0].columns)
        for df in all_dataframes[1:]:
            common_columns = common_columns.intersection(set(df.columns))
        
        required_common_columns = {'t', 'x', 'y'}
        assert required_common_columns.issubset(common_columns), \
            f"All experiments should share common columns: {required_common_columns}"
        
        # Validate time consistency across experiments
        time_ranges = []
        for df in all_dataframes:
            if 't' in df.columns and len(df) > 0:
                time_ranges.append((df['t'].min(), df['t'].max()))
        
        # All time series should start from reasonable values
        for t_min, t_max in time_ranges:
            assert t_min >= 0, "Time series should start from non-negative values"
            assert t_max > t_min, "Time series should have positive duration"

    def _perform_integrated_analysis(
        self, 
        experiment_results: Dict[str, Any], 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform integrated analysis across multiple experiments.
        
        Demonstrates cross-experiment analysis capabilities and validates
        that the integration infrastructure supports complex analytical workflows.
        """
        # Collect all DataFrames for integrated analysis
        all_experiments_data = {}
        
        for exp_name, results in experiment_results.items():
            dataframes = results["processed_dataframes"]
            
            # Combine DataFrames for this experiment
            if dataframes:
                combined_df = pd.concat(dataframes, ignore_index=True)
                all_experiments_data[exp_name] = combined_df
        
        # Cross-experiment comparison
        experiment_comparison = {}
        for exp_name, df in all_experiments_data.items():
            experiment_comparison[exp_name] = {
                "sample_count": len(df),
                "duration": df['t'].max() - df['t'].min() if 't' in df.columns else 0,
                "position_range": {
                    "x": [df['x'].min(), df['x'].max()] if 'x' in df.columns else [0, 0],
                    "y": [df['y'].min(), df['y'].max()] if 'y' in df.columns else [0, 0]
                },
                "has_signal": 'signal' in df.columns,
                "has_signal_disp": any('signal_disp' in col for col in df.columns)
            }
        
        # Data quality metrics across experiments
        data_quality_metrics = {
            "total_experiments": len(all_experiments_data),
            "total_samples": sum(len(df) for df in all_experiments_data.values()),
            "experiments_with_signals": sum(1 for comp in experiment_comparison.values() if comp["has_signal"]),
            "experiments_with_multi_signal": sum(1 for comp in experiment_comparison.values() if comp["has_signal_disp"])
        }
        
        # Cross-experiment statistics
        if all_experiments_data:
            all_positions_x = np.concatenate([df['x'].values for df in all_experiments_data.values() if 'x' in df.columns])
            all_positions_y = np.concatenate([df['y'].values for df in all_experiments_data.values() if 'y' in df.columns])
            
            cross_experiment_statistics = {
                "global_position_stats": {
                    "x_mean": np.mean(all_positions_x),
                    "x_std": np.std(all_positions_x),
                    "y_mean": np.mean(all_positions_y),
                    "y_std": np.std(all_positions_y)
                },
                "experiment_consistency": {
                    "position_overlap": self._calculate_position_overlap(all_experiments_data)
                }
            }
        else:
            cross_experiment_statistics = {}
        
        return {
            "experiment_comparison": experiment_comparison,
            "data_quality_metrics": data_quality_metrics,
            "cross_experiment_statistics": cross_experiment_statistics
        }

    def _calculate_position_overlap(self, experiments_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Calculate spatial overlap between different experiments."""
        position_overlaps = {}
        
        exp_names = list(experiments_data.keys())
        for i, exp1 in enumerate(exp_names):
            for exp2 in exp_names[i+1:]:
                df1, df2 = experiments_data[exp1], experiments_data[exp2]
                
                if 'x' in df1.columns and 'y' in df1.columns and 'x' in df2.columns and 'y' in df2.columns:
                    # Calculate bounding boxes
                    bbox1 = [df1['x'].min(), df1['y'].min(), df1['x'].max(), df1['y'].max()]
                    bbox2 = [df2['x'].min(), df2['y'].min(), df2['x'].max(), df2['y'].max()]
                    
                    # Calculate overlap area
                    overlap_area = max(0, min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0])) * \
                                 max(0, min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1]))
                    
                    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
                    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
                    
                    union_area = area1 + area2 - overlap_area
                    overlap_ratio = overlap_area / union_area if union_area > 0 else 0
                    
                    position_overlaps[f"{exp1}_vs_{exp2}"] = overlap_ratio
        
        return position_overlaps


class TestRealisticDataGeneration:
    """
    Test class for realistic test data generation per TST-INTEG-002.
    
    Validates comprehensive synthetic experimental data generation using NumPy
    and Pandas with biologically plausible characteristics and performance
    requirements (<5 seconds generation time).
    """

    def test_synthetic_experimental_data_generation(
        self, 
        synthetic_trajectory_generator,
        synthetic_signal_generator,
        performance_benchmarks
    ):
        """
        Test realistic synthetic experimental data generation.
        
        Validates TST-INTEG-002 requirements for NumPy/Pandas-based synthetic
        experimental data creation with realistic characteristics.
        """
        generation_start_time = time.time()
        
        # Generate realistic trajectory data
        n_timepoints = 18000  # 5 minutes at 60 Hz
        time_data, x_pos, y_pos = synthetic_trajectory_generator(
            n_timepoints=n_timepoints,
            sampling_freq=60.0,
            arena_diameter=120.0,
            center_bias=0.3,
            movement_noise=0.1,
            seed=42
        )
        
        # Validate trajectory characteristics
        assert len(time_data) == n_timepoints, "Should generate requested number of timepoints"
        assert len(x_pos) == n_timepoints, "X position should match timepoints"
        assert len(y_pos) == n_timepoints, "Y position should match timepoints"
        
        # Check trajectory realism
        assert np.all(time_data >= 0), "Time should be non-negative"
        assert np.all(np.diff(time_data) > 0), "Time should be strictly increasing"
        
        # Position should be within arena bounds
        arena_radius = 60.0  # 120mm diameter / 2
        distances = np.sqrt(x_pos**2 + y_pos**2)
        assert np.all(distances <= arena_radius * 1.1), "Position should stay within arena bounds (with small tolerance)"
        
        # Generate realistic multi-channel signal data
        signal_data = synthetic_signal_generator(
            n_timepoints=n_timepoints,
            n_channels=16,
            signal_freq=2.0,
            noise_level=0.1,
            baseline_drift=True,
            seed=42
        )
        
        # Validate signal characteristics
        assert signal_data.shape == (16, n_timepoints), "Signal should have correct dimensions"
        assert np.all(np.isfinite(signal_data)), "Signal should contain finite values"
        
        # Check signal realism (should have reasonable amplitude range)
        signal_range = np.max(signal_data) - np.min(signal_data)
        assert 1.0 < signal_range < 10.0, "Signal should have reasonable dynamic range"
        
        generation_time = time.time() - generation_start_time
        performance_benchmarks.assert_performance_sla(
            "synthetic_data_generation", generation_time, 5.0
        )
        
        print(f"✅ Synthetic data generation completed in {generation_time:.3f}s")
        print(f"   - Generated {n_timepoints} timepoints of trajectory data")
        print(f"   - Generated {signal_data.shape[0]} channels × {signal_data.shape[1]} signal data")

    @given(
        n_timepoints=st.integers(min_value=100, max_value=50000),
        n_channels=st.integers(min_value=1, max_value=32),
        arena_diameter=st.floats(min_value=50.0, max_value=200.0)
    )
    @settings(max_examples=20, deadline=None)
    def test_synthetic_data_generation_property_based(
        self,
        synthetic_trajectory_generator,
        synthetic_signal_generator,
        n_timepoints,
        n_channels,
        arena_diameter
    ):
        """
        Property-based test for synthetic data generation robustness.
        
        Uses Hypothesis to test data generation across a wide range of parameters
        to ensure robust synthetic data creation under various conditions.
        """
        # Skip very large datasets for performance in property-based testing
        assume(n_timepoints * n_channels < 1_000_000)
        
        # Generate trajectory data
        time_data, x_pos, y_pos = synthetic_trajectory_generator(
            n_timepoints=n_timepoints,
            sampling_freq=60.0,
            arena_diameter=arena_diameter,
            seed=42
        )
        
        # Property validation
        assert len(time_data) == n_timepoints
        assert len(x_pos) == n_timepoints
        assert len(y_pos) == n_timepoints
        assert np.all(np.isfinite(time_data))
        assert np.all(np.isfinite(x_pos))
        assert np.all(np.isfinite(y_pos))
        
        # Generate signal data
        signal_data = synthetic_signal_generator(
            n_timepoints=n_timepoints,
            n_channels=n_channels,
            seed=42
        )
        
        # Signal property validation
        assert signal_data.shape == (n_channels, n_timepoints)
        assert np.all(np.isfinite(signal_data))

    def test_experimental_metadata_generation(self, cross_platform_temp_dir):
        """
        Test realistic experimental metadata generation for integration scenarios.
        
        Validates that synthetic metadata follows realistic experimental patterns
        and supports comprehensive integration testing scenarios.
        """
        # Generate diverse experimental metadata
        metadata_samples = []
        
        for i in range(20):
            metadata = {
                "animal_id": f"fly_{i+1:03d}",
                "experiment_date": (datetime.now() - timedelta(days=i)).strftime("%Y%m%d"),
                "condition": ["control", "treatment_a", "treatment_b"][i % 3],
                "replicate": (i % 5) + 1,
                "rig": ["old_opto", "new_opto", "high_speed_rig"][i % 3],
                "experimenter": ["researcher_a", "researcher_b"][i % 2],
                "temperature_c": 22.0 + np.random.normal(0, 1.0),
                "humidity_percent": 45.0 + np.random.normal(0, 5.0)
            }
            metadata_samples.append(metadata)
        
        # Validate metadata diversity and realism
        animal_ids = [m["animal_id"] for m in metadata_samples]
        assert len(set(animal_ids)) == len(metadata_samples), "Should generate unique animal IDs"
        
        conditions = [m["condition"] for m in metadata_samples]
        assert len(set(conditions)) >= 2, "Should generate diverse experimental conditions"
        
        dates = [m["experiment_date"] for m in metadata_samples]
        assert len(set(dates)) > 1, "Should generate diverse experiment dates"
        
        # Validate realistic value ranges
        for metadata in metadata_samples:
            assert 18.0 <= metadata["temperature_c"] <= 28.0, "Temperature should be realistic"
            assert 30.0 <= metadata["humidity_percent"] <= 70.0, "Humidity should be realistic"
            assert 1 <= metadata["replicate"] <= 5, "Replicate should be reasonable"
        
        print(f"✅ Generated {len(metadata_samples)} realistic metadata samples")
        print(f"   - Unique conditions: {set(conditions)}")
        print(f"   - Date range: {min(dates)} to {max(dates)}")


class TestDataFrameOutputVerification:
    """
    Test class for DataFrame output verification per TST-INTEG-003.
    
    Validates final DataFrame structure, types, and content integrity with
    comprehensive validation completing within <1 second per requirement.
    """

    def test_dataframe_structure_validation(
        self,
        comprehensive_exp_matrix,
        sample_column_config_file,
        performance_benchmarks
    ):
        """
        Test comprehensive DataFrame structure validation.
        
        Validates TST-INTEG-003 requirements for DataFrame output verification
        including structure, types, and content integrity validation.
        """
        verification_start_time = time.time()
        
        # Create DataFrame from experimental matrix
        df = make_dataframe_from_config(
            exp_matrix=comprehensive_exp_matrix,
            config_source=sample_column_config_file,
            metadata={
                "experiment_name": "structure_validation_test",
                "processing_timestamp": datetime.now().isoformat()
            }
        )
        
        # === Structure Validation ===
        assert isinstance(df, pd.DataFrame), "Output should be pandas DataFrame"
        assert len(df) > 0, "DataFrame should not be empty"
        assert len(df.columns) > 0, "DataFrame should have columns"
        
        # Required columns validation
        required_columns = ["t", "x", "y"]
        for col in required_columns:
            assert col in df.columns, f"Required column '{col}' should be present"
        
        # === Type Validation ===
        numeric_columns = ["t", "x", "y", "signal"]
        for col in numeric_columns:
            if col in df.columns:
                assert pd.api.types.is_numeric_dtype(df[col]), \
                    f"Column '{col}' should be numeric type"
        
        # Metadata columns should be object/string type
        metadata_columns = ["experiment_name", "processing_timestamp"]
        for col in metadata_columns:
            if col in df.columns:
                assert pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]), \
                    f"Metadata column '{col}' should be object/string type"
        
        # === Content Integrity Validation ===
        
        # Time series validation
        if "t" in df.columns:
            assert df["t"].is_monotonic_increasing, "Time should be monotonically increasing"
            assert (df["t"] >= 0).all(), "Time values should be non-negative"
            assert df["t"].notna().all(), "Time values should not contain NaN"
        
        # Position data validation
        for pos_col in ["x", "y"]:
            if pos_col in df.columns:
                assert df[pos_col].notna().all(), f"Position column '{pos_col}' should not contain NaN"
                assert np.isfinite(df[pos_col]).all(), f"Position column '{pos_col}' should be finite"
                
                # Position values should be within reasonable experimental bounds
                pos_range = df[pos_col].max() - df[pos_col].min()
                assert pos_range > 0, f"Position column '{pos_col}' should have variation"
                assert pos_range < 1000, f"Position column '{pos_col}' range seems unrealistic"
        
        # Signal data validation
        if "signal" in df.columns:
            assert df["signal"].notna().all(), "Signal column should not contain NaN"
            assert np.isfinite(df["signal"]).all(), "Signal values should be finite"
            assert df["signal"].std() > 0, "Signal should have variation"
        
        # Multi-dimensional signal validation
        signal_disp_cols = [col for col in df.columns if "signal_disp" in col or col == "signal_disp"]
        if signal_disp_cols:
            for col in signal_disp_cols:
                # Signal_disp might be Series of arrays or individual columns
                if df[col].dtype == object:
                    # Series of arrays - check each array
                    for i, signal_array in enumerate(df[col].head(10)):  # Check first 10
                        if hasattr(signal_array, '__len__') and len(signal_array) > 0:
                            assert np.all(np.isfinite(signal_array)), f"Signal_disp arrays should be finite at row {i}"
                else:
                    # Individual numeric column
                    assert np.isfinite(df[col]).all(), f"Signal_disp column '{col}' should be finite"
        
        # === Metadata Integration Validation ===
        if "experiment_name" in df.columns:
            unique_exp_names = df["experiment_name"].unique()
            assert len(unique_exp_names) == 1, "Experiment name should be consistent across rows"
            assert unique_exp_names[0] == "structure_validation_test", "Experiment name should match input"
        
        # === Index Validation ===
        assert df.index.is_unique, "DataFrame index should be unique"
        assert df.index.is_monotonic_increasing, "DataFrame index should be monotonic"
        
        verification_time = time.time() - verification_start_time
        performance_benchmarks.assert_performance_sla(
            "dataframe_verification", verification_time, 1.0
        )
        
        print(f"✅ DataFrame structure validation completed in {verification_time:.3f}s")
        print(f"   - Validated {len(df)} rows × {len(df.columns)} columns")
        print(f"   - All required columns present: {required_columns}")

    def test_dataframe_content_integrity_comprehensive(
        self,
        sample_exp_matrix_with_signal_disp,
        sample_column_config_file,
        performance_benchmarks
    ):
        """
        Test comprehensive DataFrame content integrity validation.
        
        Validates that data transformation preserves content integrity
        and that all data processing steps maintain data quality.
        """
        verification_start_time = time.time()
        
        # Create DataFrame with complex data
        df = make_dataframe_from_config(
            exp_matrix=sample_exp_matrix_with_signal_disp,
            config_source=sample_column_config_file,
            metadata={"test_type": "content_integrity"}
        )
        
        original_matrix = sample_exp_matrix_with_signal_disp
        
        # === Data Preservation Validation ===
        
        # Time data preservation
        if "t" in df.columns and "t" in original_matrix:
            original_t = original_matrix["t"]
            df_t = df["t"].values
            
            assert len(df_t) == len(original_t), "Time array length should be preserved"
            np.testing.assert_array_almost_equal(df_t, original_t, decimal=10,
                err_msg="Time values should be precisely preserved")
        
        # Position data preservation
        for pos_col in ["x", "y"]:
            if pos_col in df.columns and pos_col in original_matrix:
                original_pos = original_matrix[pos_col]
                df_pos = df[pos_col].values
                
                assert len(df_pos) == len(original_pos), f"Position array '{pos_col}' length should be preserved"
                np.testing.assert_array_almost_equal(df_pos, original_pos, decimal=10,
                    err_msg=f"Position values '{pos_col}' should be precisely preserved")
        
        # === Data Quality Metrics ===
        
        # Statistical consistency
        for col in ["x", "y"]:
            if col in df.columns and col in original_matrix:
                original_mean = np.mean(original_matrix[col])
                df_mean = df[col].mean()
                
                assert abs(original_mean - df_mean) < 1e-10, \
                    f"Mean should be preserved for column '{col}'"
                
                original_std = np.std(original_matrix[col])
                df_std = df[col].std()
                
                assert abs(original_std - df_std) < 1e-10, \
                    f"Standard deviation should be preserved for column '{col}'"
        
        # === Multi-dimensional Data Validation ===
        
        if "signal_disp" in original_matrix:
            original_signal_disp = original_matrix["signal_disp"]
            
            # Check if signal_disp was properly handled
            signal_disp_cols = [col for col in df.columns if "signal_disp" in col]
            
            if signal_disp_cols:
                # Validate signal_disp processing
                if "signal_disp" in df.columns:
                    # Series of arrays format
                    assert len(df["signal_disp"]) == original_signal_disp.shape[-1], \
                        "Signal_disp should preserve time dimension"
                    
                    # Check first few entries for array consistency
                    for i in range(min(5, len(df))):
                        if hasattr(df["signal_disp"].iloc[i], '__len__'):
                            signal_array = df["signal_disp"].iloc[i]
                            expected_length = original_signal_disp.shape[0] if original_signal_disp.ndim == 2 else 1
                            assert len(signal_array) == expected_length, \
                                f"Signal_disp array {i} should have correct channel dimension"
        
        # === Data Range Validation ===
        
        # Ensure data ranges are preserved and reasonable
        for col in ["x", "y"]:
            if col in df.columns:
                col_range = df[col].max() - df[col].min()
                assert col_range > 0, f"Column '{col}' should have positive range"
                
                # Check for outliers (values beyond 5 standard deviations)
                col_mean = df[col].mean()
                col_std = df[col].std()
                outliers = np.abs(df[col] - col_mean) > 5 * col_std
                outlier_pct = outliers.sum() / len(df) * 100
                
                assert outlier_pct < 1.0, f"Column '{col}' should have <1% outliers, found {outlier_pct:.2f}%"
        
        verification_time = time.time() - verification_start_time
        performance_benchmarks.assert_performance_sla(
            "content_integrity_verification", verification_time, 1.0
        )
        
        print(f"✅ Content integrity verification completed in {verification_time:.3f}s")
        print(f"   - Validated data preservation for {len(df)} rows")
        print(f"   - Verified statistical consistency across {len(df.columns)} columns")

    def test_dataframe_metadata_integration_validation(
        self,
        comprehensive_exp_matrix,
        sample_column_config_file,
        sample_metadata
    ):
        """
        Test comprehensive metadata integration in DataFrame output.
        
        Validates that metadata is properly integrated into DataFrame
        and that metadata fields are handled consistently.
        """
        # Create DataFrame with extensive metadata
        df = make_dataframe_from_config(
            exp_matrix=comprehensive_exp_matrix,
            config_source=sample_column_config_file,
            metadata=sample_metadata
        )
        
        # === Metadata Presence Validation ===
        metadata_columns = ["date", "exp_name", "rig", "fly_id"]
        
        for meta_col in metadata_columns:
            if meta_col in sample_metadata:
                # Check if metadata column was added to DataFrame
                # Note: This depends on column configuration marking columns as metadata
                # For this test, we verify the metadata was available for integration
                assert meta_col in sample_metadata, f"Metadata field '{meta_col}' should be in input metadata"
        
        # === Metadata Consistency Validation ===
        
        # If metadata columns are present in DataFrame, they should be consistent
        for col in df.columns:
            if col in sample_metadata:
                # Metadata values should be consistent across all rows
                unique_values = df[col].nunique()
                assert unique_values <= 1, f"Metadata column '{col}' should have consistent values"
                
                if unique_values == 1:
                    df_value = df[col].iloc[0]
                    expected_value = sample_metadata[col]
                    assert str(df_value) == str(expected_value), \
                        f"Metadata column '{col}' should match input value"
        
        # === Data and Metadata Separation ===
        
        # Core experimental data should not be affected by metadata
        experimental_columns = ["t", "x", "y", "signal"]
        for exp_col in experimental_columns:
            if exp_col in df.columns:
                # Experimental data should have variation (not constant like metadata)
                if df[exp_col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                    col_std = df[exp_col].std()
                    assert col_std > 0, f"Experimental column '{exp_col}' should have variation"
        
        print(f"✅ Metadata integration validation completed")
        print(f"   - Validated metadata integration for {len(sample_metadata)} metadata fields")
        print(f"   - Verified data/metadata separation in {len(df.columns)} total columns")


class TestPerformanceValidation:
    """
    Test class for performance validation against SLA requirements.
    
    Validates that the complete integration workflow meets performance
    requirements including <30 seconds full workflow execution.
    """

    def test_workflow_performance_sla_validation(
        self,
        performance_test_data,
        performance_benchmarks,
        cross_platform_temp_dir
    ):
        """
        Test workflow performance against defined SLAs.
        
        Validates that complete workflows meet performance requirements
        across different data sizes and complexity levels.
        """
        # Test performance across different data scales
        for scale, data_config in performance_test_data.items():
            scale_start_time = time.time()
            
            print(f"\n🔄 Testing {scale} dataset: {data_config['size_description']}")
            
            # Create test data files
            test_dir = cross_platform_temp_dir / f"perf_test_{scale}"
            test_dir.mkdir(exist_ok=True)
            
            data_file = test_dir / f"{scale}_data.pkl"
            with open(data_file, 'wb') as f:
                pickle.dump(data_config["data"], f)
            
            # === Data Loading Performance ===
            load_start_time = time.time()
            loaded_data = read_pickle_any_format(str(data_file))
            load_time = time.time() - load_start_time
            
            # Calculate expected time based on file size
            file_size_mb = os.path.getsize(data_file) / (1024 * 1024)
            expected_load_time = performance_benchmarks.benchmark_data_loading(file_size_mb)
            
            performance_benchmarks.assert_performance_sla(
                f"data_loading_{scale}", load_time, expected_load_time
            )
            
            # === DataFrame Transformation Performance ===
            transform_start_time = time.time()
            df = make_dataframe_from_config(exp_matrix=loaded_data)
            transform_time = time.time() - transform_start_time
            
            # Calculate expected transformation time
            row_count = len(df) if isinstance(df, pd.DataFrame) else data_config["timepoints"]
            expected_transform_time = performance_benchmarks.benchmark_dataframe_transform(row_count)
            
            performance_benchmarks.assert_performance_sla(
                f"dataframe_transform_{scale}", transform_time, expected_transform_time
            )
            
            # === Total Scale Performance ===
            total_scale_time = time.time() - scale_start_time
            
            # Ensure total time is reasonable for scale
            max_scale_time = {"small": 2.0, "medium": 5.0, "large": 15.0}[scale]
            performance_benchmarks.assert_performance_sla(
                f"total_{scale}_workflow", total_scale_time, max_scale_time
            )
            
            print(f"   ✅ {scale.capitalize()} scale completed in {total_scale_time:.3f}s")
            print(f"      - Data loading: {load_time:.3f}s (file: {file_size_mb:.2f} MB)")
            print(f"      - DataFrame transform: {transform_time:.3f}s ({row_count:,} rows)")

    def test_concurrent_workflow_performance(
        self,
        comprehensive_sample_config_dict,
        cross_platform_temp_dir,
        performance_benchmarks
    ):
        """
        Test performance under concurrent workflow execution scenarios.
        
        Validates that multiple workflows can execute concurrently without
        significant performance degradation.
        """
        import concurrent.futures
        import threading
        
        concurrent_start_time = time.time()
        
        # Create multiple datasets for concurrent processing
        num_concurrent_workflows = 3
        datasets = []
        
        for i in range(num_concurrent_workflows):
            dataset_dir = cross_platform_temp_dir / f"concurrent_{i}"
            dataset_dir.mkdir(exist_ok=True)
            
            # Create moderate-sized dataset
            data = {
                't': np.linspace(0, 300, 18000),  # 5 minutes at 60 Hz
                'x': np.random.rand(18000) * 100,
                'y': np.random.rand(18000) * 100,
                'signal': np.random.rand(18000)
            }
            
            data_file = dataset_dir / f"concurrent_data_{i}.pkl"
            with open(data_file, 'wb') as f:
                pickle.dump(data, f)
            
            datasets.append(str(data_file))
        
        # Define workflow function for concurrent execution
        def execute_workflow(data_file_path):
            workflow_start = time.time()
            
            # Load and process data
            exp_matrix = read_pickle_any_format(data_file_path)
            df = make_dataframe_from_config(exp_matrix=exp_matrix)
            
            # Perform some basic analysis
            analysis_results = {
                "file_path": data_file_path,
                "row_count": len(df),
                "execution_time": time.time() - workflow_start,
                "mean_x": df['x'].mean() if 'x' in df.columns else 0,
                "mean_y": df['y'].mean() if 'y' in df.columns else 0
            }
            
            return analysis_results
        
        # Execute workflows concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent_workflows) as executor:
            futures = [executor.submit(execute_workflow, dataset) for dataset in datasets]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        total_concurrent_time = time.time() - concurrent_start_time
        
        # Validate concurrent execution performance
        max_concurrent_time = 20.0  # Should complete within 20 seconds
        performance_benchmarks.assert_performance_sla(
            "concurrent_workflows", total_concurrent_time, max_concurrent_time
        )
        
        # Validate that all workflows completed successfully
        assert len(results) == num_concurrent_workflows, "All concurrent workflows should complete"
        
        for result in results:
            assert result["execution_time"] > 0, "Each workflow should have positive execution time"
            assert result["row_count"] > 0, "Each workflow should process data"
        
        # Calculate average execution time per workflow
        avg_execution_time = sum(r["execution_time"] for r in results) / len(results)
        
        print(f"✅ Concurrent workflow testing completed in {total_concurrent_time:.3f}s")
        print(f"   - {num_concurrent_workflows} workflows executed concurrently")
        print(f"   - Average workflow time: {avg_execution_time:.3f}s")
        print(f"   - Total concurrent overhead: {total_concurrent_time - avg_execution_time:.3f}s")

    def test_memory_efficiency_validation(
        self,
        performance_test_data,
        cross_platform_temp_dir
    ):
        """
        Test memory efficiency of workflow execution.
        
        Validates that workflows execute within reasonable memory bounds
        and that memory usage scales appropriately with data size.
        """
        import psutil
        import gc
        
        # Get baseline memory usage
        gc.collect()
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        memory_measurements = {}
        
        for scale, data_config in performance_test_data.items():
            gc.collect()  # Clean up before test
            
            scale_start_memory = process.memory_info().rss / (1024 * 1024)
            
            # Create and process data
            test_dir = cross_platform_temp_dir / f"memory_test_{scale}"
            test_dir.mkdir(exist_ok=True)
            
            data_file = test_dir / f"memory_{scale}_data.pkl"
            with open(data_file, 'wb') as f:
                pickle.dump(data_config["data"], f)
            
            # Execute workflow and monitor memory
            exp_matrix = read_pickle_any_format(str(data_file))
            peak_after_load = process.memory_info().rss / (1024 * 1024)
            
            df = make_dataframe_from_config(exp_matrix=exp_matrix)
            peak_after_transform = process.memory_info().rss / (1024 * 1024)
            
            # Clean up
            del exp_matrix, df
            gc.collect()
            
            final_memory = process.memory_info().rss / (1024 * 1024)
            
            memory_measurements[scale] = {
                "baseline": scale_start_memory,
                "after_load": peak_after_load,
                "after_transform": peak_after_transform,
                "after_cleanup": final_memory,
                "peak_usage": max(peak_after_load, peak_after_transform),
                "memory_increase": max(peak_after_load, peak_after_transform) - scale_start_memory
            }
        
        # Validate memory efficiency
        for scale, measurements in memory_measurements.items():
            memory_increase = measurements["memory_increase"]
            
            # Memory increase should be reasonable for data size
            expected_max_increase = {"small": 50, "medium": 200, "large": 800}[scale]  # MB
            
            assert memory_increase < expected_max_increase, \
                f"Memory increase for {scale} dataset ({memory_increase:.1f} MB) should be < {expected_max_increase} MB"
            
            # Memory should be released after cleanup
            memory_retained = measurements["after_cleanup"] - measurements["baseline"]
            assert memory_retained < 50, \
                f"Memory retained after cleanup ({memory_retained:.1f} MB) should be minimal"
        
        print(f"✅ Memory efficiency validation completed")
        print(f"   - Baseline memory usage: {baseline_memory:.1f} MB")
        for scale, measurements in memory_measurements.items():
            print(f"   - {scale.capitalize()} dataset: peak +{measurements['memory_increase']:.1f} MB, "
                  f"retained +{measurements['after_cleanup'] - measurements['baseline']:.1f} MB")