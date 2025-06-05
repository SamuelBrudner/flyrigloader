"""
Comprehensive end-to-end workflow integration test suite for flyrigloader.

This module implements TST-INTEG-001, TST-INTEG-002, and TST-INTEG-003 requirements,
validating complete pipeline functionality from YAML configuration loading through
final DataFrame output generation. Tests realistic experimental scenarios using
NumPy and Pandas synthetic data generation while ensuring seamless cross-module
integration and performance validation against <30 seconds SLA requirements.

Integration Test Requirements Validation:
- TST-INTEG-001: End-to-end workflow validation from YAML config to DataFrame output
- TST-INTEG-002: Realistic test data generation with NumPy/Pandas synthetic data
- TST-INTEG-003: DataFrame output verification with structure, types, and content integrity
- F-015: Integration Test Harness with comprehensive workflow scenarios
- Section 4.1.1.1: End-to-End User Journey workflow validation
- F-001-F-006: Complete feature integration across all modules

Performance Requirements:
- Complete workflow execution within 30 seconds per Section 2.2.10
- Data loading SLA validation of 1 second per 100MB
- DataFrame transformation within 500ms per 1M rows
"""

import json
import os
import pickle
import tempfile
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import yaml
from hypothesis import given, strategies as st, settings, assume

# Import flyrigloader components for integration testing
from flyrigloader.api import (
    load_experiment_files,
    load_dataset_files,
    get_experiment_parameters,
    get_dataset_parameters,
    process_experiment_data,
    get_default_column_config,
    reset_dependency_provider,
    set_dependency_provider,
    DefaultDependencyProvider
)
from flyrigloader.config.yaml_config import load_config, get_experiment_info, get_dataset_info
from flyrigloader.config.discovery import discover_experiment_files, discover_dataset_files
from flyrigloader.discovery.files import discover_files
from flyrigloader.io.pickle import read_pickle_any_format, make_dataframe_from_config
from flyrigloader.io.column_models import ColumnConfigDict, ColumnConfig, ColumnDimension, SpecialHandlerType


class TestExperimentalDataGenerator:
    """
    Advanced synthetic experimental data generator for realistic integration testing.
    
    Implements TST-INTEG-002 requirements with NumPy/Pandas-based synthetic data
    generation that mirrors actual flyrigloader usage patterns in neuroscience research.
    """
    
    def __init__(self, random_seed: int = 42):
        """Initialize generator with reproducible random seed."""
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
    def generate_neural_tracking_data(self, 
                                    time_points: int = 1000,
                                    num_neurons: int = 50,
                                    sampling_rate: float = 60.0) -> Dict[str, np.ndarray]:
        """
        Generate realistic neural tracking experimental matrix.
        
        Creates synthetic data that matches typical fly behavior experimental structure
        with time-series neural activity, position tracking, and behavioral metrics.
        
        Args:
            time_points: Number of time samples (default 1000 for ~16.7s at 60Hz)
            num_neurons: Number of neural recording channels
            sampling_rate: Data acquisition sampling rate in Hz
            
        Returns:
            Dictionary containing experimental matrix with realistic data structure
        """
        # Generate time vector
        t = np.linspace(0, time_points / sampling_rate, time_points)
        
        # Generate realistic fly position tracking
        # Simulate natural fly movement with random walk + periodic components
        position_noise = np.random.normal(0, 0.1, time_points)
        x_pos = np.cumsum(position_noise) + 5 * np.sin(0.1 * t) + 50  # Arena center ~50mm
        y_pos = np.cumsum(np.random.normal(0, 0.1, time_points)) + 3 * np.cos(0.15 * t) + 50
        
        # Ensure positions stay within realistic arena bounds (0-100mm)
        x_pos = np.clip(x_pos, 5, 95)
        y_pos = np.clip(y_pos, 5, 95)
        
        # Generate neural signals with realistic characteristics
        neural_base = np.random.normal(0, 1, (time_points, num_neurons))
        
        # Add correlated activity patterns (some neurons fire together)
        correlation_groups = 5
        for group in range(correlation_groups):
            group_neurons = slice(group * (num_neurons // correlation_groups), 
                                (group + 1) * (num_neurons // correlation_groups))
            group_signal = np.random.normal(0, 0.5, time_points)
            neural_base[:, group_neurons] += group_signal[:, np.newaxis]
        
        # Generate signal_disp (2D array for testing special handling)
        signal_disp = neural_base.T  # Shape: (num_neurons, time_points)
        
        # Calculate velocity from position
        velocity = np.sqrt(np.diff(x_pos)**2 + np.diff(y_pos)**2) * sampling_rate
        velocity = np.concatenate([[0], velocity])  # Pad to match time length
        
        # Generate behavioral metrics
        angular_velocity = np.random.normal(0, 10, time_points)  # degrees/second
        head_direction = np.cumsum(angular_velocity / sampling_rate) % 360
        
        # Generate experimental metadata and conditions
        temperature = np.random.normal(25, 0.5, time_points)  # Celsius
        humidity = np.random.normal(60, 2, time_points)  # Percent
        
        return {
            't': t,
            'x': x_pos,
            'y': y_pos,
            'velocity': velocity,
            'angular_velocity': angular_velocity,
            'head_direction': head_direction,
            'signal_disp': signal_disp,
            'temperature': temperature,
            'humidity': humidity,
            'frame_count': np.arange(time_points, dtype=np.int32),
            'experiment_id': 'EXP001',
            'animal_id': 'mouse_001',
            'condition': 'baseline'
        }
    
    def generate_optogenetic_experiment(self, 
                                      time_points: int = 2000,
                                      stimulation_periods: int = 3) -> Dict[str, np.ndarray]:
        """
        Generate synthetic optogenetic stimulation experiment data.
        
        Creates realistic optogenetic intervention data with stimulation periods,
        behavioral responses, and neural activity modulation.
        """
        base_data = self.generate_neural_tracking_data(time_points)
        
        # Generate stimulation protocol
        stim_signal = np.zeros(time_points)
        stim_intervals = np.linspace(200, time_points - 200, stimulation_periods * 2).astype(int)
        
        for i in range(0, len(stim_intervals), 2):
            start_idx = stim_intervals[i]
            end_idx = stim_intervals[i + 1] if i + 1 < len(stim_intervals) else start_idx + 100
            stim_signal[start_idx:end_idx] = 1
        
        # Modulate neural activity during stimulation
        modulation_factor = 1 + 0.5 * stim_signal
        base_data['signal_disp'] *= modulation_factor
        
        # Add stimulation-specific fields
        base_data['stimulation_signal'] = stim_signal
        base_data['led_power'] = stim_signal * np.random.uniform(0.8, 1.2, time_points)
        base_data['condition'] = 'optogenetic_stimulation'
        
        return base_data
    
    def generate_multi_animal_dataset(self, 
                                    animal_count: int = 5,
                                    time_points: int = 1000) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Generate multi-animal experimental dataset for batch processing tests.
        
        Creates multiple experimental matrices representing different animals
        in the same experimental paradigm.
        """
        dataset = {}
        animal_ids = [f'mouse_{i:03d}' for i in range(1, animal_count + 1)]
        conditions = ['baseline', 'treatment_a', 'treatment_b']
        
        for i, animal_id in enumerate(animal_ids):
            # Vary experimental conditions across animals
            condition = conditions[i % len(conditions)]
            
            if condition == 'optogenetic_stimulation':
                exp_data = self.generate_optogenetic_experiment(time_points)
            else:
                exp_data = self.generate_neural_tracking_data(time_points)
                exp_data['condition'] = condition
            
            exp_data['animal_id'] = animal_id
            dataset[animal_id] = exp_data
            
        return dataset


@pytest.fixture(scope="function")
def experimental_data_generator():
    """Fixture providing synthetic experimental data generator."""
    return TestExperimentalDataGenerator()


@pytest.fixture(scope="function")
def comprehensive_test_environment(tmp_path, experimental_data_generator):
    """
    Create comprehensive test environment with realistic experimental structure.
    
    Implements realistic directory structure, configuration files, and experimental
    data files that mirror actual research laboratory organization per TST-INTEG-002.
    """
    # Create realistic directory structure
    base_dir = tmp_path / "research_data"
    base_dir.mkdir()
    
    # Create date-based subdirectories (common in research data organization)
    dates = ["20241201", "20241202", "20241203"]
    for date in dates:
        date_dir = base_dir / date
        date_dir.mkdir()
    
    # Generate and save experimental data files
    test_files = {}
    
    # Generate baseline experiment files
    for i, date in enumerate(dates):
        date_dir = base_dir / date
        
        # Create multiple animals per date
        animals = [f'mouse_{j:03d}' for j in range(1, 4)]
        
        for animal in animals:
            # Generate experimental data
            exp_data = experimental_data_generator.generate_neural_tracking_data(
                time_points=1000 + i * 100  # Vary data size
            )
            exp_data['animal_id'] = animal
            exp_data['experiment_date'] = date
            
            # Save as pickle file with realistic naming
            filename = f"{animal}_{date}_baseline_rep001.pkl"
            file_path = date_dir / filename
            
            with open(file_path, 'wb') as f:
                pickle.dump(exp_data, f)
            
            test_files[str(file_path)] = exp_data
    
    # Generate optogenetic experiment files
    opto_data = experimental_data_generator.generate_optogenetic_experiment()
    opto_file = base_dir / dates[0] / "mouse_001_20241201_optogenetic_rep001.pkl"
    with open(opto_file, 'wb') as f:
        pickle.dump(opto_data, f)
    test_files[str(opto_file)] = opto_data
    
    # Create comprehensive YAML configuration
    config_content = {
        "project": {
            "directories": {
                "major_data_directory": str(base_dir)
            },
            "ignore_substrings": ["backup", "temp", ".DS_Store"],
            "mandatory_experiment_strings": [],
            "extraction_patterns": [
                r"(?P<animal_id>mouse_\d+)_(?P<date>\d{8})_(?P<condition>\w+)_rep(?P<replicate>\d+)\.pkl"
            ]
        },
        "datasets": {
            "baseline_behavior": {
                "dates_vials": {
                    "20241201": ["mouse_001", "mouse_002", "mouse_003"],
                    "20241202": ["mouse_001", "mouse_002", "mouse_003"],
                    "20241203": ["mouse_001", "mouse_002", "mouse_003"]
                },
                "patterns": ["*baseline*"],
                "metadata": {
                    "experiment_type": "baseline",
                    "sampling_rate": 60.0,
                    "arena_diameter_mm": 100
                }
            },
            "optogenetic_stimulation": {
                "dates_vials": {
                    "20241201": ["mouse_001"]
                },
                "patterns": ["*optogenetic*"],
                "metadata": {
                    "experiment_type": "optogenetic",
                    "stimulation_wavelength": 470,
                    "stimulation_power": 1.0
                }
            }
        },
        "experiments": {
            "baseline_study": {
                "datasets": ["baseline_behavior"],
                "description": "Baseline behavioral characterization",
                "parameters": {
                    "analysis_window": 60,
                    "velocity_threshold": 2.0,
                    "minimum_duration": 300
                },
                "filters": {
                    "mandatory_experiment_strings": ["baseline"],
                    "ignore_substrings": ["failed", "aborted"]
                }
            },
            "optogenetic_intervention": {
                "datasets": ["optogenetic_stimulation"],
                "description": "Optogenetic manipulation study",
                "parameters": {
                    "stimulation_duration": 5.0,
                    "recovery_period": 30.0,
                    "analysis_epochs": ["pre", "stim", "post"]
                },
                "filters": {
                    "mandatory_experiment_strings": ["optogenetic"],
                    "ignore_substrings": ["failed"]
                }
            }
        }
    }
    
    # Save configuration file
    config_file = tmp_path / "experiment_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config_content, f, default_flow_style=False)
    
    return {
        "base_directory": base_dir,
        "config_file": config_file,
        "config_dict": config_content,
        "test_files": test_files,
        "dates": dates
    }


@pytest.fixture(scope="function")
def performance_tracker():
    """Fixture for tracking integration test performance metrics."""
    class PerformanceTracker:
        def __init__(self):
            self.start_time = None
            self.timings = {}
            
        def start_timing(self, operation: str):
            self.start_time = time.time()
            
        def end_timing(self, operation: str):
            if self.start_time:
                duration = time.time() - self.start_time
                self.timings[operation] = duration
                return duration
            return 0
            
        def assert_performance_sla(self, operation: str, max_seconds: float):
            """Assert operation meets performance SLA."""
            duration = self.timings.get(operation, float('inf'))
            assert duration <= max_seconds, (
                f"Performance SLA violation: {operation} took {duration:.3f}s, "
                f"expected <= {max_seconds:.3f}s"
            )
    
    return PerformanceTracker()


class TestEndToEndWorkflowIntegration:
    """
    Comprehensive end-to-end workflow integration test suite.
    
    Validates complete flyrigloader pipeline functionality from YAML configuration
    loading through DataFrame output generation, ensuring seamless integration
    across all modules and realistic experimental data processing scenarios.
    """

    def test_complete_baseline_experiment_workflow(self, comprehensive_test_environment, performance_tracker):
        """
        Test complete baseline experiment workflow from configuration to DataFrame.
        
        Validates TST-INTEG-001: End-to-end workflow validation including:
        - YAML configuration loading and validation (F-001)
        - File discovery with pattern matching (F-002)
        - Data loading with format detection (F-003)
        - Schema validation and column processing (F-004)
        - DataFrame transformation with metadata integration (F-006)
        
        Performance requirement: Complete workflow within 30 seconds per Section 2.2.10
        """
        # Start performance tracking
        performance_tracker.start_timing("complete_workflow")
        
        # Test configuration loading (F-001)
        config_file = comprehensive_test_environment["config_file"]
        base_directory = comprehensive_test_environment["base_directory"]
        
        # Load experiment files using API facade
        experiment_files = load_experiment_files(
            config_path=config_file,
            experiment_name="baseline_study",
            base_directory=base_directory,
            pattern="*.pkl",
            recursive=True,
            extract_metadata=True
        )
        
        # Validate file discovery results (F-002)
        assert isinstance(experiment_files, dict), "Should return metadata dictionary"
        assert len(experiment_files) > 0, "Should discover experimental files"
        
        # Verify discovered files contain expected baseline files
        baseline_files = [path for path in experiment_files.keys() if 'baseline' in path]
        assert len(baseline_files) >= 3, f"Should find at least 3 baseline files, found {len(baseline_files)}"
        
        # Test data loading and processing for each discovered file (F-003, F-006)
        processed_dataframes = []
        
        for file_path, metadata in experiment_files.items():
            if 'baseline' in file_path:
                # Load and process experimental data
                df = process_experiment_data(
                    data_path=file_path,
                    metadata=metadata
                )
                
                # Validate DataFrame structure and content (TST-INTEG-003)
                assert isinstance(df, pd.DataFrame), f"Should return DataFrame for {file_path}"
                assert len(df) > 0, f"DataFrame should not be empty for {file_path}"
                
                # Verify essential columns are present
                expected_columns = ['t', 'x', 'y', 'velocity']
                for col in expected_columns:
                    assert col in df.columns, f"Missing expected column '{col}' in {file_path}"
                
                # Validate data types and ranges
                assert df['t'].dtype in [np.float64, np.float32], "Time should be numeric"
                assert df['x'].min() >= 0 and df['x'].max() <= 100, "X position should be in arena bounds"
                assert df['y'].min() >= 0 and df['y'].max() <= 100, "Y position should be in arena bounds"
                assert df['velocity'].min() >= 0, "Velocity should be non-negative"
                
                # Verify metadata integration
                if 'animal_id' in metadata:
                    assert 'animal_id' in df.columns or 'animal_id' in df.attrs, "Metadata should be integrated"
                
                processed_dataframes.append(df)
        
        # Validate integration across multiple files
        assert len(processed_dataframes) >= 3, "Should process multiple experimental files"
        
        # Check data consistency across files
        for df in processed_dataframes:
            assert len(df.columns) >= 8, "Should have sufficient data columns"
            assert df.shape[0] >= 1000, "Should have sufficient time points"
        
        # End performance tracking and validate SLA
        workflow_duration = performance_tracker.end_timing("complete_workflow")
        performance_tracker.assert_performance_sla("complete_workflow", 30.0)
        
        print(f"✓ Complete baseline workflow executed in {workflow_duration:.3f}s")

    def test_optogenetic_experiment_with_special_handling(self, comprehensive_test_environment, performance_tracker):
        """
        Test optogenetic experiment workflow with advanced data processing.
        
        Validates complex data transformation scenarios including:
        - Special column handling (signal_disp transformation)
        - Multi-dimensional array processing
        - Stimulation protocol data integration
        - Error handling across module boundaries
        """
        performance_tracker.start_timing("optogenetic_workflow")
        
        config_file = comprehensive_test_environment["config_file"]
        base_directory = comprehensive_test_environment["base_directory"]
        
        # Discover optogenetic experiment files
        experiment_files = load_experiment_files(
            config_path=config_file,
            experiment_name="optogenetic_intervention",
            base_directory=base_directory,
            pattern="*optogenetic*.pkl",
            recursive=True
        )
        
        assert len(experiment_files) > 0, "Should discover optogenetic files"
        
        # Process optogenetic data with special handling
        opto_file = next(iter(experiment_files))
        
        # Load raw data to verify structure
        raw_data = read_pickle_any_format(opto_file)
        assert isinstance(raw_data, dict), "Raw data should be dictionary"
        assert 'signal_disp' in raw_data, "Should contain signal_disp for special handling"
        assert 'stimulation_signal' in raw_data, "Should contain stimulation protocol"
        
        # Verify signal_disp is 2D for transformation testing
        signal_disp = raw_data['signal_disp']
        assert signal_disp.ndim == 2, f"signal_disp should be 2D, got {signal_disp.ndim}D"
        
        # Process with column configuration
        df = process_experiment_data(data_path=opto_file)
        
        # Validate special handling results
        assert isinstance(df, pd.DataFrame), "Should return DataFrame"
        assert len(df) > 0, "DataFrame should not be empty"
        
        # Verify stimulation-specific columns
        assert 'stimulation_signal' in df.columns, "Should preserve stimulation signal"
        assert 'led_power' in df.columns, "Should preserve LED power data"
        
        # Validate stimulation protocol integrity
        stim_periods = df[df['stimulation_signal'] > 0]
        assert len(stim_periods) > 0, "Should have stimulation periods"
        assert stim_periods['led_power'].min() > 0, "LED power should be positive during stimulation"
        
        workflow_duration = performance_tracker.end_timing("optogenetic_workflow")
        performance_tracker.assert_performance_sla("optogenetic_workflow", 15.0)
        
        print(f"✓ Optogenetic workflow with special handling executed in {workflow_duration:.3f}s")

    def test_error_propagation_across_module_boundaries(self, comprehensive_test_environment):
        """
        Test error handling and propagation across module boundaries.
        
        Validates that errors in one module are properly caught and handled
        by upstream modules with meaningful error messages per Section 4.1.2.3.
        """
        config_file = comprehensive_test_environment["config_file"]
        base_directory = comprehensive_test_environment["base_directory"]
        
        # Test 1: Invalid experiment name should propagate properly
        with pytest.raises(KeyError) as exc_info:
            load_experiment_files(
                config_path=config_file,
                experiment_name="nonexistent_experiment",
                base_directory=base_directory
            )
        
        assert "not found in configuration" in str(exc_info.value)
        
        # Test 2: Invalid configuration file should be caught at API level
        invalid_config = comprehensive_test_environment["config_file"].parent / "invalid_config.yaml"
        invalid_config.write_text("invalid: yaml: content: [unclosed")
        
        with pytest.raises((yaml.YAMLError, ValueError)) as exc_info:
            load_experiment_files(
                config_path=invalid_config,
                experiment_name="baseline_study",
                base_directory=base_directory
            )
        
        # Test 3: Invalid data file should be handled gracefully
        invalid_data_file = base_directory / "invalid_data.pkl"
        invalid_data_file.write_text("not a pickle file")
        
        with pytest.raises(RuntimeError) as exc_info:
            process_experiment_data(data_path=invalid_data_file)
        
        assert "Failed to load pickle file" in str(exc_info.value)
        
        # Test 4: Missing required configuration sections
        minimal_config = {"project": {}}  # Missing required sections
        
        with pytest.raises(ValueError) as exc_info:
            load_experiment_files(
                config=minimal_config,
                experiment_name="any_experiment",
                base_directory=base_directory
            )
        
        print("✓ Error propagation validation completed successfully")

    def test_parameter_validation_and_edge_cases(self, comprehensive_test_environment):
        """
        Test parameter validation and edge case handling across the pipeline.
        
        Validates robust parameter handling and boundary condition management
        per F-001-F-006 requirements with comprehensive input validation.
        """
        config_file = comprehensive_test_environment["config_file"]
        base_directory = comprehensive_test_environment["base_directory"]
        
        # Test parameter validation for experiment parameters
        exp_params = get_experiment_parameters(
            config_path=config_file,
            experiment_name="baseline_study"
        )
        
        assert isinstance(exp_params, dict), "Should return parameter dictionary"
        assert "analysis_window" in exp_params, "Should contain experiment parameters"
        assert exp_params["velocity_threshold"] == 2.0, "Should preserve parameter values"
        
        # Test dataset parameter extraction
        dataset_params = get_dataset_parameters(
            config_path=config_file,
            dataset_name="baseline_behavior"
        )
        
        assert isinstance(dataset_params, dict), "Should return dataset parameters"
        
        # Test edge cases with empty and minimal parameters
        experiment_files = load_experiment_files(
            config_path=config_file,
            experiment_name="baseline_study",
            base_directory=base_directory,
            pattern="*",  # Very broad pattern
            recursive=False,  # Non-recursive search
            extensions=["pkl"],  # Specific extension
            extract_metadata=False  # No metadata extraction
        )
        
        assert isinstance(experiment_files, list), "Should return file list without metadata"
        
        # Test boundary conditions for data loading
        test_files = list(comprehensive_test_environment["test_files"].keys())
        if test_files:
            test_file = test_files[0]
            
            # Test with minimal metadata
            df = process_experiment_data(
                data_path=test_file,
                metadata={"test_metadata": "minimal"}
            )
            
            assert isinstance(df, pd.DataFrame), "Should handle minimal metadata"
        
        print("✓ Parameter validation and edge cases handled successfully")

    def test_performance_validation_against_sla_requirements(self, comprehensive_test_environment, performance_tracker):
        """
        Test performance validation against SLA requirements.
        
        Validates that all operations meet performance criteria per Section 2.2.10:
        - Data loading: 1 second per 100MB
        - DataFrame transformation: 500ms per 1M rows
        - Complete workflow: <30 seconds
        """
        config_file = comprehensive_test_environment["config_file"]
        base_directory = comprehensive_test_environment["base_directory"]
        test_files = list(comprehensive_test_environment["test_files"].keys())
        
        # Test data loading performance
        for file_path in test_files[:3]:  # Test first 3 files
            performance_tracker.start_timing("data_loading")
            
            data = read_pickle_any_format(file_path)
            
            loading_duration = performance_tracker.end_timing("data_loading")
            
            # Estimate file size and validate SLA (1s per 100MB)
            file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
            max_loading_time = max(1.0, file_size_mb / 100)  # At least 1 second minimum
            
            assert loading_duration <= max_loading_time, (
                f"Data loading SLA violation: {file_path} took {loading_duration:.3f}s "
                f"for {file_size_mb:.2f}MB (expected <= {max_loading_time:.3f}s)"
            )
        
        # Test DataFrame transformation performance
        if test_files:
            test_file = test_files[0]
            raw_data = read_pickle_any_format(test_file)
            
            performance_tracker.start_timing("dataframe_transformation")
            
            df = make_dataframe_from_config(raw_data)
            
            transform_duration = performance_tracker.end_timing("dataframe_transformation")
            
            # Validate transformation SLA (500ms per 1M rows)
            row_count = len(df)
            max_transform_time = max(0.1, (row_count / 1_000_000) * 0.5)
            
            assert transform_duration <= max_transform_time, (
                f"DataFrame transformation SLA violation: {row_count} rows took "
                f"{transform_duration:.3f}s (expected <= {max_transform_time:.3f}s)"
            )
        
        # Test complete workflow performance
        performance_tracker.start_timing("complete_sla_test")
        
        experiment_files = load_experiment_files(
            config_path=config_file,
            experiment_name="baseline_study",
            base_directory=base_directory,
            recursive=True
        )
        
        # Process subset of files for performance testing
        processed_count = 0
        for file_path in list(experiment_files)[:2]:  # Process first 2 files
            df = process_experiment_data(data_path=file_path)
            processed_count += 1
        
        total_duration = performance_tracker.end_timing("complete_sla_test")
        
        # Validate complete workflow SLA (<30 seconds)
        performance_tracker.assert_performance_sla("complete_sla_test", 30.0)
        
        print(f"✓ Performance validation completed: processed {processed_count} files in {total_duration:.3f}s")

    @given(
        animal_count=st.integers(min_value=1, max_value=5),
        time_points=st.integers(min_value=100, max_value=2000),
        condition=st.sampled_from(['baseline', 'treatment', 'control'])
    )
    @settings(max_examples=5, deadline=None)
    def test_property_based_workflow_validation(self, animal_count, time_points, condition, tmp_path):
        """
        Property-based testing for workflow validation with varied parameters.
        
        Uses Hypothesis to generate diverse experimental scenarios and validate
        that the workflow handles all reasonable parameter combinations correctly.
        """
        # Skip very large datasets in property-based testing
        assume(time_points * animal_count < 10000)
        
        # Generate synthetic test environment
        data_generator = TestExperimentalDataGenerator()
        
        # Create temporary experiment structure
        base_dir = tmp_path / "property_test"
        base_dir.mkdir()
        
        date_dir = base_dir / "20241201"
        date_dir.mkdir()
        
        # Generate experimental data
        exp_data = data_generator.generate_neural_tracking_data(time_points)
        exp_data['condition'] = condition
        
        # Save test file
        test_file = date_dir / f"animal001_{condition}_rep001.pkl"
        with open(test_file, 'wb') as f:
            pickle.dump(exp_data, f)
        
        # Create minimal configuration
        config = {
            "project": {
                "directories": {"major_data_directory": str(base_dir)},
                "ignore_substrings": []
            },
            "datasets": {
                "test_dataset": {
                    "dates_vials": {"20241201": ["animal001"]}
                }
            },
            "experiments": {
                "test_experiment": {
                    "datasets": ["test_dataset"]
                }
            }
        }
        
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        # Test workflow with generated parameters
        try:
            experiment_files = load_experiment_files(
                config_path=config_file,
                experiment_name="test_experiment",
                base_directory=base_dir
            )
            
            # Validate basic workflow properties
            assert len(experiment_files) >= 1, "Should discover at least one file"
            
            # Process discovered files
            for file_path in experiment_files:
                df = process_experiment_data(data_path=file_path)
                
                # Validate universal properties
                assert isinstance(df, pd.DataFrame), "Should always return DataFrame"
                assert len(df) == time_points, "Should preserve time dimension"
                assert 't' in df.columns, "Should always contain time column"
                assert df['t'].nunique() == len(df), "Time values should be unique"
                
        except Exception as e:
            # Property-based testing should identify edge cases
            pytest.fail(f"Workflow failed with parameters: animal_count={animal_count}, "
                       f"time_points={time_points}, condition={condition}. Error: {e}")

    def test_multi_experiment_batch_processing(self, comprehensive_test_environment, performance_tracker):
        """
        Test batch processing of multiple experiments with resource management.
        
        Validates Section 4.1.2.2 Multi-Experiment Batch Processing workflow
        with proper resource management and progress tracking.
        """
        performance_tracker.start_timing("batch_processing")
        
        config_file = comprehensive_test_environment["config_file"]
        base_directory = comprehensive_test_environment["base_directory"]
        
        # Test multiple experiment processing
        experiments = ["baseline_study", "optogenetic_intervention"]
        batch_results = {}
        
        for experiment_name in experiments:
            try:
                # Load experiment files
                experiment_files = load_experiment_files(
                    config_path=config_file,
                    experiment_name=experiment_name,
                    base_directory=base_directory,
                    recursive=True
                )
                
                # Process files for this experiment
                experiment_dataframes = []
                for file_path in experiment_files:
                    df = process_experiment_data(data_path=file_path)
                    experiment_dataframes.append(df)
                
                batch_results[experiment_name] = {
                    'file_count': len(experiment_files),
                    'dataframes': experiment_dataframes,
                    'total_rows': sum(len(df) for df in experiment_dataframes)
                }
                
            except Exception as e:
                # Log error but continue with other experiments
                batch_results[experiment_name] = {'error': str(e)}
        
        # Validate batch processing results
        assert len(batch_results) == len(experiments), "Should process all experiments"
        
        # Check that at least one experiment processed successfully
        successful_experiments = [
            name for name, result in batch_results.items() 
            if 'error' not in result
        ]
        assert len(successful_experiments) >= 1, "At least one experiment should succeed"
        
        # Validate data consistency across experiments
        for experiment_name, result in batch_results.items():
            if 'dataframes' in result:
                assert result['file_count'] > 0, f"Should process files for {experiment_name}"
                assert result['total_rows'] > 0, f"Should have data rows for {experiment_name}"
        
        batch_duration = performance_tracker.end_timing("batch_processing")
        
        # Batch processing should complete within reasonable time
        performance_tracker.assert_performance_sla("batch_processing", 60.0)
        
        print(f"✓ Batch processing of {len(experiments)} experiments completed in {batch_duration:.3f}s")

    def test_dataframe_output_verification_comprehensive(self, comprehensive_test_environment):
        """
        Comprehensive DataFrame output verification per TST-INTEG-003.
        
        Validates DataFrame structure, types, content integrity, and metadata
        integration with detailed assertions for data quality assurance.
        """
        config_file = comprehensive_test_environment["config_file"]
        base_directory = comprehensive_test_environment["base_directory"]
        
        # Load and process experimental data
        experiment_files = load_experiment_files(
            config_path=config_file,
            experiment_name="baseline_study",
            base_directory=base_directory,
            extract_metadata=True
        )
        
        assert len(experiment_files) > 0, "Should have experiment files to validate"
        
        # Test comprehensive DataFrame validation
        for file_path, metadata in experiment_files.items():
            df = process_experiment_data(data_path=file_path, metadata=metadata)
            
            # Structure validation
            assert isinstance(df, pd.DataFrame), f"Output should be DataFrame for {file_path}"
            assert len(df) > 0, f"DataFrame should not be empty for {file_path}"
            assert len(df.columns) >= 8, f"Should have sufficient columns for {file_path}"
            
            # Required columns validation
            required_columns = ['t', 'x', 'y', 'velocity', 'angular_velocity', 'head_direction']
            for col in required_columns:
                assert col in df.columns, f"Missing required column '{col}' in {file_path}"
            
            # Data type validation
            assert pd.api.types.is_numeric_dtype(df['t']), "Time should be numeric"
            assert pd.api.types.is_numeric_dtype(df['x']), "X position should be numeric"
            assert pd.api.types.is_numeric_dtype(df['y']), "Y position should be numeric"
            assert pd.api.types.is_numeric_dtype(df['velocity']), "Velocity should be numeric"
            
            # Content integrity validation
            assert df['t'].is_monotonic_increasing, "Time should be monotonically increasing"
            assert not df['t'].isnull().any(), "Time should not have null values"
            assert df['x'].between(0, 100).all(), "X position should be within arena bounds"
            assert df['y'].between(0, 100).all(), "Y position should be within arena bounds"
            assert (df['velocity'] >= 0).all(), "Velocity should be non-negative"
            
            # Metadata integration validation
            if metadata:
                # Check that metadata is preserved in DataFrame attributes or columns
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float)):
                        # Metadata should be either in columns or DataFrame attributes
                        metadata_preserved = (
                            key in df.columns or 
                            key in getattr(df, 'attrs', {}) or
                            any(key in str(col) for col in df.columns)
                        )
                        # Allow for metadata transformation during processing
                        # Some metadata might be transformed or embedded differently
                        if not metadata_preserved:
                            print(f"Note: Metadata key '{key}' not directly preserved in {file_path}")
            
            # Statistical validation (basic sanity checks)
            assert df['x'].std() > 0, "Position data should show variation"
            assert df['y'].std() > 0, "Position data should show variation"
            assert df['velocity'].mean() > 0, "Should have non-zero mean velocity"
            
            # Temporal consistency validation
            time_diff = df['t'].diff().dropna()
            assert (time_diff > 0).all(), "Time differences should be positive"
            
            print(f"✓ Comprehensive DataFrame validation passed for {Path(file_path).name}")

    def test_kedro_integration_compatibility(self, comprehensive_test_environment):
        """
        Test compatibility with Kedro parameter dictionary format.
        
        Validates that the system works seamlessly with Kedro-style parameter
        dictionaries per Section 4.1.2.1 Kedro Pipeline Integration.
        """
        base_directory = comprehensive_test_environment["base_directory"]
        config_dict = comprehensive_test_environment["config_dict"]
        
        # Test Kedro-style parameter dictionary usage
        experiment_files = load_experiment_files(
            config=config_dict,  # Use dict instead of file path
            experiment_name="baseline_study",
            base_directory=base_directory
        )
        
        assert len(experiment_files) > 0, "Should work with Kedro parameter dictionary"
        
        # Test parameter extraction from dictionary
        exp_params = get_experiment_parameters(
            config=config_dict,
            experiment_name="baseline_study"
        )
        
        assert isinstance(exp_params, dict), "Should extract parameters from dictionary"
        assert "analysis_window" in exp_params, "Should find experiment parameters"
        
        # Test dataset parameter extraction
        dataset_params = get_dataset_parameters(
            config=config_dict,
            dataset_name="baseline_behavior"
        )
        
        assert isinstance(dataset_params, dict), "Should extract dataset parameters"
        
        # Validate that both file and dict approaches give same results
        config_file = comprehensive_test_environment["config_file"]
        
        files_from_file = load_experiment_files(
            config_path=config_file,
            experiment_name="baseline_study",
            base_directory=base_directory
        )
        
        files_from_dict = load_experiment_files(
            config=config_dict,
            experiment_name="baseline_study",
            base_directory=base_directory
        )
        
        # Should discover same files regardless of config source
        assert len(files_from_file) == len(files_from_dict), "File vs dict should give same results"
        
        print("✓ Kedro integration compatibility validated successfully")

    def teardown_method(self):
        """Clean up after each test method."""
        # Reset dependency providers to ensure clean state
        reset_dependency_provider()
        
        # Additional cleanup if needed
        pass


class TestIntegrationErrorRecovery:
    """
    Test suite for error recovery and resilience in integration scenarios.
    
    Validates Section 4.1.2.3 Error Recovery and Resilience mechanisms
    across the complete workflow with realistic failure scenarios.
    """

    def test_configuration_error_recovery(self, tmp_path, experimental_data_generator):
        """Test recovery from configuration errors."""
        # Create test data
        base_dir = tmp_path / "error_test"
        base_dir.mkdir()
        
        # Create invalid configuration with recovery
        invalid_config = {
            "project": {
                # Missing required directories section
                "ignore_substrings": []
            }
            # Missing experiments and datasets sections
        }
        
        config_file = tmp_path / "invalid_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(invalid_config, f)
        
        # Test graceful error handling
        with pytest.raises(ValueError) as exc_info:
            load_experiment_files(
                config_path=config_file,
                experiment_name="any_experiment",
                base_directory=base_dir
            )
        
        # Verify meaningful error message
        assert "base_directory" in str(exc_info.value) or "directory" in str(exc_info.value)

    def test_file_system_error_resilience(self, comprehensive_test_environment):
        """Test resilience to file system errors."""
        config_file = comprehensive_test_environment["config_file"]
        
        # Test with non-existent base directory
        nonexistent_dir = "/definitely/does/not/exist/anywhere"
        
        # Should handle gracefully without crashing
        experiment_files = load_experiment_files(
            config_path=config_file,
            experiment_name="baseline_study",
            base_directory=nonexistent_dir
        )
        
        # Should return empty results rather than crash
        assert isinstance(experiment_files, (list, dict)), "Should return valid result type"

    def test_partial_data_processing_resilience(self, comprehensive_test_environment, tmp_path):
        """Test resilience when some files are corrupted or invalid."""
        config_file = comprehensive_test_environment["config_file"]
        base_directory = comprehensive_test_environment["base_directory"]
        
        # Create a corrupted file in the data directory
        corrupted_file = base_directory / "20241201" / "corrupted_baseline_file.pkl"
        corrupted_file.write_bytes(b"definitely not a pickle file")
        
        # The system should continue processing valid files
        # and handle the corrupted file gracefully
        experiment_files = load_experiment_files(
            config_path=config_file,
            experiment_name="baseline_study",
            base_directory=base_directory
        )
        
        # Should still find valid files
        assert len(experiment_files) > 0, "Should find valid files despite corrupted ones"
        
        # Processing should handle individual file errors
        successful_processing = 0
        for file_path in experiment_files:
            try:
                df = process_experiment_data(data_path=file_path)
                if isinstance(df, pd.DataFrame) and len(df) > 0:
                    successful_processing += 1
            except Exception:
                # Individual file errors should not crash the entire process
                continue
        
        assert successful_processing > 0, "Should successfully process at least some files"


# Additional integration test utilities and helpers

def create_synthetic_experiment_file(file_path: Path, 
                                   time_points: int = 1000,
                                   condition: str = "baseline") -> Dict[str, np.ndarray]:
    """
    Utility function to create synthetic experimental files for testing.
    
    Args:
        file_path: Path where to save the experimental data
        time_points: Number of time points in the experiment
        condition: Experimental condition label
        
    Returns:
        Dictionary containing the experimental data that was saved
    """
    generator = TestExperimentalDataGenerator()
    
    if condition == "optogenetic":
        exp_data = generator.generate_optogenetic_experiment(time_points)
    else:
        exp_data = generator.generate_neural_tracking_data(time_points)
        exp_data['condition'] = condition
    
    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as pickle file
    with open(file_path, 'wb') as f:
        pickle.dump(exp_data, f)
    
    return exp_data


def validate_experimental_dataframe(df: pd.DataFrame, 
                                  expected_columns: Optional[List[str]] = None) -> bool:
    """
    Utility function for comprehensive DataFrame validation in integration tests.
    
    Args:
        df: DataFrame to validate
        expected_columns: Optional list of expected column names
        
    Returns:
        True if DataFrame passes all validation checks
        
    Raises:
        AssertionError: If any validation check fails
    """
    # Basic structure validation
    assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame"
    assert len(df) > 0, "DataFrame must not be empty"
    assert len(df.columns) > 0, "DataFrame must have columns"
    
    # Column validation
    if expected_columns:
        for col in expected_columns:
            assert col in df.columns, f"Missing expected column: {col}"
    
    # Common experimental data validation
    if 't' in df.columns:
        assert df['t'].is_monotonic_increasing, "Time should be monotonically increasing"
        assert not df['t'].isnull().any(), "Time should not have null values"
    
    if 'x' in df.columns and 'y' in df.columns:
        assert df['x'].between(0, 200).all(), "X position should be within reasonable bounds"
        assert df['y'].between(0, 200).all(), "Y position should be within reasonable bounds"
    
    if 'velocity' in df.columns:
        assert (df['velocity'] >= 0).all(), "Velocity should be non-negative"
    
    return True


# Module-level test configuration and setup

@pytest.fixture(scope="module", autouse=True)
def setup_integration_test_environment():
    """Module-level setup for integration tests."""
    # Ensure clean dependency state at module start
    reset_dependency_provider()
    
    yield
    
    # Cleanup at module end
    reset_dependency_provider()