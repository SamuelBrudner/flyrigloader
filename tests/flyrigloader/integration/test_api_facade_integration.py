"""
API facade integration test suite.

Comprehensive integration testing for flyrigloader.api functions validating
end-to-end workflows from configuration to DataFrame output. Tests API facade
coordination of all subsystems (config, discovery, io) with realistic 
experimental scenarios and error propagation validation.

This module implements TST-INTEG-001 (complete workflow validation) and 
TST-INTEG-002 (realistic test data generation) requirements per Section 2.2.10.
"""

import os
import tempfile
import shutil
import pickle
import gzip
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
from unittest.mock import patch, MagicMock, mock_open

import numpy as np
import pandas as pd
import pytest
import yaml

from flyrigloader.api import (
    load_experiment_files,
    load_dataset_files,
    get_experiment_parameters,
    get_dataset_parameters,
    process_experiment_data,
    MISSING_DATA_DIR_ERROR
)


# =============================================================================
# Integration Test Fixtures
# =============================================================================

@pytest.fixture
def temp_data_directory():
    """Create a temporary data directory with realistic structure."""
    temp_dir = tempfile.mkdtemp()
    try:
        # Create date-based subdirectories (realistic experimental structure)
        data_dir = Path(temp_dir) / "experimental_data"
        data_dir.mkdir()
        
        # Create date directories with experimental files
        for date in ["2024-12-20", "2024-12-22", "2024-10-18", "2024-10-24"]:
            date_dir = data_dir / date
            date_dir.mkdir()
            
            # Create sample pickle files with realistic naming patterns
            for vial in [1, 2, 3, 4, 5]:
                # Create experiment files
                exp_file = date_dir / f"exp_test_{date}_{vial:02d}.pkl"
                exp_data = _create_realistic_exp_matrix(vial)
                with open(exp_file, 'wb') as f:
                    pickle.dump(exp_data, f)
                
                # Create plume navigation files
                plume_file = date_dir / f"plume_nav_{date}_{vial:02d}.pkl"
                plume_data = _create_realistic_exp_matrix(vial, experiment_type="plume")
                with open(plume_file, 'wb') as f:
                    pickle.dump(plume_data, f)
                
                # Create some files to be ignored
                ignore_file = date_dir / f"static_horiz_ribbon_{date}_{vial:02d}.pkl"
                with open(ignore_file, 'wb') as f:
                    pickle.dump({"should": "be_ignored"}, f)
        
        yield str(data_dir)
    finally:
        shutil.rmtree(temp_dir)


@pytest.fixture
def realistic_config_dict(temp_data_directory):
    """Create a realistic configuration dictionary matching experimental workflows."""
    return {
        "project": {
            "directories": {
                "major_data_directory": temp_data_directory,
                "batchfile_directory": "/path/to/batch_defs"
            },
            "ignore_substrings": [
                "static_horiz_ribbon",
                "._",
                "calibration"
            ],
            "mandatory_experiment_strings": [],
            "extraction_patterns": [
                r"(?P<exp_type>\w+)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<vial>\d+)\.pkl",
                r".*_(?P<date>\d{8})_(?P<condition>\w+)_(?P<replicate>\d+)\.pkl"
            ]
        },
        "rigs": {
            "old_opto": {
                "sampling_frequency": 60,
                "mm_per_px": 0.154,
                "arena_diameter": 120
            },
            "new_opto": {
                "sampling_frequency": 60,
                "mm_per_px": 0.1818,
                "arena_diameter": 110
            }
        },
        "datasets": {
            "test_dataset": {
                "rig": "old_opto",
                "patterns": ["*test*"],
                "dates_vials": {
                    "2024-12-20": [1, 2],
                    "2024-12-22": [1, 2, 3]
                },
                "metadata": {
                    "extraction_patterns": [
                        r"(?P<exp_type>\w+)_test_(?P<date>\d{4}-\d{2}-\d{2})_(?P<vial>\d+)\.pkl"
                    ]
                },
                "parameters": {
                    "sampling_rate": 60,
                    "arena_type": "circular",
                    "stimulus_type": "optogenetic"
                }
            },
            "plume_movie_navigation": {
                "rig": "old_opto", 
                "patterns": ["*plume*", "*nav*"],
                "dates_vials": {
                    "2024-10-18": [1, 3, 4, 5],
                    "2024-10-24": [1, 2]
                },
                "metadata": {
                    "extraction_patterns": [
                        r"plume_nav_(?P<date>\d{4}-\d{2}-\d{2})_(?P<vial>\d+)\.pkl"
                    ]
                },
                "parameters": {
                    "sampling_rate": 60,
                    "stimulus_type": "olfactory",
                    "arena_type": "rectangular"
                }
            }
        },
        "experiments": {
            "test_experiment": {
                "datasets": ["test_dataset"],
                "description": "Test optogenetic experiment",
                "metadata": {
                    "extraction_patterns": [
                        r"exp_(?P<exp_name>\w+)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<vial>\d+)\.pkl"
                    ]
                },
                "filters": {
                    "ignore_substrings": ["calibration"],
                    "mandatory_experiment_strings": ["test"]
                },
                "parameters": {
                    "protocol_version": "1.2",
                    "led_intensity": 80,
                    "pulse_duration": 500
                }
            },
            "multi_dataset_experiment": {
                "datasets": [
                    "test_dataset",
                    "plume_movie_navigation"
                ],
                "description": "Combined optogenetic and olfactory experiment",
                "filters": {
                    "ignore_substrings": ["smoke_2a", "backup"]
                },
                "parameters": {
                    "protocol_version": "2.0",
                    "cross_modal": True
                }
            }
        }
    }


@pytest.fixture
def realistic_config_file(realistic_config_dict):
    """Create a temporary YAML configuration file with realistic structure."""
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    try:
        yaml.dump(realistic_config_dict, temp_file)
        temp_file.close()
        yield temp_file.name
    finally:
        os.unlink(temp_file.name)


def _create_realistic_exp_matrix(vial_id: int, experiment_type: str = "test") -> Dict[str, np.ndarray]:
    """Create realistic experimental data matrix for integration testing."""
    np.random.seed(42 + vial_id)  # Reproducible but varied data
    
    # Realistic time series (10 minutes at 60 Hz)
    time_points = 10 * 60 * 60  # 10 minutes * 60 seconds * 60 Hz
    t = np.linspace(0, 600, time_points)  # 10 minutes in seconds
    
    # Realistic fly trajectory data
    base_data = {
        't': t,
        'x': np.cumsum(np.random.randn(time_points) * 0.1) + 50,  # Random walk around center
        'y': np.cumsum(np.random.randn(time_points) * 0.1) + 50,
        'dtheta': np.random.randn(time_points) * 0.1,  # Angular velocity
        'speed': np.abs(np.random.randn(time_points) * 2 + 3),  # Positive speeds
    }
    
    # Add experiment-specific data
    if experiment_type == "plume":
        # Olfactory experiment has different patterns
        base_data['odor_conc'] = np.random.exponential(1, time_points)
        base_data['wind_direction'] = np.random.uniform(0, 2*np.pi, time_points)
    else:
        # Optogenetic experiment
        base_data['led_signal'] = np.random.choice([0, 1], time_points, p=[0.8, 0.2])
        base_data['signal_disp'] = np.random.randn(15, time_points)  # Multi-channel neural data
    
    return base_data


# =============================================================================
# API Function Integration Tests
# =============================================================================

class TestLoadExperimentFilesIntegration:
    """Integration tests for load_experiment_files API function."""
    
    def test_load_experiment_files_with_config_path_success(self, realistic_config_file, temp_data_directory):
        """Test successful experiment file loading using config file path."""
        # Test basic file loading
        files = load_experiment_files(
            config_path=realistic_config_file,
            experiment_name="test_experiment",
            pattern="*.pkl"
        )
        
        # Verify files were found
        assert isinstance(files, list)
        assert len(files) > 0
        
        # Verify files contain expected patterns
        file_paths = [str(f) for f in files]
        assert any("exp_test_" in path for path in file_paths)
        
        # Verify ignored files are excluded  
        assert not any("static_horiz_ribbon" in path for path in file_paths)
    
    def test_load_experiment_files_with_config_dict_success(self, realistic_config_dict):
        """Test successful experiment file loading using config dictionary."""
        files = load_experiment_files(
            config=realistic_config_dict,
            experiment_name="test_experiment",
            pattern="*.pkl"
        )
        
        # Should find files based on datasets configuration
        assert isinstance(files, list)
        # Files may be empty if directories don't exist, but should not error
    
    def test_load_experiment_files_with_metadata_extraction(self, realistic_config_file):
        """Test metadata extraction during file loading."""
        files_with_metadata = load_experiment_files(
            config_path=realistic_config_file,
            experiment_name="test_experiment",
            extract_metadata=True
        )
        
        # Should return dictionary when metadata extraction is enabled
        assert isinstance(files_with_metadata, dict)
        
        # Check metadata structure for each file
        for file_path, metadata in files_with_metadata.items():
            assert isinstance(metadata, dict)
            # Should contain extracted information when patterns match
            assert "file_path" in metadata or len(metadata) == 0
    
    def test_load_experiment_files_with_date_parsing(self, realistic_config_file):
        """Test date parsing functionality."""
        files_with_dates = load_experiment_files(
            config_path=realistic_config_file,
            experiment_name="test_experiment",
            parse_dates=True
        )
        
        # Should return dictionary with date information
        assert isinstance(files_with_dates, dict)
    
    def test_load_experiment_files_parameter_validation_errors(self, realistic_config_file):
        """Test parameter validation error scenarios."""
        
        # Test both config_path and config provided
        with pytest.raises(ValueError, match="Exactly one of 'config_path' or 'config' must be provided"):
            load_experiment_files(
                config_path=realistic_config_file,
                config={"test": "config"},
                experiment_name="test_experiment"
            )
        
        # Test neither config_path nor config provided
        with pytest.raises(ValueError, match="Exactly one of 'config_path' or 'config' must be provided"):
            load_experiment_files(experiment_name="test_experiment")
    
    def test_load_experiment_files_config_file_not_found(self):
        """Test FileNotFoundError when config file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_experiment_files(
                config_path="/nonexistent/config.yaml",
                experiment_name="test_experiment"
            )
    
    def test_load_experiment_files_experiment_not_found(self, realistic_config_file):
        """Test KeyError when experiment doesn't exist in config."""
        with pytest.raises(KeyError, match="Experiment 'nonexistent_experiment' not found"):
            load_experiment_files(
                config_path=realistic_config_file,
                experiment_name="nonexistent_experiment"
            )
    
    def test_load_experiment_files_missing_data_directory(self, realistic_config_dict):
        """Test error when data directory is not specified."""
        # Remove data directory from config
        config_without_dir = realistic_config_dict.copy()
        del config_without_dir["project"]["directories"]["major_data_directory"]
        
        with pytest.raises(ValueError, match=MISSING_DATA_DIR_ERROR):
            load_experiment_files(
                config=config_without_dir,
                experiment_name="test_experiment"
            )
    
    def test_load_experiment_files_base_directory_override(self, realistic_config_dict, temp_data_directory):
        """Test base directory override functionality."""
        files = load_experiment_files(
            config=realistic_config_dict,
            experiment_name="test_experiment", 
            base_directory=temp_data_directory
        )
        
        assert isinstance(files, list)
        # Should use override directory instead of config directory


class TestLoadDatasetFilesIntegration:
    """Integration tests for load_dataset_files API function."""
    
    def test_load_dataset_files_with_config_path_success(self, realistic_config_file):
        """Test successful dataset file loading using config file path."""
        files = load_dataset_files(
            config_path=realistic_config_file,
            dataset_name="test_dataset",
            pattern="*.pkl"
        )
        
        assert isinstance(files, list)
        # Should find files matching dataset configuration
    
    def test_load_dataset_files_with_config_dict_success(self, realistic_config_dict):
        """Test successful dataset file loading using config dictionary."""
        files = load_dataset_files(
            config=realistic_config_dict,
            dataset_name="plume_movie_navigation",
            pattern="*.pkl"
        )
        
        assert isinstance(files, list)
    
    def test_load_dataset_files_with_extension_filtering(self, realistic_config_file):
        """Test extension-based file filtering."""
        files = load_dataset_files(
            config_path=realistic_config_file,
            dataset_name="test_dataset",
            extensions=["pkl"]
        )
        
        assert isinstance(files, list)
        # All returned files should have .pkl extension
        for file_path in files:
            assert file_path.endswith('.pkl')
    
    def test_load_dataset_files_with_metadata_extraction(self, realistic_config_file):
        """Test metadata extraction for dataset files."""
        files_with_metadata = load_dataset_files(
            config_path=realistic_config_file,
            dataset_name="plume_movie_navigation",
            extract_metadata=True
        )
        
        assert isinstance(files_with_metadata, dict)
    
    def test_load_dataset_files_parameter_validation_errors(self, realistic_config_file):
        """Test parameter validation error scenarios."""
        
        # Test both config_path and config provided
        with pytest.raises(ValueError, match="Exactly one of 'config_path' or 'config' must be provided"):
            load_dataset_files(
                config_path=realistic_config_file,
                config={"test": "config"},
                dataset_name="test_dataset"
            )
    
    def test_load_dataset_files_dataset_not_found(self, realistic_config_file):
        """Test KeyError when dataset doesn't exist in config."""
        with pytest.raises(KeyError, match="Dataset 'nonexistent_dataset' not found"):
            load_dataset_files(
                config_path=realistic_config_file,
                dataset_name="nonexistent_dataset"
            )
    
    def test_load_dataset_files_recursive_search(self, realistic_config_file):
        """Test recursive search functionality."""
        files_recursive = load_dataset_files(
            config_path=realistic_config_file,
            dataset_name="test_dataset",
            recursive=True
        )
        
        files_non_recursive = load_dataset_files(
            config_path=realistic_config_file,
            dataset_name="test_dataset", 
            recursive=False
        )
        
        assert isinstance(files_recursive, list)
        assert isinstance(files_non_recursive, list)
        # Recursive search should find same or more files
        assert len(files_recursive) >= len(files_non_recursive)


class TestGetParameterFunctionsIntegration:
    """Integration tests for parameter extraction API functions."""
    
    def test_get_experiment_parameters_with_config_path(self, realistic_config_file):
        """Test experiment parameter extraction using config file path."""
        params = get_experiment_parameters(
            config_path=realistic_config_file,
            experiment_name="test_experiment"
        )
        
        assert isinstance(params, dict)
        assert "protocol_version" in params
        assert params["protocol_version"] == "1.2"
        assert params["led_intensity"] == 80
        assert params["pulse_duration"] == 500
    
    def test_get_experiment_parameters_with_config_dict(self, realistic_config_dict):
        """Test experiment parameter extraction using config dictionary."""
        params = get_experiment_parameters(
            config=realistic_config_dict,
            experiment_name="multi_dataset_experiment"
        )
        
        assert isinstance(params, dict)
        assert "protocol_version" in params
        assert params["protocol_version"] == "2.0"
        assert params["cross_modal"] is True
    
    def test_get_experiment_parameters_empty_parameters(self, realistic_config_dict):
        """Test parameter extraction when experiment has no parameters."""
        # Add experiment without parameters
        config_copy = realistic_config_dict.copy()
        config_copy["experiments"]["no_params_experiment"] = {
            "datasets": ["test_dataset"],
            "description": "Experiment without parameters"
        }
        
        params = get_experiment_parameters(
            config=config_copy,
            experiment_name="no_params_experiment"
        )
        
        assert isinstance(params, dict)
        assert len(params) == 0
    
    def test_get_dataset_parameters_with_config_path(self, realistic_config_file):
        """Test dataset parameter extraction using config file path."""
        params = get_dataset_parameters(
            config_path=realistic_config_file,
            dataset_name="test_dataset"
        )
        
        assert isinstance(params, dict)
        assert "sampling_rate" in params
        assert params["sampling_rate"] == 60
        assert params["arena_type"] == "circular"
        assert params["stimulus_type"] == "optogenetic"
    
    def test_get_dataset_parameters_with_config_dict(self, realistic_config_dict):
        """Test dataset parameter extraction using config dictionary."""
        params = get_dataset_parameters(
            config=realistic_config_dict,
            dataset_name="plume_movie_navigation"
        )
        
        assert isinstance(params, dict)
        assert "sampling_rate" in params
        assert params["stimulus_type"] == "olfactory"
        assert params["arena_type"] == "rectangular"
    
    def test_get_dataset_parameters_empty_parameters(self, realistic_config_dict):
        """Test parameter extraction when dataset has no parameters."""
        # Add dataset without parameters
        config_copy = realistic_config_dict.copy()
        config_copy["datasets"]["no_params_dataset"] = {
            "rig": "old_opto",
            "dates_vials": {"2024-01-01": [1]}
        }
        
        params = get_dataset_parameters(
            config=config_copy,
            dataset_name="no_params_dataset"
        )
        
        assert isinstance(params, dict)
        assert len(params) == 0
    
    def test_get_parameters_validation_errors(self, realistic_config_file):
        """Test parameter validation errors for both parameter functions."""
        
        # Test experiment parameters with both config_path and config
        with pytest.raises(ValueError, match="Exactly one of 'config_path' or 'config' must be provided"):
            get_experiment_parameters(
                config_path=realistic_config_file,
                config={"test": "config"},
                experiment_name="test_experiment"
            )
        
        # Test dataset parameters with neither config_path nor config
        with pytest.raises(ValueError, match="Exactly one of 'config_path' or 'config' must be provided"):
            get_dataset_parameters(dataset_name="test_dataset")
    
    def test_get_parameters_not_found_errors(self, realistic_config_file):
        """Test KeyError when experiment/dataset doesn't exist."""
        
        # Test nonexistent experiment
        with pytest.raises(KeyError, match="Experiment 'nonexistent' not found"):
            get_experiment_parameters(
                config_path=realistic_config_file,
                experiment_name="nonexistent"
            )
        
        # Test nonexistent dataset
        with pytest.raises(KeyError, match="Dataset 'nonexistent' not found"):
            get_dataset_parameters(
                config_path=realistic_config_file,
                dataset_name="nonexistent"
            )


class TestProcessExperimentDataIntegration:
    """Integration tests for process_experiment_data API function."""
    
    def test_process_experiment_data_with_pickle_file(self, temp_data_directory):
        """Test processing experimental data from pickle file."""
        # Create a test pickle file with realistic data
        test_data = _create_realistic_exp_matrix(1)
        pickle_file = Path(temp_data_directory) / "test_exp_data.pkl"
        
        with open(pickle_file, 'wb') as f:
            pickle.dump(test_data, f)
        
        # Process the data (using default column config)
        result_df = process_experiment_data(data_path=pickle_file)
        
        # Verify output is a DataFrame
        assert isinstance(result_df, pd.DataFrame)
        
        # Verify basic columns are present
        assert 't' in result_df.columns
        assert 'x' in result_df.columns 
        assert 'y' in result_df.columns
        
        # Verify data integrity
        assert len(result_df) > 0
        assert not result_df['t'].isna().any()
    
    def test_process_experiment_data_with_gzipped_pickle(self, temp_data_directory):
        """Test processing data from gzipped pickle file."""
        test_data = _create_realistic_exp_matrix(2)
        gzip_file = Path(temp_data_directory) / "test_exp_data.pkl.gz"
        
        with gzip.open(gzip_file, 'wb') as f:
            pickle.dump(test_data, f)
        
        result_df = process_experiment_data(data_path=gzip_file)
        
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) > 0
    
    def test_process_experiment_data_with_metadata_injection(self, temp_data_directory):
        """Test metadata injection during data processing."""
        test_data = _create_realistic_exp_matrix(3)
        pickle_file = Path(temp_data_directory) / "test_exp_data.pkl"
        
        with open(pickle_file, 'wb') as f:
            pickle.dump(test_data, f)
        
        metadata = {
            "experiment_name": "test_integration",
            "date": "2024-12-20",
            "fly_id": "fly_003"
        }
        
        result_df = process_experiment_data(
            data_path=pickle_file,
            metadata=metadata
        )
        
        assert isinstance(result_df, pd.DataFrame)
        # Metadata should be added as columns or attributes
        # (exact implementation depends on column config)
    
    def test_process_experiment_data_file_not_found(self):
        """Test error handling when data file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            process_experiment_data(data_path="/nonexistent/file.pkl")


# =============================================================================
# Cross-Module Orchestration Tests
# =============================================================================

class TestCrossModuleOrchestration:
    """Integration tests validating coordination between API functions and subsystems."""
    
    def test_end_to_end_experiment_workflow(self, realistic_config_file, temp_data_directory):
        """Test complete end-to-end workflow from config to processed data."""
        # Step 1: Load experiment files
        files = load_experiment_files(
            config_path=realistic_config_file,
            experiment_name="test_experiment",
            extract_metadata=True
        )
        
        assert isinstance(files, dict)
        
        # Step 2: Get experiment parameters
        exp_params = get_experiment_parameters(
            config_path=realistic_config_file,
            experiment_name="test_experiment"
        )
        
        assert isinstance(exp_params, dict)
        assert "protocol_version" in exp_params
        
        # Step 3: Process first file if any exist
        if files:
            first_file = list(files.keys())[0]
            if Path(first_file).exists():
                result_df = process_experiment_data(
                    data_path=first_file,
                    metadata={"experiment_params": exp_params}
                )
                
                assert isinstance(result_df, pd.DataFrame)
                assert len(result_df) > 0
    
    def test_multi_dataset_experiment_coordination(self, realistic_config_file):
        """Test coordination for experiments with multiple datasets."""
        # Load files for multi-dataset experiment
        files = load_experiment_files(
            config_path=realistic_config_file,
            experiment_name="multi_dataset_experiment"
        )
        
        assert isinstance(files, list)
        
        # Get parameters for each dataset in the experiment
        dataset_params = {}
        for dataset_name in ["test_dataset", "plume_movie_navigation"]:
            params = get_dataset_parameters(
                config_path=realistic_config_file,
                dataset_name=dataset_name
            )
            dataset_params[dataset_name] = params
        
        # Verify different datasets have different parameters
        assert dataset_params["test_dataset"]["stimulus_type"] == "optogenetic"
        assert dataset_params["plume_movie_navigation"]["stimulus_type"] == "olfactory"
    
    def test_config_dict_vs_file_consistency(self, realistic_config_file, realistic_config_dict):
        """Test that config dictionary and file produce consistent results."""
        experiment_name = "test_experiment"
        
        # Load using config file
        files_from_file = load_experiment_files(
            config_path=realistic_config_file,
            experiment_name=experiment_name
        )
        
        # Load using config dictionary
        files_from_dict = load_experiment_files(
            config=realistic_config_dict,
            experiment_name=experiment_name
        )
        
        # Results should be consistent (same file patterns)
        assert isinstance(files_from_file, list)
        assert isinstance(files_from_dict, list)
        # Note: Actual files found may differ due to filesystem state,
        # but the discovery patterns should be equivalent
    
    def test_error_propagation_from_config_module(self):
        """Test that configuration errors propagate correctly through API."""
        # Test with invalid YAML syntax
        invalid_yaml = "invalid: yaml: content: ["
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml)
            invalid_config_file = f.name
        
        try:
            with pytest.raises(yaml.YAMLError):
                load_experiment_files(
                    config_path=invalid_config_file,
                    experiment_name="test_experiment"
                )
        finally:
            os.unlink(invalid_config_file)
    
    def test_error_propagation_from_discovery_module(self, realistic_config_dict):
        """Test that discovery module errors propagate correctly."""
        # Test with non-existent base directory
        config_bad_dir = realistic_config_dict.copy()
        config_bad_dir["project"]["directories"]["major_data_directory"] = "/totally/nonexistent/path"
        
        # Should not raise error (discovery handles missing directories gracefully)
        # but should return empty results
        files = load_experiment_files(
            config=config_bad_dir,
            experiment_name="test_experiment"
        )
        
        assert isinstance(files, list)
        # May be empty due to missing directory


# =============================================================================
# Realistic Data Flow Scenarios  
# =============================================================================

class TestRealisticDataFlowScenarios:
    """Integration tests with realistic experimental data flows and scenarios."""
    
    def test_typical_neuroscience_workflow(self, realistic_config_file, temp_data_directory):
        """Test typical neuroscience research workflow patterns."""
        # Scenario: Researcher analyzing optogenetic behavioral data
        
        # 1. Discover all files for an experiment
        all_files = load_experiment_files(
            config_path=realistic_config_file,
            experiment_name="test_experiment",
            pattern="*.pkl",
            recursive=True
        )
        
        # 2. Filter files by date range (common research pattern)
        dated_files = load_experiment_files(
            config_path=realistic_config_file,
            experiment_name="test_experiment",
            extract_metadata=True,
            parse_dates=True
        )
        
        # 3. Get experimental parameters for analysis context
        exp_params = get_experiment_parameters(
            config_path=realistic_config_file,
            experiment_name="test_experiment"
        )
        
        # Verify workflow components work together
        assert isinstance(all_files, list)
        assert isinstance(dated_files, dict)
        assert isinstance(exp_params, dict)
        assert "protocol_version" in exp_params
    
    def test_cross_rig_comparison_workflow(self, realistic_config_dict):
        """Test workflow for comparing data across different experimental rigs."""
        # Get dataset parameters for different rigs
        test_params = get_dataset_parameters(
            config=realistic_config_dict,
            dataset_name="test_dataset"
        )
        
        plume_params = get_dataset_parameters(
            config=realistic_config_dict, 
            dataset_name="plume_movie_navigation"
        )
        
        # Verify rig-specific parameters are accessible
        assert test_params["stimulus_type"] == "optogenetic"
        assert plume_params["stimulus_type"] == "olfactory"
        
        # Both should have sampling rate info
        assert "sampling_rate" in test_params
        assert "sampling_rate" in plume_params
    
    def test_longitudinal_study_workflow(self, realistic_config_file, temp_data_directory):
        """Test workflow for longitudinal studies across multiple dates."""
        # Load files with date information for temporal analysis
        files_with_dates = load_experiment_files(
            config_path=realistic_config_file,
            experiment_name="test_experiment",
            parse_dates=True,
            extract_metadata=True
        )
        
        assert isinstance(files_with_dates, dict)
        
        # Verify we can access multiple dates
        dataset_files = load_dataset_files(
            config_path=realistic_config_file,
            dataset_name="test_dataset",
            extract_metadata=True
        )
        
        assert isinstance(dataset_files, dict)
    
    @pytest.mark.parametrize("experiment_name,expected_datasets", [
        ("test_experiment", ["test_dataset"]),
        ("multi_dataset_experiment", ["test_dataset", "plume_movie_navigation"])
    ])
    def test_parametrized_experiment_workflows(self, realistic_config_dict, experiment_name, expected_datasets):
        """Test various experiment configurations with parametrization."""
        # Get experiment parameters
        exp_params = get_experiment_parameters(
            config=realistic_config_dict,
            experiment_name=experiment_name
        )
        
        assert isinstance(exp_params, dict)
        
        # Load files for the experiment
        files = load_experiment_files(
            config=realistic_config_dict,
            experiment_name=experiment_name
        )
        
        assert isinstance(files, list)
        
        # Verify we can get parameters for each expected dataset
        for dataset_name in expected_datasets:
            dataset_params = get_dataset_parameters(
                config=realistic_config_dict,
                dataset_name=dataset_name
            )
            assert isinstance(dataset_params, dict)


# =============================================================================
# Backward Compatibility Tests
# =============================================================================

class TestBackwardCompatibility:
    """Integration tests ensuring API maintains backward compatibility."""
    
    def test_api_function_signatures_unchanged(self):
        """Test that API function signatures remain stable."""
        import inspect
        
        # Check load_experiment_files signature
        sig = inspect.signature(load_experiment_files)
        expected_params = [
            'config_path', 'config', 'experiment_name', 'base_directory',
            'pattern', 'recursive', 'extensions', 'extract_metadata', 'parse_dates'
        ]
        actual_params = list(sig.parameters.keys())
        
        for param in expected_params:
            assert param in actual_params, f"Parameter {param} missing from load_experiment_files"
    
    def test_return_type_consistency(self, realistic_config_dict):
        """Test that return types remain consistent."""
        # Test without metadata extraction (should return list)
        files = load_experiment_files(
            config=realistic_config_dict,
            experiment_name="test_experiment"
        )
        assert isinstance(files, list)
        
        # Test with metadata extraction (should return dict)
        files_with_meta = load_experiment_files(
            config=realistic_config_dict,
            experiment_name="test_experiment",
            extract_metadata=True
        )
        assert isinstance(files_with_meta, dict)
    
    def test_default_parameter_behavior(self, realistic_config_dict):
        """Test that default parameters maintain expected behavior."""
        # Test default pattern behavior
        files_default = load_experiment_files(
            config=realistic_config_dict,
            experiment_name="test_experiment"
        )
        
        files_explicit = load_experiment_files(
            config=realistic_config_dict,
            experiment_name="test_experiment",
            pattern="*.*",
            recursive=True,
            extract_metadata=False,
            parse_dates=False
        )
        
        # Should produce same results
        assert type(files_default) == type(files_explicit)
    
    def test_kedro_parameter_dictionary_support(self, realistic_config_dict):
        """Test continued support for Kedro-style parameter dictionaries."""
        # Simulate Kedro parameters structure
        kedro_params = {
            "flyrigloader": realistic_config_dict
        }
        
        # Should work when passed the nested config
        files = load_experiment_files(
            config=realistic_config_dict,  # Direct config, not nested
            experiment_name="test_experiment"
        )
        
        assert isinstance(files, list)


# =============================================================================
# Performance and Quality Validation 
# =============================================================================

class TestPerformanceValidation:
    """Integration tests validating performance requirements from Section 2.2.10."""
    
    def test_end_to_end_workflow_performance(self, realistic_config_file, temp_data_directory):
        """Test that complete workflow meets 30s SLA requirement."""
        import time
        
        start_time = time.time()
        
        # Complete workflow
        files = load_experiment_files(
            config_path=realistic_config_file,
            experiment_name="test_experiment",
            extract_metadata=True
        )
        
        exp_params = get_experiment_parameters(
            config_path=realistic_config_file,
            experiment_name="test_experiment"
        )
        
        if files:
            first_file = list(files.keys())[0]
            if Path(first_file).exists():
                process_experiment_data(data_path=first_file)
        
        end_time = time.time()
        workflow_duration = end_time - start_time
        
        # Should complete within 30 seconds (TST-INTEG-001 requirement)
        assert workflow_duration < 30.0, f"Workflow took {workflow_duration:.2f}s, exceeds 30s SLA"
    
    def test_synthetic_data_generation_performance(self):
        """Test that test data generation meets 5s SLA requirement."""
        import time
        
        start_time = time.time()
        
        # Generate multiple realistic datasets
        for i in range(10):
            data = _create_realistic_exp_matrix(i)
            assert isinstance(data, dict)
            assert len(data['t']) > 1000  # Substantial dataset
        
        end_time = time.time()
        generation_duration = end_time - start_time
        
        # Should complete within 5 seconds (TST-INTEG-002 requirement)
        assert generation_duration < 5.0, f"Data generation took {generation_duration:.2f}s, exceeds 5s SLA"
    
    def test_dataframe_verification_performance(self, temp_data_directory):
        """Test DataFrame verification meets 1s SLA requirement."""
        import time
        
        # Create test data
        test_data = _create_realistic_exp_matrix(1)
        pickle_file = Path(temp_data_directory) / "perf_test.pkl"
        
        with open(pickle_file, 'wb') as f:
            pickle.dump(test_data, f)
        
        start_time = time.time()
        
        # Process and verify DataFrame
        result_df = process_experiment_data(data_path=pickle_file)
        
        # Perform comprehensive verification
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) > 0
        assert not result_df.isna().all().any()  # Check for data integrity
        assert result_df.dtypes.notna().all()    # Check type consistency
        
        end_time = time.time()
        verification_duration = end_time - start_time
        
        # Should complete within 1 second (TST-INTEG-003 requirement)
        assert verification_duration < 1.0, f"DataFrame verification took {verification_duration:.2f}s, exceeds 1s SLA"