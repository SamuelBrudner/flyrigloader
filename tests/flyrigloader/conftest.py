"""
Flyrigloader-specific fixtures for specialized neuroscience testing scenarios.

This module contains pytest fixtures specifically for flyrigloader testing that cannot
be generalized to the centralized fixture management system in tests/conftest.py.
Following the fixture consolidation strategy from Section 0, this module now:

- Maintains only domain-specific fixtures unique to flyrigloader neuroscience data patterns
- Redirects to centralized implementations from tests/conftest.py and tests/utils.py
- Follows standardized fixture naming conventions (mock_*, temp_*, sample_*)
- Preserves backwards compatibility during fixture consolidation transition
- Implements flyrigloader-specific mock behaviors for experimental data structures

For general testing utilities, configuration providers, and mock implementations,
import from tests.conftest.py and tests.utils module instead.
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import yaml

# Import centralized testing infrastructure
from tests.utils import (
    create_mock_config_provider,
    create_mock_dataloader, 
    create_mock_filesystem,
    MockConfigurationProvider,
    MockDataLoading,
    MockFilesystem
)

# Import specialized neuroscience data generators
try:
    from tests.conftest import test_data_generator, temp_experiment_directory
except ImportError:
    # Fallback for transition period
    test_data_generator = None
    temp_experiment_directory = None


# --- Flyrigloader-Specific Configuration Fixtures ---
# Redirected from centralized MockConfigurationProvider for enhanced functionality

@pytest.fixture
def mock_neuroscience_config_provider():
    """
    Neuroscience-specific configuration provider using centralized MockConfigurationProvider.
    
    Returns comprehensive flyrigloader configuration with neuroscience experiment patterns,
    rig specifications, and behavioral data validation rules. Uses centralized mock
    implementation with flyrigloader-specific enhancements.
    
    Returns:
        MockConfigurationProvider: Configured provider with neuroscience data patterns
    """
    provider = create_mock_config_provider('comprehensive', include_errors=False)
    
    # Add flyrigloader-specific configuration enhancements
    provider.add_configuration('flyrigloader_neuroscience', {
        "project": {
            "directories": {
                "major_data_directory": "/research/data/neuroscience",
                "batchfile_directory": "/research/batch_definitions",
                "backup_directory": "/research/backups",
                "processed_data_directory": "/research/processed"
            },
            "ignore_substrings": [
                "static_horiz_ribbon", "._", ".DS_Store", "__pycache__",
                ".tmp", "backup_", "test_calibration"
            ],
            "mandatory_substrings": ["experiment_", "data_"],
            "extraction_patterns": [
                r".*_(?P<date>\d{8})_(?P<condition>\w+)_(?P<replicate>\d+)\.csv",
                r"(?P<rig>\w+)_(?P<date>\d{8})_(?P<animal_id>\w+)_(?P<trial>\d+)",
                r"exp_(?P<experiment_id>\d+)_(?P<dataset>\w+)_(?P<timestamp>\d{14})"
            ],
            "file_extensions": [".csv", ".pkl", ".pickle", ".json"],
            "max_file_size_mb": 500,
            "parallel_processing": True
        },
        "rigs": {
            "old_opto": {
                "sampling_frequency": 60, "mm_per_px": 0.154,
                "camera_resolution": [1024, 768], "calibration_date": "2024-01-15",
                "arena_diameter_mm": 120, "led_wavelength_nm": 470
            },
            "new_opto": {
                "sampling_frequency": 60, "mm_per_px": 0.1818,
                "camera_resolution": [1280, 1024], "calibration_date": "2024-06-01",
                "arena_diameter_mm": 150, "led_wavelength_nm": 470
            },
            "high_speed_rig": {
                "sampling_frequency": 200, "mm_per_px": 0.05,
                "camera_resolution": [2048, 2048], "calibration_date": "2024-08-15",
                "arena_diameter_mm": 200, "led_wavelength_nm": 590
            }
        },
        "datasets": {
            "baseline_behavior": {
                "rig": "old_opto", "patterns": ["*baseline*", "*control*"],
                "dates_vials": {"2024-12-20": [1, 2, 3, 4, 5], "2024-12-21": [1, 2, 3]},
                "metadata": {
                    "extraction_patterns": [r".*_(?P<dataset>\w+)_(?P<date>\d{8})_(?P<vial>\d+)\.csv"],
                    "required_fields": ["dataset", "date", "vial"],
                    "experiment_type": "baseline"
                },
                "filters": {
                    "min_duration_seconds": 300, "max_duration_seconds": 3600,
                    "required_columns": ["t", "x", "y"]
                }
            },
            "optogenetic_stimulation": {
                "rig": "new_opto", "patterns": ["*opto*", "*stim*"],
                "dates_vials": {"2024-12-18": [1, 2, 3, 4], "2024-12-19": [1, 2, 3, 4, 5, 6]},
                "metadata": {
                    "extraction_patterns": [r".*_(?P<dataset>\w+)_(?P<stimulation_type>\w+)_(?P<date>\d{8})\.csv"],
                    "required_fields": ["dataset", "stimulation_type", "date"],
                    "experiment_type": "optogenetic"
                }
            }
        },
        "experiments": {
            "baseline_control_study": {
                "datasets": ["baseline_behavior"],
                "metadata": {
                    "extraction_patterns": [r".*_(?P<experiment>baseline)_(?P<date>\d{8})\.csv"],
                    "required_fields": ["experiment", "date"],
                    "study_type": "control", "principal_investigator": "Dr. Research"
                }
            }
        },
        "validation": {
            "required_columns": ["t", "x", "y"],
            "optional_columns": ["signal", "signal_disp", "dtheta"],
            "metadata_columns": ["date", "exp_name", "rig", "fly_id"],
            "data_quality_checks": {
                "max_missing_data_percent": 5.0,
                "min_trajectory_length": 100,
                "velocity_outlier_threshold": 3.0
            }
        }
    })
    
    return provider

@pytest.fixture
def sample_neuroscience_config_dict(mock_neuroscience_config_provider):
    """
    Comprehensive neuroscience configuration dictionary for flyrigloader testing.
    
    Redirects to centralized MockConfigurationProvider with flyrigloader-specific
    neuroscience experiment patterns. Maintains backwards compatibility with
    existing test modules during fixture consolidation transition.
    
    Returns:
        Dict[str, Any]: Comprehensive neuroscience configuration dictionary
    """
    return mock_neuroscience_config_provider.load_config('flyrigloader_neuroscience')

# Backwards compatibility aliases for transition period
@pytest.fixture  
def comprehensive_sample_config_dict(sample_neuroscience_config_dict):
    """Backwards compatibility alias for comprehensive_sample_config_dict."""
    return sample_neuroscience_config_dict

@pytest.fixture
def temp_neuroscience_config_file(sample_neuroscience_config_dict, temp_experiment_directory):
    """
    Create temporary neuroscience configuration file using centralized temp directory.
    
    Uses centralized temp_experiment_directory fixture from tests/conftest.py
    instead of duplicated cross_platform_temp_dir. Maintains flyrigloader-specific
    neuroscience configuration structure.
    
    Args:
        sample_neuroscience_config_dict: Neuroscience configuration data
        temp_experiment_directory: Centralized temporary directory fixture
    
    Returns:
        str: Path to the temporary config file
    """
    if temp_experiment_directory is None:
        # Fallback for transition period
        import tempfile
        temp_dir = Path(tempfile.mkdtemp())
        config_path = temp_dir / "neuroscience_config.yaml"
    else:
        config_path = temp_experiment_directory["directory"] / "neuroscience_config.yaml"
    
    # Write the config to the file
    with open(config_path, 'w') as f:
        yaml.dump(sample_neuroscience_config_dict, f, default_flow_style=False)
    
    return str(config_path)

@pytest.fixture
def sample_minimal_config_dict(sample_neuroscience_config_dict):
    """
    Simplified neuroscience configuration dictionary for basic flyrigloader testing.
    
    Provides subset of comprehensive configuration for tests that don't need
    full complexity. Focuses on essential flyrigloader configuration elements.
    
    Returns:
        Dict[str, Any]: Simplified neuroscience configuration dictionary
    """
    return {
        "project": sample_neuroscience_config_dict["project"],
        "rigs": {
            "old_opto": sample_neuroscience_config_dict["rigs"]["old_opto"]
        },
        "datasets": {
            "test_dataset": sample_neuroscience_config_dict["datasets"]["baseline_behavior"]
        },
        "experiments": {
            "test_experiment": sample_neuroscience_config_dict["experiments"]["baseline_control_study"]
        }
    }

# Backwards compatibility aliases for transition period
@pytest.fixture
def sample_config_file(temp_neuroscience_config_file):
    """Backwards compatibility alias for sample_config_file."""
    return temp_neuroscience_config_file

@pytest.fixture  
def sample_config_dict(sample_minimal_config_dict):
    """Backwards compatibility alias for sample_config_dict."""
    return sample_minimal_config_dict

# --- Flyrigloader-Specific Temporary Filesystem Fixtures ---
# Redirected to centralized temp_experiment_directory with neuroscience enhancements

@pytest.fixture
def temp_neuroscience_filesystem(temp_experiment_directory):
    """
    Neuroscience-specific filesystem structure using centralized temp directory.
    
    Enhances centralized temp_experiment_directory with flyrigloader-specific
    neuroscience data organization patterns. Removes duplicate cross-platform
    temporary directory implementation in favor of centralized approach.
    
    Returns:
        Dict[str, Path]: Neuroscience experiment filesystem structure
    """
    if temp_experiment_directory is None:
        # Fallback for transition period
        import tempfile
        import shutil
        temp_dir = Path(tempfile.mkdtemp(prefix="flyrigloader_neuroscience_"))
        base_dir = temp_dir
        # Minimal cleanup function for fallback
        import atexit
        atexit.register(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
    else:
        base_dir = temp_experiment_directory["directory"]
    
    # Create neuroscience-specific directory structure
    structure = {
        "data_root": base_dir / "neuroscience_data",
        "experiments": base_dir / "neuroscience_data" / "experiments",
        "baselines": base_dir / "neuroscience_data" / "experiments" / "baseline",
        "optogenetics": base_dir / "neuroscience_data" / "experiments" / "optogenetics",
        "navigation": base_dir / "neuroscience_data" / "experiments" / "navigation",
        "rigs": base_dir / "neuroscience_data" / "rigs",
        "configs": base_dir / "configs",
        "processed": base_dir / "processed_data"
    }
    
    # Create all directories
    for dir_path in structure.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create flyrigloader-specific sample data files
    sample_files = {
        # Baseline neuroscience experiment files
        "baseline_file_1": structure["baselines"] / "baseline_20241220_control_1.csv",
        "baseline_file_2": structure["baselines"] / "baseline_20241221_control_2.csv",
        
        # Optogenetic stimulation experiment files  
        "opto_file_1": structure["optogenetics"] / "opto_stim_20241218_treatment_1.csv",
        "opto_file_2": structure["optogenetics"] / "opto_stim_20241219_treatment_2.csv",
        
        # Navigation behavior experiment files
        "nav_file_1": structure["navigation"] / "plume_navigation_20241025_trial_1.csv",
        "nav_file_2": structure["navigation"] / "plume_navigation_20241025_trial_2.csv",
        
        # Rig-specific files
        "rig_config_1": structure["rigs"] / "old_opto_calibration.yaml",
        "rig_config_2": structure["rigs"] / "new_opto_calibration.yaml",
        
        # Files to be ignored by flyrigloader patterns
        "ignored_file_1": structure["baselines"] / "static_horiz_ribbon_calibration.csv",
        "ignored_file_2": structure["optogenetics"] / "._temp_file.csv",
        
        # Configuration file
        "config_file": structure["configs"] / "neuroscience_experiment_config.yaml"
    }
    
    # Neuroscience-specific CSV content with trajectory data
    neuroscience_csv_content = """t,x,y,signal,dtheta
0.0,60.5,62.3,0.1,0.05
0.016,60.6,62.2,0.15,0.02
0.032,60.7,62.1,0.12,-0.01
0.048,60.8,62.0,0.18,0.03
0.064,60.9,61.9,0.14,-0.02
"""
    
    # Write sample files with neuroscience data patterns
    for file_key, file_path in sample_files.items():
        if file_path.suffix == ".csv":
            file_path.write_text(neuroscience_csv_content)
        elif file_path.suffix == ".yaml":
            file_path.write_text("# Neuroscience experiment config file")
    
    return {**structure, **sample_files}

# Backwards compatibility aliases for transition period
@pytest.fixture
def cross_platform_temp_dir(temp_neuroscience_filesystem):
    """Backwards compatibility alias redirecting to neuroscience filesystem."""
    return temp_neuroscience_filesystem["data_root"].parent

@pytest.fixture
def temp_filesystem_structure(temp_neuroscience_filesystem):
    """Backwards compatibility alias for temp_filesystem_structure.""" 
    return temp_neuroscience_filesystem


# --- Flyrigloader-Specific Column Configuration Fixtures ---

@pytest.fixture
def sample_flyrigloader_column_config():
    """
    Flyrigloader-specific column configuration for neuroscience data structures.
    
    Defines column schema for typical flyrigloader experimental data including
    trajectory data, signal channels, and metadata fields specific to 
    neuroscience experiments.
    
    Returns:
        Dict[str, Any]: Column configuration dictionary
    """
    return {
        'columns': {
            't': {
                'type': 'numpy.ndarray', 'dimension': 1, 'required': True,
                'description': 'Time values in seconds'
            },
            'x': {
                'type': 'numpy.ndarray', 'dimension': 1, 'required': True,
                'description': 'X position in mm'
            },
            'y': {
                'type': 'numpy.ndarray', 'dimension': 1, 'required': True,
                'description': 'Y position in mm'
            },
            'dtheta': {
                'type': 'numpy.ndarray', 'dimension': 1, 'required': False,
                'description': 'Change in heading angle', 'alias': 'dtheta_smooth'
            },
            'signal': {
                'type': 'numpy.ndarray', 'dimension': 1, 'required': False,
                'description': 'Single channel signal values', 'default_value': None
            },
            'signal_disp': {
                'type': 'numpy.ndarray', 'dimension': 2, 'required': False,
                'description': 'Multi-channel signal display data',
                'special_handling': 'transform_to_match_time_dimension'
            },
            'date': {
                'type': 'string', 'required': False, 'is_metadata': True,
                'description': 'Experiment date (YYYYMMDD)'
            },
            'exp_name': {
                'type': 'string', 'required': False, 'is_metadata': True,
                'description': 'Experiment name identifier'
            },
            'rig': {
                'type': 'string', 'required': False, 'is_metadata': True,
                'description': 'Rig identifier (old_opto, new_opto, etc.)'
            },
            'fly_id': {
                'type': 'string', 'required': False, 'is_metadata': True,
                'description': 'Individual animal identifier'
            }
        },
        'special_handlers': {
            'transform_to_match_time_dimension': '_handle_signal_disp'
        }
    }

@pytest.fixture  
def temp_flyrigloader_column_config_file(sample_flyrigloader_column_config, temp_neuroscience_filesystem):
    """
    Create temporary flyrigloader column configuration file.
    
    Uses centralized temp filesystem and flyrigloader-specific column schema.
    Maintains backwards compatibility with existing column config file expectations.
    
    Returns:
        str: Path to the temporary column config file
    """
    config_path = temp_neuroscience_filesystem["configs"] / "flyrigloader_columns.yaml"
    
    with open(config_path, 'w') as f:
        yaml.dump(sample_flyrigloader_column_config, f, default_flow_style=False)
    
    return str(config_path)

# Backwards compatibility alias for transition period
@pytest.fixture
def sample_column_config_file(temp_flyrigloader_column_config_file):
    """Backwards compatibility alias for sample_column_config_file."""
    return temp_flyrigloader_column_config_file


# --- Flyrigloader-Specific Neuroscience Data Generation ---
# Enhanced with centralized test_data_generator and neuroscience patterns

@pytest.fixture
def sample_neuroscience_time_series_params():
    """
    Parameters for generating realistic neuroscience experimental time series data.
    
    Defines parameters specific to flyrigloader neuroscience experiments including
    sampling frequencies typical of behavioral recording systems and arena specifications.
    
    Returns:
        Dict: Neuroscience-specific parameters for synthetic data generation
    """
    return {
        "sampling_frequency": 60.0,  # Hz - typical behavioral tracking rate
        "duration_seconds": 300.0,   # 5 minutes - standard experiment duration
        "arena_diameter_mm": 120.0,  # Standard circular arena
        "center_bias": 0.3,          # Fly tendency to stay near center
        "movement_noise": 0.1,       # Biological movement smoothness
        "velocity_max": 15.0,        # mm/s maximum realistic fly velocity
        "signal_channels": 16,       # Multi-channel neural/calcium imaging
        "signal_noise_level": 0.05,  # Realistic signal-to-noise ratio
        "experimental_conditions": ["control", "optogenetic", "baseline"],
        "rig_types": ["old_opto", "new_opto", "high_speed_rig"]
    }

@pytest.fixture
def mock_neuroscience_trajectory_generator(test_data_generator):
    """
    Neuroscience-specific trajectory generator using centralized test data infrastructure.
    
    Enhances centralized test_data_generator with flyrigloader-specific neuroscience
    trajectory patterns including realistic fly movement behaviors, arena constraints,
    and biologically plausible velocity profiles.
    
    Returns:
        Callable: Function that generates neuroscience trajectory data
    """
    def generate_neuroscience_trajectory(
        n_timepoints: int = 1000,
        sampling_freq: float = 60.0,
        arena_diameter: float = 120.0,
        center_bias: float = 0.3,
        movement_noise: float = 0.1,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate realistic neuroscience fly trajectory with biological constraints.
        
        Args:
            n_timepoints: Number of time points to generate
            sampling_freq: Sampling frequency in Hz
            arena_diameter: Arena diameter in mm
            center_bias: Bias toward arena center (0=random walk, 1=strong center bias)
            movement_noise: Movement noise level (higher = more erratic)
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (time, x_position, y_position) arrays
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Use centralized data generator if available
        if test_data_generator is not None:
            # Generate base experimental matrix and extract trajectory
            base_matrix = test_data_generator.generate_experimental_matrix(
                rows=n_timepoints, cols=3, data_type="behavioral"
            )
            if base_matrix is not None:
                # Extract time, x, y and apply neuroscience constraints
                dt = 1.0 / sampling_freq
                time = np.arange(n_timepoints) * dt
                
                # Apply arena constraints to base trajectory
                x_pos = (base_matrix[:, 1] - 0.5) * arena_diameter
                y_pos = (base_matrix[:, 2] - 0.5) * arena_diameter
                
                # Apply center bias and boundary constraints
                arena_radius = arena_diameter / 2.0
                for i in range(1, n_timepoints):
                    current_radius = np.sqrt(x_pos[i]**2 + y_pos[i]**2)
                    if current_radius > arena_radius:
                        # Constrain to arena boundaries
                        x_pos[i] *= arena_radius / current_radius * 0.95
                        y_pos[i] *= arena_radius / current_radius * 0.95
                
                return time, x_pos, y_pos
        
        # Fallback implementation for transition period
        dt = 1.0 / sampling_freq
        time = np.arange(n_timepoints) * dt
        arena_radius = arena_diameter / 2.0
        
        # Generate simple constrained random walk
        x_pos = np.cumsum(np.random.normal(0, movement_noise, n_timepoints))
        y_pos = np.cumsum(np.random.normal(0, movement_noise, n_timepoints))
        
        # Apply arena constraints
        for i in range(n_timepoints):
            radius = np.sqrt(x_pos[i]**2 + y_pos[i]**2)
            if radius > arena_radius:
                x_pos[i] *= arena_radius / radius * 0.95
                y_pos[i] *= arena_radius / radius * 0.95
        
        return time, x_pos, y_pos
    
    return generate_neuroscience_trajectory

# Backwards compatibility alias for transition period
@pytest.fixture
def synthetic_trajectory_generator(mock_neuroscience_trajectory_generator):
    """Backwards compatibility alias for synthetic_trajectory_generator."""
    return mock_neuroscience_trajectory_generator

@pytest.fixture
def mock_neuroscience_signal_generator(test_data_generator):
    """
    Neuroscience-specific signal generator using centralized test data infrastructure.
    
    Generates multi-channel signals mimicking calcium imaging or electrophysiological
    recordings with neuroscience-specific characteristics using centralized test_data_generator
    when available, with flyrigloader-specific enhancements.
    
    Returns:
        Callable: Function that generates neuroscience multi-channel signal data
    """
    def generate_neuroscience_signals(
        n_timepoints: int,
        n_channels: int = 16,
        signal_freq: float = 2.0,
        noise_level: float = 0.1,
        baseline_drift: bool = True,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate realistic neuroscience multi-channel signal data.
        
        Args:
            n_timepoints: Number of time points
            n_channels: Number of signal channels (typical for calcium imaging)
            signal_freq: Characteristic frequency of neural oscillations (Hz)
            noise_level: Noise amplitude relative to signal
            baseline_drift: Whether to include slow baseline drift (photobleaching)
            seed: Random seed for reproducibility
            
        Returns:
            Array of shape (n_channels, n_timepoints) with neuroscience signal data
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Use centralized data generator if available
        if test_data_generator is not None:
            # Generate base experimental matrix
            base_matrix = test_data_generator.generate_experimental_matrix(
                rows=n_timepoints, cols=n_channels, data_type="neural"
            )
            if base_matrix is not None:
                # Reshape to (n_channels, n_timepoints) and apply neuroscience constraints
                signals = base_matrix.T
                
                # Apply neuroscience-specific signal characteristics
                for ch in range(n_channels):
                    # Add channel-specific phase offset for realistic correlations
                    phase_offset = 2 * np.pi * ch / n_channels
                    t = np.linspace(0, n_timepoints/60.0, n_timepoints)
                    
                    # Add harmonic components typical of neural signals
                    harmonics = 0.3 * np.sin(4 * np.pi * signal_freq * t + phase_offset)
                    signals[ch, :] += harmonics
                    
                    # Add baseline drift for photobleaching simulation
                    if baseline_drift:
                        drift_freq = 0.01
                        drift = 0.2 * np.sin(2 * np.pi * drift_freq * t + np.random.random() * 2 * np.pi)
                        signals[ch, :] += drift
                
                return signals
        
        # Fallback implementation for transition period
        signals = np.zeros((n_channels, n_timepoints))
        
        for ch in range(n_channels):
            # Generate neuroscience-realistic signal patterns
            phase_offset = 2 * np.pi * ch / n_channels
            amplitude = 0.8 + 0.4 * np.random.random()
            
            t = np.linspace(0, n_timepoints/60.0, n_timepoints)
            base_signal = amplitude * np.sin(2 * np.pi * signal_freq * t + phase_offset)
            
            # Add noise characteristic of calcium imaging
            noise = noise_level * np.random.normal(0, 1, n_timepoints)
            signals[ch, :] = base_signal + noise
        
        return signals
    
    return generate_neuroscience_signals

# Backwards compatibility alias for transition period
@pytest.fixture
def synthetic_signal_generator(mock_neuroscience_signal_generator):
    """Backwards compatibility alias for synthetic_signal_generator."""
    return mock_neuroscience_signal_generator

@pytest.fixture
def sample_neuroscience_exp_matrix(mock_neuroscience_trajectory_generator, sample_neuroscience_time_series_params):
    """
    Create sample neuroscience experimental data matrix using centralized infrastructure.
    
    Uses centralized trajectory generator enhanced with neuroscience-specific patterns.
    Maintains backwards compatibility while utilizing improved centralized infrastructure.
    
    Returns:
        Dict[str, np.ndarray]: Sample neuroscience experimental data
    """
    params = sample_neuroscience_time_series_params
    n_points = int(params["duration_seconds"] * params["sampling_frequency"])
    
    time, x_pos, y_pos = mock_neuroscience_trajectory_generator(
        n_timepoints=n_points,
        sampling_freq=params["sampling_frequency"],
        arena_diameter=params["arena_diameter_mm"],
        seed=42  # Reproducible for tests
    )
    
    return {
        't': time,
        'x': x_pos,
        'y': y_pos,
        'metadata': {
            'rig': 'old_opto',
            'condition': 'control',
            'sampling_frequency': params["sampling_frequency"]
        }
    }

@pytest.fixture
def sample_neuroscience_exp_matrix_with_signals(sample_neuroscience_exp_matrix, mock_neuroscience_signal_generator):
    """
    Create sample neuroscience experimental data matrix with multi-channel signal data.
    
    Enhances basic trajectory data with realistic multi-channel signals typical
    of calcium imaging or electrophysiological recordings in neuroscience experiments.
    
    Returns:
        Dict[str, np.ndarray]: Sample neuroscience experimental data with signals
    """
    matrix = sample_neuroscience_exp_matrix.copy()
    n_timepoints = len(matrix['t'])
    
    # Generate neuroscience-specific multi-channel signal data
    signal_data = mock_neuroscience_signal_generator(
        n_timepoints=n_timepoints,
        n_channels=16,  # Standard for calcium imaging
        seed=42
    )
    
    matrix['signal_disp'] = signal_data
    
    # Add single channel signal as well
    matrix['signal'] = signal_data[0, :]  # First channel as single signal
    
    return matrix

@pytest.fixture  
def sample_neuroscience_exp_matrix_with_derivatives(sample_neuroscience_exp_matrix):
    """
    Create sample neuroscience experimental data matrix with derived measures.
    
    Adds common neuroscience analysis derivatives like angular velocity (dtheta),
    speed, and distance measures typical in behavioral analysis.
    
    Returns:
        Dict[str, np.ndarray]: Sample neuroscience experimental data with derivatives
    """
    matrix = sample_neuroscience_exp_matrix.copy()
    
    # Calculate movement derivatives common in neuroscience analysis
    x_diff = np.diff(matrix['x'], prepend=matrix['x'][0])
    y_diff = np.diff(matrix['y'], prepend=matrix['y'][0])
    dt = np.diff(matrix['t'], prepend=matrix['t'][1] - matrix['t'][0])
    
    # Angular velocity (heading change)
    dtheta_smooth = np.arctan2(y_diff, x_diff) + 0.05 * np.random.normal(0, 1, len(matrix['t']))
    matrix['dtheta_smooth'] = dtheta_smooth
    matrix['dtheta'] = dtheta_smooth  # Alias
    
    # Speed calculation
    matrix['speed'] = np.sqrt(x_diff**2 + y_diff**2) / dt
    
    # Distance from center (common neuroscience measure)
    matrix['distance_from_center'] = np.sqrt(matrix['x']**2 + matrix['y']**2)
    
    return matrix

# Backwards compatibility aliases for transition period
@pytest.fixture
def sample_exp_matrix(sample_neuroscience_exp_matrix):
    """Backwards compatibility alias for sample_exp_matrix."""
    return sample_neuroscience_exp_matrix

@pytest.fixture
def sample_exp_matrix_with_signal_disp(sample_neuroscience_exp_matrix_with_signals):
    """Backwards compatibility alias for sample_exp_matrix_with_signal_disp."""
    return sample_neuroscience_exp_matrix_with_signals

@pytest.fixture  
def sample_exp_matrix_with_aliases(sample_neuroscience_exp_matrix_with_derivatives):
    """Backwards compatibility alias for sample_exp_matrix_with_aliases."""
    return sample_neuroscience_exp_matrix_with_derivatives

@pytest.fixture
def sample_neuroscience_metadata():
    """
    Create sample neuroscience metadata dictionary for flyrigloader tests.
    
    Provides realistic metadata structure typical of neuroscience experiments
    with flyrigloader-specific fields and naming conventions.
    
    Returns:
        Dict[str, str]: Sample neuroscience metadata
    """
    return {
        'date': '20241201',
        'exp_name': 'baseline_control_study',
        'rig': 'old_opto',
        'fly_id': 'fly_001',
        'condition': 'control',
        'replicate': '1',
        'experimenter': 'researcher_a',
        'temperature_c': '23.5',
        'humidity_percent': '45.2',
        'arena_diameter_mm': '120',
        'sampling_frequency_hz': '60'
    }

@pytest.fixture
def sample_neuroscience_dataframe(sample_neuroscience_exp_matrix_with_signals, sample_neuroscience_metadata):
    """
    Create sample neuroscience pandas DataFrame using centralized infrastructure.
    
    Demonstrates expected flyrigloader output format after data processing.
    Uses centralized infrastructure while maintaining neuroscience-specific
    data structure and metadata patterns.
    
    Returns:
        pd.DataFrame: Sample neuroscience DataFrame with experimental data
    """
    # Convert neuroscience matrix to DataFrame
    df_data = {}
    
    # Add time series data
    for col, data in sample_neuroscience_exp_matrix_with_signals.items():
        if col == 'metadata':
            continue  # Skip metadata dict, handled separately
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                df_data[col] = data
            elif data.ndim == 2:
                # For 2D data like signal_disp, add as columns with channel indices
                for ch in range(data.shape[0]):
                    df_data[f"{col}_ch{ch:02d}"] = data[ch, :]
    
    df = pd.DataFrame(df_data)
    
    # Add neuroscience metadata as constant columns
    for key, value in sample_neuroscience_metadata.items():
        df[key] = value
    
    return df

# Backwards compatibility aliases for transition period
@pytest.fixture
def comprehensive_exp_matrix(sample_neuroscience_exp_matrix_with_signals):
    """Backwards compatibility alias for comprehensive_exp_matrix."""
    return sample_neuroscience_exp_matrix_with_signals

@pytest.fixture
def sample_metadata(sample_neuroscience_metadata):
    """Backwards compatibility alias for sample_metadata."""
    return sample_neuroscience_metadata

@pytest.fixture
def sample_pandas_dataframe(sample_neuroscience_dataframe):
    """Backwards compatibility alias for sample_pandas_dataframe."""
    return sample_neuroscience_dataframe


# --- Flyrigloader-Specific Mock Integration Fixtures ---
# Using centralized mock infrastructure from tests/utils.py

@pytest.fixture
def mock_flyrigloader_integration(sample_neuroscience_config_dict, temp_neuroscience_filesystem):
    """
    Flyrigloader-specific integration mocks using centralized mock infrastructure.
    
    Replaces large custom mock implementation with centralized MockConfigurationProvider,
    MockDataLoading, and MockFilesystem from tests/utils.py. Maintains flyrigloader-specific
    neuroscience patterns while leveraging standardized mock infrastructure.
    
    Returns:
        Dict: Dictionary containing all flyrigloader-specific mock objects
    """
    # Use centralized mock providers with flyrigloader-specific configurations
    config_provider = create_mock_config_provider('comprehensive', include_errors=True)
    
    # Add flyrigloader-specific configuration
    config_provider.add_configuration('flyrigloader_integration', sample_neuroscience_config_dict)
    
    # Create centralized data loader with neuroscience scenarios
    data_loader = create_mock_dataloader(
        scenarios=['basic', 'corrupted', 'network'],
        include_experimental_data=True
    )
    
    # Add flyrigloader-specific experimental data
    for file_path, file_info in temp_neuroscience_filesystem.items():
        if isinstance(file_info, Path) and file_info.suffix == '.csv':
            # Add neuroscience experimental matrices
            data_loader.add_experimental_matrix(
                str(file_info),
                n_timepoints=18000 if 'baseline' in str(file_info) else 10800,
                include_signal='opto' in str(file_info),
                include_metadata=True
            )
    
    # Create centralized filesystem with neuroscience structure
    filesystem_structure = {
        'files': {
            str(file_path): {'size': 2048 + i * 512} 
            for i, (name, file_path) in enumerate(temp_neuroscience_filesystem.items())
            if isinstance(file_path, Path)
        },
        'directories': [str(dir_path) for dir_path in temp_neuroscience_filesystem.values() 
                       if isinstance(dir_path, Path) and dir_path.is_dir()]
    }
    
    filesystem = create_mock_filesystem(
        structure=filesystem_structure,
        unicode_files=True,
        corrupted_files=True
    )
    
    # Create discovered files mapping for flyrigloader
    discovered_files = {}
    for name, file_path in temp_neuroscience_filesystem.items():
        if isinstance(file_path, Path) and 'file' in name:
            discovered_files[str(file_path)] = {
                "date": "20241220" if 'baseline' in name else "20241218",
                "condition": "control" if 'baseline' in name else "treatment",
                "replicate": "1",
                "dataset": "baseline" if 'baseline' in name else "optogenetic",
                "file_size": 2048,
                "modification_time": datetime.now().isoformat()
            }
    
    return {
        "config_provider": config_provider,
        "data_loader": data_loader,
        "filesystem": filesystem,
        "discovered_files": discovered_files,
        "neuroscience_config": sample_neuroscience_config_dict
    }

# Backwards compatibility alias for transition period  
@pytest.fixture
def mock_config_and_discovery(mock_flyrigloader_integration):
    """Backwards compatibility alias for mock_config_and_discovery."""
    return mock_flyrigloader_integration

# --- Redirected Mock Fixtures to Centralized Infrastructure ---
# These fixtures now redirect to centralized mock implementations

@pytest.fixture
def mock_flyrigloader_filesystem_operations(temp_neuroscience_filesystem):
    """
    Flyrigloader-specific filesystem operations using centralized MockFilesystem.
    
    Redirects to centralized MockFilesystem from tests/utils.py with flyrigloader-specific
    filesystem structure. Eliminates duplicate mock implementation while maintaining
    flyrigloader-specific behavior patterns.
    
    Returns:
        MockFilesystem: Configured filesystem mock for flyrigloader operations
    """
    # Create filesystem structure for flyrigloader
    filesystem_structure = {
        'files': {
            str(file_path): {'size': 2048} 
            for name, file_path in temp_neuroscience_filesystem.items()
            if isinstance(file_path, Path) and file_path.suffix in ['.csv', '.pkl', '.yaml']
        },
        'directories': [
            str(dir_path) for name, dir_path in temp_neuroscience_filesystem.items()
            if isinstance(dir_path, Path) and name.endswith('s')  # Directory names typically plural
        ]
    }
    
    return create_mock_filesystem(
        structure=filesystem_structure,
        unicode_files=True,
        corrupted_files=False  # Keep clean for basic operations
    )

# Backwards compatibility alias for transition period
@pytest.fixture
def mock_filesystem_operations(mock_flyrigloader_filesystem_operations):
    """Backwards compatibility alias - redirects to centralized implementation."""
    # Return dictionary format expected by legacy tests
    filesystem = mock_flyrigloader_filesystem_operations
    
    # Create mock objects that match the legacy interface
    from unittest.mock import MagicMock
    
    return {
        "path_exists": MagicMock(side_effect=filesystem.exists),
        "path_is_file": MagicMock(side_effect=filesystem.is_file),
        "path_glob": MagicMock(side_effect=filesystem.glob),
        "path_rglob": MagicMock(side_effect=filesystem.rglob),
        "stat": MagicMock(side_effect=filesystem.stat),
        "open": MagicMock(side_effect=filesystem.open_file)
    }

# External dependencies are now handled by centralized infrastructure
# No need for local mock_external_dependencies - use centralized mock providers instead

# --- Flyrigloader-Specific Integration Test Scenarios ---
# Performance and hypothesis testing now handled by centralized infrastructure

@pytest.fixture
def sample_flyrigloader_pickle_files(temp_neuroscience_filesystem, sample_neuroscience_exp_matrix_with_signals):
    """
    Create flyrigloader-specific pickle files using centralized infrastructure.
    
    Uses centralized MockDataLoading from tests/utils.py to create various pickle file
    formats for flyrigloader testing. Maintains neuroscience-specific data patterns.
    
    Returns:
        MockDataLoading: Configured data loader with flyrigloader pickle scenarios
    """
    data_loader = create_mock_dataloader(
        scenarios=['basic', 'corrupted'], 
        include_experimental_data=True
    )
    
    # Add flyrigloader-specific pickle files
    base_dir = temp_neuroscience_filesystem["data_root"]
    
    # Standard neuroscience experimental pickle
    data_loader.add_experimental_matrix(
        str(base_dir / "standard_neuroscience_data.pkl"),
        n_timepoints=18000,  # 5 minutes at 60 Hz
        include_signal=True,
        include_metadata=True
    )
    
    # Gzipped experimental data
    data_loader.add_experimental_matrix(
        str(base_dir / "gzipped_neuroscience_data.pkl.gz"),
        n_timepoints=36000,  # 10 minutes at 60 Hz
        include_signal=True,
        include_metadata=True
    )
    
    # Pandas-format DataFrame
    data_loader.add_dataframe_file(
        str(base_dir / "pandas_neuroscience_data.pkl"),
        shape=(10800, 5),  # 3 minutes worth of data
        columns=['t', 'x', 'y', 'signal', 'dtheta']
    )
    
    return data_loader

@pytest.fixture
def flyrigloader_integration_scenario(
    sample_neuroscience_config_dict,
    temp_neuroscience_filesystem,
    sample_flyrigloader_pickle_files
):
    """
    Complete flyrigloader integration test scenario using centralized infrastructure.
    
    Combines neuroscience configuration, filesystem structure, and pickle files
    into comprehensive integration test environment. Uses centralized mock
    infrastructure while maintaining flyrigloader-specific patterns.
    
    Returns:
        Dict[str, Any]: Complete flyrigloader integration test scenario
    """
    scenario = {
        "config": sample_neuroscience_config_dict,
        "filesystem": temp_neuroscience_filesystem,
        "data_loader": sample_flyrigloader_pickle_files,
        "expected_files": {
            "baseline_experiments": [
                temp_neuroscience_filesystem.get("baseline_file_1"),
                temp_neuroscience_filesystem.get("baseline_file_2")
            ],
            "optogenetic_experiments": [
                temp_neuroscience_filesystem.get("opto_file_1"),
                temp_neuroscience_filesystem.get("opto_file_2")
            ],
            "navigation_experiments": [
                temp_neuroscience_filesystem.get("nav_file_1"),
                temp_neuroscience_filesystem.get("nav_file_2")
            ]
        },
        "expected_metadata_extractions": {
            "baseline_experiments": {
                "dataset": "baseline_behavior",
                "date": "20241220",
                "condition": "control",
                "rig": "old_opto"
            },
            "optogenetic_experiments": {
                "dataset": "optogenetic_stimulation",
                "date": "20241218", 
                "condition": "treatment",
                "rig": "new_opto"
            }
        },
        "neuroscience_specific": {
            "arena_diameter_mm": 120,
            "sampling_frequency_hz": 60,
            "expected_columns": ["t", "x", "y", "signal", "dtheta"],
            "rig_types": ["old_opto", "new_opto", "high_speed_rig"]
        }
    }
    
    return scenario

# Backwards compatibility aliases for transition period
@pytest.fixture
def integration_test_scenario(flyrigloader_integration_scenario):
    """Backwards compatibility alias for integration_test_scenario."""
    return flyrigloader_integration_scenario

@pytest.fixture
def sample_pickle_files(sample_flyrigloader_pickle_files):
    """Backwards compatibility alias for sample_pickle_files."""
    # Return dictionary format expected by legacy tests
    return {
        "standard": "standard_neuroscience_data.pkl",
        "gzipped": "gzipped_neuroscience_data.pkl.gz", 
        "pandas": "pandas_neuroscience_data.pkl",
        "data_loader": sample_flyrigloader_pickle_files
    }

# Hypothesis strategies and performance test data are now available through:
# - tests.utils.create_hypothesis_strategies() for domain-specific strategies
# - tests.utils.create_performance_test_utilities() for performance testing
# - tests.conftest.py performance_benchmarks fixture for SLA validation