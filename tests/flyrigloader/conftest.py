"""
Shared fixtures for flyrigloader tests.

This module contains pytest fixtures that are shared across multiple test files
to reduce code duplication and ensure consistency in test data.

Enhanced with comprehensive mocking scenarios for integration testing,
advanced synthetic data generation, property-based testing support,
and cross-platform temporary filesystem management.
"""
import importlib.util
import os
import tempfile
import platform
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Generator, Tuple, Iterator
from datetime import datetime, timedelta
import gzip
import pickle

import numpy as np
import pandas as pd
import pytest
import yaml
from unittest.mock import MagicMock, patch, mock_open

HYPOTHESIS_AVAILABLE = importlib.util.find_spec("hypothesis") is not None

if HYPOTHESIS_AVAILABLE:
    from hypothesis import strategies as st
    from hypothesis import given, settings, assume
else:  # pragma: no cover - executed when Hypothesis is missing
    st = None  # type: ignore[assignment]
    settings = None  # type: ignore[assignment]
    assume = None  # type: ignore[assignment]


# --- Enhanced Configuration Fixtures ---

@pytest.fixture
def comprehensive_sample_config_dict():
    """
    Return a comprehensive sample configuration dictionary with all supported features.
    
    This fixture provides the full range of configuration options including:
    - Project directories and global settings
    - Ignore patterns and extraction patterns
    - Multiple rig configurations with different parameters
    - Complex dataset definitions with filters and metadata
    - Experiment hierarchies with nested datasets
    
    Returns:
        Dict[str, Any]: Comprehensive sample configuration dictionary
    """
    return {
        "project": {
            "directories": {
                "major_data_directory": "/research/data/neuroscience",
                "batchfile_directory": "/research/batch_definitions",
                "backup_directory": "/research/backups",
                "processed_data_directory": "/research/processed"
            },
            "ignore_substrings": [
                "static_horiz_ribbon",
                "._",
                ".DS_Store",
                "__pycache__",
                ".tmp",
                "backup_",
                "test_calibration"
            ],
            "mandatory_substrings": [
                "experiment_",
                "data_"
            ],
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
                "sampling_frequency": 60,
                "mm_per_px": 0.154,
                "camera_resolution": [1024, 768],
                "calibration_date": "2024-01-15",
                "arena_diameter_mm": 120,
                "led_wavelength_nm": 470
            },
            "new_opto": {
                "sampling_frequency": 60,
                "mm_per_px": 0.1818,
                "camera_resolution": [1280, 1024],
                "calibration_date": "2024-06-01",
                "arena_diameter_mm": 150,
                "led_wavelength_nm": 470
            },
            "high_speed_rig": {
                "sampling_frequency": 200,
                "mm_per_px": 0.05,
                "camera_resolution": [2048, 2048],
                "calibration_date": "2024-08-15",
                "arena_diameter_mm": 200,
                "led_wavelength_nm": 590
            }
        },
        "datasets": {
            "baseline_behavior": {
                "rig": "old_opto",
                "patterns": ["*baseline*", "*control*"],
                "dates_vials": {
                    "2024-12-20": [1, 2, 3, 4, 5],
                    "2024-12-21": [1, 2, 3],
                    "2024-12-22": [1, 2]
                },
                "metadata": {
                    "extraction_patterns": [
                        r".*_(?P<dataset>\w+)_(?P<date>\d{8})_(?P<vial>\d+)\.csv"
                    ],
                    "required_fields": ["dataset", "date", "vial"],
                    "experiment_type": "baseline"
                },
                "filters": {
                    "min_duration_seconds": 300,
                    "max_duration_seconds": 3600,
                    "required_columns": ["t", "x", "y"]
                }
            },
            "optogenetic_stimulation": {
                "rig": "new_opto",
                "patterns": ["*opto*", "*stim*"],
                "dates_vials": {
                    "2024-12-18": [1, 2, 3, 4],
                    "2024-12-19": [1, 2, 3, 4, 5, 6]
                },
                "metadata": {
                    "extraction_patterns": [
                        r".*_(?P<dataset>\w+)_(?P<stimulation_type>\w+)_(?P<date>\d{8})\.csv"
                    ],
                    "required_fields": ["dataset", "stimulation_type", "date"],
                    "experiment_type": "optogenetic"
                },
                "filters": {
                    "ignore_substrings": ["failed", "aborted"],
                    "min_file_size_bytes": 10000
                }
            },
            "plume_movie_navigation": {
                "rig": "old_opto",
                "patterns": ["*plume*", "*navigation*"],
                "dates_vials": {
                    "2024-10-18": [1, 3, 4, 5],
                    "2024-10-24": [1, 2],
                    "2024-10-25": [1, 2, 3, 4, 5, 6, 7, 8]
                },
                "metadata": {
                    "extraction_patterns": [
                        r".*_(?P<plume_type>\w+)_(?P<date>\d{8})_(?P<trial>\d+)\.csv"
                    ],
                    "required_fields": ["plume_type", "date", "trial"],
                    "experiment_type": "navigation"
                }
            },
            "high_resolution_tracking": {
                "rig": "high_speed_rig",
                "patterns": ["*highres*", "*200hz*"],
                "dates_vials": {
                    "2024-11-01": [1, 2, 3],
                    "2024-11-02": [1, 2, 3, 4, 5]
                },
                "metadata": {
                    "extraction_patterns": [
                        r".*_(?P<resolution>\d+hz)_(?P<date>\d{8})_(?P<session>\d+)\.csv"
                    ],
                    "required_fields": ["resolution", "date", "session"],
                    "experiment_type": "high_resolution"
                }
            }
        },
        "experiments": {
            "baseline_control_study": {
                "datasets": ["baseline_behavior"],
                "metadata": {
                    "extraction_patterns": [
                        r".*_(?P<experiment>baseline)_(?P<date>\d{8})\.csv"
                    ],
                    "required_fields": ["experiment", "date"],
                    "study_type": "control",
                    "principal_investigator": "Dr. Research",
                    "grant_number": "NSF-123456"
                },
                "analysis_parameters": {
                    "velocity_threshold": 2.0,
                    "smoothing_window": 5,
                    "edge_exclusion_mm": 10
                }
            },
            "optogenetic_manipulation": {
                "datasets": ["optogenetic_stimulation", "baseline_behavior"],
                "metadata": {
                    "extraction_patterns": [
                        r".*_(?P<experiment>opto)_(?P<treatment>\w+)_(?P<date>\d{8})\.csv"
                    ],
                    "required_fields": ["experiment", "treatment", "date"],
                    "study_type": "intervention"
                },
                "filters": {
                    "ignore_substrings": ["smoke_2a", "calibration"],
                    "mandatory_substrings": ["opto"]
                }
            },
            "multi_modal_navigation": {
                "datasets": [
                    "plume_movie_navigation",
                    "baseline_behavior",
                    "high_resolution_tracking"
                ],
                "metadata": {
                    "extraction_patterns": [
                        r".*_(?P<experiment>navigation)_(?P<modality>\w+)_(?P<date>\d{8})\.csv"
                    ],
                    "required_fields": ["experiment", "modality", "date"],
                    "study_type": "comparative"
                },
                "analysis_parameters": {
                    "spatial_bins": 50,
                    "temporal_resolution_ms": 16.67,
                    "trajectory_smoothing": True
                }
            },
            "longitudinal_development": {
                "datasets": ["baseline_behavior", "optogenetic_stimulation"],
                "metadata": {
                    "extraction_patterns": [
                        r".*_(?P<age_group>\w+)_(?P<date>\d{8})_(?P<timepoint>\d+)\.csv"
                    ],
                    "required_fields": ["age_group", "date", "timepoint"],
                    "study_type": "longitudinal"
                },
                "temporal_grouping": {
                    "timepoint_1": ["2024-10-01", "2024-10-07"],
                    "timepoint_2": ["2024-10-15", "2024-10-21"],
                    "timepoint_3": ["2024-11-01", "2024-11-07"]
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
    }

@pytest.fixture
def sample_config_file(comprehensive_sample_config_dict, cross_platform_temp_dir):
    """
    Create a temporary config file with comprehensive sample configuration.
    
    Args:
        comprehensive_sample_config_dict: The comprehensive configuration data
        cross_platform_temp_dir: Cross-platform temporary directory fixture
    
    Returns:
        str: Path to the temporary config file
    """
    config_path = cross_platform_temp_dir / "config.yaml"
    
    # Write the config to the file
    with open(config_path, 'w') as f:
        yaml.dump(comprehensive_sample_config_dict, f, default_flow_style=False)
    
    return str(config_path)

@pytest.fixture
def sample_config_dict(comprehensive_sample_config_dict):
    """
    Return a simplified sample configuration dictionary for basic testing.
    
    This fixture provides a subset of the comprehensive configuration
    for tests that don't need the full complexity.
    
    Returns:
        Dict[str, Any]: Simplified sample configuration dictionary
    """
    return {
        "project": comprehensive_sample_config_dict["project"],
        "rigs": {
            "old_opto": comprehensive_sample_config_dict["rigs"]["old_opto"]
        },
        "datasets": {
            "test_dataset": comprehensive_sample_config_dict["datasets"]["baseline_behavior"]
        },
        "experiments": {
            "test_experiment": comprehensive_sample_config_dict["experiments"]["baseline_control_study"]
        }
    }

# --- Cross-Platform Temporary Filesystem Fixtures ---

@pytest.fixture
def cross_platform_temp_dir():
    """
    Create a cross-platform temporary directory with proper cleanup.
    
    This fixture handles platform-specific considerations for Windows, Linux, and macOS:
    - Uses appropriate temporary directory locations per OS
    - Ensures proper permissions across platforms
    - Handles long path limitations on Windows
    - Provides proper cleanup even on test failures
    
    Returns:
        Path: Cross-platform temporary directory path
    """
    import shutil
    
    # Create platform-appropriate temporary directory
    if platform.system() == "Windows":
        # Use shorter paths to avoid Windows MAX_PATH limitations
        temp_base = Path(tempfile.gettempdir()) / "flyrig_test"
        temp_base.mkdir(exist_ok=True)
        temp_dir = tempfile.mkdtemp(dir=temp_base, prefix="test_")
    else:
        # Unix-like systems (Linux, macOS) can handle longer paths
        temp_dir = tempfile.mkdtemp(prefix="flyrigloader_test_")
    
    temp_path = Path(temp_dir)
    
    try:
        # Ensure the directory is writable
        test_file = temp_path / "write_test.tmp"
        test_file.write_text("test")
        test_file.unlink()
        
        yield temp_path
    finally:
        # Comprehensive cleanup with error handling
        try:
            if temp_path.exists():
                shutil.rmtree(temp_path, ignore_errors=True)
        except Exception:
            # On Windows, files might be locked; try again with onerror handler
            if platform.system() == "Windows":
                def handle_remove_readonly(func, path, exc):
                    import stat
                    os.chmod(path, stat.S_IWRITE)
                    func(path)
                
                try:
                    shutil.rmtree(temp_path, onerror=handle_remove_readonly)
                except Exception:
                    pass  # Best effort cleanup

@pytest.fixture
def temp_filesystem_structure(cross_platform_temp_dir):
    """
    Create a realistic temporary filesystem structure for integration testing.
    
    This fixture creates a directory structure that mimics a real research
    data organization with multiple experiments, datasets, and file types.
    
    Returns:
        Dict[str, Path]: Dictionary mapping logical names to filesystem paths
    """
    base_dir = cross_platform_temp_dir
    
    # Create directory structure
    structure = {
        "data_root": base_dir / "research_data",
        "experiments": base_dir / "research_data" / "experiments",
        "baselines": base_dir / "research_data" / "experiments" / "baseline",
        "optogenetics": base_dir / "research_data" / "experiments" / "optogenetics",
        "navigation": base_dir / "research_data" / "experiments" / "navigation",
        "batch_files": base_dir / "batch_definitions",
        "configs": base_dir / "configs",
        "processed": base_dir / "processed_data"
    }
    
    # Create all directories
    for dir_path in structure.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create sample data files
    sample_files = {
        # Baseline experiment files
        "baseline_file_1": structure["baselines"] / "baseline_20241220_control_1.csv",
        "baseline_file_2": structure["baselines"] / "baseline_20241221_control_2.csv",
        
        # Optogenetic experiment files  
        "opto_file_1": structure["optogenetics"] / "opto_stim_20241218_treatment_1.csv",
        "opto_file_2": structure["optogenetics"] / "opto_stim_20241219_treatment_2.csv",
        
        # Navigation experiment files
        "nav_file_1": structure["navigation"] / "plume_navigation_20241025_trial_1.csv",
        "nav_file_2": structure["navigation"] / "plume_navigation_20241025_trial_2.csv",
        
        # Ignore pattern test files (should be filtered out)
        "ignored_file_1": structure["baselines"] / "static_horiz_ribbon_calibration.csv",
        "ignored_file_2": structure["optogenetics"] / "._temp_file.csv",
        
        # Configuration file
        "config_file": structure["configs"] / "experiment_config.yaml"
    }
    
    # Create sample CSV content
    sample_csv_content = """t,x,y,signal
0.0,10.5,20.3,0.1
0.016,10.6,20.2,0.2
0.032,10.7,20.1,0.3
0.048,10.8,20.0,0.4
"""
    
    # Write sample files
    for file_key, file_path in sample_files.items():
        if file_path.suffix == ".csv":
            file_path.write_text(sample_csv_content)
        elif file_path.suffix == ".yaml":
            file_path.write_text("# Sample config file")
    
    return {**structure, **sample_files}


# --- Column Configuration Fixtures ---

@pytest.fixture
def sample_column_config_file():
    """
    Create a temporary column config file.
    
    Returns:
        str: Path to the temporary column config file
    """
    # Create a temporary configuration file
    temp_file = tempfile.NamedTemporaryFile(suffix='.yaml', mode='w', delete=False)
    config_path = temp_file.name
    
    # Define test configuration
    test_config = {
        'columns': {
            't': {
                'type': 'numpy.ndarray',
                'dimension': 1,
                'required': True,
                'description': 'Time values'
            },
            'x': {
                'type': 'numpy.ndarray',
                'dimension': 1,
                'required': True,
                'description': 'X position'
            },
            'y': {
                'type': 'numpy.ndarray',
                'dimension': 1,
                'required': True,
                'description': 'Y position'
            },
            'dtheta': {
                'type': 'numpy.ndarray',
                'dimension': 1,
                'required': False,
                'description': 'Change in heading',
                'alias': 'dtheta_smooth'
            },
            'signal': {
                'type': 'numpy.ndarray',
                'dimension': 1,
                'required': False,
                'description': 'Signal values',
                'default_value': None
            },
            'signal_disp': {
                'type': 'numpy.ndarray',
                'dimension': 2,
                'required': False,
                'description': 'Signal display data',
                'special_handling': 'transform_to_match_time_dimension'
            },
            'date': {
                'type': 'string',
                'required': False,
                'is_metadata': True,
                'description': 'Experiment date'
            },
            'exp_name': {
                'type': 'string',
                'required': False,
                'is_metadata': True,
                'description': 'Experiment name'
            },
            'rig': {
                'type': 'string',
                'required': False,
                'is_metadata': True,
                'description': 'Rig identifier'
            },
            'fly_id': {
                'type': 'string',
                'required': False,
                'is_metadata': True,
                'description': 'Fly ID'
            }
        },
        'special_handlers': {
            'transform_to_match_time_dimension': '_handle_signal_disp'
        }
    }
    
    yaml.dump(test_config, temp_file)
    temp_file.close()
    
    yield config_path
    
    # Clean up
    os.unlink(config_path)


# --- Advanced Synthetic Experimental Data Generation Fixtures ---

@pytest.fixture
def realistic_time_series_params():
    """
    Parameters for generating realistic experimental time series data.
    
    Returns:
        Dict: Parameters for synthetic data generation
    """
    return {
        "sampling_frequency": 60.0,  # Hz
        "duration_seconds": 300.0,   # 5 minutes
        "arena_diameter_mm": 120.0,
        "center_bias": 0.3,          # Tendency to stay near center
        "movement_noise": 0.1,       # Movement smoothness
        "velocity_max": 15.0,        # mm/s maximum velocity
        "signal_channels": 16,       # Number of signal channels
        "signal_noise_level": 0.05   # Signal-to-noise ratio
    }

@pytest.fixture
def synthetic_trajectory_generator():
    """
    Factory function for generating realistic synthetic fly trajectories.
    
    This fixture generates biologically plausible movement patterns including:
    - Brownian motion with drift toward center
    - Realistic velocity profiles
    - Arena boundary constraints
    - Temporally correlated movement patterns
    
    Returns:
        Callable: Function that generates trajectory data
    """
    def generate_trajectory(
        n_timepoints: int = 1000,
        sampling_freq: float = 60.0,
        arena_diameter: float = 120.0,
        center_bias: float = 0.3,
        movement_noise: float = 0.1,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a realistic fly trajectory.
        
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
        
        # Initialize time array
        dt = 1.0 / sampling_freq
        time = np.arange(n_timepoints) * dt
        
        # Initialize position at arena center
        arena_radius = arena_diameter / 2.0
        x_pos = np.zeros(n_timepoints)
        y_pos = np.zeros(n_timepoints)
        
        # Generate correlated random walk with center bias
        for i in range(1, n_timepoints):
            # Current distance from center
            current_radius = np.sqrt(x_pos[i-1]**2 + y_pos[i-1]**2)
            
            # Center bias force (stronger near edges)
            bias_strength = center_bias * (current_radius / arena_radius)**2
            center_force_x = -bias_strength * x_pos[i-1] / max(current_radius, 0.1)
            center_force_y = -bias_strength * y_pos[i-1] / max(current_radius, 0.1)
            
            # Random movement component
            random_x = np.random.normal(0, movement_noise)
            random_y = np.random.normal(0, movement_noise)
            
            # Update position
            dx = (center_force_x + random_x) * dt
            dy = (center_force_y + random_y) * dt
            
            new_x = x_pos[i-1] + dx
            new_y = y_pos[i-1] + dy
            
            # Enforce arena boundaries with reflection
            new_radius = np.sqrt(new_x**2 + new_y**2)
            if new_radius > arena_radius:
                # Reflect off boundary
                reflection_factor = arena_radius / new_radius
                new_x *= reflection_factor * 0.95  # Slight inward bias
                new_y *= reflection_factor * 0.95
            
            x_pos[i] = new_x
            y_pos[i] = new_y
        
        return time, x_pos, y_pos
    
    return generate_trajectory

@pytest.fixture
def synthetic_signal_generator():
    """
    Factory function for generating realistic multi-channel signal data.
    
    Generates signals that mimic calcium imaging or electrophysiological recordings
    with realistic noise characteristics and temporal correlations.
    
    Returns:
        Callable: Function that generates multi-channel signal data
    """
    def generate_signals(
        n_timepoints: int,
        n_channels: int = 16,
        signal_freq: float = 2.0,
        noise_level: float = 0.1,
        baseline_drift: bool = True,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate realistic multi-channel signal data.
        
        Args:
            n_timepoints: Number of time points
            n_channels: Number of signal channels  
            signal_freq: Characteristic frequency of signal oscillations (Hz)
            noise_level: Noise amplitude relative to signal
            baseline_drift: Whether to include slow baseline drift
            seed: Random seed for reproducibility
            
        Returns:
            Array of shape (n_channels, n_timepoints) with signal data
        """
        if seed is not None:
            np.random.seed(seed)
        
        signals = np.zeros((n_channels, n_timepoints))
        
        for ch in range(n_channels):
            # Base signal with channel-specific phase and amplitude
            phase_offset = 2 * np.pi * ch / n_channels
            amplitude = 0.8 + 0.4 * np.random.random()  # Random amplitude 0.8-1.2
            
            t = np.linspace(0, n_timepoints/60.0, n_timepoints)  # Assume 60 Hz
            base_signal = amplitude * np.sin(2 * np.pi * signal_freq * t + phase_offset)
            
            # Add harmonic components
            base_signal += 0.3 * amplitude * np.sin(4 * np.pi * signal_freq * t + phase_offset)
            base_signal += 0.1 * amplitude * np.sin(6 * np.pi * signal_freq * t + phase_offset)
            
            # Add baseline drift if requested
            if baseline_drift:
                drift_freq = 0.01  # Very slow drift
                drift = 0.2 * np.sin(2 * np.pi * drift_freq * t + np.random.random() * 2 * np.pi)
                base_signal += drift
            
            # Add noise
            noise = noise_level * np.random.normal(0, 1, n_timepoints)
            
            signals[ch, :] = base_signal + noise
        
        return signals
    
    return generate_signals

@pytest.fixture
def sample_exp_matrix(synthetic_trajectory_generator, realistic_time_series_params):
    """
    Create sample experimental data matrix using realistic synthetic generation.
    
    Returns:
        Dict[str, np.ndarray]: Sample experimental data
    """
    params = realistic_time_series_params
    n_points = int(params["duration_seconds"] * params["sampling_frequency"])
    
    time, x_pos, y_pos = synthetic_trajectory_generator(
        n_timepoints=n_points,
        sampling_freq=params["sampling_frequency"],
        arena_diameter=params["arena_diameter_mm"],
        seed=42  # Reproducible for tests
    )
    
    return {
        't': time,
        'x': x_pos,
        'y': y_pos
    }

@pytest.fixture
def sample_exp_matrix_with_signal_disp(sample_exp_matrix, synthetic_signal_generator):
    """
    Create sample experimental data matrix with realistic signal_disp data.
    
    Returns:
        Dict[str, np.ndarray]: Sample experimental data with signal_disp
    """
    matrix = sample_exp_matrix.copy()
    n_timepoints = len(matrix['t'])
    
    # Generate realistic multi-channel signal data
    signal_data = synthetic_signal_generator(
        n_timepoints=n_timepoints,
        n_channels=15,  # 15 channels as in original
        seed=42
    )
    
    matrix['signal_disp'] = signal_data
    return matrix

@pytest.fixture  
def sample_exp_matrix_with_aliases(sample_exp_matrix):
    """
    Create sample experimental data matrix with aliased column names.
    
    Returns:
        Dict[str, np.ndarray]: Sample experimental data with aliased columns
    """
    matrix = sample_exp_matrix.copy()
    
    # Add aliased column (dtheta_smooth instead of dtheta)
    # Generate realistic angular velocity data
    x_diff = np.diff(matrix['x'], prepend=matrix['x'][0])
    y_diff = np.diff(matrix['y'], prepend=matrix['y'][0])
    dtheta_smooth = np.arctan2(y_diff, x_diff) + 0.1 * np.random.normal(0, 1, len(matrix['t']))
    
    matrix['dtheta_smooth'] = dtheta_smooth
    return matrix

@pytest.fixture
def comprehensive_exp_matrix(sample_exp_matrix_with_signal_disp, synthetic_signal_generator):
    """
    Create a comprehensive experimental data matrix with all possible columns.
    
    This fixture provides a complete dataset that tests can use to validate
    all column handling and transformation functionality.
    
    Returns:
        Dict[str, np.ndarray]: Comprehensive experimental data matrix
    """
    matrix = sample_exp_matrix_with_signal_disp.copy()
    n_timepoints = len(matrix['t'])
    
    # Add single-channel signal
    matrix['signal'] = synthetic_signal_generator(
        n_timepoints=n_timepoints,
        n_channels=1,
        seed=43
    )[0, :]
    
    # Add velocity components
    x_diff = np.diff(matrix['x'], prepend=matrix['x'][0])
    y_diff = np.diff(matrix['y'], prepend=matrix['y'][0])
    dt = np.diff(matrix['t'], prepend=matrix['t'][1] - matrix['t'][0])
    
    matrix['vx'] = x_diff / dt
    matrix['vy'] = y_diff / dt
    matrix['speed'] = np.sqrt(matrix['vx']**2 + matrix['vy']**2)
    
    # Add angular measures
    matrix['dtheta'] = np.arctan2(y_diff, x_diff)
    matrix['dtheta_smooth'] = matrix['dtheta']  # Alias
    
    # Add derived measures
    matrix['distance_from_center'] = np.sqrt(matrix['x']**2 + matrix['y']**2)
    matrix['cumulative_distance'] = np.cumsum(np.sqrt(x_diff**2 + y_diff**2))
    
    return matrix

@pytest.fixture
def sample_metadata():
    """
    Create sample metadata dictionary for tests.
    
    Returns:
        Dict[str, str]: Sample metadata
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
        'humidity_percent': '45.2'
    }

@pytest.fixture
def sample_pandas_dataframe(comprehensive_exp_matrix, sample_metadata):
    """
    Create a sample pandas DataFrame with experimental data and metadata.
    
    This fixture demonstrates the expected output format after data processing.
    
    Returns:
        pd.DataFrame: Sample DataFrame with experimental data
    """
    # Convert matrix to DataFrame
    df_data = {}
    
    # Add time series data
    for col, data in comprehensive_exp_matrix.items():
        if data.ndim == 1:
            df_data[col] = data
        elif data.ndim == 2:
            # For 2D data like signal_disp, add as columns with channel indices
            for ch in range(data.shape[0]):
                df_data[f"{col}_ch{ch:02d}"] = data[ch, :]
    
    df = pd.DataFrame(df_data)
    
    # Add metadata as constant columns
    for key, value in sample_metadata.items():
        df[key] = value
    
    return df


# --- Enhanced Mock Fixtures for Comprehensive Integration Testing ---

@pytest.fixture
def mock_config_and_discovery(mocker, comprehensive_sample_config_dict, temp_filesystem_structure):
    """
    Setup comprehensive mocks for config loading and file discovery per TST-INTEG-002.
    
    This fixture provides sophisticated mocking of the entire configuration and discovery
    pipeline with realistic return values that support complex integration testing scenarios.
    
    Args:
        mocker: pytest-mock fixture for enhanced mocking capabilities
        comprehensive_sample_config_dict: Comprehensive configuration data
        temp_filesystem_structure: Realistic filesystem structure
        
    Returns:
        Dict: Dictionary containing all mock objects and helper functions
    """
    # Mock the config loading function with comprehensive return values
    mock_load_config = mocker.patch("flyrigloader.api.load_config")
    mock_load_config.return_value = comprehensive_sample_config_dict
    
    # Create realistic file discovery return values
    discovered_files = {
        str(temp_filesystem_structure["baseline_file_1"]): {
            "date": "20241220",
            "condition": "control", 
            "replicate": "1",
            "dataset": "baseline",
            "file_size": 1024,
            "modification_time": datetime.now().isoformat()
        },
        str(temp_filesystem_structure["baseline_file_2"]): {
            "date": "20241221",
            "condition": "control",
            "replicate": "2", 
            "dataset": "baseline",
            "file_size": 1536,
            "modification_time": datetime.now().isoformat()
        },
        str(temp_filesystem_structure["opto_file_1"]): {
            "date": "20241218",
            "condition": "treatment",
            "replicate": "1",
            "dataset": "optogenetic",
            "stimulation_type": "stim",
            "file_size": 2048,
            "modification_time": datetime.now().isoformat()
        },
        str(temp_filesystem_structure["nav_file_1"]): {
            "date": "20241025",
            "condition": "navigation",
            "replicate": "1",
            "dataset": "plume",
            "plume_type": "plume",
            "trial": "1",
            "file_size": 3072,
            "modification_time": datetime.now().isoformat()
        }
    }
    
    # Mock discovery functions with realistic return values
    mock_discover_experiment_files = mocker.patch("flyrigloader.api.discover_experiment_files")
    mock_discover_experiment_files.return_value = discovered_files
    
    mock_discover_dataset_files = mocker.patch("flyrigloader.api.discover_dataset_files")
    mock_discover_dataset_files.return_value = discovered_files
    
    # Mock individual discovery components for granular testing
    mock_file_discoverer = mocker.patch("flyrigloader.discovery.files.FileDiscoverer")
    mock_file_discoverer_instance = MagicMock()
    mock_file_discoverer.return_value = mock_file_discoverer_instance
    mock_file_discoverer_instance.find_files.return_value = list(discovered_files.keys())
    
    # Mock YAML configuration functions
    mock_yaml_load_config = mocker.patch("flyrigloader.config.yaml_config.load_config")
    mock_yaml_load_config.return_value = comprehensive_sample_config_dict
    
    # Mock pickle loading for data files
    mock_pickle_loader = mocker.patch("flyrigloader.io.pickle.read_pickle_any_format")
    
    def pickle_loader_side_effect(path):
        """Dynamic side effect for pickle loading based on file path."""
        if "baseline" in str(path):
            return {
                't': np.linspace(0, 300, 18000),  # 5 minutes at 60 Hz
                'x': np.random.rand(18000) * 100,
                'y': np.random.rand(18000) * 100
            }
        elif "opto" in str(path):
            return {
                't': np.linspace(0, 600, 36000),  # 10 minutes at 60 Hz  
                'x': np.random.rand(36000) * 100,
                'y': np.random.rand(36000) * 100,
                'signal': np.random.rand(36000)
            }
        elif "nav" in str(path):
            return {
                't': np.linspace(0, 180, 10800),  # 3 minutes at 60 Hz
                'x': np.random.rand(10800) * 120,
                'y': np.random.rand(10800) * 120,
                'signal_disp': np.random.rand(16, 10800)
            }
        else:
            return {'t': np.array([0, 1, 2]), 'x': np.array([0, 1, 2]), 'y': np.array([0, 1, 2])}
    
    mock_pickle_loader.side_effect = pickle_loader_side_effect
    
    # Mock column configuration loading
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
    
    return {
        "load_config": mock_load_config,
        "discover_experiment_files": mock_discover_experiment_files,
        "discover_dataset_files": mock_discover_dataset_files,
        "file_discoverer": mock_file_discoverer,
        "yaml_load_config": mock_yaml_load_config,
        "pickle_loader": mock_pickle_loader,
        "column_config": mock_column_config,
        "discovered_files": discovered_files
    }

@pytest.fixture
def mock_filesystem_operations(mocker, temp_filesystem_structure):
    """
    Mock filesystem operations for isolated testing.
    
    This fixture mocks common filesystem operations to enable testing
    without requiring actual file I/O while maintaining realistic behavior.
    
    Args:
        mocker: pytest-mock fixture
        temp_filesystem_structure: Temporary filesystem for realistic paths
        
    Returns:
        Dict: Dictionary containing filesystem operation mocks
    """
    # Mock pathlib Path operations
    mock_path_exists = mocker.patch("pathlib.Path.exists")
    mock_path_exists.return_value = True
    
    mock_path_is_file = mocker.patch("pathlib.Path.is_file") 
    mock_path_is_file.return_value = True
    
    mock_path_glob = mocker.patch("pathlib.Path.glob")
    mock_path_glob.return_value = [
        temp_filesystem_structure["baseline_file_1"],
        temp_filesystem_structure["baseline_file_2"]
    ]
    
    mock_path_rglob = mocker.patch("pathlib.Path.rglob")
    mock_path_rglob.return_value = [
        temp_filesystem_structure["baseline_file_1"],
        temp_filesystem_structure["baseline_file_2"],
        temp_filesystem_structure["opto_file_1"],
        temp_filesystem_structure["nav_file_1"]
    ]
    
    # Mock file statistics
    mock_stat = mocker.patch("pathlib.Path.stat")
    mock_stat_result = MagicMock()
    mock_stat_result.st_size = 2048
    mock_stat_result.st_mtime = datetime.now().timestamp()
    mock_stat_result.st_ctime = datetime.now().timestamp()
    mock_stat.return_value = mock_stat_result
    
    # Mock file reading operations
    mock_open_func = mocker.patch("builtins.open", mock_open(read_data="t,x,y\n0,1,2\n1,2,3\n"))
    
    return {
        "path_exists": mock_path_exists,
        "path_is_file": mock_path_is_file,
        "path_glob": mock_path_glob,
        "path_rglob": mock_path_rglob,
        "stat": mock_stat,
        "open": mock_open_func
    }

@pytest.fixture  
def mock_external_dependencies(mocker):
    """
    Mock external library dependencies for isolated unit testing.
    
    This fixture mocks external dependencies like NumPy, Pandas, and YAML
    operations to enable fast unit testing without full library overhead.
    
    Args:
        mocker: pytest-mock fixture
        
    Returns:
        Dict: Dictionary containing external dependency mocks
    """
    # Mock numpy operations
    mock_numpy_array = mocker.patch("numpy.array")
    mock_numpy_array.side_effect = lambda x: np.array(x)  # Pass through to real numpy
    
    mock_numpy_linspace = mocker.patch("numpy.linspace")
    mock_numpy_linspace.side_effect = lambda start, stop, num: np.linspace(start, stop, num)
    
    # Mock pandas operations
    mock_pandas_dataframe = mocker.patch("pandas.DataFrame")
    mock_pandas_dataframe.side_effect = lambda data: pd.DataFrame(data)
    
    # Mock YAML operations  
    mock_yaml_safe_load = mocker.patch("yaml.safe_load")
    mock_yaml_safe_load.return_value = {"test": "config"}
    
    mock_yaml_dump = mocker.patch("yaml.dump")
    mock_yaml_dump.return_value = "test: config\n"
    
    return {
        "numpy_array": mock_numpy_array,
        "numpy_linspace": mock_numpy_linspace,
        "pandas_dataframe": mock_pandas_dataframe,
        "yaml_safe_load": mock_yaml_safe_load,
        "yaml_dump": mock_yaml_dump
    }

# --- Property-Based Testing Fixtures Using Hypothesis ---

@pytest.fixture
def hypothesis_config_strategy():
    """
    Hypothesis strategy for generating valid configuration dictionaries.
    
    This fixture enables property-based testing of configuration validation
    and processing logic by generating diverse valid configuration structures.
    
    Returns:
        hypothesis.strategies.SearchStrategy: Strategy for configuration generation
    """
    # Strategy for project directories
    directory_strategy = st.fixed_dict({
        "major_data_directory": st.text(min_size=5, max_size=50),
        "batchfile_directory": st.text(min_size=5, max_size=50)
    })
    
    # Strategy for project configuration
    project_strategy = st.fixed_dict({
        "directories": directory_strategy,
        "ignore_substrings": st.lists(st.text(min_size=1, max_size=20), min_size=0, max_size=5),
        "extraction_patterns": st.lists(st.text(min_size=10, max_size=100), min_size=1, max_size=3)
    })
    
    # Strategy for rig configuration
    rig_strategy = st.fixed_dict({
        "sampling_frequency": st.floats(min_value=1.0, max_value=1000.0),
        "mm_per_px": st.floats(min_value=0.01, max_value=1.0),
        "camera_resolution": st.lists(st.integers(min_value=100, max_value=4000), min_size=2, max_size=2)
    })
    
    # Strategy for dataset configuration
    dataset_strategy = st.fixed_dict({
        "rig": st.text(min_size=1, max_size=20),
        "patterns": st.lists(st.text(min_size=1, max_size=30), min_size=1, max_size=5),
        "dates_vials": st.dictionaries(
            st.text(min_size=8, max_size=8).filter(lambda s: s.isdigit()),
            st.lists(st.integers(min_value=1, max_value=10), min_size=1, max_size=8),
            min_size=1, max_size=5
        )
    })
    
    # Strategy for complete configuration
    config_strategy = st.fixed_dict({
        "project": project_strategy,
        "rigs": st.dictionaries(st.text(min_size=1, max_size=20), rig_strategy, min_size=1, max_size=3),
        "datasets": st.dictionaries(st.text(min_size=1, max_size=20), dataset_strategy, min_size=1, max_size=5)
    })
    
    return config_strategy

@pytest.fixture
def hypothesis_experimental_data_strategy():
    """
    Hypothesis strategy for generating valid experimental data matrices.
    
    This fixture enables property-based testing of data processing and validation
    logic by generating diverse experimental data structures.
    
    Returns:
        hypothesis.strategies.SearchStrategy: Strategy for experimental data generation
    """
    # Time series strategy
    time_strategy = st.builds(
        np.linspace,
        start=st.floats(min_value=0.0, max_value=1.0),
        stop=st.floats(min_value=10.0, max_value=1000.0),
        num=st.integers(min_value=10, max_value=10000)
    )
    
    # Position data strategy
    position_strategy = st.builds(
        np.random.uniform,
        low=st.floats(min_value=-200.0, max_value=-50.0),
        high=st.floats(min_value=50.0, max_value=200.0),
        size=st.integers(min_value=10, max_value=10000)
    )
    
    # Signal data strategy
    signal_strategy = st.builds(
        np.random.normal,
        loc=st.floats(min_value=-1.0, max_value=1.0),
        scale=st.floats(min_value=0.1, max_value=2.0),
        size=st.integers(min_value=10, max_value=10000)
    )
    
    # Multi-channel signal strategy
    multichannel_strategy = st.builds(
        np.random.random,
        size=st.tuples(
            st.integers(min_value=1, max_value=32),  # channels
            st.integers(min_value=10, max_value=10000)  # timepoints
        )
    )
    
    # Complete experimental matrix strategy
    exp_matrix_strategy = st.fixed_dict({
        "t": time_strategy,
        "x": position_strategy,
        "y": position_strategy
    }).flatmap(lambda base: st.fixed_dict({
        **base,
        "signal": st.one_of(st.none(), signal_strategy),
        "signal_disp": st.one_of(st.none(), multichannel_strategy)
    }))
    
    return exp_matrix_strategy

@pytest.fixture
def hypothesis_metadata_strategy():
    """
    Hypothesis strategy for generating valid metadata dictionaries.
    
    Returns:
        hypothesis.strategies.SearchStrategy: Strategy for metadata generation
    """
    date_strategy = st.dates(
        min_value=datetime(2020, 1, 1).date(),
        max_value=datetime(2030, 12, 31).date()
    ).map(lambda d: d.strftime("%Y%m%d"))
    
    metadata_strategy = st.fixed_dict({
        "date": date_strategy,
        "exp_name": st.text(min_size=1, max_size=50),
        "rig": st.sampled_from(["old_opto", "new_opto", "high_speed_rig"]),
        "fly_id": st.text(min_size=1, max_size=20),
        "condition": st.sampled_from(["control", "treatment", "baseline", "stimulation"]),
        "replicate": st.integers(min_value=1, max_value=20).map(str)
    })
    
    return metadata_strategy

@pytest.fixture
def hypothesis_file_path_strategy():
    """
    Hypothesis strategy for generating realistic file paths.
    
    Returns:
        hypothesis.strategies.SearchStrategy: Strategy for file path generation
    """
    filename_components = st.tuples(
        st.text(min_size=3, max_size=20, alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"))),
        st.dates(min_value=datetime(2020, 1, 1).date(), max_value=datetime(2030, 12, 31).date()).map(
            lambda d: d.strftime("%Y%m%d")
        ),
        st.text(min_size=3, max_size=15, alphabet=st.characters(whitelist_categories=("Ll", "Lu"))),
        st.integers(min_value=1, max_value=99).map(str)
    )
    
    file_extension = st.sampled_from([".csv", ".pkl", ".pickle"])
    
    file_path_strategy = filename_components.flatmap(
        lambda components: st.builds(
            lambda name, date, condition, replicate, ext: f"/data/{name}_{date}_{condition}_{replicate}{ext}",
            st.just(components[0]),
            st.just(components[1]), 
            st.just(components[2]),
            st.just(components[3]),
            file_extension
        )
    )
    
    return file_path_strategy


# --- Pickle File Testing Fixtures ---

@pytest.fixture
def sample_pickle_files(cross_platform_temp_dir, comprehensive_exp_matrix):
    """
    Create sample pickle files in various formats for testing.
    
    This fixture creates pickle files in different formats (standard, gzipped, pandas)
    to test the multi-format pickle loading functionality.
    
    Args:
        cross_platform_temp_dir: Cross-platform temporary directory
        comprehensive_exp_matrix: Complete experimental data matrix
        
    Returns:
        Dict[str, Path]: Dictionary mapping format names to file paths
    """
    pickle_dir = cross_platform_temp_dir / "pickle_files"
    pickle_dir.mkdir(exist_ok=True)
    
    pickle_files = {}
    
    # Standard pickle file
    standard_pickle = pickle_dir / "standard_data.pkl"
    with open(standard_pickle, 'wb') as f:
        pickle.dump(comprehensive_exp_matrix, f)
    pickle_files["standard"] = standard_pickle
    
    # Gzipped pickle file
    gzipped_pickle = pickle_dir / "gzipped_data.pkl.gz"
    with gzip.open(gzipped_pickle, 'wb') as f:
        pickle.dump(comprehensive_exp_matrix, f)
    pickle_files["gzipped"] = gzipped_pickle
    
    # Pandas pickle file (using pandas to save)
    pandas_pickle = pickle_dir / "pandas_data.pkl"
    df = pd.DataFrame({
        't': comprehensive_exp_matrix['t'],
        'x': comprehensive_exp_matrix['x'],
        'y': comprehensive_exp_matrix['y']
    })
    df.to_pickle(pandas_pickle)
    pickle_files["pandas"] = pandas_pickle
    
    # Corrupted pickle file for error testing
    corrupted_pickle = pickle_dir / "corrupted_data.pkl"
    with open(corrupted_pickle, 'wb') as f:
        f.write(b"not a pickle file")
    pickle_files["corrupted"] = corrupted_pickle
    
    return pickle_files


@pytest.fixture
def integration_test_scenario(
    comprehensive_sample_config_dict,
    temp_filesystem_structure,
    sample_pickle_files,
    cross_platform_temp_dir
):
    """
    Create a complete integration test scenario.
    
    This fixture sets up a comprehensive test environment that includes:
    - Realistic configuration
    - Structured filesystem with data files
    - Various pickle file formats
    - Complete metadata extraction scenarios
    
    This supports end-to-end integration testing as required by Section 2.1.15.
    
    Returns:
        Dict[str, Any]: Complete integration test scenario data
    """
    scenario = {
        "config": comprehensive_sample_config_dict,
        "filesystem": temp_filesystem_structure,
        "pickle_files": sample_pickle_files,
        "temp_dir": cross_platform_temp_dir,
        "expected_files": {
            "baseline_experiments": [
                temp_filesystem_structure["baseline_file_1"],
                temp_filesystem_structure["baseline_file_2"]
            ],
            "optogenetic_experiments": [
                temp_filesystem_structure["opto_file_1"], 
                temp_filesystem_structure["opto_file_2"]
            ],
            "navigation_experiments": [
                temp_filesystem_structure["nav_file_1"],
                temp_filesystem_structure["nav_file_2"] 
            ]
        },
        "expected_metadata_extractions": {
            str(temp_filesystem_structure["baseline_file_1"]): {
                "dataset": "baseline",
                "date": "20241220",
                "condition": "control",
                "replicate": "1"
            },
            str(temp_filesystem_structure["opto_file_1"]): {
                "dataset": "opto",
                "stimulation_type": "stim", 
                "date": "20241218",
                "condition": "treatment",
                "replicate": "1"
            }
        }
    }
    
    return scenario


# --- Performance Testing Support Fixtures ---

@pytest.fixture
def performance_test_data():
    """
    Generate large-scale test data for performance benchmarking.
    
    This fixture creates datasets of various sizes to support performance
    testing against the SLAs defined in Section 2.1.14.
    
    Returns:
        Dict[str, Dict]: Performance test datasets with different scales
    """
    performance_data = {}
    
    # Small dataset (baseline)
    performance_data["small"] = {
        "size_description": "1 minute at 60Hz",
        "timepoints": 3600,
        "data": {
            't': np.linspace(0, 60, 3600),
            'x': np.random.rand(3600) * 100,
            'y': np.random.rand(3600) * 100,
            'signal_disp': np.random.rand(16, 3600)
        }
    }
    
    # Medium dataset
    performance_data["medium"] = {
        "size_description": "10 minutes at 60Hz", 
        "timepoints": 36000,
        "data": {
            't': np.linspace(0, 600, 36000),
            'x': np.random.rand(36000) * 100,
            'y': np.random.rand(36000) * 100,
            'signal_disp': np.random.rand(16, 36000)
        }
    }
    
    # Large dataset
    performance_data["large"] = {
        "size_description": "1 hour at 60Hz",
        "timepoints": 216000,
        "data": {
            't': np.linspace(0, 3600, 216000),
            'x': np.random.rand(216000) * 100,
            'y': np.random.rand(216000) * 100,
            'signal_disp': np.random.rand(16, 216000)
        }
    }
    
    return performance_data