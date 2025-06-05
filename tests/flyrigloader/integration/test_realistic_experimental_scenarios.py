"""
Realistic experimental scenario integration test suite for flyrigloader.

This module validates flyrigloader functionality with comprehensive synthetic datasets 
that mirror actual neuroscience research workflows. Implements diverse experimental 
conditions including multi-day studies, various rig configurations, complex metadata 
patterns, and realistic data scales.

Tests complete workflows with synthetic but realistic experimental matrices, time series 
data, multi-dimensional signal arrays, and complex directory structures that represent 
actual optical fly rig experimental setups. Validates system behavior under realistic 
data loads, complex filtering scenarios, and ensures robustness across different 
experimental design patterns used in neuroscience research.

Requirements Coverage:
- TST-INTEG-002: Realistic test data generation representing experimental scenarios
- F-015: Realistic experimental data flows validation  
- Section 4.1.2.2: Multi-Experiment Batch Processing workflow validation
- F-007: Realistic metadata extraction pattern validation
- TST-PERF-002: Realistic data scale performance validation
- Section 4.1.2.3: Error recovery validation with realistic failure scenarios
"""

import pytest
import numpy as np
import pandas as pd
import pickle
import gzip
import yaml
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union, Generator
from unittest.mock import patch, MagicMock, Mock

from loguru import logger

# Import the modules under test
import flyrigloader.api as api
from flyrigloader.config.yaml_config import load_config
from flyrigloader.discovery.files import discover_files
from flyrigloader.io.pickle import read_pickle_any_format
from flyrigloader.io.column_models import get_config_from_source
from flyrigloader.utils.dataframe import combine_metadata_and_data


class RealisticExperimentalDataGenerator:
    """
    Advanced generator for creating realistic experimental datasets that mirror 
    actual neuroscience research workflows and data characteristics.
    
    Features:
    - Biologically plausible trajectory patterns
    - Realistic multi-channel neural signals
    - Complex experimental metadata structures
    - Date-based experimental organization
    - Multi-rig configuration scenarios
    """
    
    def __init__(self, random_seed: int = 42):
        """Initialize with reproducible random seed for consistent test data."""
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Experimental design constants
        self.rig_configurations = {
            "old_opto": {
                "sampling_frequency": 60.0,
                "arena_diameter_mm": 120.0,
                "camera_resolution": [1024, 768],
                "signal_channels": 16,
                "mm_per_px": 0.154
            },
            "new_opto": {
                "sampling_frequency": 60.0,
                "arena_diameter_mm": 150.0,
                "camera_resolution": [1280, 1024],
                "signal_channels": 32,
                "mm_per_px": 0.1818
            },
            "high_speed_rig": {
                "sampling_frequency": 200.0,
                "arena_diameter_mm": 200.0,
                "camera_resolution": [2048, 2048],
                "signal_channels": 64,
                "mm_per_px": 0.05
            }
        }
        
        self.experimental_conditions = [
            "baseline", "control", "treatment_a", "treatment_b", 
            "optogenetic_stim", "thermal_stim", "odor_gradient",
            "visual_pattern", "recovery", "sham"
        ]
        
        self.animal_populations = {
            "wild_type": {"strain": "CS", "prefix": "wt"},
            "mutant_line_1": {"strain": "UAS-ChR2", "prefix": "chr2"},
            "mutant_line_2": {"strain": "Gal4-VNC", "prefix": "gal4"},
            "control_line": {"strain": "Berlin-K", "prefix": "bk"}
        }
    
    def generate_realistic_trajectory(
        self,
        duration_seconds: float,
        sampling_freq: float,
        arena_diameter: float,
        behavioral_context: str = "baseline",
        seed_offset: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate biologically plausible fly trajectory with context-dependent behavior.
        
        Args:
            duration_seconds: Duration of the trajectory in seconds
            sampling_freq: Sampling frequency in Hz
            arena_diameter: Arena diameter in mm
            behavioral_context: Type of behavioral experiment context
            seed_offset: Offset for reproducible variation
            
        Returns:
            Tuple of (time, x_position, y_position) arrays
        """
        np.random.seed(self.random_seed + seed_offset)
        
        n_points = int(duration_seconds * sampling_freq)
        dt = 1.0 / sampling_freq
        time = np.arange(n_points) * dt
        
        arena_radius = arena_diameter / 2.0
        
        # Initialize position arrays
        x_pos = np.zeros(n_points)
        y_pos = np.zeros(n_points)
        
        # Behavioral context parameters
        if behavioral_context == "baseline":
            center_bias = 0.3
            movement_noise = 0.8
            velocity_scale = 1.0
        elif behavioral_context == "optogenetic_stim":
            center_bias = 0.1  # More wall-following during stimulation
            movement_noise = 1.2  # More erratic movement
            velocity_scale = 1.5  # Faster movement
        elif behavioral_context == "thermal_stim":
            center_bias = 0.8  # Strong center preference
            movement_noise = 0.5  # Reduced movement
            velocity_scale = 0.6  # Slower movement
        elif behavioral_context == "odor_gradient":
            center_bias = 0.0  # No center bias, following gradient
            movement_noise = 0.3  # Directed movement
            velocity_scale = 1.2
        else:
            center_bias = 0.4
            movement_noise = 0.6
            velocity_scale = 1.0
        
        # Generate correlated random walk with behavioral context
        for i in range(1, n_points):
            current_radius = np.sqrt(x_pos[i-1]**2 + y_pos[i-1]**2)
            
            # Context-dependent forces
            if behavioral_context == "odor_gradient":
                # Simulate gradient following - bias toward upper-right quadrant
                gradient_force_x = 0.2 * (1 - x_pos[i-1] / arena_radius)
                gradient_force_y = 0.2 * (1 - y_pos[i-1] / arena_radius)
            else:
                gradient_force_x = 0
                gradient_force_y = 0
            
            # Center bias force
            bias_strength = center_bias * (current_radius / arena_radius)**2
            center_force_x = -bias_strength * x_pos[i-1] / max(current_radius, 0.1)
            center_force_y = -bias_strength * y_pos[i-1] / max(current_radius, 0.1)
            
            # Random movement component
            random_x = np.random.normal(0, movement_noise) * velocity_scale
            random_y = np.random.normal(0, movement_noise) * velocity_scale
            
            # Total force
            total_force_x = center_force_x + gradient_force_x + random_x
            total_force_y = center_force_y + gradient_force_y + random_y
            
            # Update position with velocity constraints
            max_velocity = 20.0 * velocity_scale  # mm/s
            dx = np.clip(total_force_x * dt, -max_velocity * dt, max_velocity * dt)
            dy = np.clip(total_force_y * dt, -max_velocity * dt, max_velocity * dt)
            
            new_x = x_pos[i-1] + dx
            new_y = y_pos[i-1] + dy
            
            # Enforce arena boundaries with reflection
            new_radius = np.sqrt(new_x**2 + new_y**2)
            if new_radius > arena_radius:
                reflection_factor = arena_radius / new_radius
                new_x *= reflection_factor * 0.95
                new_y *= reflection_factor * 0.95
            
            x_pos[i] = new_x
            y_pos[i] = new_y
        
        return time, x_pos, y_pos
    
    def generate_realistic_neural_signals(
        self,
        n_timepoints: int,
        n_channels: int,
        behavioral_context: str = "baseline",
        rig_type: str = "old_opto",
        seed_offset: int = 0
    ) -> np.ndarray:
        """
        Generate realistic multi-channel neural signal data with context-dependent patterns.
        
        Args:
            n_timepoints: Number of time points
            n_channels: Number of signal channels
            behavioral_context: Experimental context
            rig_type: Type of recording rig
            seed_offset: Seed offset for variation
            
        Returns:
            Array of shape (n_channels, n_timepoints) with signal data
        """
        np.random.seed(self.random_seed + seed_offset)
        
        rig_config = self.rig_configurations[rig_type]
        sampling_freq = rig_config["sampling_frequency"]
        
        signals = np.zeros((n_channels, n_timepoints))
        t = np.linspace(0, n_timepoints / sampling_freq, n_timepoints)
        
        # Context-dependent signal characteristics
        if behavioral_context == "optogenetic_stim":
            base_amplitude = 1.5
            noise_level = 0.2
            stimulation_freq = 10.0  # Hz stimulation
        elif behavioral_context == "thermal_stim":
            base_amplitude = 0.8
            noise_level = 0.1
            stimulation_freq = 0.1  # Very slow thermal changes
        elif behavioral_context == "baseline":
            base_amplitude = 1.0
            noise_level = 0.15
            stimulation_freq = 2.0  # Intrinsic neural oscillations
        else:
            base_amplitude = 1.0
            noise_level = 0.15
            stimulation_freq = 2.0
        
        for ch in range(n_channels):
            # Channel-specific properties
            channel_phase = 2 * np.pi * ch / n_channels
            channel_amplitude = base_amplitude * (0.7 + 0.6 * np.random.random())
            
            # Base signal with harmonics
            base_signal = (
                channel_amplitude * np.sin(2 * np.pi * stimulation_freq * t + channel_phase) +
                0.3 * channel_amplitude * np.sin(4 * np.pi * stimulation_freq * t + channel_phase) +
                0.1 * channel_amplitude * np.sin(6 * np.pi * stimulation_freq * t + channel_phase)
            )
            
            # Context-specific modulation
            if behavioral_context == "optogenetic_stim":
                # Add stimulation epochs
                stim_epochs = np.where(
                    (t % 20 < 5) & (t > 30),  # 5s stim every 20s after 30s baseline
                    2.0, 1.0
                )
                base_signal *= stim_epochs
            
            # Baseline drift
            drift = 0.2 * np.sin(2 * np.pi * 0.01 * t + np.random.random() * 2 * np.pi)
            
            # Noise
            noise = noise_level * np.random.normal(0, 1, n_timepoints)
            
            signals[ch, :] = base_signal + drift + noise
        
        return signals
    
    def generate_experimental_metadata(
        self,
        experiment_type: str,
        date: datetime,
        animal_line: str = "wild_type",
        replicate: int = 1,
        rig_type: str = "old_opto"
    ) -> Dict[str, Any]:
        """
        Generate realistic experimental metadata with proper hierarchical structure.
        
        Args:
            experiment_type: Type of experiment being conducted
            date: Experiment date
            animal_line: Genetic line of the animal
            replicate: Replicate number
            rig_type: Type of rig used
            
        Returns:
            Dictionary containing comprehensive experimental metadata
        """
        animal_info = self.animal_populations[animal_line]
        rig_config = self.rig_configurations[rig_type]
        
        # Generate realistic animal ID
        animal_id = f"{animal_info['prefix']}_{date.strftime('%m%d')}_{replicate:02d}"
        
        # Generate session-specific parameters
        session_duration = {
            "baseline": np.random.randint(300, 900),  # 5-15 minutes
            "optogenetic_stim": np.random.randint(600, 1800),  # 10-30 minutes
            "thermal_stim": np.random.randint(900, 2700),  # 15-45 minutes
            "odor_gradient": np.random.randint(300, 600)  # 5-10 minutes
        }.get(experiment_type, 600)
        
        metadata = {
            # Animal information
            "animal_id": animal_id,
            "strain": animal_info["strain"],
            "genetic_line": animal_line,
            "age_days": np.random.randint(3, 7),
            "sex": np.random.choice(["male", "female"]),
            
            # Experimental information
            "experiment_type": experiment_type,
            "date": date.strftime("%Y%m%d"),
            "time": date.strftime("%H%M%S"),
            "replicate": replicate,
            "session_duration_seconds": session_duration,
            "condition": experiment_type.replace("_", "-"),
            
            # Rig information
            "rig": rig_type,
            "sampling_frequency": rig_config["sampling_frequency"],
            "arena_diameter_mm": rig_config["arena_diameter_mm"],
            "mm_per_px": rig_config["mm_per_px"],
            "signal_channels": rig_config["signal_channels"],
            
            # Environmental conditions
            "temperature_c": np.random.normal(23.0, 1.0),
            "humidity_percent": np.random.normal(50.0, 5.0),
            "light_intensity_lux": np.random.normal(100.0, 10.0),
            
            # Experimenter information
            "experimenter": np.random.choice(["researcher_a", "researcher_b", "researcher_c"]),
            "protocol_version": "v2.1",
            "notes": f"Standard {experiment_type} protocol"
        }
        
        return metadata
    
    def create_experimental_matrix(
        self,
        experiment_type: str,
        date: datetime,
        animal_line: str = "wild_type",
        replicate: int = 1,
        rig_type: str = "old_opto",
        include_signals: bool = True,
        seed_offset: int = 0
    ) -> Dict[str, Any]:
        """
        Create a complete experimental data matrix with metadata.
        
        Args:
            experiment_type: Type of experiment
            date: Experiment date
            animal_line: Genetic line
            replicate: Replicate number
            rig_type: Rig configuration
            include_signals: Whether to include neural signals
            seed_offset: Seed offset for variation
            
        Returns:
            Complete experimental data dictionary
        """
        metadata = self.generate_experimental_metadata(
            experiment_type, date, animal_line, replicate, rig_type
        )
        
        rig_config = self.rig_configurations[rig_type]
        duration = metadata["session_duration_seconds"]
        sampling_freq = rig_config["sampling_frequency"]
        
        # Generate trajectory data
        time, x_pos, y_pos = self.generate_realistic_trajectory(
            duration, sampling_freq, rig_config["arena_diameter_mm"],
            experiment_type, seed_offset
        )
        
        # Create experimental matrix
        exp_matrix = {
            "t": time,
            "x": x_pos,
            "y": y_pos
        }
        
        # Add derived kinematic measures
        dt = np.diff(time, prepend=time[1] - time[0])
        dx = np.diff(x_pos, prepend=0)
        dy = np.diff(y_pos, prepend=0)
        
        exp_matrix["vx"] = dx / dt
        exp_matrix["vy"] = dy / dt
        exp_matrix["speed"] = np.sqrt(exp_matrix["vx"]**2 + exp_matrix["vy"]**2)
        exp_matrix["distance_from_center"] = np.sqrt(x_pos**2 + y_pos**2)
        
        # Add angular measures
        exp_matrix["heading"] = np.arctan2(dy, dx)
        exp_matrix["dtheta"] = np.diff(exp_matrix["heading"], prepend=0)
        
        # Add neural signals if requested
        if include_signals:
            n_timepoints = len(time)
            n_channels = rig_config["signal_channels"]
            
            signals = self.generate_realistic_neural_signals(
                n_timepoints, n_channels, experiment_type, rig_type, seed_offset
            )
            
            if n_channels == 1:
                exp_matrix["signal"] = signals[0, :]
            else:
                exp_matrix["signal_disp"] = signals
                # Also add individual channel access
                for ch in range(min(n_channels, 4)):  # First 4 channels as examples
                    exp_matrix[f"signal_ch{ch:02d}"] = signals[ch, :]
        
        # Add metadata to matrix
        for key, value in metadata.items():
            if not isinstance(value, (np.ndarray, list)):
                exp_matrix[key] = value
        
        return exp_matrix


@pytest.fixture(scope="session")
def realistic_data_generator():
    """Session-scoped fixture providing realistic experimental data generator."""
    return RealisticExperimentalDataGenerator(random_seed=42)


@pytest.fixture(scope="function")
def realistic_experiment_directory(tmp_path, realistic_data_generator):
    """
    Create a realistic experimental directory structure with multiple experiments,
    dates, conditions, and animals representing actual research workflows.
    """
    base_dir = tmp_path / "neuroscience_data"
    base_dir.mkdir()
    
    # Create realistic directory structure
    directories = {
        "experiments": base_dir / "experiments",
        "raw_data": base_dir / "experiments" / "raw_data",
        "processed": base_dir / "experiments" / "processed",
        "configs": base_dir / "configs",
        "batch_definitions": base_dir / "batch_definitions",
        "analysis_results": base_dir / "analysis_results"
    }
    
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Generate experimental data across multiple scenarios
    experiment_scenarios = [
        # Multi-day baseline study
        {"type": "baseline", "dates": ["2024-01-15", "2024-01-16", "2024-01-17"], 
         "animals": ["wild_type", "wild_type", "control_line"], "rigs": ["old_opto"]},
        
        # Optogenetic stimulation series
        {"type": "optogenetic_stim", "dates": ["2024-01-20", "2024-01-21"], 
         "animals": ["mutant_line_1", "mutant_line_1"], "rigs": ["new_opto"]},
        
        # Thermal stimulation study
        {"type": "thermal_stim", "dates": ["2024-01-25"], 
         "animals": ["wild_type", "mutant_line_2"], "rigs": ["old_opto", "new_opto"]},
        
        # High-resolution tracking
        {"type": "baseline", "dates": ["2024-02-01"], 
         "animals": ["wild_type"], "rigs": ["high_speed_rig"]},
        
        # Complex multi-rig comparison
        {"type": "odor_gradient", "dates": ["2024-02-05", "2024-02-06"], 
         "animals": ["wild_type", "mutant_line_1"], "rigs": ["old_opto", "new_opto"]}
    ]
    
    created_files = []
    experiment_metadata = []
    
    for scenario in experiment_scenarios:
        for date_str in scenario["dates"]:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            
            # Create date-based directory structure
            date_dir = directories["raw_data"] / date_str
            date_dir.mkdir(exist_ok=True)
            
            for animal_line in scenario["animals"]:
                for rig in scenario["rigs"]:
                    for replicate in range(1, 4):  # 3 replicates per condition
                        # Generate experimental data
                        exp_matrix = realistic_data_generator.create_experimental_matrix(
                            experiment_type=scenario["type"],
                            date=date_obj,
                            animal_line=animal_line,
                            replicate=replicate,
                            rig_type=rig,
                            seed_offset=hash(f"{date_str}_{animal_line}_{rig}_{replicate}") % 1000
                        )
                        
                        # Create realistic filename
                        animal_id = exp_matrix["animal_id"]
                        filename = f"{scenario['type']}_{animal_id}_{rig}_{date_str}_rep{replicate:02d}.pkl"
                        file_path = date_dir / filename
                        
                        # Save as pickle file
                        with open(file_path, 'wb') as f:
                            pickle.dump(exp_matrix, f)
                        
                        created_files.append(file_path)
                        experiment_metadata.append({
                            "file_path": file_path,
                            "experiment_type": scenario["type"],
                            "date": date_str,
                            "animal_line": animal_line,
                            "rig": rig,
                            "replicate": replicate,
                            "animal_id": animal_id,
                            "filename": filename
                        })
    
    # Create some corrupted files for error testing
    corrupted_dir = directories["raw_data"] / "corrupted"
    corrupted_dir.mkdir(exist_ok=True)
    
    corrupted_files = []
    
    # Empty file
    empty_file = corrupted_dir / "empty_experiment.pkl"
    empty_file.touch()
    corrupted_files.append(empty_file)
    
    # Invalid pickle data
    invalid_pickle = corrupted_dir / "invalid_data.pkl"
    with open(invalid_pickle, 'wb') as f:
        f.write(b"not a pickle file")
    corrupted_files.append(invalid_pickle)
    
    # Incomplete data structure
    incomplete_data = {"t": np.array([0, 1, 2]), "x": np.array([0, 1])}  # Mismatched lengths
    incomplete_file = corrupted_dir / "incomplete_experiment.pkl"
    with open(incomplete_file, 'wb') as f:
        pickle.dump(incomplete_data, f)
    corrupted_files.append(incomplete_file)
    
    # Create comprehensive configuration file
    config_data = {
        "project": {
            "name": "realistic_neuroscience_study",
            "directories": {
                "major_data_directory": str(directories["raw_data"]),
                "processed_data_directory": str(directories["processed"]),
                "analysis_results_directory": str(directories["analysis_results"])
            },
            "ignore_substrings": ["backup", "temp", "corrupted", "._", "__pycache__"],
            "mandatory_substrings": [],
            "extraction_patterns": [
                r"(?P<experiment_type>\w+)_(?P<animal_id>\w+)_(?P<rig>\w+)_(?P<date>\d{4}-\d{2}-\d{2})_rep(?P<replicate>\d+)\.pkl",
                r"(?P<experiment_type>\w+)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<animal_id>\w+)_(?P<condition>\w+)\.pkl"
            ]
        },
        "rigs": realistic_data_generator.rig_configurations,
        "datasets": {
            "baseline_studies": {
                "rig": "old_opto",
                "patterns": ["*baseline*"],
                "dates_vials": {
                    "2024-01-15": [1, 2, 3],
                    "2024-01-16": [1, 2, 3],
                    "2024-01-17": [1, 2, 3]
                },
                "metadata": {
                    "extraction_patterns": [
                        r"baseline_(?P<animal_id>\w+)_(?P<rig>\w+)_(?P<date>\d{4}-\d{2}-\d{2})_rep(?P<replicate>\d+)\.pkl"
                    ],
                    "required_fields": ["animal_id", "rig", "date", "replicate"]
                }
            },
            "optogenetic_experiments": {
                "rig": "new_opto",
                "patterns": ["*optogenetic*"],
                "dates_vials": {
                    "2024-01-20": [1, 2, 3],
                    "2024-01-21": [1, 2, 3]
                },
                "metadata": {
                    "extraction_patterns": [
                        r"optogenetic_stim_(?P<animal_id>\w+)_(?P<rig>\w+)_(?P<date>\d{4}-\d{2}-\d{2})_rep(?P<replicate>\d+)\.pkl"
                    ],
                    "required_fields": ["animal_id", "rig", "date", "replicate"]
                }
            },
            "multi_rig_comparison": {
                "rig": ["old_opto", "new_opto", "high_speed_rig"],
                "patterns": ["*thermal*", "*odor*", "*baseline*"],
                "metadata": {
                    "extraction_patterns": [
                        r"(?P<experiment_type>\w+)_(?P<animal_id>\w+)_(?P<rig>\w+)_(?P<date>\d{4}-\d{2}-\d{2})_rep(?P<replicate>\d+)\.pkl"
                    ],
                    "required_fields": ["experiment_type", "animal_id", "rig", "date", "replicate"]
                }
            }
        },
        "experiments": {
            "longitudinal_baseline": {
                "datasets": ["baseline_studies"],
                "metadata": {
                    "study_type": "longitudinal",
                    "duration_days": 3
                }
            },
            "optogenetic_manipulation": {
                "datasets": ["optogenetic_experiments"],
                "metadata": {
                    "study_type": "intervention",
                    "stimulation_protocol": "10Hz_5s_on_15s_off"
                }
            },
            "cross_rig_validation": {
                "datasets": ["multi_rig_comparison"],
                "metadata": {
                    "study_type": "validation",
                    "comparison_type": "equipment"
                }
            }
        }
    }
    
    config_file = directories["configs"] / "realistic_experiment_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False)
    
    return {
        "base_directory": base_dir,
        "directories": directories,
        "config_file": config_file,
        "config_data": config_data,
        "created_files": created_files,
        "corrupted_files": corrupted_files,
        "experiment_metadata": experiment_metadata,
        "total_experiments": len(experiment_metadata)
    }


class TestRealisticSingleExperimentScenarios:
    """
    Test realistic single experiment scenarios with various experimental contexts
    and comprehensive validation of data loading and processing workflows.
    """
    
    def test_baseline_experiment_complete_workflow(self, realistic_experiment_directory):
        """
        Test complete workflow for a single baseline experiment including
        configuration loading, file discovery, data loading, and validation.
        
        Validates:
        - TST-INTEG-002: Realistic test data generation
        - F-015: Complete experimental data flows
        - Section 4.1.1.1: End-to-end user journey
        """
        config_file = realistic_experiment_directory["config_file"]
        
        # Test configuration loading
        config = load_config(config_file)
        assert config is not None
        assert "experiments" in config
        assert "datasets" in config
        assert "project" in config
        
        # Test experiment file discovery
        experiment_files = api.load_experiment_files(
            config_path=config_file,
            experiment_name="longitudinal_baseline",
            extract_metadata=True
        )
        
        assert isinstance(experiment_files, dict), "Should return metadata dictionary"
        assert len(experiment_files) > 0, "Should find baseline experiment files"
        
        # Validate metadata extraction
        for file_path, metadata in experiment_files.items():
            assert "animal_id" in metadata
            assert "rig" in metadata
            assert "date" in metadata
            assert "replicate" in metadata
            assert Path(file_path).suffix == ".pkl"
        
        # Test data loading for one file
        first_file = list(experiment_files.keys())[0]
        exp_data = read_pickle_any_format(first_file)
        
        # Validate experimental data structure
        assert isinstance(exp_data, dict)
        assert "t" in exp_data
        assert "x" in exp_data
        assert "y" in exp_data
        assert len(exp_data["t"]) == len(exp_data["x"]) == len(exp_data["y"])
        
        # Validate data quality
        assert np.all(np.isfinite(exp_data["t"]))
        assert np.all(np.isfinite(exp_data["x"]))
        assert np.all(np.isfinite(exp_data["y"]))
        
        logger.info(f"Successfully validated baseline experiment workflow with {len(experiment_files)} files")
    
    def test_optogenetic_experiment_signal_processing(self, realistic_experiment_directory):
        """
        Test optogenetic stimulation experiment with multi-channel signal processing.
        
        Validates:
        - F-007: Complex metadata extraction with stimulation parameters
        - TST-PERF-002: Signal processing performance with multi-channel data
        """
        config_file = realistic_experiment_directory["config_file"]
        
        # Load optogenetic experiment files
        experiment_files = api.load_experiment_files(
            config_path=config_file,
            experiment_name="optogenetic_manipulation",
            extract_metadata=True
        )
        
        assert len(experiment_files) > 0
        
        # Test signal processing for optogenetic experiments
        for file_path, metadata in experiment_files.items():
            if "optogenetic" in str(file_path):
                exp_data = read_pickle_any_format(file_path)
                
                # Validate multi-channel signal data
                if "signal_disp" in exp_data:
                    signal_data = exp_data["signal_disp"]
                    assert signal_data.ndim == 2, "Multi-channel signals should be 2D"
                    assert signal_data.shape[1] == len(exp_data["t"])
                    
                    # Validate signal characteristics
                    assert np.all(np.isfinite(signal_data))
                    assert signal_data.std() > 0, "Signals should have variation"
                
                # Validate experimental metadata
                assert "experiment_type" in exp_data
                assert exp_data["experiment_type"] == "optogenetic_stim"
                assert "rig" in exp_data
                assert "sampling_frequency" in exp_data
                
                break
        
        logger.info("Successfully validated optogenetic experiment signal processing")
    
    def test_high_speed_tracking_performance(self, realistic_experiment_directory):
        """
        Test high-speed tracking experiments with performance validation.
        
        Validates:
        - TST-PERF-002: Performance with high-frequency data (200 Hz)
        - Complex trajectory analysis with high temporal resolution
        """
        config_file = realistic_experiment_directory["config_file"]
        
        # Find high-speed rig experiments
        all_files = realistic_experiment_directory["created_files"]
        high_speed_files = [f for f in all_files if "high_speed_rig" in str(f)]
        
        if not high_speed_files:
            pytest.skip("No high-speed rig experiments found")
        
        # Test performance with high-frequency data
        start_time = time.time()
        
        for file_path in high_speed_files:
            exp_data = read_pickle_any_format(file_path)
            
            # Validate high-frequency characteristics
            sampling_freq = exp_data.get("sampling_frequency", 60)
            assert sampling_freq >= 200, f"Expected high sampling frequency, got {sampling_freq}"
            
            # Validate data density
            duration = exp_data["t"][-1] - exp_data["t"][0]
            expected_points = int(duration * sampling_freq)
            actual_points = len(exp_data["t"])
            assert abs(actual_points - expected_points) < sampling_freq, "Data density should match sampling frequency"
            
            # Validate temporal resolution
            dt = np.median(np.diff(exp_data["t"]))
            expected_dt = 1.0 / sampling_freq
            assert abs(dt - expected_dt) < expected_dt * 0.1, "Temporal resolution should be consistent"
        
        load_time = time.time() - start_time
        
        # Performance assertion - should load high-speed data efficiently
        data_size_mb = sum(f.stat().st_size for f in high_speed_files) / (1024 * 1024)
        max_load_time = data_size_mb * 1.0  # 1 second per MB SLA
        assert load_time <= max_load_time, f"Loading took {load_time:.2f}s, expected <= {max_load_time:.2f}s"
        
        logger.info(f"Successfully validated high-speed tracking performance: {data_size_mb:.1f}MB in {load_time:.2f}s")


class TestRealisticMultiDayStudyScenarios:
    """
    Test realistic multi-day experimental study scenarios with temporal organization
    and longitudinal data analysis validation.
    """
    
    def test_longitudinal_baseline_study(self, realistic_experiment_directory):
        """
        Test multi-day longitudinal baseline study with date-based organization.
        
        Validates:
        - Section 4.1.2.2: Multi-experiment batch processing
        - F-002-RQ-005: Date-based directory resolution
        - Temporal data organization and aggregation
        """
        config_file = realistic_experiment_directory["config_file"]
        
        # Load all baseline experiments
        baseline_files = api.load_experiment_files(
            config_path=config_file,
            experiment_name="longitudinal_baseline",
            extract_metadata=True
        )
        
        # Group by date for longitudinal analysis
        date_groups = {}
        for file_path, metadata in baseline_files.items():
            date = metadata["date"]
            if date not in date_groups:
                date_groups[date] = []
            date_groups[date].append((file_path, metadata))
        
        # Validate multi-day organization
        assert len(date_groups) >= 3, "Should have multiple experimental days"
        
        # Validate date-based file discovery
        expected_dates = ["2024-01-15", "2024-01-16", "2024-01-17"]
        for expected_date in expected_dates:
            if expected_date in date_groups:
                day_files = date_groups[expected_date]
                assert len(day_files) > 0, f"Should have files for {expected_date}"
                
                # Validate consistency within day
                for file_path, metadata in day_files:
                    assert metadata["date"] == expected_date
                    assert "baseline" in str(file_path).lower()
        
        # Test batch processing performance
        start_time = time.time()
        all_data = []
        
        for file_path, metadata in baseline_files.items():
            exp_data = read_pickle_any_format(file_path)
            all_data.append({
                "data": exp_data,
                "metadata": metadata,
                "file_path": file_path
            })
        
        batch_time = time.time() - start_time
        
        # Performance validation
        total_files = len(baseline_files)
        max_batch_time = total_files * 0.5  # 0.5 seconds per file SLA
        assert batch_time <= max_batch_time, f"Batch processing took {batch_time:.2f}s, expected <= {max_batch_time:.2f}s"
        
        logger.info(f"Successfully validated longitudinal study: {total_files} files across {len(date_groups)} days in {batch_time:.2f}s")
    
    def test_cross_day_data_consistency(self, realistic_experiment_directory):
        """
        Test data consistency and quality across multiple experimental days.
        
        Validates:
        - Data quality consistency across time
        - Metadata consistency in longitudinal studies
        - Animal tracking across multiple sessions
        """
        config_file = realistic_experiment_directory["config_file"]
        
        # Load baseline study data
        baseline_files = api.load_experiment_files(
            config_path=config_file,
            experiment_name="longitudinal_baseline",
            extract_metadata=True
        )
        
        # Group by animal for consistency checking
        animal_sessions = {}
        for file_path, metadata in baseline_files.items():
            animal_id = metadata["animal_id"]
            if animal_id not in animal_sessions:
                animal_sessions[animal_id] = []
            
            exp_data = read_pickle_any_format(file_path)
            animal_sessions[animal_id].append({
                "date": metadata["date"],
                "data": exp_data,
                "metadata": metadata
            })
        
        # Validate cross-session consistency for each animal
        for animal_id, sessions in animal_sessions.items():
            if len(sessions) > 1:
                # Sort by date
                sessions.sort(key=lambda x: x["date"])
                
                # Check data structure consistency
                base_keys = set(sessions[0]["data"].keys())
                for session in sessions[1:]:
                    session_keys = set(session["data"].keys())
                    common_keys = base_keys.intersection(session_keys)
                    assert len(common_keys) >= 3, f"Sessions should have common data keys for {animal_id}"
                
                # Check metadata consistency
                base_rig = sessions[0]["metadata"]["rig"]
                for session in sessions[1:]:
                    # Rig should be consistent within animal
                    assert session["metadata"]["rig"] == base_rig, f"Rig should be consistent for {animal_id}"
                
                # Check data quality trends
                durations = [len(session["data"]["t"]) for session in sessions]
                # Durations should be reasonable (not dramatically different)
                duration_cv = np.std(durations) / np.mean(durations)
                assert duration_cv < 0.5, f"Session durations too variable for {animal_id}: CV={duration_cv:.2f}"
        
        logger.info(f"Successfully validated cross-day consistency for {len(animal_sessions)} animals")
    
    def test_temporal_batch_processing_workflow(self, realistic_experiment_directory):
        """
        Test temporal batch processing with date-based filtering and aggregation.
        
        Validates:
        - Section 4.1.2.2: Multi-experiment batch processing workflow
        - Date range filtering and selection
        - Aggregate statistics across time periods
        """
        config_file = realistic_experiment_directory["config_file"]
        
        # Test date-based filtering using patterns
        config = load_config(config_file)
        base_dir = config["project"]["directories"]["major_data_directory"]
        
        # Discover files with date-based patterns
        from flyrigloader.discovery.files import discover_files
        
        # Test specific date filtering
        date_pattern = "2024-01-15"
        files_2024_01_15 = discover_files(
            directory=base_dir,
            pattern=f"*{date_pattern}*",
            recursive=True,
            extensions=[".pkl"]
        )
        
        assert len(files_2024_01_15) > 0, "Should find files for specific date"
        
        # Validate all files match date pattern
        for file_path in files_2024_01_15:
            assert date_pattern in str(file_path), f"File {file_path} should contain date pattern"
        
        # Test date range batch processing
        date_range_files = []
        for date in ["2024-01-15", "2024-01-16", "2024-01-17"]:
            daily_files = discover_files(
                directory=base_dir,
                pattern=f"*{date}*",
                recursive=True,
                extensions=[".pkl"]
            )
            date_range_files.extend(daily_files)
        
        # Process batch with performance monitoring
        start_time = time.time()
        batch_results = []
        
        for file_path in date_range_files:
            try:
                exp_data = read_pickle_any_format(file_path)
                
                # Extract summary statistics
                summary = {
                    "file_path": str(file_path),
                    "duration_seconds": exp_data["t"][-1] - exp_data["t"][0] if len(exp_data["t"]) > 0 else 0,
                    "total_distance": np.sum(np.sqrt(np.diff(exp_data["x"])**2 + np.diff(exp_data["y"])**2)),
                    "mean_speed": np.mean(exp_data.get("speed", [0])),
                    "data_points": len(exp_data["t"])
                }
                batch_results.append(summary)
                
            except Exception as e:
                logger.warning(f"Failed to process {file_path}: {e}")
        
        batch_time = time.time() - start_time
        
        # Validate batch processing results
        assert len(batch_results) > 0, "Should successfully process some files"
        assert len(batch_results) >= len(date_range_files) * 0.8, "Should process at least 80% of files successfully"
        
        # Performance validation
        files_per_second = len(batch_results) / batch_time
        assert files_per_second >= 2.0, f"Should process at least 2 files/second, got {files_per_second:.2f}"
        
        logger.info(f"Successfully processed {len(batch_results)} files in batch in {batch_time:.2f}s ({files_per_second:.1f} files/s)")


class TestRealisticComplexMetadataScenarios:
    """
    Test realistic complex metadata extraction scenarios with diverse filename
    patterns and experimental hierarchies.
    """
    
    def test_complex_filename_pattern_extraction(self, realistic_experiment_directory):
        """
        Test complex metadata extraction from realistic experimental filename patterns.
        
        Validates:
        - F-007: Realistic metadata extraction pattern validation
        - Complex regex pattern matching with named groups
        - Multi-pattern extraction scenarios
        """
        config_file = realistic_experiment_directory["config_file"]
        
        # Load configuration with extraction patterns
        config = load_config(config_file)
        patterns = config["project"]["extraction_patterns"]
        
        # Test pattern matching on actual files
        all_files = realistic_experiment_directory["created_files"]
        extraction_results = {}
        
        for file_path in all_files:
            filename = file_path.name
            extracted_metadata = {}
            
            # Try each pattern
            for pattern in patterns:
                try:
                    match = re.match(pattern, filename)
                    if match:
                        extracted_metadata.update(match.groupdict())
                        break
                except Exception as e:
                    logger.debug(f"Pattern {pattern} failed on {filename}: {e}")
            
            if extracted_metadata:
                extraction_results[str(file_path)] = extracted_metadata
        
        # Validate extraction success rate
        total_files = len([f for f in all_files if f.suffix == ".pkl" and "corrupted" not in str(f)])
        extraction_rate = len(extraction_results) / total_files
        assert extraction_rate >= 0.8, f"Should extract metadata from at least 80% of files, got {extraction_rate:.2%}"
        
        # Validate extracted metadata quality
        required_fields = ["experiment_type", "animal_id", "rig", "date", "replicate"]
        for file_path, metadata in extraction_results.items():
            found_fields = [field for field in required_fields if field in metadata]
            assert len(found_fields) >= 4, f"Should extract at least 4 required fields from {file_path}, got {found_fields}"
            
            # Validate field formats
            if "date" in metadata:
                date_str = metadata["date"]
                assert re.match(r"\d{4}-\d{2}-\d{2}", date_str), f"Date should be in YYYY-MM-DD format: {date_str}"
            
            if "replicate" in metadata:
                rep_str = metadata["replicate"]
                assert rep_str.isdigit(), f"Replicate should be numeric: {rep_str}"
        
        logger.info(f"Successfully extracted metadata from {len(extraction_results)} files ({extraction_rate:.1%} success rate)")
    
    def test_hierarchical_experiment_organization(self, realistic_experiment_directory):
        """
        Test hierarchical experimental organization with nested datasets and experiments.
        
        Validates:
        - Complex experiment-dataset relationships
        - Hierarchical metadata inheritance
        - Multi-level filtering and organization
        """
        config_file = realistic_experiment_directory["config_file"]
        config = load_config(config_file)
        
        # Test hierarchical experiment structure
        experiments = config["experiments"]
        datasets = config["datasets"]
        
        # Validate experiment-dataset relationships
        for exp_name, exp_config in experiments.items():
            assert "datasets" in exp_config, f"Experiment {exp_name} should specify datasets"
            
            exp_datasets = exp_config["datasets"]
            for dataset_name in exp_datasets:
                assert dataset_name in datasets, f"Dataset {dataset_name} should exist in config"
        
        # Test cross-rig validation experiment
        cross_rig_files = api.load_experiment_files(
            config_path=config_file,
            experiment_name="cross_rig_validation",
            extract_metadata=True
        )
        
        # Group by rig for comparison
        rig_groups = {}
        for file_path, metadata in cross_rig_files.items():
            rig = metadata["rig"]
            if rig not in rig_groups:
                rig_groups[rig] = []
            rig_groups[rig].append((file_path, metadata))
        
        # Validate multi-rig data availability
        expected_rigs = ["old_opto", "new_opto"]
        found_rigs = list(rig_groups.keys())
        common_rigs = set(expected_rigs).intersection(set(found_rigs))
        assert len(common_rigs) >= 1, f"Should have data from multiple rigs, found: {found_rigs}"
        
        # Test rig-specific characteristics
        for rig, files in rig_groups.items():
            rig_config = config["rigs"][rig]
            
            for file_path, metadata in files[:3]:  # Test first 3 files per rig
                exp_data = read_pickle_any_format(file_path)
                
                # Validate rig-specific parameters
                if "sampling_frequency" in exp_data:
                    assert exp_data["sampling_frequency"] == rig_config["sampling_frequency"]
                
                if "arena_diameter_mm" in exp_data:
                    assert exp_data["arena_diameter_mm"] == rig_config["arena_diameter_mm"]
                
                # Validate signal channel counts
                if "signal_disp" in exp_data:
                    expected_channels = rig_config["signal_channels"]
                    actual_channels = exp_data["signal_disp"].shape[0]
                    assert actual_channels == expected_channels, f"Expected {expected_channels} channels for {rig}, got {actual_channels}"
        
        logger.info(f"Successfully validated hierarchical organization across {len(rig_groups)} rigs")
    
    def test_animal_identifier_consistency(self, realistic_experiment_directory):
        """
        Test animal identifier consistency and tracking across experiments.
        
        Validates:
        - Animal ID generation and consistency
        - Genetic line tracking
        - Cross-experiment animal identification
        """
        config_file = realistic_experiment_directory["config_file"]
        
        # Collect all animal data across experiments
        all_experiments = ["longitudinal_baseline", "optogenetic_manipulation", "cross_rig_validation"]
        animal_database = {}
        
        for exp_name in all_experiments:
            try:
                exp_files = api.load_experiment_files(
                    config_path=config_file,
                    experiment_name=exp_name,
                    extract_metadata=True
                )
                
                for file_path, metadata in exp_files.items():
                    animal_id = metadata["animal_id"]
                    
                    if animal_id not in animal_database:
                        animal_database[animal_id] = {
                            "experiments": [],
                            "genetic_line": None,
                            "strain": None,
                            "sessions": []
                        }
                    
                    # Load actual data for genetic information
                    exp_data = read_pickle_any_format(file_path)
                    
                    animal_info = animal_database[animal_id]
                    animal_info["experiments"].append(exp_name)
                    animal_info["sessions"].append({
                        "file_path": file_path,
                        "date": metadata["date"],
                        "experiment": exp_name
                    })
                    
                    # Extract genetic information
                    if "genetic_line" in exp_data:
                        if animal_info["genetic_line"] is None:
                            animal_info["genetic_line"] = exp_data["genetic_line"]
                        else:
                            # Consistency check
                            assert animal_info["genetic_line"] == exp_data["genetic_line"], \
                                f"Genetic line inconsistency for {animal_id}"
                    
                    if "strain" in exp_data:
                        if animal_info["strain"] is None:
                            animal_info["strain"] = exp_data["strain"]
                        else:
                            # Consistency check
                            assert animal_info["strain"] == exp_data["strain"], \
                                f"Strain inconsistency for {animal_id}"
            
            except KeyError:
                # Experiment might not exist in config, skip
                continue
        
        # Validate animal database
        assert len(animal_database) > 0, "Should identify multiple animals"
        
        # Validate animal ID patterns
        for animal_id, animal_info in animal_database.items():
            # Animal IDs should follow consistent pattern
            assert "_" in animal_id, f"Animal ID should contain underscores: {animal_id}"
            
            # Animals should have multiple sessions for longitudinal studies
            sessions = animal_info["sessions"]
            if len(sessions) > 1:
                # Sort by date
                sessions.sort(key=lambda x: x["date"])
                
                # Validate temporal consistency
                dates = [s["date"] for s in sessions]
                assert len(set(dates)) >= 1, f"Animal {animal_id} should have sessions across time"
        
        # Generate summary statistics
        multi_session_animals = [aid for aid, info in animal_database.items() if len(info["sessions"]) > 1]
        genetic_lines = set(info["genetic_line"] for info in animal_database.values() if info["genetic_line"])
        
        logger.info(f"Successfully validated {len(animal_database)} animals, {len(multi_session_animals)} with multiple sessions, {len(genetic_lines)} genetic lines")


class TestRealisticDataScalePerformance:
    """
    Test realistic data scale performance validation scenarios with large
    experimental datasets and complex processing workflows.
    """
    
    def test_large_dataset_batch_processing(self, realistic_experiment_directory, performance_benchmarks):
        """
        Test performance with large-scale experimental datasets.
        
        Validates:
        - TST-PERF-002: Realistic data scale performance validation
        - Batch processing of multiple large experiments
        - Memory efficiency with large signal arrays
        """
        config_file = realistic_experiment_directory["config_file"]
        
        # Load all available experiments for performance testing
        all_files = realistic_experiment_directory["created_files"]
        
        # Measure batch loading performance
        start_time = time.time()
        total_data_size = 0
        processed_files = 0
        memory_efficient_processing = True
        
        for file_path in all_files:
            try:
                file_size = file_path.stat().st_size
                total_data_size += file_size
                
                # Load and process data
                exp_data = read_pickle_any_format(file_path)
                
                # Validate data structure and compute basic statistics
                assert "t" in exp_data and "x" in exp_data and "y" in exp_data
                
                # Memory-efficient processing check
                data_points = len(exp_data["t"])
                if data_points > 10000:  # Large dataset
                    # Should have multi-channel signals for large datasets
                    if "signal_disp" in exp_data:
                        signal_array = exp_data["signal_disp"]
                        # Memory footprint check
                        expected_size = signal_array.nbytes
                        if expected_size > 50 * 1024 * 1024:  # > 50MB
                            # Should use appropriate data types
                            assert signal_array.dtype in [np.float32, np.float64]
                
                processed_files += 1
                
            except Exception as e:
                logger.warning(f"Failed to process {file_path}: {e}")
        
        total_time = time.time() - start_time
        
        # Performance validation
        data_size_mb = total_data_size / (1024 * 1024)
        performance_benchmarks.assert_performance_sla(
            "large_dataset_batch_processing",
            total_time,
            performance_benchmarks.benchmark_data_loading(data_size_mb)
        )
        
        # Throughput validation
        files_per_second = processed_files / total_time
        assert files_per_second >= 1.0, f"Should process at least 1 file/second, got {files_per_second:.2f}"
        
        logger.info(f"Successfully processed {processed_files} files ({data_size_mb:.1f}MB) in {total_time:.2f}s")
    
    def test_high_frequency_signal_processing(self, realistic_experiment_directory, performance_benchmarks):
        """
        Test performance with high-frequency multi-channel signal data.
        
        Validates:
        - High-frequency signal processing performance
        - Multi-channel data handling efficiency
        - Signal array transformation benchmarks
        """
        config_file = realistic_experiment_directory["config_file"]
        
        # Find files with high-frequency signal data
        all_files = realistic_experiment_directory["created_files"]
        signal_files = []
        
        for file_path in all_files:
            try:
                exp_data = read_pickle_any_format(file_path)
                if "signal_disp" in exp_data and exp_data["signal_disp"].shape[0] >= 16:
                    signal_files.append((file_path, exp_data))
                    if len(signal_files) >= 5:  # Test with 5 files
                        break
            except:
                continue
        
        if not signal_files:
            pytest.skip("No multi-channel signal files found for performance testing")
        
        # Test signal processing performance
        start_time = time.time()
        total_signal_points = 0
        
        for file_path, exp_data in signal_files:
            signal_data = exp_data["signal_disp"]
            n_channels, n_timepoints = signal_data.shape
            total_signal_points += n_channels * n_timepoints
            
            # Simulate realistic signal processing operations
            # 1. Channel-wise statistics
            channel_means = np.mean(signal_data, axis=1)
            channel_stds = np.std(signal_data, axis=1)
            
            # 2. Temporal filtering (simplified)
            filtered_signals = np.zeros_like(signal_data)
            for ch in range(n_channels):
                # Simple smoothing filter
                kernel_size = min(5, n_timepoints // 10)
                if kernel_size > 1:
                    kernel = np.ones(kernel_size) / kernel_size
                    filtered_signals[ch, :] = np.convolve(signal_data[ch, :], kernel, mode='same')
                else:
                    filtered_signals[ch, :] = signal_data[ch, :]
            
            # 3. Cross-channel correlation (subset)
            if n_channels >= 4:
                sample_channels = signal_data[:4, :]  # First 4 channels
                correlation_matrix = np.corrcoef(sample_channels)
            
            # Validate processing results
            assert np.all(np.isfinite(channel_means))
            assert np.all(np.isfinite(channel_stds))
            assert np.all(np.isfinite(filtered_signals))
        
        processing_time = time.time() - start_time
        
        # Performance validation
        points_per_second = total_signal_points / processing_time
        min_throughput = 1_000_000  # 1M points per second minimum
        assert points_per_second >= min_throughput, \
            f"Signal processing should handle at least {min_throughput:,} points/s, got {points_per_second:,.0f}"
        
        # SLA validation
        estimated_mb = (total_signal_points * 8) / (1024 * 1024)  # 8 bytes per float64
        performance_benchmarks.assert_performance_sla(
            "signal_processing",
            processing_time,
            performance_benchmarks.benchmark_dataframe_transform(total_signal_points)
        )
        
        logger.info(f"Successfully processed {total_signal_points:,} signal points in {processing_time:.2f}s ({points_per_second:,.0f} points/s)")
    
    def test_memory_efficient_large_experiment_loading(self, realistic_experiment_directory):
        """
        Test memory-efficient loading of large experimental datasets.
        
        Validates:
        - Memory usage optimization for large datasets
        - Streaming/incremental loading capabilities
        - Resource cleanup and management
        """
        config_file = realistic_experiment_directory["config_file"]
        
        # Find largest experiment files
        all_files = realistic_experiment_directory["created_files"]
        file_sizes = [(f, f.stat().st_size) for f in all_files]
        file_sizes.sort(key=lambda x: x[1], reverse=True)
        
        largest_files = file_sizes[:3]  # Test with 3 largest files
        
        initial_memory = None
        try:
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        except ImportError:
            logger.warning("psutil not available, skipping memory monitoring")
        
        # Test memory-efficient loading
        for file_path, file_size in largest_files:
            # Load data
            exp_data = read_pickle_any_format(file_path)
            
            # Validate data completeness
            required_keys = ["t", "x", "y"]
            for key in required_keys:
                assert key in exp_data, f"Missing required key: {key}"
                assert len(exp_data[key]) > 0, f"Empty data for key: {key}"
            
            # Memory usage check
            if initial_memory is not None:
                current_memory = process.memory_info().rss / (1024 * 1024)
                memory_increase = current_memory - initial_memory
                file_size_mb = file_size / (1024 * 1024)
                
                # Memory increase should be reasonable (not more than 3x file size)
                max_memory_increase = file_size_mb * 3
                assert memory_increase <= max_memory_increase, \
                    f"Memory increase {memory_increase:.1f}MB exceeds limit {max_memory_increase:.1f}MB for file {file_size_mb:.1f}MB"
            
            # Clean up references to help GC
            del exp_data
        
        logger.info(f"Successfully validated memory-efficient loading of {len(largest_files)} large files")


class TestRealisticErrorRecoveryScenarios:
    """
    Test realistic error recovery and robustness scenarios with various
    failure modes and recovery mechanisms.
    """
    
    def test_corrupted_file_handling(self, realistic_experiment_directory):
        """
        Test robust handling of corrupted or invalid experimental files.
        
        Validates:
        - Section 4.1.2.3: Error recovery validation with realistic failure scenarios
        - Graceful degradation with partial data loss
        - Comprehensive error reporting and logging
        """
        config_file = realistic_experiment_directory["config_file"]
        corrupted_files = realistic_experiment_directory["corrupted_files"]
        
        # Test individual corrupted file handling
        for corrupted_file in corrupted_files:
            logger.info(f"Testing corrupted file handling: {corrupted_file.name}")
            
            with pytest.raises((pickle.UnpicklingError, EOFError, FileNotFoundError, ValueError)):
                read_pickle_any_format(corrupted_file)
        
        # Test batch processing with mixed valid/corrupted files
        all_files = realistic_experiment_directory["created_files"] + corrupted_files
        
        successful_loads = 0
        failed_loads = 0
        error_summary = {}
        
        for file_path in all_files:
            try:
                exp_data = read_pickle_any_format(file_path)
                # Validate basic structure
                if isinstance(exp_data, dict) and "t" in exp_data:
                    successful_loads += 1
                else:
                    failed_loads += 1
                    error_type = "invalid_structure"
                    error_summary[error_type] = error_summary.get(error_type, 0) + 1
            
            except Exception as e:
                failed_loads += 1
                error_type = type(e).__name__
                error_summary[error_type] = error_summary.get(error_type, 0) + 1
        
        # Validate error recovery behavior
        total_files = len(all_files)
        success_rate = successful_loads / total_files
        
        # Should successfully process most valid files
        expected_corrupted = len(corrupted_files)
        expected_success_rate = (total_files - expected_corrupted) / total_files
        assert success_rate >= expected_success_rate * 0.9, \
            f"Success rate {success_rate:.2%} below expected {expected_success_rate:.2%}"
        
        logger.info(f"Error recovery test: {successful_loads}/{total_files} successful loads, error summary: {error_summary}")
    
    def test_incomplete_experiment_recovery(self, realistic_experiment_directory):
        """
        Test recovery from incomplete or partially corrupted experimental data.
        
        Validates:
        - Partial data recovery and validation
        - Missing field handling and defaults
        - Data quality assessment and filtering
        """
        config_file = realistic_experiment_directory["config_file"]
        
        # Create test cases with incomplete data
        temp_dir = realistic_experiment_directory["base_directory"] / "incomplete_test"
        temp_dir.mkdir(exist_ok=True)
        
        incomplete_scenarios = [
            # Missing required fields
            {"t": np.array([0, 1, 2]), "x": np.array([0, 1, 2])},  # Missing y
            
            # Mismatched array lengths
            {"t": np.array([0, 1, 2, 3]), "x": np.array([0, 1]), "y": np.array([0, 1])},
            
            # Empty arrays
            {"t": np.array([]), "x": np.array([]), "y": np.array([])},
            
            # Wrong data types
            {"t": [0, 1, 2], "x": "invalid", "y": np.array([0, 1, 2])},
            
            # Partially valid data
            {"t": np.array([0, 1, 2]), "x": np.array([0, 1, 2]), "y": np.array([0, 1, 2]), 
             "signal": np.array([np.nan, 1, 2])},
        ]
        
        recovery_results = []
        
        for i, incomplete_data in enumerate(incomplete_scenarios):
            test_file = temp_dir / f"incomplete_{i}.pkl"
            
            # Save incomplete data
            with open(test_file, 'wb') as f:
                pickle.dump(incomplete_data, f)
            
            # Test recovery
            try:
                loaded_data = read_pickle_any_format(test_file)
                
                # Attempt basic validation
                recovery_status = "partial"
                issues = []
                
                # Check required fields
                required_fields = ["t", "x", "y"]
                missing_fields = [field for field in required_fields if field not in loaded_data]
                if missing_fields:
                    issues.append(f"missing_fields: {missing_fields}")
                
                # Check array lengths
                if all(field in loaded_data for field in required_fields):
                    try:
                        lengths = [len(loaded_data[field]) for field in required_fields]
                        if len(set(lengths)) > 1:
                            issues.append(f"length_mismatch: {lengths}")
                    except (TypeError, AttributeError):
                        issues.append("invalid_array_types")
                
                # Check for NaN values
                for field in required_fields:
                    if field in loaded_data:
                        try:
                            if isinstance(loaded_data[field], np.ndarray) and np.any(np.isnan(loaded_data[field])):
                                issues.append(f"nan_values_in_{field}")
                        except (TypeError, AttributeError):
                            pass
                
                if not issues:
                    recovery_status = "success"
                
                recovery_results.append({
                    "scenario": i,
                    "status": recovery_status,
                    "issues": issues
                })
                
            except Exception as e:
                recovery_results.append({
                    "scenario": i,
                    "status": "failed",
                    "error": str(e)
                })
        
        # Analyze recovery results
        total_scenarios = len(incomplete_scenarios)
        successful_recoveries = sum(1 for r in recovery_results if r["status"] in ["success", "partial"])
        
        # Should be able to load most files even if they have issues
        recovery_rate = successful_recoveries / total_scenarios
        assert recovery_rate >= 0.8, f"Should recover from at least 80% of incomplete data scenarios, got {recovery_rate:.2%}"
        
        logger.info(f"Incomplete data recovery: {successful_recoveries}/{total_scenarios} scenarios handled, recovery rate: {recovery_rate:.2%}")
    
    def test_configuration_error_recovery(self, realistic_experiment_directory):
        """
        Test recovery from configuration errors and invalid settings.
        
        Validates:
        - Configuration validation and error reporting
        - Fallback configuration mechanisms
        - Invalid parameter handling
        """
        base_config = realistic_experiment_directory["config_data"]
        temp_dir = realistic_experiment_directory["base_directory"] / "config_test"
        temp_dir.mkdir(exist_ok=True)
        
        # Test various configuration error scenarios
        config_error_scenarios = [
            # Missing required sections
            {"project": {"name": "test"}},  # Missing directories
            
            # Invalid directory paths
            {"project": {"directories": {"major_data_directory": "/nonexistent/path"}}},
            
            # Malformed experiments section
            {"project": base_config["project"], "experiments": {"invalid": "not_a_dict"}},
            
            # Invalid extraction patterns
            {"project": {"extraction_patterns": ["[invalid regex("], "directories": base_config["project"]["directories"]}},
            
            # Missing datasets referenced by experiments
            {
                "project": base_config["project"],
                "experiments": {"test_exp": {"datasets": ["nonexistent_dataset"]}},
                "datasets": {}
            }
        ]
        
        error_handling_results = []
        
        for i, error_config in enumerate(config_error_scenarios):
            config_file = temp_dir / f"error_config_{i}.yaml"
            
            # Save error configuration
            with open(config_file, 'w') as f:
                yaml.dump(error_config, f)
            
            # Test error handling
            try:
                # Test configuration loading
                config = load_config(config_file)
                
                # Test API functions with error config
                if "experiments" in error_config:
                    experiment_names = list(error_config["experiments"].keys())
                    if experiment_names:
                        try:
                            result = api.load_experiment_files(
                                config_path=config_file,
                                experiment_name=experiment_names[0]
                            )
                            error_handling_results.append({
                                "scenario": i,
                                "config_load": "success",
                                "api_call": "success" if result is not None else "failed"
                            })
                        except Exception as api_error:
                            error_handling_results.append({
                                "scenario": i,
                                "config_load": "success",
                                "api_call": "failed",
                                "api_error": str(api_error)
                            })
                    else:
                        error_handling_results.append({
                            "scenario": i,
                            "config_load": "success",
                            "api_call": "skipped"
                        })
                else:
                    error_handling_results.append({
                        "scenario": i,
                        "config_load": "success",
                        "api_call": "not_applicable"
                    })
                    
            except Exception as config_error:
                error_handling_results.append({
                    "scenario": i,
                    "config_load": "failed",
                    "config_error": str(config_error)
                })
        
        # Validate error handling
        total_scenarios = len(config_error_scenarios)
        
        # Should gracefully handle configuration errors
        handled_errors = sum(1 for r in error_handling_results if "error" in str(r).lower())
        assert handled_errors >= total_scenarios * 0.6, \
            f"Should handle at least 60% of configuration errors gracefully"
        
        # Should provide meaningful error messages
        for result in error_handling_results:
            if "error" in result:
                error_msg = result.get("config_error") or result.get("api_error", "")
                assert len(error_msg) > 10, "Error messages should be descriptive"
        
        logger.info(f"Configuration error recovery: handled {len(error_handling_results)} scenarios")


class TestRealisticWorkflowIntegration:
    """
    Test realistic end-to-end workflow integration scenarios combining
    multiple system components in realistic research workflows.
    """
    
    def test_complete_neuroscience_research_workflow(self, realistic_experiment_directory, performance_benchmarks):
        """
        Test complete neuroscience research workflow from configuration to analysis.
        
        Validates:
        - Section 4.1.1.1: End-to-end user journey
        - F-015: Complete experimental data flows validation
        - Integration of all system components
        """
        config_file = realistic_experiment_directory["config_file"]
        
        # Step 1: Configuration and validation
        config = load_config(config_file)
        assert config is not None
        
        # Step 2: Experiment discovery and organization
        all_experiments = list(config["experiments"].keys())
        experiment_results = {}
        
        start_time = time.time()
        
        for experiment_name in all_experiments:
            try:
                # Discover experiment files
                exp_files = api.load_experiment_files(
                    config_path=config_file,
                    experiment_name=experiment_name,
                    extract_metadata=True
                )
                
                # Process experimental data
                processed_data = []
                for file_path, metadata in list(exp_files.items())[:5]:  # Process first 5 files per experiment
                    exp_data = read_pickle_any_format(file_path)
                    
                    # Basic analysis pipeline
                    analysis_result = {
                        "file_path": file_path,
                        "metadata": metadata,
                        "duration_seconds": exp_data["t"][-1] - exp_data["t"][0] if len(exp_data["t"]) > 0 else 0,
                        "trajectory_length": np.sum(np.sqrt(np.diff(exp_data["x"])**2 + np.diff(exp_data["y"])**2)),
                        "mean_speed": np.mean(exp_data.get("speed", [0])),
                        "arena_coverage": len(np.unique(np.round(exp_data["x"], 1))) * len(np.unique(np.round(exp_data["y"], 1))),
                        "data_quality_score": 1.0 - (np.sum(np.isnan(exp_data["x"])) + np.sum(np.isnan(exp_data["y"]))) / (2 * len(exp_data["x"]))
                    }
                    
                    processed_data.append(analysis_result)
                
                experiment_results[experiment_name] = {
                    "total_files": len(exp_files),
                    "processed_files": len(processed_data),
                    "processed_data": processed_data
                }
                
            except Exception as e:
                logger.warning(f"Failed to process experiment {experiment_name}: {e}")
                experiment_results[experiment_name] = {"error": str(e)}
        
        total_time = time.time() - start_time
        
        # Step 3: Validate workflow results
        successful_experiments = [name for name, result in experiment_results.items() if "error" not in result]
        assert len(successful_experiments) >= 2, f"Should successfully process at least 2 experiments, got {len(successful_experiments)}"
        
        # Step 4: Cross-experiment analysis
        all_processed_data = []
        for exp_name in successful_experiments:
            all_processed_data.extend(experiment_results[exp_name]["processed_data"])
        
        # Aggregate statistics
        if all_processed_data:
            durations = [d["duration_seconds"] for d in all_processed_data]
            speeds = [d["mean_speed"] for d in all_processed_data]
            quality_scores = [d["data_quality_score"] for d in all_processed_data]
            
            workflow_summary = {
                "total_experiments": len(successful_experiments),
                "total_files_processed": len(all_processed_data),
                "mean_duration": np.mean(durations),
                "mean_speed": np.mean(speeds),
                "mean_quality_score": np.mean(quality_scores),
                "processing_time": total_time
            }
            
            # Validate workflow quality
            assert workflow_summary["mean_quality_score"] >= 0.9, "Data quality should be high"
            assert workflow_summary["mean_duration"] > 0, "Should have positive duration data"
            
            # Performance validation
            files_per_second = workflow_summary["total_files_processed"] / total_time
            assert files_per_second >= 0.5, f"Should process at least 0.5 files/second in complete workflow, got {files_per_second:.2f}"
        
        logger.info(f"Complete workflow validation: {len(successful_experiments)} experiments, {len(all_processed_data)} files in {total_time:.2f}s")
    
    def test_multi_experimenter_collaborative_workflow(self, realistic_experiment_directory):
        """
        Test collaborative research workflow with multiple experimenters and datasets.
        
        Validates:
        - Multi-user data organization and access
        - Cross-experimenter data consistency
        - Collaborative metadata standards
        """
        config_file = realistic_experiment_directory["config_file"]
        
        # Group data by experimenter
        all_files = realistic_experiment_directory["created_files"]
        experimenter_data = {}
        
        for file_path in all_files[:10]:  # Test with subset for performance
            try:
                exp_data = read_pickle_any_format(file_path)
                experimenter = exp_data.get("experimenter", "unknown")
                
                if experimenter not in experimenter_data:
                    experimenter_data[experimenter] = {
                        "files": [],
                        "experiments": set(),
                        "animals": set(),
                        "dates": set()
                    }
                
                experimenter_data[experimenter]["files"].append(file_path)
                experimenter_data[experimenter]["experiments"].add(exp_data.get("experiment_type", "unknown"))
                experimenter_data[experimenter]["animals"].add(exp_data.get("animal_id", "unknown"))
                experimenter_data[experimenter]["dates"].add(exp_data.get("date", "unknown"))
                
            except Exception as e:
                logger.debug(f"Could not process {file_path}: {e}")
        
        # Validate multi-experimenter organization
        assert len(experimenter_data) >= 2, f"Should have multiple experimenters, found: {list(experimenter_data.keys())}"
        
        # Validate experimenter data quality
        for experimenter, data in experimenter_data.items():
            assert len(data["files"]) > 0, f"Experimenter {experimenter} should have data files"
            assert len(data["experiments"]) > 0, f"Experimenter {experimenter} should have experiments"
            
            # Check data consistency within experimenter
            file_count = len(data["files"])
            if file_count > 1:
                # Should have some consistency in experimental approach
                experiment_count = len(data["experiments"])
                assert experiment_count <= file_count, f"Experimenter {experimenter} has inconsistent experiment organization"
        
        # Test cross-experimenter consistency
        all_experimenters = list(experimenter_data.keys())
        if len(all_experimenters) >= 2:
            exp1_data = experimenter_data[all_experimenters[0]]
            exp2_data = experimenter_data[all_experimenters[1]]
            
            # Should have some overlap in experimental types (collaborative project)
            common_experiments = exp1_data["experiments"].intersection(exp2_data["experiments"])
            # At least some shared experimental approaches in collaborative work
            total_experiments = exp1_data["experiments"].union(exp2_data["experiments"])
            if len(total_experiments) > 0:
                overlap_rate = len(common_experiments) / len(total_experiments)
                # Allow for some experimenter specialization
                assert overlap_rate >= 0.2, f"Should have some experimental overlap between experimenters"
        
        logger.info(f"Validated collaborative workflow: {len(experimenter_data)} experimenters")
    
    def test_longitudinal_study_temporal_analysis(self, realistic_experiment_directory):
        """
        Test longitudinal study analysis with temporal organization and trends.
        
        Validates:
        - Temporal data organization and analysis
        - Longitudinal tracking and consistency
        - Time-series analysis capabilities
        """
        config_file = realistic_experiment_directory["config_file"]
        
        # Load longitudinal baseline study
        baseline_files = api.load_experiment_files(
            config_path=config_file,
            experiment_name="longitudinal_baseline",
            extract_metadata=True
        )
        
        # Organize data temporally
        temporal_data = {}
        
        for file_path, metadata in baseline_files.items():
            date = metadata["date"]
            animal_id = metadata["animal_id"]
            
            key = (animal_id, date)
            
            if key not in temporal_data:
                exp_data = read_pickle_any_format(file_path)
                temporal_data[key] = {
                    "animal_id": animal_id,
                    "date": date,
                    "file_path": file_path,
                    "duration": exp_data["t"][-1] - exp_data["t"][0] if len(exp_data["t"]) > 0 else 0,
                    "total_distance": np.sum(np.sqrt(np.diff(exp_data["x"])**2 + np.diff(exp_data["y"])**2)),
                    "mean_speed": np.mean(exp_data.get("speed", [0])),
                    "arena_center_time": np.sum(np.sqrt(exp_data["x"]**2 + exp_data["y"]**2) < 30) / len(exp_data["x"]) if len(exp_data["x"]) > 0 else 0
                }
        
        # Group by animal for longitudinal analysis
        animal_timeseries = {}
        for (animal_id, date), data in temporal_data.items():
            if animal_id not in animal_timeseries:
                animal_timeseries[animal_id] = []
            animal_timeseries[animal_id].append(data)
        
        # Sort by date within each animal
        for animal_id in animal_timeseries:
            animal_timeseries[animal_id].sort(key=lambda x: x["date"])
        
        # Validate longitudinal patterns
        multi_session_animals = [aid for aid, sessions in animal_timeseries.items() if len(sessions) > 1]
        assert len(multi_session_animals) > 0, "Should have animals with multiple sessions"
        
        # Analyze temporal trends
        temporal_consistency_results = []
        
        for animal_id in multi_session_animals:
            sessions = animal_timeseries[animal_id]
            
            # Extract time series
            dates = [s["date"] for s in sessions]
            durations = [s["duration"] for s in sessions]
            distances = [s["total_distance"] for s in sessions]
            speeds = [s["mean_speed"] for s in sessions]
            center_times = [s["arena_center_time"] for s in sessions]
            
            # Calculate consistency metrics
            duration_cv = np.std(durations) / np.mean(durations) if np.mean(durations) > 0 else 0
            speed_cv = np.std(speeds) / np.mean(speeds) if np.mean(speeds) > 0 else 0
            
            temporal_consistency_results.append({
                "animal_id": animal_id,
                "sessions": len(sessions),
                "date_range": f"{min(dates)} to {max(dates)}",
                "duration_cv": duration_cv,
                "speed_cv": speed_cv,
                "mean_duration": np.mean(durations),
                "mean_speed": np.mean(speeds)
            })
        
        # Validate temporal consistency
        for result in temporal_consistency_results:
            # Behavioral measures should be reasonably consistent within animals
            assert result["duration_cv"] < 1.0, f"Duration too variable for {result['animal_id']}: CV={result['duration_cv']:.2f}"
            assert result["speed_cv"] < 2.0, f"Speed too variable for {result['animal_id']}: CV={result['speed_cv']:.2f}"
        
        logger.info(f"Longitudinal analysis: {len(multi_session_animals)} animals across time, consistency validated")


# Performance benchmark configuration for realistic scenarios
@pytest.mark.performance
class TestRealisticPerformanceBenchmarks:
    """
    Comprehensive performance benchmarks for realistic experimental scenarios.
    These tests validate performance against defined SLAs and establish
    performance baselines for various operational scenarios.
    """
    
    def test_realistic_discovery_performance_benchmark(self, realistic_experiment_directory, performance_benchmarks):
        """
        Benchmark file discovery performance with realistic directory structures.
        
        SLA: File discovery should complete within 5 seconds for 10,000 files
        """
        config_file = realistic_experiment_directory["config_file"]
        base_dir = realistic_experiment_directory["directories"]["raw_data"]
        
        # Count available files for baseline
        from flyrigloader.discovery.files import discover_files
        
        start_time = time.time()
        discovered_files = discover_files(
            directory=base_dir,
            pattern="*.pkl",
            recursive=True
        )
        discovery_time = time.time() - start_time
        
        file_count = len(discovered_files)
        performance_benchmarks.assert_performance_sla(
            "realistic_file_discovery",
            discovery_time,
            performance_benchmarks.benchmark_file_discovery(file_count)
        )
        
        logger.info(f"Discovery benchmark: {file_count} files in {discovery_time:.3f}s")
    
    def test_realistic_data_loading_performance_benchmark(self, realistic_experiment_directory, performance_benchmarks):
        """
        Benchmark data loading performance with realistic experimental data.
        
        SLA: Data loading should achieve 1 second per 100MB performance
        """
        all_files = realistic_experiment_directory["created_files"]
        test_files = all_files[:10]  # Test with subset
        
        total_size = sum(f.stat().st_size for f in test_files)
        
        start_time = time.time()
        loaded_count = 0
        
        for file_path in test_files:
            try:
                exp_data = read_pickle_any_format(file_path)
                if isinstance(exp_data, dict) and "t" in exp_data:
                    loaded_count += 1
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
        
        loading_time = time.time() - start_time
        
        size_mb = total_size / (1024 * 1024)
        performance_benchmarks.assert_performance_sla(
            "realistic_data_loading",
            loading_time,
            performance_benchmarks.benchmark_data_loading(size_mb)
        )
        
        logger.info(f"Loading benchmark: {loaded_count} files ({size_mb:.1f}MB) in {loading_time:.3f}s")


if __name__ == "__main__":
    # Allow running specific test classes or methods for development
    import sys
    if len(sys.argv) > 1:
        pytest.main([__file__] + sys.argv[1:])
    else:
        pytest.main([__file__, "-v"])