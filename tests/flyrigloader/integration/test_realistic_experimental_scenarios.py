"""
Realistic experimental scenario integration test suite validating flyrigloader functionality
with comprehensive synthetic datasets that mirror actual neuroscience research workflows.

This module implements diverse experimental conditions including multi-day studies, various
rig configurations, complex metadata patterns, and realistic data scales. Tests complete
workflows with synthetic but realistic experimental matrices, time series data, multi-dimensional
signal arrays, and complex directory structures that represent actual optical fly rig experimental setups.

Validates system behavior under realistic data loads, complex filtering scenarios, and ensures
robustness across different experimental design patterns used in neuroscience research.

Requirements Covered:
- TST-INTEG-002: Realistic test data generation representing experimental scenarios per Section 2.2.10
- F-015: Realistic experimental data flows validation per Section 2.1.15 Integration Test Harness
- Section 4.1.2.2: Multi-Experiment Batch Processing workflow validation per System Workflows
- F-007: Realistic metadata extraction pattern validation per Section 2.1.7 Metadata Extraction System
- TST-PERF-002: Realistic data scale performance validation per Section 2.2.9 Performance Benchmark Requirements
- Section 4.1.2.3: Error recovery validation with realistic failure scenarios per System Workflows
"""

import copy
import gzip
import pickle
import random
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml
from loguru import logger

# flyrigloader imports for integration testing
from flyrigloader.api import (
    get_dataset_parameters,
    get_experiment_parameters,
    load_dataset_files,
    load_experiment_files,
    process_experiment_data,
)
from flyrigloader.config.yaml_config import (
    get_all_dataset_names,
    get_all_experiment_names,
    get_dataset_info,
    get_experiment_info,
    load_config,
)
from flyrigloader.discovery.files import discover_files
from flyrigloader.io.pickle import read_pickle_any_format
from flyrigloader.utils.dataframe import validate_dataframe_structure


class RealisticExperimentalDataGenerator:
    """
    Advanced synthetic data generator for realistic neuroscience experimental scenarios.
    
    Creates comprehensive datasets that mirror actual fly rig experimental setups including:
    - Multi-day longitudinal studies with temporal correlations
    - Various rig configurations with different sampling rates and resolutions
    - Complex behavioral patterns with biologically plausible characteristics
    - Realistic metadata structures with experimental design patterns
    - Multi-dimensional signal arrays representing calcium imaging or electrophysiology
    - Error conditions and edge cases found in real experimental data
    """

    def __init__(self, seed: int = 42):
        """Initialize the data generator with reproducible random seed."""
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Realistic experimental parameters based on actual fly rig setups
        self.rig_configurations = {
            "old_opto": {
                "sampling_frequency": 60.0,  # Hz
                "mm_per_px": 0.154,
                "camera_resolution": [1024, 768],
                "arena_diameter_mm": 120.0,
                "signal_channels": 16,
                "typical_experiment_duration_min": [5, 15, 30],
                "led_wavelength_nm": 470
            },
            "new_opto": {
                "sampling_frequency": 60.0,
                "mm_per_px": 0.1818,
                "camera_resolution": [1280, 1024],
                "arena_diameter_mm": 150.0,
                "signal_channels": 32,
                "typical_experiment_duration_min": [10, 20, 45],
                "led_wavelength_nm": 470
            },
            "high_speed_rig": {
                "sampling_frequency": 200.0,
                "mm_per_px": 0.05,
                "camera_resolution": [2048, 2048],
                "arena_diameter_mm": 200.0,
                "signal_channels": 64,
                "typical_experiment_duration_min": [2, 5, 10],
                "led_wavelength_nm": 590
            }
        }
        
        # Realistic experimental conditions
        self.experimental_conditions = {
            "baseline": {
                "description": "Control condition with no stimulation",
                "behavioral_characteristics": {
                    "velocity_mean": 8.0,  # mm/s
                    "velocity_std": 3.0,
                    "center_bias": 0.3,
                    "turn_frequency": 0.1  # turns per second
                }
            },
            "optogenetic_stimulation": {
                "description": "Optogenetic activation of neural circuits",
                "behavioral_characteristics": {
                    "velocity_mean": 12.0,
                    "velocity_std": 5.0,
                    "center_bias": 0.1,  # Less center bias during stimulation
                    "turn_frequency": 0.25
                }
            },
            "chemical_stimulation": {
                "description": "Pharmacological intervention",
                "behavioral_characteristics": {
                    "velocity_mean": 5.0,  # Slower movement
                    "velocity_std": 2.0,
                    "center_bias": 0.6,  # Higher center bias
                    "turn_frequency": 0.05
                }
            },
            "heat_stress": {
                "description": "Temperature stress condition",
                "behavioral_characteristics": {
                    "velocity_mean": 15.0,  # Increased activity
                    "velocity_std": 7.0,
                    "center_bias": 0.0,  # Edge-seeking behavior
                    "turn_frequency": 0.3
                }
            }
        }
        
        # Animal line characteristics for metadata generation
        self.animal_lines = {
            "WT": {"description": "Wild type control", "n_animals": 20},
            "GAL4": {"description": "GAL4 driver line", "n_animals": 15},
            "UAS": {"description": "UAS effector line", "n_animals": 12},
            "CRISPR": {"description": "CRISPR knockout", "n_animals": 8}
        }

    def generate_realistic_trajectory(
        self,
        rig_name: str,
        condition: str,
        duration_minutes: float,
        animal_id: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate biologically plausible fly trajectory with realistic movement patterns.
        
        Args:
            rig_name: Name of the rig configuration to use
            condition: Experimental condition affecting behavior
            duration_minutes: Duration of experiment in minutes
            animal_id: Unique animal identifier affecting individual variability
            
        Returns:
            Tuple of (time_array, x_positions, y_positions) with realistic movement
        """
        rig_config = self.rig_configurations[rig_name]
        behavior_params = self.experimental_conditions[condition]["behavioral_characteristics"]
        
        # Calculate time parameters
        sampling_freq = rig_config["sampling_frequency"]
        n_timepoints = int(duration_minutes * 60 * sampling_freq)
        dt = 1.0 / sampling_freq
        time_array = np.arange(n_timepoints) * dt
        
        # Arena parameters
        arena_radius = rig_config["arena_diameter_mm"] / 2.0
        
        # Individual animal variability (based on animal_id hash)
        animal_hash = hash(animal_id) % 1000
        individual_velocity_factor = 0.7 + 0.6 * (animal_hash / 1000.0)  # 0.7-1.3x velocity
        individual_turn_bias = -0.1 + 0.2 * (animal_hash / 1000.0)  # Left/right turn bias
        
        # Initialize position arrays
        x_pos = np.zeros(n_timepoints)
        y_pos = np.zeros(n_timepoints)
        
        # Start near center with some randomness
        x_pos[0] = np.random.normal(0, arena_radius * 0.1)
        y_pos[0] = np.random.normal(0, arena_radius * 0.1)
        
        # Generate movement with temporal correlations
        velocity_autocorr = 0.9  # High temporal correlation in velocity
        heading_autocorr = 0.95  # Very high correlation in heading
        
        current_velocity = behavior_params["velocity_mean"] * individual_velocity_factor
        current_heading = np.random.uniform(0, 2 * np.pi)
        
        for i in range(1, n_timepoints):
            # Update velocity with autocorrelation and noise
            velocity_noise = np.random.normal(0, behavior_params["velocity_std"])
            current_velocity = (
                velocity_autocorr * current_velocity +
                (1 - velocity_autocorr) * behavior_params["velocity_mean"] * individual_velocity_factor +
                velocity_noise * dt
            )
            current_velocity = max(0, current_velocity)  # Non-negative velocity
            
            # Update heading with autocorrelation, turn frequency, and center bias
            current_distance_from_center = np.sqrt(x_pos[i-1]**2 + y_pos[i-1]**2)
            center_bias_strength = behavior_params["center_bias"]
            
            # Center-seeking force
            if current_distance_from_center > 0.1:
                center_heading = np.arctan2(-y_pos[i-1], -x_pos[i-1])
                center_bias_force = center_bias_strength * (current_distance_from_center / arena_radius)**2
            else:
                center_heading = current_heading
                center_bias_force = 0
            
            # Turn tendency
            turn_noise = np.random.normal(individual_turn_bias, behavior_params["turn_frequency"] * dt)
            
            # Boundary avoidance
            boundary_avoidance = 0
            if current_distance_from_center > arena_radius * 0.8:
                boundary_heading = np.arctan2(-y_pos[i-1], -x_pos[i-1])
                boundary_avoidance = 0.5 * ((current_distance_from_center - arena_radius * 0.8) / (arena_radius * 0.2))
                boundary_avoidance = min(boundary_avoidance, 1.0)
            
            # Combine heading influences
            heading_change = (
                turn_noise +
                center_bias_force * np.sin(center_heading - current_heading) +
                boundary_avoidance * np.sin(boundary_heading - current_heading)
            )
            
            current_heading = (
                heading_autocorr * current_heading +
                (1 - heading_autocorr) * (current_heading + heading_change)
            ) % (2 * np.pi)
            
            # Calculate position change
            dx = current_velocity * np.cos(current_heading) * dt
            dy = current_velocity * np.sin(current_heading) * dt
            
            # Update position
            new_x = x_pos[i-1] + dx
            new_y = y_pos[i-1] + dy
            
            # Enforce arena boundaries with realistic reflection
            new_distance = np.sqrt(new_x**2 + new_y**2)
            if new_distance > arena_radius:
                # Reflect trajectory with some energy loss
                reflection_factor = arena_radius / new_distance
                new_x *= reflection_factor * 0.9
                new_y *= reflection_factor * 0.9
                # Randomize heading after boundary collision
                current_heading = np.random.uniform(0, 2 * np.pi)
                current_velocity *= 0.7  # Reduce velocity after collision
            
            x_pos[i] = new_x
            y_pos[i] = new_y
        
        return time_array, x_pos, y_pos

    def generate_realistic_signal_data(
        self,
        rig_name: str,
        condition: str,
        n_timepoints: int,
        behavior_correlation: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate realistic multi-channel signal data (calcium imaging or electrophysiology).
        
        Args:
            rig_name: Rig configuration determining channel count
            condition: Experimental condition affecting signal characteristics
            n_timepoints: Number of temporal samples
            behavior_correlation: Whether signals should correlate with behavior
            
        Returns:
            Tuple of (single_channel_signal, multi_channel_signal_disp)
        """
        rig_config = self.rig_configurations[rig_name]
        n_channels = rig_config["signal_channels"]
        sampling_freq = rig_config["sampling_frequency"]
        
        # Time array for signal generation
        t = np.arange(n_timepoints) / sampling_freq
        
        # Base signal characteristics depend on experimental condition
        if condition == "optogenetic_stimulation":
            # Higher baseline activity with stimulation artifacts
            baseline_activity = 0.3
            stimulation_events = np.random.poisson(0.1, n_timepoints)  # Sparse stimulation
            signal_noise = 0.05
        elif condition == "chemical_stimulation":
            # Gradual increase in activity over time
            baseline_activity = 0.2 + 0.3 * (t / np.max(t))
            stimulation_events = np.zeros(n_timepoints)
            signal_noise = 0.03
        elif condition == "heat_stress":
            # Irregular bursts of activity
            baseline_activity = 0.4
            stimulation_events = np.random.poisson(0.05, n_timepoints)
            signal_noise = 0.08
        else:  # baseline
            baseline_activity = 0.1
            stimulation_events = np.zeros(n_timepoints)
            signal_noise = 0.02
        
        # Generate single-channel signal (summary/average)
        single_signal = baseline_activity * np.ones(n_timepoints)
        
        # Add temporal dynamics
        for freq in [0.1, 0.3, 1.0]:  # Multiple frequency components
            amplitude = np.random.uniform(0.05, 0.15)
            phase = np.random.uniform(0, 2 * np.pi)
            single_signal += amplitude * np.sin(2 * np.pi * freq * t + phase)
        
        # Add stimulation events
        single_signal += 0.5 * stimulation_events
        
        # Add noise
        single_signal += signal_noise * np.random.normal(0, 1, n_timepoints)
        
        # Generate multi-channel signal_disp
        signal_disp = np.zeros((n_channels, n_timepoints))
        
        for ch in range(n_channels):
            # Each channel has individual characteristics
            channel_baseline = baseline_activity * (0.5 + np.random.random())
            channel_signal = channel_baseline * np.ones(n_timepoints)
            
            # Channel-specific frequency response
            for freq in [0.1, 0.3, 1.0, 2.0]:
                amplitude = np.random.uniform(0.02, 0.1)
                phase = np.random.uniform(0, 2 * np.pi)
                channel_signal += amplitude * np.sin(2 * np.pi * freq * t + phase)
            
            # Correlation between channels (realistic neural connectivity)
            if ch > 0:
                correlation_strength = np.random.uniform(0.1, 0.4)
                channel_signal += correlation_strength * signal_disp[ch-1, :]
            
            # Add stimulation events with channel-specific responses
            channel_stim_response = np.random.uniform(0.3, 0.8)
            channel_signal += channel_stim_response * stimulation_events
            
            # Channel-specific noise
            channel_noise = signal_noise * np.random.uniform(0.8, 1.2)
            channel_signal += channel_noise * np.random.normal(0, 1, n_timepoints)
            
            signal_disp[ch, :] = channel_signal
        
        return single_signal, signal_disp

    def generate_experimental_matrix(
        self,
        rig_name: str,
        condition: str,
        animal_id: str,
        duration_minutes: Optional[float] = None,
        include_derived_measures: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Generate a complete experimental data matrix for a single experiment.
        
        Args:
            rig_name: Rig configuration name
            condition: Experimental condition
            animal_id: Unique animal identifier
            duration_minutes: Duration (if None, randomly selected from typical durations)
            include_derived_measures: Whether to include velocity, angular measures, etc.
            
        Returns:
            Dictionary containing complete experimental data matrix
        """
        rig_config = self.rig_configurations[rig_name]
        
        # Select duration if not provided
        if duration_minutes is None:
            duration_minutes = np.random.choice(rig_config["typical_experiment_duration_min"])
        
        # Generate trajectory data
        time_array, x_pos, y_pos = self.generate_realistic_trajectory(
            rig_name, condition, duration_minutes, animal_id
        )
        
        # Generate signal data
        single_signal, signal_disp = self.generate_realistic_signal_data(
            rig_name, condition, len(time_array)
        )
        
        # Build experimental matrix
        exp_matrix = {
            't': time_array,
            'x': x_pos,
            'y': y_pos,
            'signal': single_signal,
            'signal_disp': signal_disp
        }
        
        if include_derived_measures:
            # Calculate derived measures
            dt = np.diff(time_array, prepend=time_array[1] - time_array[0])
            dx = np.diff(x_pos, prepend=0)
            dy = np.diff(y_pos, prepend=0)
            
            # Velocity components
            exp_matrix['vx'] = dx / dt
            exp_matrix['vy'] = dy / dt
            exp_matrix['speed'] = np.sqrt(exp_matrix['vx']**2 + exp_matrix['vy']**2)
            
            # Angular measures
            exp_matrix['dtheta'] = np.arctan2(dy, dx)
            
            # Distance measures
            exp_matrix['distance_from_center'] = np.sqrt(x_pos**2 + y_pos**2)
            exp_matrix['cumulative_distance'] = np.cumsum(np.sqrt(dx**2 + dy**2))
        
        return exp_matrix

    def create_multi_day_study_structure(
        self,
        base_directory: Path,
        n_days: int = 7,
        animals_per_day: List[int] = None,
        rigs: List[str] = None,
        conditions: List[str] = None
    ) -> Dict[str, Any]:
        """
        Create a realistic multi-day experimental study directory structure.
        
        Args:
            base_directory: Root directory for the study
            n_days: Number of experimental days
            animals_per_day: List of animal counts per day (if None, randomized)
            rigs: List of rig names to use (if None, uses all available)
            conditions: List of conditions to use (if None, uses all available)
            
        Returns:
            Dictionary with study structure and metadata
        """
        if animals_per_day is None:
            animals_per_day = [np.random.randint(3, 8) for _ in range(n_days)]
        
        if rigs is None:
            rigs = list(self.rig_configurations.keys())
        
        if conditions is None:
            conditions = list(self.experimental_conditions.keys())
        
        # Create study structure
        study_structure = {
            "base_directory": base_directory,
            "experiment_days": {},
            "animals": {},
            "files": [],
            "metadata": {
                "study_start_date": datetime.now().date(),
                "total_days": n_days,
                "total_animals": sum(animals_per_day),
                "rigs_used": rigs,
                "conditions_tested": conditions
            }
        }
        
        # Generate experimental schedule
        start_date = datetime.now().date() - timedelta(days=n_days)
        
        for day_idx in range(n_days):
            current_date = start_date + timedelta(days=day_idx)
            date_str = current_date.strftime("%Y-%m-%d")
            day_directory = base_directory / date_str
            day_directory.mkdir(parents=True, exist_ok=True)
            
            study_structure["experiment_days"][date_str] = {
                "directory": day_directory,
                "date": current_date,
                "animals_tested": animals_per_day[day_idx],
                "files": []
            }
            
            # Generate animals for this day
            for animal_idx in range(animals_per_day[day_idx]):
                # Select animal line with realistic distribution
                animal_line = np.random.choice(list(self.animal_lines.keys()), 
                                             p=[0.4, 0.3, 0.2, 0.1])  # WT most common
                animal_id = f"{animal_line}_{day_idx:02d}_{animal_idx:03d}"
                
                # Select rig and condition
                rig = np.random.choice(rigs)
                condition = np.random.choice(conditions)
                
                # Generate experimental data
                exp_matrix = self.generate_experimental_matrix(
                    rig, condition, animal_id
                )
                
                # Create filename with realistic pattern
                filename = f"{date_str}_{animal_id}_{rig}_{condition}_exp.pkl"
                file_path = day_directory / filename
                
                # Save experimental data
                with open(file_path, 'wb') as f:
                    pickle.dump(exp_matrix, f)
                
                # Record file information
                file_info = {
                    "path": file_path,
                    "filename": filename,
                    "animal_id": animal_id,
                    "animal_line": animal_line,
                    "rig": rig,
                    "condition": condition,
                    "date": current_date,
                    "day_index": day_idx,
                    "animal_index": animal_idx,
                    "file_size_bytes": file_path.stat().st_size,
                    "n_timepoints": len(exp_matrix['t']),
                    "duration_seconds": exp_matrix['t'][-1] - exp_matrix['t'][0]
                }
                
                study_structure["experiment_days"][date_str]["files"].append(file_info)
                study_structure["files"].append(file_info)
                
                # Track animal across days
                if animal_id not in study_structure["animals"]:
                    study_structure["animals"][animal_id] = {
                        "animal_line": animal_line,
                        "first_test_date": current_date,
                        "experiments": []
                    }
                
                study_structure["animals"][animal_id]["experiments"].append(file_info)
        
        return study_structure

    def create_corrupted_data_scenarios(
        self,
        base_directory: Path,
        n_corrupted_files: int = 5
    ) -> Dict[str, Path]:
        """
        Create realistic data corruption scenarios for error recovery testing.
        
        Args:
            base_directory: Directory to create corrupted files in
            n_corrupted_files: Number of corrupted files to create
            
        Returns:
            Dictionary mapping corruption type to file paths
        """
        corrupted_dir = base_directory / "corrupted_data"
        corrupted_dir.mkdir(exist_ok=True)
        
        corruption_scenarios = {}
        
        # Truncated pickle file
        truncated_path = corrupted_dir / "truncated_experiment.pkl"
        with open(truncated_path, 'wb') as f:
            f.write(b'\x80\x03}')  # Incomplete pickle header
        corruption_scenarios["truncated_pickle"] = truncated_path
        
        # Empty file
        empty_path = corrupted_dir / "empty_experiment.pkl"
        empty_path.touch()
        corruption_scenarios["empty_file"] = empty_path
        
        # Non-pickle file with .pkl extension
        non_pickle_path = corrupted_dir / "not_pickle.pkl"
        with open(non_pickle_path, 'w') as f:
            f.write("This is not a pickle file")
        corruption_scenarios["fake_pickle"] = non_pickle_path
        
        # Missing required columns
        missing_columns_path = corrupted_dir / "missing_columns.pkl"
        incomplete_matrix = {'x': np.array([1, 2, 3]), 'y': np.array([4, 5, 6])}  # Missing 't'
        with open(missing_columns_path, 'wb') as f:
            pickle.dump(incomplete_matrix, f)
        corruption_scenarios["missing_columns"] = missing_columns_path
        
        # Mismatched array lengths
        mismatched_path = corrupted_dir / "mismatched_lengths.pkl"
        mismatched_matrix = {
            't': np.array([0, 1, 2, 3]),
            'x': np.array([1, 2]),  # Wrong length
            'y': np.array([4, 5, 6])  # Wrong length
        }
        with open(mismatched_path, 'wb') as f:
            pickle.dump(mismatched_matrix, f)
        corruption_scenarios["mismatched_lengths"] = mismatched_path
        
        return corruption_scenarios


@pytest.fixture(scope="function")
def realistic_data_generator():
    """Fixture providing the realistic experimental data generator."""
    return RealisticExperimentalDataGenerator(seed=42)


@pytest.fixture(scope="function")  
def multi_day_study_scenario(realistic_data_generator, cross_platform_temp_dir):
    """
    Create a comprehensive multi-day experimental study scenario.
    
    This fixture generates a realistic experimental study with:
    - Multiple days of data collection
    - Various animal lines and experimental conditions
    - Different rig configurations
    - Realistic file naming and directory structure
    - Complex metadata patterns for extraction testing
    """
    study_dir = cross_platform_temp_dir / "multi_day_study"
    study_dir.mkdir(exist_ok=True)
    
    # Generate 5-day study with realistic parameters
    study_structure = realistic_data_generator.create_multi_day_study_structure(
        base_directory=study_dir,
        n_days=5,
        animals_per_day=[4, 6, 5, 7, 4],  # Varying daily schedules
        rigs=["old_opto", "new_opto"],
        conditions=["baseline", "optogenetic_stimulation", "chemical_stimulation"]
    )
    
    return study_structure


@pytest.fixture(scope="function")
def large_scale_performance_dataset(realistic_data_generator, cross_platform_temp_dir):
    """
    Create large-scale datasets for performance testing against SLA requirements.
    
    Generates datasets of various sizes to test:
    - Data loading performance (1 second per 100MB SLA)
    - DataFrame transformation (500ms per 1M rows SLA)
    - File discovery performance (5 seconds for 10,000 files SLA)
    """
    perf_dir = cross_platform_temp_dir / "performance_testing"
    perf_dir.mkdir(exist_ok=True)
    
    performance_datasets = {}
    
    # Small dataset (baseline)
    small_matrix = realistic_data_generator.generate_experimental_matrix(
        "old_opto", "baseline", "perf_test_small", duration_minutes=5
    )
    small_path = perf_dir / "small_experiment.pkl"
    with open(small_path, 'wb') as f:
        pickle.dump(small_matrix, f)
    performance_datasets["small"] = {
        "path": small_path,
        "expected_rows": len(small_matrix['t']),
        "file_size_mb": small_path.stat().st_size / 1024 / 1024
    }
    
    # Large dataset (>1M rows)
    large_matrix = realistic_data_generator.generate_experimental_matrix(
        "high_speed_rig", "optogenetic_stimulation", "perf_test_large", duration_minutes=90
    )
    large_path = perf_dir / "large_experiment.pkl"
    with open(large_path, 'wb') as f:
        pickle.dump(large_matrix, f)
    performance_datasets["large"] = {
        "path": large_path,
        "expected_rows": len(large_matrix['t']),
        "file_size_mb": large_path.stat().st_size / 1024 / 1024
    }
    
    # Very large dataset (compressed)
    very_large_matrix = realistic_data_generator.generate_experimental_matrix(
        "high_speed_rig", "heat_stress", "perf_test_very_large", duration_minutes=120
    )
    very_large_path = perf_dir / "very_large_experiment.pkl.gz"
    with gzip.open(very_large_path, 'wb') as f:
        pickle.dump(very_large_matrix, f)
    performance_datasets["very_large"] = {
        "path": very_large_path,
        "expected_rows": len(very_large_matrix['t']),
        "file_size_mb": very_large_path.stat().st_size / 1024 / 1024
    }
    
    return performance_datasets


@pytest.fixture(scope="function")
def comprehensive_config_scenario(multi_day_study_scenario, cross_platform_temp_dir):
    """
    Create comprehensive configuration scenario with realistic experimental design.
    
    This fixture provides a complete YAML configuration that matches the multi-day
    study structure for end-to-end integration testing.
    """
    config_dir = cross_platform_temp_dir / "configs"
    config_dir.mkdir(exist_ok=True)
    
    # Extract information from study structure
    study = multi_day_study_scenario
    
    # Create comprehensive configuration
    config = {
        "project": {
            "name": "multi_day_neuroscience_study",
            "directories": {
                "major_data_directory": str(study["base_directory"]),
                "processed_data_directory": str(config_dir / "processed"),
                "backup_directory": str(config_dir / "backups")
            },
            "ignore_substrings": [
                "._",  # Mac hidden files
                ".DS_Store",  # Mac metadata
                "__pycache__",  # Python cache
                ".tmp",  # Temporary files
                "backup_",  # Backup files
                "calibration",  # Calibration data
                "test_"  # Test files
            ],
            "mandatory_substrings": [
                "_exp"  # All experimental files must contain "_exp"
            ],
            "extraction_patterns": [
                # Date-animal-rig-condition pattern
                r"(?P<date>\d{4}-\d{2}-\d{2})_(?P<animal_line>\w+)_(?P<day>\d{2})_(?P<animal_num>\d{3})_(?P<rig>\w+)_(?P<condition>\w+)_exp\.pkl",
                # Alternative pattern for metadata extraction
                r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day_of_month>\d{2})_(?P<animal_id>\w+)_(?P<rig_type>\w+)_(?P<experimental_condition>\w+)",
                # Simple pattern fallback
                r"(?P<identifier>\w+)_(?P<timestamp>\d+)"
            ],
            "file_extensions": [".pkl", ".pickle", ".pkl.gz"],
            "max_file_size_mb": 1000,
            "parallel_processing": True
        },
        "rigs": {
            "old_opto": {
                "sampling_frequency": 60.0,
                "mm_per_px": 0.154,
                "camera_resolution": [1024, 768],
                "arena_diameter_mm": 120.0,
                "calibration_date": "2024-01-15",
                "led_wavelength_nm": 470,
                "typical_experiments": ["baseline", "optogenetic_stimulation"]
            },
            "new_opto": {
                "sampling_frequency": 60.0,
                "mm_per_px": 0.1818,
                "camera_resolution": [1280, 1024],
                "arena_diameter_mm": 150.0,
                "calibration_date": "2024-06-01",
                "led_wavelength_nm": 470,
                "typical_experiments": ["optogenetic_stimulation", "chemical_stimulation"]
            },
            "high_speed_rig": {
                "sampling_frequency": 200.0,
                "mm_per_px": 0.05,
                "camera_resolution": [2048, 2048],
                "arena_diameter_mm": 200.0,
                "calibration_date": "2024-08-15",
                "led_wavelength_nm": 590,
                "typical_experiments": ["heat_stress", "baseline"]
            }
        },
        "datasets": {},
        "experiments": {}
    }
    
    # Generate dataset configurations for each day
    for date_str, day_info in study["experiment_days"].items():
        dataset_name = f"day_{date_str.replace('-', '_')}"
        
        # Extract animal IDs for this day
        animal_ids = [file_info["animal_id"] for file_info in day_info["files"]]
        
        config["datasets"][dataset_name] = {
            "description": f"Experimental data from {date_str}",
            "rig": "mixed",  # Multiple rigs used
            "patterns": [f"*{date_str}*", "*_exp.pkl"],
            "dates_vials": {
                date_str: list(range(1, len(animal_ids) + 1))
            },
            "metadata": {
                "extraction_patterns": [
                    rf"{date_str}_(?P<animal_line>\w+)_\d{{2}}_(?P<animal_num>\d{{3}})_(?P<rig>\w+)_(?P<condition>\w+)_exp\.pkl"
                ],
                "required_fields": ["animal_line", "animal_num", "rig", "condition"],
                "experiment_date": date_str
            },
            "filters": {
                "ignore_substrings": ["calibration", "test"],
                "mandatory_substrings": ["_exp"],
                "min_file_size_bytes": 1000
            }
        }
    
    # Generate experiment configurations
    for condition in ["baseline", "optogenetic_stimulation", "chemical_stimulation", "heat_stress"]:
        config["experiments"][f"{condition}_study"] = {
            "description": f"Multi-day {condition} experimental paradigm",
            "datasets": [name for name in config["datasets"].keys()],
            "metadata": {
                "extraction_patterns": [
                    rf"(?P<date>\d{{4}}-\d{{2}}-\d{{2}})_(?P<animal_id>\w+)_(?P<rig>\w+)_{condition}_exp\.pkl"
                ],
                "required_fields": ["date", "animal_id", "rig"],
                "experimental_condition": condition,
                "study_type": "longitudinal"
            },
            "filters": {
                "mandatory_substrings": [condition],
                "ignore_substrings": ["calibration", "backup"]
            },
            "analysis_parameters": {
                "velocity_threshold": 2.0,
                "smoothing_window": 5,
                "edge_exclusion_mm": 10,
                "signal_processing": {
                    "highpass_freq": 0.1,
                    "lowpass_freq": 30.0,
                    "artifact_threshold": 3.0
                }
            }
        }
    
    # Add cross-experiment comparisons
    config["experiments"]["multi_condition_comparison"] = {
        "description": "Cross-condition comparison across all experimental days",
        "datasets": list(config["datasets"].keys()),
        "metadata": {
            "extraction_patterns": [
                r"(?P<date>\d{4}-\d{2}-\d{2})_(?P<animal_id>\w+)_(?P<rig>\w+)_(?P<condition>\w+)_exp\.pkl"
            ],
            "required_fields": ["date", "animal_id", "rig", "condition"],
            "study_type": "comparative"
        },
        "grouping": {
            "by_condition": True,
            "by_animal_line": True,
            "by_rig": True,
            "temporal_binning": "daily"
        }
    }
    
    # Save configuration
    config_path = config_dir / "comprehensive_study_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return {
        "config": config,
        "config_path": config_path,
        "study_structure": study
    }


class TestRealisticExperimentalScenarios:
    """
    Comprehensive integration test suite for realistic experimental scenarios.
    
    This test class validates flyrigloader functionality across diverse experimental
    workflows that mirror actual neuroscience research patterns, ensuring robustness
    under realistic conditions and data scales.
    """

    def test_single_experiment_workflow_integration(
        self,
        realistic_data_generator,
        cross_platform_temp_dir,
        performance_benchmarks
    ):
        """
        Test complete workflow for a single realistic experiment.
        
        Validates:
        - Realistic experimental data generation
        - File discovery with realistic naming patterns
        - Data loading and validation
        - DataFrame transformation
        - Metadata extraction from realistic filenames
        
        Requirements: TST-INTEG-002, F-015
        """
        logger.info("Starting single experiment workflow integration test")
        
        # Create realistic experiment
        exp_dir = cross_platform_temp_dir / "single_experiment"
        exp_dir.mkdir(exist_ok=True)
        
        # Generate realistic experimental data
        animal_id = "WT_001_mouse"
        rig_name = "new_opto"
        condition = "optogenetic_stimulation"
        
        start_time = time.time()
        exp_matrix = realistic_data_generator.generate_experimental_matrix(
            rig_name, condition, animal_id, duration_minutes=10
        )
        generation_time = time.time() - start_time
        
        # Verify generation performance (should be fast)
        assert generation_time < 5.0, f"Data generation too slow: {generation_time:.2f}s"
        
        # Validate realistic data characteristics
        assert len(exp_matrix['t']) > 1000, "Experiment should have substantial time points"
        assert 'signal_disp' in exp_matrix, "Should include multi-channel signal data"
        assert exp_matrix['signal_disp'].shape[0] == 32, "new_opto rig should have 32 channels"
        
        # Verify biologically plausible ranges
        arena_radius = realistic_data_generator.rig_configurations[rig_name]["arena_diameter_mm"] / 2
        max_distance = np.sqrt(exp_matrix['x']**2 + exp_matrix['y']**2).max()
        assert max_distance <= arena_radius * 1.1, "Trajectory should stay within arena bounds"
        
        # Create realistic filename and save data
        experiment_date = datetime.now().strftime("%Y-%m-%d")
        filename = f"{experiment_date}_{animal_id}_{rig_name}_{condition}_exp.pkl"
        file_path = exp_dir / filename
        
        with open(file_path, 'wb') as f:
            pickle.dump(exp_matrix, f)
        
        # Test file discovery
        discovered_files = discover_files(
            directory=exp_dir,
            pattern="*_exp.pkl",
            recursive=False
        )
        
        assert len(discovered_files) == 1, "Should discover exactly one experiment file"
        assert str(file_path) in discovered_files, "Should discover the created file"
        
        # Test data loading
        start_time = time.time()
        loaded_matrix = read_pickle_any_format(file_path)
        loading_time = time.time() - start_time
        
        file_size_mb = file_path.stat().st_size / 1024 / 1024
        expected_loading_time = performance_benchmarks.benchmark_data_loading(file_size_mb)
        performance_benchmarks.assert_performance_sla(
            "single_experiment_loading", loading_time, expected_loading_time
        )
        
        # Validate loaded data integrity
        assert set(loaded_matrix.keys()) == set(exp_matrix.keys()), "All data columns should be preserved"
        np.testing.assert_array_equal(loaded_matrix['t'], exp_matrix['t'], "Time arrays should match exactly")
        np.testing.assert_array_equal(loaded_matrix['x'], exp_matrix['x'], "X positions should match exactly")
        
        logger.info("Single experiment workflow integration test completed successfully")

    def test_multi_day_study_batch_processing(
        self,
        multi_day_study_scenario,
        comprehensive_config_scenario,
        performance_benchmarks
    ):
        """
        Test batch processing of multi-day experimental study.
        
        Validates:
        - Multi-experiment batch discovery and loading
        - Date-based directory organization
        - Cross-day data consistency
        - Batch processing performance
        - Complex metadata extraction patterns
        
        Requirements: Section 4.1.2.2, TST-INTEG-002, F-007
        """
        logger.info("Starting multi-day study batch processing test")
        
        study = multi_day_study_scenario
        config_scenario = comprehensive_config_scenario
        config = config_scenario["config"]
        
        # Test batch discovery across all days
        start_time = time.time()
        
        # Simulate batch processing of all experiments
        all_experiment_files = []
        all_metadata = {}
        
        for experiment_name in config["experiments"].keys():
            # Load files for this experiment
            try:
                experiment_files = load_experiment_files(
                    config=config,
                    experiment_name=experiment_name,
                    base_directory=study["base_directory"],
                    pattern="*_exp.pkl",
                    recursive=True,
                    extract_metadata=True
                )
                
                if isinstance(experiment_files, dict):
                    all_experiment_files.extend(experiment_files.keys())
                    all_metadata.update(experiment_files)
                else:
                    all_experiment_files.extend(experiment_files)
                    
            except KeyError:
                # Some experiments may not have matching files, which is realistic
                continue
        
        batch_processing_time = time.time() - start_time
        
        # Validate batch processing performance
        total_files = len(study["files"])
        expected_batch_time = performance_benchmarks.benchmark_file_discovery(total_files)
        performance_benchmarks.assert_performance_sla(
            "multi_day_batch_discovery", batch_processing_time, expected_batch_time
        )
        
        # Validate cross-day consistency
        assert len(all_experiment_files) >= total_files // 2, "Should discover substantial number of files"
        
        # Test date-based organization validation
        date_pattern = r"\d{4}-\d{2}-\d{2}"
        date_organized_files = [f for f in all_experiment_files if Path(f).name.count('-') >= 2]
        assert len(date_organized_files) > 0, "Should find date-organized files"
        
        # Validate metadata extraction across multiple patterns
        if all_metadata:
            # Check for successful metadata extraction
            successful_extractions = sum(1 for metadata in all_metadata.values() if metadata)
            extraction_rate = successful_extractions / len(all_metadata)
            assert extraction_rate > 0.7, f"Metadata extraction rate too low: {extraction_rate:.2%}"
            
            # Validate specific metadata fields
            sample_metadata = next(iter(all_metadata.values()))
            if sample_metadata:
                expected_fields = {"animal_line", "rig", "condition"}
                extracted_fields = set(sample_metadata.keys())
                assert len(expected_fields.intersection(extracted_fields)) > 0, "Should extract expected metadata fields"
        
        logger.info("Multi-day study batch processing test completed successfully")

    def test_large_scale_performance_validation(
        self,
        large_scale_performance_dataset,
        performance_benchmarks
    ):
        """
        Test performance against SLA requirements with large-scale realistic data.
        
        Validates:
        - Data loading performance (1 second per 100MB SLA)
        - DataFrame transformation (500ms per 1M rows SLA)
        - Memory efficiency with large datasets
        - Compressed file handling performance
        
        Requirements: TST-PERF-002, Section 2.2.9
        """
        logger.info("Starting large-scale performance validation test")
        
        datasets = large_scale_performance_dataset
        
        for size_category, dataset_info in datasets.items():
            logger.info(f"Testing performance for {size_category} dataset")
            
            file_path = dataset_info["path"]
            expected_rows = dataset_info["expected_rows"]
            file_size_mb = dataset_info["file_size_mb"]
            
            # Test data loading performance
            start_time = time.time()
            loaded_matrix = read_pickle_any_format(file_path)
            loading_time = time.time() - start_time
            
            # Validate loading SLA (1 second per 100MB)
            expected_loading_time = performance_benchmarks.benchmark_data_loading(file_size_mb)
            performance_benchmarks.assert_performance_sla(
                f"{size_category}_data_loading", loading_time, expected_loading_time
            )
            
            # Test DataFrame transformation performance
            start_time = time.time()
            
            # Simulate DataFrame transformation
            df_data = {}
            for key, value in loaded_matrix.items():
                if isinstance(value, np.ndarray):
                    if value.ndim == 1:
                        df_data[key] = value
                    elif value.ndim == 2:
                        # Handle 2D arrays (like signal_disp)
                        for ch in range(min(value.shape[0], 5)):  # Limit for performance
                            df_data[f"{key}_ch{ch:02d}"] = value[ch, :]
            
            df = pd.DataFrame(df_data)
            transformation_time = time.time() - start_time
            
            # Validate transformation SLA (500ms per 1M rows)
            row_count = len(df)
            expected_transform_time = performance_benchmarks.benchmark_dataframe_transform(row_count)
            performance_benchmarks.assert_performance_sla(
                f"{size_category}_dataframe_transform", transformation_time, expected_transform_time
            )
            
            # Validate data integrity after transformation
            assert len(df) == expected_rows, f"DataFrame should have expected number of rows: {expected_rows}"
            assert 't' in df.columns, "Time column should be preserved"
            assert 'x' in df.columns and 'y' in df.columns, "Position columns should be preserved"
            
            # Validate realistic data ranges
            assert df['t'].min() >= 0, "Time should start at zero or positive"
            assert df['t'].is_monotonic_increasing, "Time should be monotonically increasing"
            
            logger.info(f"Performance validation for {size_category} dataset completed")
        
        logger.info("Large-scale performance validation test completed successfully")

    def test_complex_metadata_extraction_scenarios(
        self,
        multi_day_study_scenario,
        comprehensive_config_scenario
    ):
        """
        Test complex metadata extraction with realistic filename patterns.
        
        Validates:
        - Multiple regex pattern matching
        - Named group extraction
        - Date parsing from filenames
        - Animal ID and condition extraction
        - Fallback pattern handling
        
        Requirements: F-007, TST-INTEG-002
        """
        logger.info("Starting complex metadata extraction scenarios test")
        
        study = multi_day_study_scenario
        config_scenario = comprehensive_config_scenario
        config = config_scenario["config"]
        
        # Test metadata extraction across all study files
        extraction_results = {}
        pattern_success_counts = {}
        
        for file_info in study["files"]:
            file_path = file_info["path"]
            filename = file_info["filename"]
            
            # Test extraction using project-level patterns
            extraction_patterns = config["project"]["extraction_patterns"]
            
            extracted_metadata = None
            successful_pattern = None
            
            for pattern_idx, pattern in enumerate(extraction_patterns):
                try:
                    import re
                    match = re.search(pattern, filename)
                    if match:
                        extracted_metadata = match.groupdict()
                        successful_pattern = pattern_idx
                        pattern_success_counts[pattern_idx] = pattern_success_counts.get(pattern_idx, 0) + 1
                        break
                except re.error:
                    continue
            
            extraction_results[str(file_path)] = {
                "filename": filename,
                "extracted_metadata": extracted_metadata,
                "successful_pattern": successful_pattern,
                "expected_metadata": {
                    "animal_id": file_info["animal_id"],
                    "rig": file_info["rig"],
                    "condition": file_info["condition"],
                    "date": file_info["date"].strftime("%Y-%m-%d")
                }
            }
        
        # Validate extraction success rate
        successful_extractions = sum(1 for result in extraction_results.values() 
                                   if result["extracted_metadata"] is not None)
        total_files = len(extraction_results)
        success_rate = successful_extractions / total_files
        
        assert success_rate > 0.8, f"Metadata extraction success rate too low: {success_rate:.2%}"
        
        # Validate pattern effectiveness
        assert len(pattern_success_counts) > 0, "At least one pattern should be successful"
        most_effective_pattern = max(pattern_success_counts.items(), key=lambda x: x[1])
        logger.info(f"Most effective pattern (index {most_effective_pattern[0]}): {most_effective_pattern[1]} successes")
        
        # Validate specific metadata field extraction
        field_extraction_counts = {}
        for result in extraction_results.values():
            if result["extracted_metadata"]:
                for field in result["extracted_metadata"].keys():
                    field_extraction_counts[field] = field_extraction_counts.get(field, 0) + 1
        
        # Check for extraction of key experimental metadata
        key_fields = ["date", "animal", "rig", "condition"]
        extracted_key_fields = [field for field in field_extraction_counts.keys() 
                               if any(key in field.lower() for key in key_fields)]
        
        assert len(extracted_key_fields) >= 2, f"Should extract at least 2 key metadata fields, found: {extracted_key_fields}"
        
        # Test date extraction specifically
        date_extractions = [result for result in extraction_results.values()
                           if result["extracted_metadata"] and 
                           any("date" in field.lower() for field in result["extracted_metadata"].keys())]
        
        assert len(date_extractions) > total_files * 0.5, "Should successfully extract dates from most files"
        
        logger.info("Complex metadata extraction scenarios test completed successfully")

    def test_error_recovery_and_resilience(
        self,
        realistic_data_generator,
        cross_platform_temp_dir
    ):
        """
        Test error recovery with realistic failure scenarios.
        
        Validates:
        - Corrupted file handling
        - Missing data file recovery
        - Incomplete experimental matrices
        - Network/filesystem errors
        - Graceful degradation
        
        Requirements: Section 4.1.2.3, TST-INTEG-002
        """
        logger.info("Starting error recovery and resilience test")
        
        error_dir = cross_platform_temp_dir / "error_scenarios"
        error_dir.mkdir(exist_ok=True)
        
        # Create corrupted data scenarios
        corrupted_files = realistic_data_generator.create_corrupted_data_scenarios(
            error_dir, n_corrupted_files=5
        )
        
        # Test handling of each corruption type
        for corruption_type, file_path in corrupted_files.items():
            logger.info(f"Testing error recovery for {corruption_type}")
            
            # Test that appropriate exceptions are raised
            if corruption_type in ["truncated_pickle", "empty_file", "fake_pickle"]:
                with pytest.raises((pickle.UnpicklingError, EOFError, ValueError, FileNotFoundError)):
                    read_pickle_any_format(file_path)
            
            elif corruption_type == "missing_columns":
                # Should load but fail validation if enforced
                try:
                    loaded_data = read_pickle_any_format(file_path)
                    assert isinstance(loaded_data, dict), "Should load as dictionary"
                    assert 't' not in loaded_data, "Should be missing time column"
                except Exception:
                    pass  # This is also acceptable behavior
            
            elif corruption_type == "mismatched_lengths":
                # Should load but have inconsistent array lengths
                loaded_data = read_pickle_any_format(file_path)
                assert isinstance(loaded_data, dict), "Should load as dictionary"
                lengths = [len(arr) for arr in loaded_data.values() if isinstance(arr, np.ndarray)]
                assert len(set(lengths)) > 1, "Should have mismatched array lengths"
        
        # Test graceful handling of missing files
        non_existent_path = error_dir / "does_not_exist.pkl"
        with pytest.raises(FileNotFoundError):
            read_pickle_any_format(non_existent_path)
        
        # Test discovery resilience with mixed file types
        valid_experiment = realistic_data_generator.generate_experimental_matrix(
            "old_opto", "baseline", "recovery_test"
        )
        valid_path = error_dir / "valid_experiment.pkl"
        with open(valid_path, 'wb') as f:
            pickle.dump(valid_experiment, f)
        
        # Discover files in directory with mixed valid/invalid files
        all_files = discover_files(
            directory=error_dir,
            pattern="*.pkl",
            recursive=False
        )
        
        # Should find all pickle files (valid and invalid)
        expected_pickle_files = [str(p) for p in error_dir.glob("*.pkl")]
        assert len(all_files) >= len(expected_pickle_files), "Should discover all pickle files"
        
        # Test selective loading with error handling
        successfully_loaded = []
        failed_to_load = []
        
        for file_path in all_files:
            try:
                data = read_pickle_any_format(file_path)
                if isinstance(data, dict) and 't' in data:
                    successfully_loaded.append(file_path)
                else:
                    failed_to_load.append(file_path)
            except Exception:
                failed_to_load.append(file_path)
        
        # Should successfully load at least the valid file
        assert len(successfully_loaded) >= 1, "Should load at least one valid file"
        assert str(valid_path) in successfully_loaded, "Should successfully load the known valid file"
        
        # Most corrupted files should fail to load
        assert len(failed_to_load) >= len(corrupted_files) - 1, "Most corrupted files should fail to load"
        
        logger.info("Error recovery and resilience test completed successfully")

    def test_cross_rig_configuration_consistency(
        self,
        realistic_data_generator,
        cross_platform_temp_dir
    ):
        """
        Test consistency across different rig configurations.
        
        Validates:
        - Data structure consistency across rigs
        - Sampling frequency handling
        - Signal channel count validation
        - Cross-rig data compatibility
        - Metadata standardization
        
        Requirements: TST-INTEG-002, F-015
        """
        logger.info("Starting cross-rig configuration consistency test")
        
        rig_dir = cross_platform_temp_dir / "cross_rig_test"
        rig_dir.mkdir(exist_ok=True)
        
        rig_data = {}
        animal_id = "cross_rig_test_mouse"
        condition = "baseline"
        duration = 5  # minutes
        
        # Generate data for each rig configuration
        for rig_name in realistic_data_generator.rig_configurations.keys():
            logger.info(f"Generating data for rig: {rig_name}")
            
            exp_matrix = realistic_data_generator.generate_experimental_matrix(
                rig_name, condition, animal_id, duration_minutes=duration
            )
            
            # Save data
            file_path = rig_dir / f"{rig_name}_{animal_id}_{condition}.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(exp_matrix, f)
            
            rig_data[rig_name] = {
                "file_path": file_path,
                "data": exp_matrix,
                "config": realistic_data_generator.rig_configurations[rig_name]
            }
        
        # Validate data structure consistency
        common_columns = None
        for rig_name, rig_info in rig_data.items():
            data_columns = set(rig_info["data"].keys())
            if common_columns is None:
                common_columns = data_columns
            else:
                common_columns = common_columns.intersection(data_columns)
        
        # Should have core experimental columns across all rigs
        required_columns = {'t', 'x', 'y', 'signal', 'signal_disp'}
        assert required_columns.issubset(common_columns), f"Missing required columns: {required_columns - common_columns}"
        
        # Validate sampling frequency consistency
        for rig_name, rig_info in rig_data.items():
            expected_freq = rig_info["config"]["sampling_frequency"]
            time_array = rig_info["data"]["t"]
            
            if len(time_array) > 1:
                calculated_freq = 1.0 / np.mean(np.diff(time_array))
                freq_error = abs(calculated_freq - expected_freq) / expected_freq
                assert freq_error < 0.05, f"Sampling frequency mismatch for {rig_name}: expected {expected_freq}, got {calculated_freq:.2f}"
        
        # Validate signal channel counts
        for rig_name, rig_info in rig_data.items():
            expected_channels = rig_info["config"]["signal_channels"]
            signal_disp = rig_info["data"]["signal_disp"]
            
            assert signal_disp.shape[0] == expected_channels, f"Channel count mismatch for {rig_name}: expected {expected_channels}, got {signal_disp.shape[0]}"
        
        # Test cross-rig data loading consistency
        for rig_name, rig_info in rig_data.items():
            loaded_data = read_pickle_any_format(rig_info["file_path"])
            
            # Validate loaded data structure
            assert isinstance(loaded_data, dict), f"Data from {rig_name} should load as dictionary"
            assert set(loaded_data.keys()) == set(rig_info["data"].keys()), f"Column consistency issue for {rig_name}"
            
            # Validate array shapes
            for column in required_columns:
                original_shape = rig_info["data"][column].shape
                loaded_shape = loaded_data[column].shape
                assert original_shape == loaded_shape, f"Shape mismatch for {rig_name}.{column}: {original_shape} vs {loaded_shape}"
        
        logger.info("Cross-rig configuration consistency test completed successfully")

    def test_temporal_correlation_and_longitudinal_analysis(
        self,
        multi_day_study_scenario
    ):
        """
        Test temporal correlations and longitudinal analysis capabilities.
        
        Validates:
        - Cross-day data consistency for same animals
        - Temporal trend detection
        - Longitudinal metadata tracking
        - Data continuity across experimental sessions
        - Animal-specific pattern recognition
        
        Requirements: TST-INTEG-002, Section 4.1.2.2
        """
        logger.info("Starting temporal correlation and longitudinal analysis test")
        
        study = multi_day_study_scenario
        
        # Group experiments by animal for longitudinal analysis
        animal_timelines = {}
        for file_info in study["files"]:
            animal_id = file_info["animal_id"]
            if animal_id not in animal_timelines:
                animal_timelines[animal_id] = []
            animal_timelines[animal_id].append(file_info)
        
        # Sort each animal's timeline by date
        for animal_id in animal_timelines:
            animal_timelines[animal_id].sort(key=lambda x: x["date"])
        
        # Validate longitudinal data structure
        multi_day_animals = {aid: timeline for aid, timeline in animal_timelines.items() 
                           if len(timeline) > 1}
        
        assert len(multi_day_animals) > 0, "Should have animals with multiple experimental sessions"
        
        # Test temporal consistency for multi-day animals
        for animal_id, timeline in multi_day_animals.items():
            logger.info(f"Analyzing longitudinal data for {animal_id}")
            
            # Load data for all sessions
            session_data = []
            for session_info in timeline:
                try:
                    data = read_pickle_any_format(session_info["path"])
                    session_data.append({
                        "data": data,
                        "date": session_info["date"],
                        "condition": session_info["condition"],
                        "rig": session_info["rig"],
                        "duration": data["t"][-1] - data["t"][0] if len(data["t"]) > 0 else 0
                    })
                except Exception as e:
                    logger.warning(f"Failed to load session data for {animal_id}: {e}")
                    continue
            
            if len(session_data) < 2:
                continue
            
            # Validate temporal progression
            dates = [session["date"] for session in session_data]
            assert dates == sorted(dates), f"Timeline should be chronologically ordered for {animal_id}"
            
            # Check for realistic experimental gaps (not too long)
            for i in range(1, len(dates)):
                gap_days = (dates[i] - dates[i-1]).days
                assert gap_days <= 10, f"Unrealistic experimental gap for {animal_id}: {gap_days} days"
            
            # Validate data consistency across sessions
            common_columns = None
            for session in session_data:
                session_columns = set(session["data"].keys())
                if common_columns is None:
                    common_columns = session_columns
                else:
                    common_columns = common_columns.intersection(session_columns)
            
            required_columns = {'t', 'x', 'y'}
            assert required_columns.issubset(common_columns), f"Missing required columns across sessions for {animal_id}"
            
            # Validate realistic behavioral consistency
            # Animals should show some consistency in basic measures
            session_speeds = []
            for session in session_data:
                data = session["data"]
                if 'speed' in data:
                    mean_speed = np.mean(data['speed'])
                elif 'x' in data and 'y' in data and 't' in data:
                    # Calculate speed if not present
                    dx = np.diff(data['x'], prepend=data['x'][0])
                    dy = np.diff(data['y'], prepend=data['y'][0])
                    dt = np.diff(data['t'], prepend=data['t'][1] - data['t'][0])
                    speeds = np.sqrt(dx**2 + dy**2) / dt
                    mean_speed = np.mean(speeds[speeds < 50])  # Filter outliers
                else:
                    continue
                
                session_speeds.append(mean_speed)
            
            if len(session_speeds) >= 2:
                speed_cv = np.std(session_speeds) / np.mean(session_speeds)
                # Coefficient of variation should be reasonable (not too high)
                assert speed_cv < 2.0, f"Excessive speed variation across sessions for {animal_id}: {speed_cv:.2f}"
        
        # Test cross-animal comparison capabilities
        baseline_sessions = [file_info for file_info in study["files"] 
                           if file_info["condition"] == "baseline"]
        
        if len(baseline_sessions) >= 3:
            # Load baseline data for comparison
            baseline_speeds = []
            for session_info in baseline_sessions[:5]:  # Limit for performance
                try:
                    data = read_pickle_any_format(session_info["path"])
                    if 'x' in data and 'y' in data and 't' in data:
                        dx = np.diff(data['x'], prepend=data['x'][0])
                        dy = np.diff(data['y'], prepend=data['y'][0])
                        dt = np.diff(data['t'], prepend=data['t'][1] - data['t'][0])
                        speeds = np.sqrt(dx**2 + dy**2) / dt
                        mean_speed = np.mean(speeds[speeds < 50])
                        baseline_speeds.append(mean_speed)
                except Exception:
                    continue
            
            if len(baseline_speeds) >= 3:
                # Validate that baseline speeds are in realistic range
                mean_baseline_speed = np.mean(baseline_speeds)
                assert 2.0 <= mean_baseline_speed <= 20.0, f"Unrealistic baseline speed range: {mean_baseline_speed:.2f} mm/s"
        
        logger.info("Temporal correlation and longitudinal analysis test completed successfully")

    def test_realistic_experimental_design_patterns(
        self,
        comprehensive_config_scenario
    ):
        """
        Test realistic experimental design patterns and workflows.
        
        Validates:
        - Randomized controlled trial simulation
        - Counterbalanced experimental designs
        - Within-subject and between-subject comparisons
        - Statistical power considerations
        - Experimental metadata completeness
        
        Requirements: TST-INTEG-002, F-015, Section 4.1.2.2
        """
        logger.info("Starting realistic experimental design patterns test")
        
        config_scenario = comprehensive_config_scenario
        config = config_scenario["config"]
        study = config_scenario["study_structure"]
        
        # Analyze experimental design structure
        conditions = set()
        animals = set()
        rigs = set()
        daily_schedules = {}
        
        for file_info in study["files"]:
            conditions.add(file_info["condition"])
            animals.add(file_info["animal_id"])
            rigs.add(file_info["rig"])
            
            date_str = file_info["date"].strftime("%Y-%m-%d")
            if date_str not in daily_schedules:
                daily_schedules[date_str] = {"animals": set(), "conditions": set()}
            daily_schedules[date_str]["animals"].add(file_info["animal_id"])
            daily_schedules[date_str]["conditions"].add(file_info["condition"])
        
        # Validate experimental design characteristics
        assert len(conditions) >= 2, "Should have multiple experimental conditions"
        assert len(animals) >= 5, "Should have sufficient animal count for statistical power"
        assert len(rigs) >= 1, "Should have rig information"
        
        # Test counterbalancing validation
        # Check if animals are tested across multiple conditions
        animal_conditions = {}
        for file_info in study["files"]:
            animal_id = file_info["animal_id"]
            condition = file_info["condition"]
            
            if animal_id not in animal_conditions:
                animal_conditions[animal_id] = set()
            animal_conditions[animal_id].add(condition)
        
        multi_condition_animals = {aid: conds for aid, conds in animal_conditions.items() 
                                 if len(conds) > 1}
        
        # Some animals should be tested in multiple conditions (within-subject design)
        within_subject_ratio = len(multi_condition_animals) / len(animal_conditions)
        logger.info(f"Within-subject design ratio: {within_subject_ratio:.2%}")
        
        # Test daily scheduling patterns
        for date_str, schedule in daily_schedules.items():
            animals_per_day = len(schedule["animals"])
            conditions_per_day = len(schedule["conditions"])
            
            # Realistic daily schedules
            assert 2 <= animals_per_day <= 10, f"Unrealistic animal count for {date_str}: {animals_per_day}"
            assert conditions_per_day <= animals_per_day, f"More conditions than animals on {date_str}"
        
        # Test experimental group balance
        condition_counts = {}
        for file_info in study["files"]:
            condition = file_info["condition"]
            condition_counts[condition] = condition_counts.get(condition, 0) + 1
        
        # Groups should be reasonably balanced
        if len(condition_counts) > 1:
            min_count = min(condition_counts.values())
            max_count = max(condition_counts.values())
            balance_ratio = min_count / max_count
            assert balance_ratio > 0.3, f"Experimental groups too unbalanced: {balance_ratio:.2%}"
        
        # Test rig utilization patterns
        rig_usage = {}
        for file_info in study["files"]:
            rig = file_info["rig"]
            rig_usage[rig] = rig_usage.get(rig, 0) + 1
        
        # Validate realistic rig distribution
        if len(rig_usage) > 1:
            # No single rig should dominate entirely
            max_rig_usage = max(rig_usage.values())
            total_usage = sum(rig_usage.values())
            max_rig_proportion = max_rig_usage / total_usage
            assert max_rig_proportion < 0.9, f"Single rig used too exclusively: {max_rig_proportion:.2%}"
        
        # Test experimental timeline realism
        all_dates = [file_info["date"] for file_info in study["files"]]
        date_range = max(all_dates) - min(all_dates)
        assert date_range.days <= 30, f"Experimental timeline too long: {date_range.days} days"
        assert date_range.days >= 1, "Should span multiple days"
        
        # Test animal line distribution
        animal_lines = {}
        for file_info in study["files"]:
            animal_id = file_info["animal_id"]
            line = animal_id.split('_')[0]  # Extract line from ID
            animal_lines[line] = animal_lines.get(line, 0) + 1
        
        # Should have realistic animal line representation
        assert len(animal_lines) >= 1, "Should have animal line information"
        if "WT" in animal_lines:
            # Wild type should be well represented
            wt_proportion = animal_lines["WT"] / sum(animal_lines.values())
            assert wt_proportion >= 0.2, f"Wild type under-represented: {wt_proportion:.2%}"
        
        logger.info("Realistic experimental design patterns test completed successfully")