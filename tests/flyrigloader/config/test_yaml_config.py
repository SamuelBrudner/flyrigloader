"""
Tests for YAML configuration handling functionality.
"""
import os
import tempfile
import pytest
import yaml
from pathlib import Path

# Import the functionality we want to test
# (We'll implement this after the tests)
from flyrigloader.config.yaml_config import (
    load_config,
    get_ignore_patterns,
    get_mandatory_substrings,
    get_dataset_info,
    get_experiment_info
)


class TestYamlConfig:
    
    @pytest.fixture
    def sample_config_file(self):
        """Create a temporary config file with sample configuration."""
        temp_dir = tempfile.mkdtemp()
        try:
            # Create a sample configuration file
            config_path = os.path.join(temp_dir, "config.yaml")
            
            # Sample config structure that mimics our base_config.yaml
            config_data = {
                "project": {
                    "directories": {
                        "major_data_directory": "/path/to/data",
                        "batchfile_directory": "/path/to/batch_defs"
                    },
                    "ignore_substrings": [
                        "static_horiz_ribbon",
                        "._"
                    ],
                    "nonstandard_folders": [
                        "2023-4-28",
                        "2023-4-27"
                    ]
                },
                "rigs": {
                    "old_opto": {
                        "sampling_frequency": 60,
                        "mm_per_px": 0.154
                    },
                    "new_opto": {
                        "sampling_frequency": 60,
                        "mm_per_px": 0.1818
                    }
                },
                "datasets": {
                    "no_green_light": {
                        "rig": "old_opto",
                        "dates_vials": {
                            "2024-12-20": [2],
                            "2024-12-22": [1, 2],
                            "2024-12-30": [1, 2, 3]
                        }
                    },
                    "plume_movie_navigation": {
                        "rig": "old_opto",
                        "dates_vials": {
                            "2024-10-18": [1, 3, 4, 5, 6, 7, 8],
                            "2024-10-24": [1, 2]
                        }
                    }
                },
                "experiments": {
                    "no_green_light_expt": {
                        "datasets": ["no_green_light"]
                    },
                    "first_nirag_expt": {
                        "datasets": ["nirag_submission_1"],
                        "analysis_params": {
                            "upwind": 180,
                            "clustering": {
                                "dbscan_eps": 0.1,
                                "dbscan_min_samples": 900,
                                "turn_cluster_id": 0
                            }
                        }
                    },
                    "second_nirag_expt": {
                        "datasets": ["nirag_submission_2"],
                        "filters": {
                            "mandatory_experiment_strings": ["smoke_2a"]
                        },
                        "analysis_params": {
                            "upwind": 180
                        }
                    },
                    "multi_plume": {
                        "datasets": [
                            "my_original_nagel_data",
                            "no_green_light"
                        ],
                        "filters": {
                            "ignore_substrings": ["smoke_2a"]
                        }
                    }
                }
            }
            
            # Write the config to the file
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            yield config_path
        finally:
            # Clean up after the test
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_load_config(self, sample_config_file):
        """Test basic config loading functionality."""
        # Load the configuration
        config = load_config(sample_config_file)
        
        # Verify the structure is loaded correctly
        assert "project" in config
        assert "rigs" in config
        assert "datasets" in config
        assert "experiments" in config
        
        # Check a few nested values
        assert config["project"]["directories"]["major_data_directory"] == "/path/to/data"
        assert "old_opto" in config["rigs"]
        assert config["rigs"]["old_opto"]["sampling_frequency"] == 60
        
        # Check datasets
        assert "no_green_light" in config["datasets"]
        assert config["datasets"]["no_green_light"]["rig"] == "old_opto"
        
        # Check experiments
        assert "multi_plume" in config["experiments"]
        assert "no_green_light" in config["experiments"]["multi_plume"]["datasets"]
    
    def test_get_ignore_patterns(self, sample_config_file):
        """Test extracting ignore patterns from config."""
        config = load_config(sample_config_file)
        
        # Get project-level ignore patterns
        project_patterns = get_ignore_patterns(config)
        assert "static_horiz_ribbon" in project_patterns
        assert "._" in project_patterns
        
        # Get experiment-level ignore patterns (inherits project patterns)
        multi_plume_patterns = get_ignore_patterns(config, experiment="multi_plume")
        assert "static_horiz_ribbon" in multi_plume_patterns
        assert "._" in multi_plume_patterns
        assert "smoke_2a" in multi_plume_patterns  # Experiment-specific pattern
        
        # Experiment without specific patterns should just return project patterns
        first_nirag_patterns = get_ignore_patterns(config, experiment="first_nirag_expt")
        assert "static_horiz_ribbon" in first_nirag_patterns
        assert "._" in first_nirag_patterns
        assert "smoke_2a" not in first_nirag_patterns
    
    def test_get_mandatory_substrings(self, sample_config_file):
        """Test extracting mandatory substrings from config."""
        config = load_config(sample_config_file)
        
        # Get project-level mandatory substrings (none defined in sample)
        project_strings = get_mandatory_substrings(config)
        assert len(project_strings) == 0
        
        # Get experiment-level mandatory substrings
        second_nirag_strings = get_mandatory_substrings(config, experiment="second_nirag_expt")
        assert "smoke_2a" in second_nirag_strings
        
        # Experiment without specific mandatory substrings should return empty list
        multi_plume_strings = get_mandatory_substrings(config, experiment="multi_plume")
        assert len(multi_plume_strings) == 0
    
    def test_get_dataset_info(self, sample_config_file):
        """Test extracting dataset information from config."""
        config = load_config(sample_config_file)
        
        # Get info for a specific dataset
        dataset_info = get_dataset_info(config, "no_green_light")
        assert dataset_info["rig"] == "old_opto"
        assert "2024-12-20" in dataset_info["dates_vials"]
        assert 2 in dataset_info["dates_vials"]["2024-12-20"]
        
        # Test with non-existent dataset - should raise KeyError
        with pytest.raises(KeyError):
            get_dataset_info(config, "non_existent_dataset")
    
    def test_get_experiment_info(self, sample_config_file):
        """Test extracting experiment information from config."""
        config = load_config(sample_config_file)
        
        # Get info for a specific experiment
        experiment_info = get_experiment_info(config, "multi_plume")
        assert "datasets" in experiment_info
        assert "my_original_nagel_data" in experiment_info["datasets"]
        assert "no_green_light" in experiment_info["datasets"]
        assert "filters" in experiment_info
        assert "ignore_substrings" in experiment_info["filters"]
        
        # Test with non-existent experiment - should raise KeyError
        with pytest.raises(KeyError):
            get_experiment_info(config, "non_existent_experiment")
