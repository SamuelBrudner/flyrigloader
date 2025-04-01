"""
Tests for config-aware file discovery functionality.
"""
import os
import tempfile
import pytest
import yaml
from pathlib import Path

# Import the functionality we want to test
from flyrigloader.config.discovery import (
    discover_files_with_config,
    discover_experiment_files,
    discover_dataset_files
)
from flyrigloader.config.yaml_config import load_config


class TestConfigDiscovery:
    
    @pytest.fixture
    def sample_config(self):
        """Create a sample configuration dictionary for testing."""
        return {
            "project": {
                "directories": {
                    "major_data_directory": "/path/to/data"
                },
                "ignore_substrings": [
                    "._",
                    "temp_"
                ]
            },
            "datasets": {
                "test_dataset": {
                    "rig": "test_rig",
                    "dates_vials": {
                        "2023-01-01": [1, 2],
                        "2023-01-02": [3, 4]
                    }
                }
            },
            "experiments": {
                "test_experiment": {
                    "datasets": ["test_dataset"],
                    "filters": {
                        "ignore_substrings": ["exclude_me"],
                        "mandatory_experiment_strings": ["include_me"]
                    }
                },
                "basic_experiment": {
                    "datasets": ["test_dataset"]
                }
            }
        }
    
    @pytest.fixture
    def test_directory_structure(self):
        """Create a temporary directory structure for testing discovery."""
        temp_dir = tempfile.mkdtemp()
        try:
            # Create date directories
            date1_dir = os.path.join(temp_dir, "2023-01-01")
            date2_dir = os.path.join(temp_dir, "2023-01-02")
            os.makedirs(date1_dir)
            os.makedirs(date2_dir)
            
            # Create files in date1_dir
            with open(os.path.join(date1_dir, "include_me_data1.csv"), "w") as f:
                f.write("data1")
            with open(os.path.join(date1_dir, "include_me_data2.csv"), "w") as f:
                f.write("data2")
            with open(os.path.join(date1_dir, "regular_data.csv"), "w") as f:
                f.write("regular data")
            with open(os.path.join(date1_dir, "exclude_me_data.csv"), "w") as f:
                f.write("excluded data")
            with open(os.path.join(date1_dir, "._hidden_file.txt"), "w") as f:
                f.write("hidden data")
            with open(os.path.join(date1_dir, "temp_notes.txt"), "w") as f:
                f.write("temp notes")
                
            # Create files in date2_dir
            with open(os.path.join(date2_dir, "include_me_data3.csv"), "w") as f:
                f.write("data3")
            with open(os.path.join(date2_dir, "include_me_exclude_me_data.csv"), "w") as f:
                f.write("mixed data")
            with open(os.path.join(date2_dir, "other_data.csv"), "w") as f:
                f.write("other data")
            
            yield temp_dir
        finally:
            # Clean up after the test
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_discover_files_with_config(self, sample_config, test_directory_structure):
        """Test discovering files using config-aware filtering."""
        # Test with project-level ignore patterns only
        files = discover_files_with_config(
            config=sample_config,
            directory=test_directory_structure,
            pattern="**/*.*",
            recursive=True
        )
        
        # Should exclude files with "._" and "temp_" but include everything else
        assert len(files) == 7
        assert not any("._" in f for f in files)
        assert not any("temp_" in f for f in files)
        
        # Test with experiment-specific filters
        files = discover_files_with_config(
            config=sample_config,
            directory=test_directory_structure,
            pattern="**/*.*",
            recursive=True,
            experiment="test_experiment"
        )
        
        # Should enforce both ignore patterns and mandatory substrings
        assert len(files) == 3
        assert all("include_me" in f for f in files)
        assert not any("exclude_me" in f for f in files)
        assert not any("._" in f for f in files)
        assert not any("temp_" in f for f in files)
    
    def test_discover_dataset_files(self, sample_config, test_directory_structure, monkeypatch):
        """Test discovering files for a specific dataset."""
        # Update the sample config to use our test directory
        sample_config["project"]["directories"]["major_data_directory"] = test_directory_structure
        
        # Test dataset file discovery
        files = discover_dataset_files(
            config=sample_config,
            dataset_name="test_dataset",
            base_directory=test_directory_structure
        )
        
        # Should find files in both date directories, excluding ignored patterns
        assert len(files) == 7
        assert any("2023-01-01" in f for f in files)
        assert any("2023-01-02" in f for f in files)
        assert not any("._" in f for f in files)
        assert not any("temp_" in f for f in files)
        
        # Test with extension filtering
        files = discover_dataset_files(
            config=sample_config,
            dataset_name="test_dataset",
            base_directory=test_directory_structure,
            extensions=["csv"]
        )
        
        # Should only find CSV files
        assert len(files) == 7
        assert all(f.endswith(".csv") for f in files)
    
    def test_discover_experiment_files(self, sample_config, test_directory_structure, monkeypatch):
        """Test discovering files for a specific experiment."""
        # Update the sample config to use our test directory
        sample_config["project"]["directories"]["major_data_directory"] = test_directory_structure
        
        # Test experiment file discovery with filters
        files = discover_experiment_files(
            config=sample_config,
            experiment_name="test_experiment",
            base_directory=test_directory_structure
        )
        
        # Should use experiment-specific filters
        assert len(files) == 3
        assert all("include_me" in f for f in files)
        assert not any("exclude_me" in f for f in files)
        
        # Test with a basic experiment (no specific filters)
        files = discover_experiment_files(
            config=sample_config,
            experiment_name="basic_experiment",
            base_directory=test_directory_structure,
            extensions=["csv"]
        )
        
        # Should only apply project-level filters
        assert len(files) == 7
        assert all(f.endswith(".csv") for f in files)
        assert not any("._" in f for f in files)
        assert not any("temp_" in f for f in files)
