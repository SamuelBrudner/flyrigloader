"""
Tests for basic file discovery functionality.
"""
import os
import tempfile
import pytest
from pathlib import Path
from typing import List


# Import the function we want to test
from flyrigloader.discovery.files import discover_files


class TestFileDiscovery:
    
    @pytest.fixture
    def test_dir(self):
        """Create a temporary directory with test files."""
        temp_dir = tempfile.mkdtemp()
        try:
            # Create test files without using loops
            file_paths = [
                os.path.join(temp_dir, "test_0.txt"),
                os.path.join(temp_dir, "test_1.txt"),
                os.path.join(temp_dir, "test_2.txt")
            ]
            
            # Write content to each file
            with open(file_paths[0], "w") as f:
                f.write("Test content 0")
            with open(file_paths[1], "w") as f:
                f.write("Test content 1")
            with open(file_paths[2], "w") as f:
                f.write("Test content 2")
            
            yield temp_dir
        finally:
            # Clean up after the test
            import shutil
            shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def nested_test_dir(self):
        """Create a temporary directory with nested subdirectories and test files."""
        temp_dir = tempfile.mkdtemp()
        subdir = os.path.join(temp_dir, "subdir")
        deepdir = os.path.join(subdir, "deepdir")
        
        try:
            # Create directories
            os.makedirs(subdir)
            os.makedirs(deepdir)
            
            # Create root files
            with open(os.path.join(temp_dir, "root_0.txt"), "w") as f:
                f.write("Root content 0")
            with open(os.path.join(temp_dir, "root_1.txt"), "w") as f:
                f.write("Root content 1")
            
            # Create subdir files
            with open(os.path.join(subdir, "sub_0.txt"), "w") as f:
                f.write("Subdir content 0")
            with open(os.path.join(subdir, "sub_1.txt"), "w") as f:
                f.write("Subdir content 1")
            with open(os.path.join(subdir, "sub_2.txt"), "w") as f:
                f.write("Subdir content 2")
            
            # Create deepdir files
            with open(os.path.join(deepdir, "deep_0.txt"), "w") as f:
                f.write("Deep content 0")
            with open(os.path.join(deepdir, "deep_1.txt"), "w") as f:
                f.write("Deep content 1")
            
            yield temp_dir
        finally:
            # Clean up after the test
            import shutil
            shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mixed_files_dir(self):
        """Create a temporary directory with mixed file types."""
        temp_dir = tempfile.mkdtemp()
        try:
            # Create text files
            with open(os.path.join(temp_dir, "data_0.txt"), "w") as f:
                f.write("Text data 0")
            with open(os.path.join(temp_dir, "data_1.txt"), "w") as f:
                f.write("Text data 1")
            
            # Create CSV files
            with open(os.path.join(temp_dir, "data_0.csv"), "w") as f:
                f.write("col1,col2\nvalue0a,value0b")
            with open(os.path.join(temp_dir, "data_1.csv"), "w") as f:
                f.write("col1,col2\nvalue1a,value1b")
            with open(os.path.join(temp_dir, "data_2.csv"), "w") as f:
                f.write("col1,col2\nvalue2a,value2b")
            
            # Create JSON files
            with open(os.path.join(temp_dir, "config_0.json"), "w") as f:
                f.write('{"name": "config0", "value": 0}')
            with open(os.path.join(temp_dir, "config_1.json"), "w") as f:
                f.write('{"name": "config1", "value": 1}')
            
            yield temp_dir
        finally:
            # Clean up after the test
            import shutil
            shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def multi_dir_setup(self):
        """Create multiple temporary directories with different file types."""
        # First directory with txt files
        dir1 = tempfile.mkdtemp()
        
        # Create txt files in dir1
        with open(os.path.join(dir1, "dir1_file_0.txt"), "w") as f:
            f.write("Dir1 content 0")
        with open(os.path.join(dir1, "dir1_file_1.txt"), "w") as f:
            f.write("Dir1 content 1")
        
        # Second directory with json files
        dir2 = tempfile.mkdtemp()
        
        # Create json files in dir2
        with open(os.path.join(dir2, "dir2_file_0.json"), "w") as f:
            f.write('{"name": "dir2_0", "value": 0}')
        with open(os.path.join(dir2, "dir2_file_1.json"), "w") as f:
            f.write('{"name": "dir2_1", "value": 1}')
        with open(os.path.join(dir2, "dir2_file_2.json"), "w") as f:
            f.write('{"name": "dir2_2", "value": 2}')
        
        try:
            yield (dir1, dir2)
        finally:
            # Clean up after the test
            import shutil
            shutil.rmtree(dir1)
            shutil.rmtree(dir2)
    
    @pytest.fixture
    def ignore_patterns_dir(self):
        """Create a directory with files that should be ignored based on patterns."""
        temp_dir = tempfile.mkdtemp()
        try:
            # Create standard files
            with open(os.path.join(temp_dir, "normal_file.txt"), "w") as f:
                f.write("Normal file content")
            with open(os.path.join(temp_dir, "important_data.csv"), "w") as f:
                f.write("important,data")
            
            # Create files that should be ignored
            with open(os.path.join(temp_dir, "._hidden_file.txt"), "w") as f:
                f.write("Hidden file that should be ignored")
            with open(os.path.join(temp_dir, "static_horiz_ribbon_data.csv"), "w") as f:
                f.write("static,horiz,ribbon,data")
            with open(os.path.join(temp_dir, "temp_smoke_2a_experiment.json"), "w") as f:
                f.write('{"type": "smoke_2a", "temp": true}')
            
            # Create a subdirectory with more files
            subdir = os.path.join(temp_dir, "subdir")
            os.makedirs(subdir)
            with open(os.path.join(subdir, "subdir_normal.txt"), "w") as f:
                f.write("Subdir normal file")
            with open(os.path.join(subdir, "._subdir_hidden.txt"), "w") as f:
                f.write("Subdir hidden file")
            
            yield temp_dir
        finally:
            # Clean up after the test
            import shutil
            shutil.rmtree(temp_dir)
            
    @pytest.fixture
    def mandatory_substrings_dir(self):
        """Create a directory with files for testing mandatory substrings."""
        temp_dir = tempfile.mkdtemp()
        try:
            # Create various experimental files
            with open(os.path.join(temp_dir, "experiment_smoke_2a_001.csv"), "w") as f:
                f.write("smoke_2a experiment data")
            with open(os.path.join(temp_dir, "experiment_smoke_2a_002.csv"), "w") as f:
                f.write("more smoke_2a experiment data")
            with open(os.path.join(temp_dir, "experiment_control_001.csv"), "w") as f:
                f.write("control experiment data")
            with open(os.path.join(temp_dir, "experiment_nagel_001.csv"), "w") as f:
                f.write("nagel experiment data")
            with open(os.path.join(temp_dir, "setup_notes.txt"), "w") as f:
                f.write("general experiment notes")
            
            # Create subdirectory with more files
            subdir = os.path.join(temp_dir, "subdir")
            os.makedirs(subdir)
            with open(os.path.join(subdir, "experiment_smoke_2a_003.csv"), "w") as f:
                f.write("nested smoke_2a experiment data")
            with open(os.path.join(subdir, "experiment_control_002.csv"), "w") as f:
                f.write("nested control experiment data")
            
            yield temp_dir
        finally:
            # Clean up after the test
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_basic_file_discovery(self, test_dir):
        """Test basic file discovery with a pattern."""
        # Call the function we want to test
        files = discover_files(test_dir, "*.txt")
        
        # Assert that we found the expected files
        assert len(files) == 3
        assert all(f.endswith(".txt") for f in files)
        
    def test_recursive_file_discovery(self, nested_test_dir):
        """Test recursive file discovery across nested directories."""
        # Call the function we want to test with recursive pattern
        files = discover_files(nested_test_dir, "**/*.txt", recursive=True)
        
        # Assert that we found all expected files
        assert len(files) == 7  # 2 in root + 3 in subdir + 2 in deepdir
        
        # Count files by type
        root_files = [f for f in files if "root_" in f]
        sub_files = [f for f in files if "sub_" in f]
        deep_files = [f for f in files if "deep_" in f]
        
        assert len(root_files) == 2
        assert len(sub_files) == 3
        assert len(deep_files) == 2
        
    def test_file_extension_filtering(self, mixed_files_dir):
        """Test filtering files by extension."""
        # Filter by single extension
        txt_files = discover_files(mixed_files_dir, "*", extensions=["txt"])
        assert len(txt_files) == 2
        assert all(f.endswith(".txt") for f in txt_files)
        
        # Filter by multiple extensions
        data_files = discover_files(mixed_files_dir, "*", extensions=["csv", "json"])
        assert len(data_files) == 5  # 3 csv + 2 json
        
        # Count files by type
        csv_files = [f for f in data_files if f.endswith(".csv")]
        json_files = [f for f in data_files if f.endswith(".json")]
        
        assert len(csv_files) == 3
        assert len(json_files) == 2
        
    def test_multi_directory_discovery(self, multi_dir_setup):
        """Test file discovery across multiple directories."""
        # Unpack the directories
        dir1, dir2 = multi_dir_setup
        
        # Call the function with multiple directories
        files = discover_files([dir1, dir2], "*.txt")
        
        # We should only find the 2 txt files from dir1
        assert len(files) == 2
        assert all(f.endswith(".txt") for f in files)
        assert all("dir1_file" in f for f in files)
        
        # Now search for json files which should come from dir2
        files = discover_files([dir1, dir2], "*.json")
        
        # We should find 3 json files from dir2
        assert len(files) == 3
        assert all(f.endswith(".json") for f in files)
        assert all("dir2_file" in f for f in files)
        
        # Search for all files using a wildcard
        files = discover_files([dir1, dir2], "*.*")
        
        # We should find all 5 files (2 txt + 3 json)
        assert len(files) == 5
        txt_files = [f for f in files if f.endswith(".txt")]
        json_files = [f for f in files if f.endswith(".json")]
        
        assert len(txt_files) == 2
        assert len(json_files) == 3
        
    def test_ignore_patterns(self, ignore_patterns_dir):
        """Test file discovery with ignore patterns."""
        # Test without any ignore patterns first - should find all files
        all_files = discover_files(ignore_patterns_dir, "**/*.*", recursive=True)
        assert len(all_files) == 7  # 5 in root + 2 in subdir
        
        # Test with single ignore pattern
        files = discover_files(
            ignore_patterns_dir, 
            "**/*.*", 
            recursive=True,
            ignore_patterns=["._"]
        )
        # Should exclude the 2 hidden files
        assert len(files) == 5
        assert all("._" not in f for f in files)
        
        # Test with multiple ignore patterns
        files = discover_files(
            ignore_patterns_dir, 
            "**/*.*", 
            recursive=True,
            ignore_patterns=["._", "static_horiz_ribbon", "smoke_2a"]
        )
        # Should exclude all 4 files matching patterns
        assert len(files) == 3
        assert all("._" not in f for f in files)
        assert all("static_horiz_ribbon" not in f for f in files)
        assert all("smoke_2a" not in f for f in files)
        
        # Test with different pattern and ignore combination
        files = discover_files(
            ignore_patterns_dir, 
            "*.txt", 
            ignore_patterns=["._"]
        )
        # Should find only normal_file.txt in root
        assert len(files) == 1
        assert "normal_file.txt" in files[0]
        
    def test_mandatory_substrings(self, mandatory_substrings_dir):
        """Test file discovery with mandatory substrings."""
        # Test without any mandatory substrings - should find all files
        all_files = discover_files(mandatory_substrings_dir, "**/*.*", recursive=True)
        assert len(all_files) == 7  # 5 in root + 2 in subdir
        
        # Test with single mandatory substring - only smoke_2a experiments
        files = discover_files(
            mandatory_substrings_dir,
            "**/*.*",
            recursive=True,
            mandatory_substrings=["smoke_2a"]
        )
        # Should only include the 3 smoke_2a files
        assert len(files) == 3
        assert all("smoke_2a" in f for f in files)
        
        # Test with multiple mandatory substrings - OR logic
        files = discover_files(
            mandatory_substrings_dir,
            "**/*.*",
            recursive=True,
            mandatory_substrings=["nagel", "control"]
        )
        # Should include both nagel and control files (1 nagel + 2 control = 3)
        assert len(files) == 3
        assert any("nagel" in f for f in files)
        assert any("control" in f for f in files)
        
        # Test with mandatory AND ignore patterns together
        files = discover_files(
            mandatory_substrings_dir,
            "**/*.*",
            recursive=True,
            mandatory_substrings=["experiment"],
            ignore_patterns=["smoke_2a"]
        )
        # Should include experiment files but not smoke_2a ones (6 experiments - 3 smoke_2a = 3)
        assert len(files) == 3
        assert all("experiment" in f for f in files)
        assert all("smoke_2a" not in f for f in files)
        
        # Test with extension filter and mandatory substrings
        files = discover_files(
            mandatory_substrings_dir,
            "**/*.*",
            recursive=True,
            extensions=["csv"],
            mandatory_substrings=["experiment"]
        )
        # Should find only CSV files with "experiment" in name (all experimental files)
        assert len(files) == 6
        assert all(f.endswith(".csv") for f in files)
        assert all("experiment" in f for f in files)