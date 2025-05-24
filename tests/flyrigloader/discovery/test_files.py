"""
Tests for basic file discovery functionality.
"""
import os
import tempfile
import pytest
from pathlib import Path
from typing import List
import re
from datetime import datetime


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
    def case_extension_dir(self):
        """Create files with mixed-case extensions for case-insensitive testing."""
        temp_dir = tempfile.mkdtemp()
        try:
            with open(os.path.join(temp_dir, "test.TXT"), "w") as f:
                f.write("Upper case text")
            with open(os.path.join(temp_dir, "data.Csv"), "w") as f:
                f.write("Mixed case csv")
            with open(os.path.join(temp_dir, "image.PnG"), "w") as f:
                f.write("Mixed case png")

            yield temp_dir
        finally:
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
    
    @pytest.fixture
    def pattern_extraction_dir(self):
        """Create a directory with files containing extractable patterns."""
        temp_dir = tempfile.mkdtemp()
        try:
            # Create files with animal IDs, dates, and other metadata in filenames
            files = [
                # Format: animalID_YYYYMMDD_condition.ext
                "mouse_20240315_control_1.csv",
                "mouse_20240316_control_2.csv",
                "mouse_20240320_treatment_1.csv",
                # Format: YYYYMMDD_animalID_condition.ext
                "20240321_rat_control_1.csv",
                "20240322_rat_treatment_1.csv",
                # Format with experiment ID: expID_animalID_condition.ext
                "exp001_mouse_baseline.csv",
                "exp002_rat_baseline.csv",
                # Files with batch info in directories
                "batch1/mouse_20240401_treatment_1.csv",
                "batch2/mouse_20240402_treatment_2.csv"
            ]
            
            # Create the files with some content
            for file_path in files:
                # Handle nested paths
                full_path = os.path.join(temp_dir, file_path)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                
                with open(full_path, "w") as f:
                    f.write(f"Data for {file_path}")
            
            yield temp_dir
        finally:
            # Clean up
            import shutil
            shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def date_files_dir(self):
        """Create a directory with files containing dates in different formats."""
        temp_dir = tempfile.mkdtemp()
        try:
            # Create test files with various date formats in names
            date_files = {
                # Standard format YYYYMMDD
                "data_20240315.csv": "2024-03-15",
                "data_20240316.csv": "2024-03-16",
                # ISO format YYYY-MM-DD
                "data_2024-03-17.csv": "2024-03-17",
                "data_2024-03-18.csv": "2024-03-18",
                # US format MM-DD-YYYY
                "data_03-19-2024.csv": "2024-03-19",
                "data_03-20-2024.csv": "2024-03-20",
                # With timestamp YYYYMMDD_HHMMSS
                "data_20240321_142030.csv": "2024-03-21",
                "data_20240322_152030.csv": "2024-03-22"
            }
            
            # Different versions of the same file (for latest file testing)
            versions = {
                "experiment_v1_20240301.csv": "Version 1",
                "experiment_v2_20240305.csv": "Version 2",
                "experiment_v3_20240310.csv": "Version 3"
            }
            
            # Create all files
            for filename, content in (date_files | versions).items():
                filepath = os.path.join(temp_dir, filename)
                with open(filepath, "w") as f:
                    f.write(content)
            
            yield temp_dir
        finally:
            # Clean up
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

    def test_case_insensitive_extension_filtering(self, case_extension_dir):
        """Extensions filter should be case-insensitive."""
        txt_files = discover_files(case_extension_dir, "*", extensions=["txt"])
        assert len(txt_files) == 1
        assert any(Path(f).name == "test.TXT" for f in txt_files)

        other_files = discover_files(case_extension_dir, "*", extensions=["csv", "png"])
        assert len(other_files) == 2
        assert any(Path(f).name == "data.Csv" for f in other_files)
        assert any(Path(f).name == "image.PnG" for f in other_files)
        
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
            ignore_patterns=["._*"]  # Use glob pattern to match files starting with "._"
        )
        # Should exclude the 2 hidden files
        assert len(files) == 5
        assert all("._" not in f for f in files)
        
        # Test with multiple ignore patterns
        files = discover_files(
            ignore_patterns_dir, 
            "**/*.*", 
            recursive=True,
            ignore_patterns=["._*", "*ribbon*"]  # Use glob patterns
        )
        # Should exclude the 2 hidden files and 1 ribbon file
        assert len(files) == 4
        assert all("._" not in f for f in files)
        assert all("ribbon" not in f for f in files)
        
        # Test with different pattern and ignore combination
        files = discover_files(
            ignore_patterns_dir, 
            "*.txt", 
            ignore_patterns=["._*"]
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
            ignore_patterns=["*smoke_2a*"]  # Use glob pattern to match filenames containing "smoke_2a"
        )
        # Should include experiment files but not smoke_2a ones (6 experiments - 3 smoke_2a = 3)
        assert len(files) == 3
        assert all("smoke_2a" not in f for f in files)
        assert all("experiment" in f for f in files)
        
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
    
    def test_pattern_extraction(self, pattern_extraction_dir):
        """Test extracting metadata from file paths using patterns."""
        # Get all files
        files = discover_files(pattern_extraction_dir, "**/*.csv", recursive=True)
        assert len(files) == 9
        
        # Extract pattern information
        patterns = [
            # Mouse pattern (start of filename)
            r".*/(mouse)_(\d{8})_(\w+)_(\d+)\.csv",
            # Rat pattern (date at start)
            r".*/(\d{8})_(rat)_(\w+)_(\d+)\.csv",
            # Experiment ID pattern
            r".*/(exp\d+)_(\w+)_(\w+)\.csv"
        ]
        
        # The function we're testing should return file information with extracted patterns
        file_info = discover_files(
            pattern_extraction_dir, 
            "**/*.csv", 
            recursive=True,
            extract_patterns=patterns
        )
        
        # Instead of a list of strings, we should now get a dictionary with metadata
        assert isinstance(file_info, dict)
        assert len(file_info) == 9
        
        # Check pattern extraction results
        mouse_files = [info for path, info in file_info.items() if "animal" in info and info["animal"] == "mouse"]
        assert len(mouse_files) == 5  # 3 in root + 2 in batch dirs
        
        rat_files = [info for path, info in file_info.items() if "animal" in info and info["animal"] == "rat"]
        assert len(rat_files) == 3  # 2 in root + 1 in experiment
        
        # Check specific metadata examples
        # Find a specific file by partial path match
        mouse_treatment_info = next(
            info for path, info in file_info.items() 
            if "mouse_20240320_treatment_1.csv" in path
        )
        assert mouse_treatment_info["animal"] == "mouse"
        assert mouse_treatment_info["date"] == "20240320"
        assert mouse_treatment_info["condition"] == "treatment"
        assert mouse_treatment_info["replicate"] == "1"
    
    def test_date_handling(self, date_files_dir):
        """Test date extraction and parsing from filenames."""
        # Get file info with date parsing enabled
        file_info = discover_files(
            date_files_dir,
            "*.csv",
            parse_dates=True
        )
        
        assert isinstance(file_info, dict)
        assert len(file_info) == 11  # 8 date files + 3 version files
        
        # Check that dates were parsed - all files should have parsed_date as datetime
        assert all("parsed_date" in info for _, info in file_info.items()), "All files should have parsed_date"
        assert all(isinstance(info["parsed_date"], datetime) for _, info in file_info.items()), "All parsed_date values should be datetime objects"
        
        # Check specific date parsing
        iso_file_info = next(
            info for path, info in file_info.items() 
            if "data_2024-03-17.csv" in path
        )
        assert iso_file_info["parsed_date"].year == 2024
        assert iso_file_info["parsed_date"].month == 3
        assert iso_file_info["parsed_date"].day == 17
    
    def test_glob_pattern_ignore(self, ignore_patterns_dir):
        """Test that glob pattern matching works correctly for ignore_patterns."""
        # Get all test files
        all_files = discover_files(ignore_patterns_dir, "*.*", recursive=True)
        
        # Check that we have all the expected files from the fixture setup
        assert len(all_files) > 5, "Should have found all test files"
        
        # Test with glob pattern that should match only specific files
        # Old substring behavior would also match 'important_data.csv' because it contains 'temp'
        # New glob behavior will only match files with temp* at the start of filename
        matched_files = discover_files(
            ignore_patterns_dir, 
            "*.*", 
            recursive=True, 
            ignore_patterns=["temp*"]
        )
        
        # Verify important_data.csv is NOT ignored (would have been with substring matching)
        assert any("important_data.csv" in f for f in matched_files), \
            "important_data.csv should not be ignored by glob pattern 'temp*'"
        
        # Verify temp_smoke_2a_experiment.json IS ignored
        assert all("temp_smoke_2a_experiment.json" not in f for f in matched_files), \
            "temp_smoke_2a_experiment.json should be ignored by glob pattern 'temp*'"
        
        # Test with multiple patterns 
        matched_files = discover_files(
            ignore_patterns_dir, 
            "*.*", 
            recursive=True, 
            ignore_patterns=[".*_*", "temp*"]  # Match hidden files and temp files
        )
        
        # Should not match normal files
        assert any("normal_file.txt" in f for f in matched_files)
        assert any("important_data.csv" in f for f in matched_files)
        
        # Should ignore temp files and hidden files
        assert all("temp_smoke_2a_experiment.json" not in f for f in matched_files)
        assert all("._hidden_file.txt" not in f for f in matched_files)
        assert all("._subdir_hidden.txt" not in f for f in matched_files)
        
        # Test more complex patterns
        matched_files = discover_files(
            ignore_patterns_dir, 
            "*.*", 
            recursive=True, 
            ignore_patterns=["*temp*", "*hidden*"]  # Match anywhere in filename
        )
        
        # Should match and ignore files with temp or hidden anywhere in the name
        assert all("temp_smoke_2a_experiment.json" not in f for f in matched_files)
        assert all("._hidden_file.txt" not in f for f in matched_files)