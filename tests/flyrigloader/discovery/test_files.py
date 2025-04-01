"""
Tests for basic file discovery functionality.
"""
import os
import tempfile
import pytest
from pathlib import Path

# Import the function we want to test (doesn't exist yet)
from flyrigloader.discovery.files import discover_files


class TestFileDiscovery:
    
    @pytest.fixture
    def test_dir(self):
        """Create a temporary directory with test files."""
        temp_dir = tempfile.mkdtemp()
        try:
            # Create a few test files
            for i in range(3):
                with open(os.path.join(temp_dir, f"test_{i}.txt"), "w") as f:
                    f.write(f"Test content {i}")
            
            yield temp_dir
        finally:
            # Clean up after the test
            import shutil
            shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def nested_test_dir(self):
        """Create a temporary directory with nested subdirectories and test files."""
        temp_dir = tempfile.mkdtemp()
        try:
            # Create files in root directory
            for i in range(2):
                with open(os.path.join(temp_dir, f"root_{i}.txt"), "w") as f:
                    f.write(f"Root content {i}")
            
            # Create subdirectory and files
            subdir = os.path.join(temp_dir, "subdir")
            os.makedirs(subdir)
            for i in range(3):
                with open(os.path.join(subdir, f"sub_{i}.txt"), "w") as f:
                    f.write(f"Subdir content {i}")
                    
            # Create deeper subdirectory and files
            deepdir = os.path.join(subdir, "deepdir")
            os.makedirs(deepdir)
            for i in range(2):
                with open(os.path.join(deepdir, f"deep_{i}.txt"), "w") as f:
                    f.write(f"Deep content {i}")
            
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
            for i in range(2):
                with open(os.path.join(temp_dir, f"data_{i}.txt"), "w") as f:
                    f.write(f"Text data {i}")
            
            # Create CSV files
            for i in range(3):
                with open(os.path.join(temp_dir, f"data_{i}.csv"), "w") as f:
                    f.write(f"col1,col2\nvalue{i}a,value{i}b")
            
            # Create JSON files
            for i in range(2):
                with open(os.path.join(temp_dir, f"config_{i}.json"), "w") as f:
                    f.write(f'{{"name": "config{i}", "value": {i}}}')
            
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
        
        # Count files by type (addressing Sourcery lint warnings)
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
        
        # Count files by type (addressing Sourcery lint warnings)
        csv_files = [f for f in data_files if f.endswith(".csv")]
        json_files = [f for f in data_files if f.endswith(".json")]
        
        assert len(csv_files) == 3
        assert len(json_files) == 2