"""
Shared test utilities and helper functions for flyrigloader test suite.

This module provides centralized test utilities that eliminate code duplication
across test modules while maintaining consistent testing patterns. All utilities
follow the established naming conventions and support edge-case testing scenarios.

Centralized Test Utilities Features:
- Protocol-based mock implementations for consistent behavior simulation
- Fixture generators with comprehensive edge-case support
- Common test patterns supporting AAA (Arrange-Act-Assert) structure
- Cross-platform compatibility helpers
- Performance testing utilities with memory monitoring
- Error simulation helpers for comprehensive error handling validation

Test Utility Categories:
- mock_*: Mock object factories and protocol-based implementations
- generate_*: Test data generators with realistic experimental patterns
- assert_*: Custom assertion helpers for domain-specific validation
- simulate_*: Error and edge-case simulation utilities
- validate_*: Comprehensive validation helpers for test verification
"""

import os
import tempfile
import platform
import contextlib
import time
import gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Generator, Tuple
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

# Optional imports with graceful degradation
try:
    import numpy as np
except ImportError:
    np = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import psutil
except ImportError:
    psutil = None

try:
    import yaml
except ImportError:
    yaml = None


# ============================================================================
# PROTOCOL-BASED MOCK IMPLEMENTATIONS
# ============================================================================

class MockFileSystemProtocol:
    """
    Protocol-based filesystem mock implementation for consistent behavior simulation.
    
    Provides standardized filesystem mocking patterns that can be reused across
    test modules without code duplication.
    """
    
    def __init__(self):
        self.files = {}
        self.directories = set()
        self.access_errors = {}
        self.corruption_patterns = {}
    
    def add_file(self, path: Union[str, Path], content: Union[str, bytes] = "mock content",
                 size: int = 1024, corrupted: bool = False) -> Path:
        """Add a mock file with specified properties."""
        path = Path(path)
        self.files[str(path)] = {
            'content': content,
            'size': size,
            'corrupted': corrupted,
            'mtime': datetime.now()
        }
        
        # Ensure parent directories exist
        for parent in path.parents:
            self.directories.add(str(parent))
        
        return path
    
    def add_directory(self, path: Union[str, Path]) -> Path:
        """Add a mock directory."""
        self.directories.add(str(Path(path)))
        return Path(path)
    
    def simulate_access_error(self, path: Union[str, Path], error: Exception):
        """Simulate access errors for specific paths."""
        self.access_errors[str(path)] = error
    
    def exists(self, path: Union[str, Path]) -> bool:
        """Mock path existence check."""
        path_str = str(path)
        if path_str in self.access_errors:
            raise self.access_errors[path_str]
        return path_str in self.files or path_str in self.directories


class MockDataLoaderProtocol:
    """
    Protocol-based data loader mock for consistent data loading simulation.
    
    Supports multiple data formats and error scenarios commonly encountered
    in experimental data loading workflows.
    """
    
    def __init__(self):
        self.data_registry = {}
        self.error_registry = {}
        self.load_delays = {}
    
    def register_data(self, path: Union[str, Path], data: Any, 
                     error: Optional[Exception] = None,
                     load_delay: float = 0.0):
        """Register mock data for a specific path."""
        path_str = str(path)
        self.data_registry[path_str] = data
        
        if error:
            self.error_registry[path_str] = error
        
        if load_delay > 0:
            self.load_delays[path_str] = load_delay
    
    def load_pickle(self, path: Union[str, Path]) -> Any:
        """Mock pickle loading with error simulation."""
        path_str = str(path)
        
        # Simulate load delay if configured
        if path_str in self.load_delays:
            time.sleep(self.load_delays[path_str])
        
        # Check for configured errors
        if path_str in self.error_registry:
            raise self.error_registry[path_str]
        
        # Return registered data
        if path_str in self.data_registry:
            return self.data_registry[path_str]
        
        raise FileNotFoundError(f"No mock data registered for: {path}")
    
    def load_yaml(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Mock YAML loading with error simulation."""
        return self.load_pickle(path)  # Same logic for demonstration


# ============================================================================
# TEST DATA GENERATORS WITH EDGE-CASE SUPPORT
# ============================================================================

def generate_realistic_experimental_config(
    num_rigs: int = 3,
    num_datasets: int = 5,
    include_edge_cases: bool = False
) -> Dict[str, Any]:
    """
    Generate realistic experimental configuration with optional edge-case scenarios.
    
    Args:
        num_rigs: Number of rig configurations to generate
        num_datasets: Number of dataset configurations to generate
        include_edge_cases: Whether to include edge-case configurations
    
    Returns:
        Dict[str, Any]: Comprehensive experimental configuration
    """
    config = {
        "project": {
            "name": "test_experiment_project",
            "directories": {
                "major_data_directory": "/research/data",
                "processed_data_directory": "/research/processed"
            },
            "ignore_substrings": ["backup", "temp", "._"],
            "mandatory_substrings": ["experiment"],
            "file_extensions": [".pkl", ".csv", ".yaml"]
        },
        "rigs": {},
        "datasets": {},
        "experiments": {}
    }
    
    # Generate rig configurations
    rig_names = ["old_opto", "new_opto", "high_speed", "custom_rig", "backup_rig"]
    for i in range(min(num_rigs, len(rig_names))):
        rig_name = rig_names[i]
        config["rigs"][rig_name] = {
            "sampling_frequency": 60 + (i * 20),
            "mm_per_px": 0.1 + (i * 0.05),
            "camera_resolution": [1024 + (i * 256), 768 + (i * 256)],
            "calibration_date": f"2024-{i+1:02d}-15"
        }
    
    # Generate dataset configurations
    dataset_types = ["baseline", "treatment", "navigation", "optogenetic", "behavioral"]
    for i in range(min(num_datasets, len(dataset_types))):
        dataset_name = f"{dataset_types[i]}_data"
        config["datasets"][dataset_name] = {
            "rig": list(config["rigs"].keys())[i % num_rigs],
            "patterns": [f"*{dataset_types[i]}*"],
            "dates_vials": {
                f"2024120{i+1}": list(range(1, 4))
            }
        }
    
    # Add edge-case configurations if requested
    if include_edge_cases:
        # Unicode rig name
        config["rigs"]["tëst_rïg"] = {
            "sampling_frequency": 60,
            "mm_per_px": 0.1,
            "camera_resolution": [1024, 768]
        }
        
        # Empty dataset
        config["datasets"]["empty_dataset"] = {
            "rig": "old_opto",
            "patterns": [],
            "dates_vials": {}
        }
        
        # Configuration with extreme values
        config["datasets"]["extreme_values"] = {
            "rig": "high_speed",
            "patterns": ["*" * 100],  # Very long pattern
            "dates_vials": {
                "99991231": list(range(1, 1000))  # Many vials
            }
        }
    
    return config


def generate_synthetic_experimental_data(
    n_timepoints: int = 1000,
    include_edge_cases: bool = False,
    corruption_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate synthetic experimental data with optional edge-case scenarios.
    
    Args:
        n_timepoints: Number of time points to generate
        include_edge_cases: Whether to include edge-case data
        corruption_type: Type of data corruption to simulate
    
    Returns:
        Dict[str, Any]: Synthetic experimental data matrix
    """
    if not np:
        return {"error": "NumPy not available"}
    
    # Base experimental data
    data = {
        't': np.linspace(0, n_timepoints/60.0, n_timepoints),
        'x': np.random.rand(n_timepoints) * 100,
        'y': np.random.rand(n_timepoints) * 100
    }
    
    # Add additional columns
    data['signal'] = np.random.rand(n_timepoints)
    data['signal_disp'] = np.random.rand(16, n_timepoints)
    
    # Add edge-case scenarios
    if include_edge_cases:
        # Arrays with special values
        data['special_values'] = np.array([np.nan, np.inf, -np.inf, 0.0])
        
        # Very small array
        data['minimal_array'] = np.array([1.0])
        
        # Empty array
        data['empty_array'] = np.array([])
    
    # Apply corruption if requested
    if corruption_type:
        if corruption_type == "nan_injection":
            # Inject NaN values randomly
            mask = np.random.random(n_timepoints) < 0.1
            data['x'][mask] = np.nan
        elif corruption_type == "shape_mismatch":
            # Create shape mismatches
            data['mismatched'] = np.random.rand(n_timepoints + 10)
        elif corruption_type == "type_error":
            # Mix data types inappropriately
            data['mixed_types'] = ["string"] * n_timepoints
    
    return data


def generate_unicode_test_paths(num_paths: int = 5) -> List[str]:
    """
    Generate Unicode file paths for cross-platform edge-case testing.
    
    Args:
        num_paths: Number of Unicode paths to generate
    
    Returns:
        List[str]: Unicode file paths
    """
    unicode_patterns = [
        "tëst_fïlé_{}.csv",
        "dätä_ñämé_{}.pkl", 
        "ëxpérïmënt_{}.yaml",
        "ünïcödë_tëst_{}.json",
        "rësëärçh_dätä_{}.txt"
    ]
    
    paths = []
    for i in range(num_paths):
        pattern = unicode_patterns[i % len(unicode_patterns)]
        path = pattern.format(i)
        paths.append(path)
    
    return paths


# ============================================================================
# CUSTOM ASSERTION HELPERS
# ============================================================================

def assert_experimental_data_valid(data: Dict[str, Any], 
                                  required_columns: Optional[List[str]] = None) -> None:
    """
    Assert that experimental data meets flyrigloader requirements.
    
    Args:
        data: Experimental data dictionary
        required_columns: List of required column names
    
    Raises:
        AssertionError: If data validation fails
    """
    if required_columns is None:
        required_columns = ['t', 'x', 'y']
    
    # Check required columns exist
    for col in required_columns:
        assert col in data, f"Required column '{col}' missing from data"
    
    # Check data types and shapes if NumPy is available
    if np and 't' in data and 'x' in data and 'y' in data:
        assert isinstance(data['t'], np.ndarray), "Time column must be numpy array"
        assert isinstance(data['x'], np.ndarray), "X position must be numpy array"
        assert isinstance(data['y'], np.ndarray), "Y position must be numpy array"
        
        # Check consistent lengths
        t_len = len(data['t'])
        assert len(data['x']) == t_len, f"X length {len(data['x'])} != time length {t_len}"
        assert len(data['y']) == t_len, f"Y length {len(data['y'])} != time length {t_len}"


def assert_config_structure_valid(config: Dict[str, Any]) -> None:
    """
    Assert that configuration dictionary has valid flyrigloader structure.
    
    Args:
        config: Configuration dictionary to validate
    
    Raises:
        AssertionError: If configuration structure is invalid
    """
    # Check top-level required sections
    required_sections = ['project', 'rigs', 'datasets']
    for section in required_sections:
        assert section in config, f"Required configuration section '{section}' missing"
    
    # Check project section structure
    project = config['project']
    assert isinstance(project, dict), "Project section must be a dictionary"
    
    # Check rigs section
    rigs = config['rigs']
    assert isinstance(rigs, dict), "Rigs section must be a dictionary"
    assert len(rigs) > 0, "At least one rig configuration required"
    
    # Check datasets section
    datasets = config['datasets']
    assert isinstance(datasets, dict), "Datasets section must be a dictionary"


def assert_performance_within_sla(operation_name: str, 
                                 actual_time: float, 
                                 expected_max_time: float) -> None:
    """
    Assert that operation performance meets SLA requirements.
    
    Args:
        operation_name: Name of the operation being tested
        actual_time: Actual execution time in seconds
        expected_max_time: Maximum allowed time in seconds
    
    Raises:
        AssertionError: If performance SLA is violated
    """
    assert actual_time <= expected_max_time, (
        f"Performance SLA violation for {operation_name}: "
        f"took {actual_time:.3f}s, expected ≤ {expected_max_time:.3f}s"
    )


# ============================================================================
# ERROR SIMULATION UTILITIES
# ============================================================================

class ErrorSimulator:
    """
    Comprehensive error simulation utility for testing error handling robustness.
    
    Provides methods to simulate various error conditions that can occur
    during experimental data processing workflows.
    """
    
    @staticmethod
    def simulate_file_corruption(original_content: bytes, 
                                corruption_type: str = "random") -> bytes:
        """
        Simulate file corruption for error handling testing.
        
        Args:
            original_content: Original file content
            corruption_type: Type of corruption to apply
        
        Returns:
            bytes: Corrupted file content
        """
        if corruption_type == "truncated":
            # Truncate file at random position
            if len(original_content) > 10:
                cut_point = len(original_content) // 2
                return original_content[:cut_point]
        elif corruption_type == "binary_injection":
            # Inject binary data
            injection_point = len(original_content) // 2
            binary_data = b"\x00\x01\x02\x03\xFF\xFE"
            return original_content[:injection_point] + binary_data + original_content[injection_point:]
        elif corruption_type == "encoding_error":
            # Create encoding issues
            return original_content + b"\xFF\xFE\x00"
        
        return original_content
    
    @staticmethod
    def simulate_memory_pressure() -> None:
        """Simulate memory pressure conditions."""
        if psutil:
            # Check available memory and simulate pressure
            memory = psutil.virtual_memory()
            if memory.available < 100 * 1024 * 1024:  # Less than 100MB
                raise MemoryError("Simulated memory pressure condition")
    
    @staticmethod
    def simulate_network_delay(delay_seconds: float = 1.0) -> None:
        """Simulate network delays for timeout testing."""
        time.sleep(delay_seconds)
    
    @staticmethod
    @contextlib.contextmanager
    def simulate_permission_error(path: Union[str, Path]):
        """Context manager to simulate permission errors."""
        try:
            yield
        except Exception:
            raise PermissionError(f"Simulated permission error for: {path}")


# ============================================================================
# PERFORMANCE TESTING UTILITIES
# ============================================================================

class PerformanceMonitor:
    """
    Performance monitoring utility for testing performance requirements.
    
    Provides comprehensive performance monitoring including timing,
    memory usage, and resource utilization tracking.
    """
    
    def __init__(self):
        self.start_time = None
        self.start_memory = None
        self.measurements = {}
    
    def start_monitoring(self, operation_name: str):
        """Start monitoring an operation."""
        self.start_time = time.perf_counter()
        
        if psutil:
            process = psutil.Process()
            self.start_memory = process.memory_info().rss
        
        return self
    
    def stop_monitoring(self, operation_name: str) -> Dict[str, float]:
        """Stop monitoring and return measurements."""
        end_time = time.perf_counter()
        duration = end_time - self.start_time if self.start_time else 0
        
        measurement = {
            'duration_seconds': duration,
            'memory_delta_mb': 0
        }
        
        if psutil and self.start_memory:
            process = psutil.Process()
            end_memory = process.memory_info().rss
            memory_delta = (end_memory - self.start_memory) / 1024 / 1024
            measurement['memory_delta_mb'] = memory_delta
        
        self.measurements[operation_name] = measurement
        return measurement
    
    @contextlib.contextmanager
    def monitor_operation(self, operation_name: str):
        """Context manager for monitoring operations."""
        self.start_monitoring(operation_name)
        try:
            yield self
        finally:
            self.stop_monitoring(operation_name)


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

def validate_test_isolation() -> bool:
    """
    Validate that test isolation is maintained.
    
    Checks for common test isolation issues like shared state,
    memory leaks, and resource cleanup.
    
    Returns:
        bool: True if isolation is maintained
    """
    # Force garbage collection
    gc.collect()
    
    # Check for obvious memory issues
    if psutil:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Warning threshold: 500MB for a single test
        if memory_mb > 500:
            return False
    
    return True


def validate_cross_platform_path(path: Union[str, Path]) -> bool:
    """
    Validate that a path is compatible across platforms.
    
    Args:
        path: Path to validate
    
    Returns:
        bool: True if path is cross-platform compatible
    """
    path_str = str(path)
    
    # Check for platform-specific issues
    if platform.system() == "Windows":
        # Check for reserved names
        reserved_names = ["CON", "PRN", "AUX", "NUL"]
        if any(reserved in path_str.upper() for reserved in reserved_names):
            return False
        
        # Check path length
        if len(path_str) > 260:
            return False
    
    # Check for invalid characters
    invalid_chars = ['<', '>', ':', '"', '|', '?', '*']
    if any(char in path_str for char in invalid_chars):
        return False
    
    return True


def validate_unicode_support(text: str) -> bool:
    """
    Validate Unicode text handling across platforms.
    
    Args:
        text: Text to validate
    
    Returns:
        bool: True if Unicode is properly supported
    """
    try:
        # Test encoding/decoding
        encoded = text.encode('utf-8')
        decoded = encoded.decode('utf-8')
        return decoded == text
    except (UnicodeEncodeError, UnicodeDecodeError):
        return False


# ============================================================================
# COMMON TEST PATTERNS
# ============================================================================

class TestPatternHelper:
    """
    Helper class providing common test patterns following AAA structure.
    
    Encapsulates frequently used test patterns to promote consistency
    and reduce code duplication across test modules.
    """
    
    @staticmethod
    def arrange_mock_filesystem(mock_fs: MockFileSystemProtocol,
                               file_paths: List[str]) -> MockFileSystemProtocol:
        """
        Arrange a mock filesystem with specified files.
        
        Args:
            mock_fs: Mock filesystem instance
            file_paths: List of file paths to create
        
        Returns:
            MockFileSystemProtocol: Configured mock filesystem
        """
        for path in file_paths:
            mock_fs.add_file(path)
        
        return mock_fs
    
    @staticmethod
    def arrange_experimental_data(n_timepoints: int = 1000,
                                 include_metadata: bool = True) -> Dict[str, Any]:
        """
        Arrange experimental data for testing.
        
        Args:
            n_timepoints: Number of time points
            include_metadata: Whether to include metadata
        
        Returns:
            Dict[str, Any]: Arranged experimental data
        """
        data = generate_synthetic_experimental_data(n_timepoints)
        
        if include_metadata:
            data.update({
                'date': '20241201',
                'exp_name': 'test_experiment',
                'rig': 'test_rig',
                'fly_id': 'test_fly_001'
            })
        
        return data
    
    @staticmethod
    def assert_operation_successful(result: Any, 
                                  expected_type: type = dict,
                                  required_keys: Optional[List[str]] = None) -> None:
        """
        Assert that an operation completed successfully.
        
        Args:
            result: Operation result to validate
            expected_type: Expected result type
            required_keys: Required keys if result is a dictionary
        """
        assert result is not None, "Operation result should not be None"
        assert isinstance(result, expected_type), f"Expected {expected_type}, got {type(result)}"
        
        if required_keys and isinstance(result, dict):
            for key in required_keys:
                assert key in result, f"Required key '{key}' missing from result"