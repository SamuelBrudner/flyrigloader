"""
Centralized test utilities module for flyrigloader test suite.

Provides comprehensive testing infrastructure with shared protocol-based mock implementations,
standardized fixture generators, and common test patterns to eliminate code duplication across
the test suite while maintaining consistent testing practices throughout unit, integration,
and performance test layers per testing strategy requirements.

This module implements the testing strategy Section 6.6 requirements for:
- Centralized fixture management to eliminate code duplication across test modules
- Protocol-based mock implementations for dependency injection and interface mocking
- Shared test utilities supporting edge-case coverage enhancement through parameterized tests
- Standardized fixture naming conventions and AAA pattern enforcement utilities
- Consistent test structure maintenance across all test categories

Key Components:
- Protocol-based mock factories for filesystem, data loading, and configuration providers
- Shared fixture generators for edge-case testing scenarios including Unicode path handling
- Corrupted file scenarios and boundary condition validation utilities
- Centralized AAA pattern enforcement utilities and test structure validation helpers
- Reusable hypothesis strategies for domain-specific property-based testing
- Memory constraint and performance testing support utilities

Usage:
    from tests.utils import (
        create_mock_filesystem,
        create_mock_dataloader,
        create_mock_config_provider,
        generate_edge_case_scenarios,
        validate_test_structure
    )

Fixture Naming Conventions:
- mock_* : Mock objects and simulated components
- generate_* : Data generation and factory functions
- validate_* : Test structure and pattern validation utilities
- create_* : Factory functions for creating test objects
- scenario_* : Edge-case and boundary condition test scenarios
"""

import contextlib
import copy
import gc
import gzip
import io
import json
import pickle
import platform
import random
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import (
    Any, Dict, List, Optional, Union, Generator, Tuple, Set, Callable,
    Protocol, runtime_checkable, Type, NamedTuple, Iterable
)
from unittest.mock import MagicMock, Mock, patch, mock_open
import threading
import queue

# Core testing imports
import pytest
from hypothesis import strategies as st
from hypothesis import given, assume, settings

# Third-party testing utilities with graceful fallbacks
try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

# Enhanced Loguru integration with fallback
try:
    from loguru import logger
except ModuleNotFoundError:  # pragma: no cover
    import logging
    logger = logging.getLogger("tests.utils")


# ============================================================================
# PROTOCOL-BASED MOCK IMPLEMENTATIONS
# ============================================================================

@runtime_checkable
class MockConfigurationProvider(Protocol):
    """Protocol-based mock implementation for configuration providers."""
    
    def load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from path."""
        ...
    
    def get_ignore_patterns(self, config: Dict[str, Any], experiment: Optional[str] = None) -> List[str]:
        """Get ignore patterns from configuration."""
        ...
    
    def get_mandatory_substrings(self, config: Dict[str, Any], experiment: Optional[str] = None) -> List[str]:
        """Get mandatory substrings from configuration."""
        ...
    
    def get_dataset_info(self, config: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
        """Get dataset information."""
        ...
    
    def get_experiment_info(self, config: Dict[str, Any], experiment_name: str) -> Dict[str, Any]:
        """Get experiment information."""
        ...


@runtime_checkable
class MockFilesystemProvider(Protocol):
    """Protocol-based mock implementation for filesystem operations."""
    
    def glob(self, path: Path, pattern: str) -> List[Path]:
        """Execute glob operation on the given path."""
        ...
    
    def rglob(self, path: Path, pattern: str) -> List[Path]:
        """Execute recursive glob operation on the given path."""
        ...
    
    def stat(self, path: Path) -> Any:
        """Get file statistics for the given path."""
        ...
    
    def exists(self, path: Path) -> bool:
        """Check if path exists."""
        ...
    
    def path_exists(self, path: Union[str, Path]) -> bool:
        """Check if a path exists."""
        ...
    
    def open_file(self, path: Union[str, Path], mode: str) -> Any:
        """Open a file with the specified mode."""
        ...


@runtime_checkable
class MockDataLoadingProvider(Protocol):
    """Protocol-based mock implementation for data loading operations."""
    
    def read_pickle_any_format(self, path: Union[str, Path]) -> Any:
        """Read pickle files in any format."""
        ...
    
    def load(self, file_obj) -> Any:
        """Load data from a pickle file object."""
        ...
    
    def read_pickle(self, path: Union[str, Path]) -> Any:
        """Read a pickle file using pandas."""
        ...
    
    def open_gzip(self, path: Union[str, Path], mode: str) -> Any:
        """Open a gzip file with the specified mode."""
        ...


# ============================================================================
# MOCK FACTORY FUNCTIONS
# ============================================================================

class MockFilesystem:
    """
    Centralized mock filesystem implementation for consistent behavior simulation
    across unit, integration, and performance test layers.
    
    Features:
    - Cross-platform path mocking with Unicode support
    - File existence and permission simulation including access errors
    - Directory traversal mocking with realistic metadata
    - Corrupted file scenario simulation
    - Memory constraint simulation for large files
    - Concurrent access simulation with file locking
    - Edge-case handling for Unicode paths and special characters
    """
    
    def __init__(self):
        """Initialize mock filesystem with empty state."""
        self.mock_files: Dict[str, Dict[str, Any]] = {}
        self.mock_dirs: Set[str] = set()
        self.access_errors: Dict[str, Exception] = {}
        self.corrupted_files: Set[str] = set()
        self.file_locks: Dict[str, bool] = {}
        self.unicode_support = True
        self._setup_default_behaviors()
    
    def _setup_default_behaviors(self):
        """Setup default filesystem behaviors."""
        # Add platform-specific root directories
        if platform.system() == "Windows":
            self.mock_dirs.add("C:\\")
            self.mock_dirs.add("C:\\temp")
        else:
            self.mock_dirs.add("/")
            self.mock_dirs.add("/tmp")
            self.mock_dirs.add("/home")
    
    def add_file(self, path: Union[str, Path], 
                size: int = 1024,
                mtime: Optional[datetime] = None,
                content: Optional[Union[str, bytes]] = None,
                corrupted: bool = False,
                access_error: Optional[Exception] = None) -> Path:
        """
        Add a mock file with comprehensive properties and edge-case support.
        
        Args:
            path: File path to add
            size: File size in bytes
            mtime: Modification time (defaults to current time)
            content: File content (text or binary)
            corrupted: Whether file should appear corrupted
            access_error: Exception to raise on access attempts
            
        Returns:
            Path object for the added file
        """
        path = Path(path)
        path_str = str(path)
        
        # Handle Unicode paths gracefully
        try:
            path_normalized = path.resolve()
            path_str = str(path_normalized)
        except (OSError, UnicodeError):
            if not self.unicode_support:
                logger.debug(f"Unicode path not supported: {path}")
                return path
            # Continue with original path
        
        # Generate realistic content if not provided
        if content is None:
            if corrupted:
                content = b"corrupted\x00\x01\x02binary data" if size > 20 else b"\x00\x01"
            else:
                content = f"mock file content for {path.name}".encode('utf-8')
                if len(content) < size:
                    content += b" " * (size - len(content))
        
        self.mock_files[path_str] = {
            'size': size,
            'mtime': mtime or datetime.now(),
            'exists': True,
            'corrupted': corrupted,
            'content': content,
            'is_file': True,
            'is_dir': False
        }
        
        # Handle special scenarios
        if corrupted:
            self.corrupted_files.add(path_str)
        
        if access_error:
            self.access_errors[path_str] = access_error
        
        # Ensure parent directories exist
        parent = path.parent
        while parent != Path('.') and str(parent) not in ('/', 'C:\\'):
            self.mock_dirs.add(str(parent))
            parent = parent.parent
        
        return path
    
    def add_directory(self, path: Union[str, Path], 
                     access_error: Optional[Exception] = None) -> Path:
        """
        Add a mock directory with optional access restrictions.
        
        Args:
            path: Directory path to add
            access_error: Exception to raise on access attempts
            
        Returns:
            Path object for the added directory
        """
        path = Path(path)
        path_str = str(path)
        
        self.mock_dirs.add(path_str)
        
        # Also add to mock_files for comprehensive stat() support
        self.mock_files[path_str] = {
            'size': 4096,  # Standard directory size
            'mtime': datetime.now(),
            'exists': True,
            'corrupted': False,
            'content': None,
            'is_file': False,
            'is_dir': True
        }
        
        if access_error:
            self.access_errors[path_str] = access_error
        
        return path
    
    def add_unicode_files(self, base_dir: Union[str, Path], count: int = 3) -> List[Path]:
        """
        Add multiple Unicode test files for cross-platform edge-case testing.
        
        Args:
            base_dir: Base directory for Unicode files
            count: Number of Unicode files to create
            
        Returns:
            List of created Unicode file paths
        """
        unicode_patterns = [
            "tëst_fïlé_{}.csv",
            "dàtä_ñãmé_{}.pkl", 
            "ëxpérîmént_{}.yaml",
            "ūnïcōdė_tėst_{}.json",
            "rėsėärçh_dātā_{}.txt"
        ]
        
        created_files = []
        base_path = Path(base_dir)
        
        for i in range(min(count, len(unicode_patterns))):
            try:
                filename = unicode_patterns[i].format(i)
                file_path = base_path / filename
                
                # Test Unicode support
                str(file_path)
                
                created_path = self.add_file(
                    file_path,
                    size=512 + i * 256,
                    content=f"Unicode test content {i}".encode('utf-8')
                )
                created_files.append(created_path)
                
            except (OSError, UnicodeError):
                logger.debug(f"Unicode file creation failed for pattern {i}")
                continue
        
        return created_files
    
    def add_corrupted_scenarios(self, base_dir: Union[str, Path]) -> Dict[str, Path]:
        """
        Add various corrupted file scenarios for comprehensive error handling testing.
        
        Args:
            base_dir: Base directory for corrupted files
            
        Returns:
            Dictionary mapping scenario names to file paths
        """
        base_path = Path(base_dir)
        scenarios = {}
        
        # Corrupted pickle file
        corrupted_pickle = self.add_file(
            base_path / "corrupted_data.pkl",
            size=100,
            corrupted=True,
            content=b"not a pickle file - corrupted data"
        )
        scenarios["corrupted_pickle"] = corrupted_pickle
        
        # Malformed YAML file
        malformed_yaml = self.add_file(
            base_path / "malformed_config.yaml",
            size=200,
            content="invalid: yaml: content: [\n  - missing closing".encode('utf-8')
        )
        scenarios["malformed_yaml"] = malformed_yaml
        
        # Truncated CSV file
        truncated_csv = self.add_file(
            base_path / "truncated_data.csv",
            size=50,
            content="t,x,y\n0.0,10.5,".encode('utf-8')
        )
        scenarios["truncated_csv"] = truncated_csv
        
        # Binary contaminated text file
        binary_contaminated = self.add_file(
            base_path / "binary_contaminated.csv",
            size=150,
            content=b"t,x,y\n0.0,10.5,20.3\n\x00\x01\x02invalid binary data"
        )
        scenarios["binary_contaminated"] = binary_contaminated
        
        # Empty file with expected extension
        empty_file = self.add_file(
            base_path / "empty_file.pkl",
            size=0,
            content=b""
        )
        scenarios["empty_file"] = empty_file
        
        # Access permission error file
        permission_error_file = self.add_file(
            base_path / "permission_denied.pkl",
            size=1024,
            access_error=PermissionError("Access denied for testing")
        )
        scenarios["permission_error"] = permission_error_file
        
        return scenarios
    
    def simulate_file_lock(self, path: Union[str, Path], locked: bool = True):
        """Simulate file lock for concurrent access testing."""
        self.file_locks[str(path)] = locked
    
    def exists(self, path: Union[str, Path]) -> bool:
        """Mock pathlib.Path.exists() with error simulation."""
        path_str = str(path)
        
        # Check for access errors first
        if path_str in self.access_errors:
            raise self.access_errors[path_str]
        
        return path_str in self.mock_files or path_str in self.mock_dirs
    
    def is_file(self, path: Union[str, Path]) -> bool:
        """Mock pathlib.Path.is_file() with error handling."""
        path_str = str(path)
        
        if path_str in self.access_errors:
            raise self.access_errors[path_str]
        
        file_info = self.mock_files.get(path_str)
        return file_info is not None and file_info.get('is_file', False)
    
    def is_dir(self, path: Union[str, Path]) -> bool:
        """Mock pathlib.Path.is_dir() with access control."""
        path_str = str(path)
        
        if path_str in self.access_errors:
            raise self.access_errors[path_str]
        
        file_info = self.mock_files.get(path_str)
        return file_info is not None and file_info.get('is_dir', False)
    
    def stat(self, path: Union[str, Path]) -> MagicMock:
        """Mock pathlib.Path.stat() with comprehensive metadata."""
        path_str = str(path)
        
        if path_str in self.access_errors:
            raise self.access_errors[path_str]
        
        if path_str in self.mock_files:
            file_info = self.mock_files[path_str]
            mock_stat = MagicMock()
            mock_stat.st_size = file_info['size']
            mock_stat.st_mtime = file_info['mtime'].timestamp()
            mock_stat.st_ctime = file_info['mtime'].timestamp()
            mock_stat.st_atime = file_info['mtime'].timestamp()
            mock_stat.st_mode = 0o644 if file_info.get('is_file') else 0o755
            mock_stat.st_uid = 1000
            mock_stat.st_gid = 1000
            mock_stat.st_nlink = 1
            return mock_stat
        
        raise FileNotFoundError(f"Mocked file not found: {path}")
    
    def open_file(self, path: Union[str, Path], mode: str = 'r', **kwargs) -> io.IOBase:
        """
        Mock file opening with corruption and lock simulation.
        
        Args:
            path: File path to open
            mode: File mode (r, w, rb, wb, etc.)
            **kwargs: Additional keyword arguments
            
        Returns:
            File-like object for mocked file content
        """
        path_str = str(path)
        
        # Check for access errors
        if path_str in self.access_errors:
            raise self.access_errors[path_str]
        
        # Check for file locks
        if path_str in self.file_locks and self.file_locks[path_str]:
            raise PermissionError(f"File is locked: {path}")
        
        # Handle corrupted files
        if path_str in self.corrupted_files:
            if 'b' in mode:
                return io.BytesIO(b"corrupted\x00\x01\x02binary data")
            else:
                return io.StringIO("corrupted text data with\x00null chars")
        
        # Normal file handling
        if path_str in self.mock_files:
            file_info = self.mock_files[path_str]
            content = file_info.get('content', b'')
            
            if 'b' in mode:
                if isinstance(content, str):
                    content = content.encode('utf-8')
                return io.BytesIO(content)
            else:
                if isinstance(content, bytes):
                    content = content.decode('utf-8', errors='replace')
                return io.StringIO(content)
        
        # Handle write modes by creating new file
        if 'w' in mode or 'a' in mode:
            if 'b' in mode:
                return io.BytesIO()
            else:
                return io.StringIO()
        
        raise FileNotFoundError(f"Mocked file not found: {path}")
    
    def glob(self, path: Path, pattern: str) -> List[Path]:
        """Mock pathlib.Path.glob() with pattern matching."""
        path_str = str(path)
        
        if path_str in self.access_errors:
            raise self.access_errors[path_str]
        
        results = []
        import fnmatch
        
        # Simple pattern matching for mock files
        for file_path in self.mock_files:
            file_path_obj = Path(file_path)
            if file_path_obj.parent == path or str(path) in str(file_path_obj.parent):
                if fnmatch.fnmatch(file_path_obj.name, pattern):
                    results.append(file_path_obj)
        
        return results
    
    def rglob(self, path: Path, pattern: str) -> List[Path]:
        """Mock pathlib.Path.rglob() with recursive pattern matching."""
        path_str = str(path)
        
        if path_str in self.access_errors:
            raise self.access_errors[path_str]
        
        results = []
        import fnmatch
        
        # Recursive pattern matching for mock files
        for file_path in self.mock_files:
            file_path_obj = Path(file_path)
            # Check if file is under the search path
            try:
                file_path_obj.relative_to(path)
                if fnmatch.fnmatch(file_path_obj.name, pattern):
                    results.append(file_path_obj)
            except ValueError:
                # File is not under the search path
                continue
        
        return results
    
    def reset(self):
        """Reset all mock filesystem state for clean test isolation."""
        self.mock_files.clear()
        self.mock_dirs.clear()
        self.access_errors.clear()
        self.corrupted_files.clear()
        self.file_locks.clear()
        self._setup_default_behaviors()


class MockDataLoading:
    """
    Centralized mock data loading implementation for consistent behavior simulation
    across unit, integration, and performance test layers.
    
    Features:
    - Pickle file loading (standard, gzipped, pandas-specific, corrupted)
    - YAML configuration loading with malformed file handling
    - DataFrame operations with memory constraints
    - Network-dependent data loading simulation
    - Concurrent access and file locking scenarios
    - Error simulation for all data loading paths
    """
    
    def __init__(self):
        """Initialize mock data loading with empty state."""
        self.mock_data: Dict[str, Any] = {}
        self.error_scenarios: Dict[str, Exception] = {}
        self.load_delays: Dict[str, float] = {}
        self.memory_constraints: Dict[str, float] = {}
        self.corruption_scenarios: Set[str] = set()
        self.network_delays: Dict[str, float] = {}
    
    def add_pickle_file(self, file_path: Union[str, Path], 
                       data: Any,
                       error: Optional[Exception] = None,
                       delay_seconds: float = 0.0,
                       memory_limit_mb: Optional[float] = None,
                       corrupted: bool = False) -> Path:
        """
        Add mock pickle file with comprehensive configuration options.
        
        Args:
            file_path: Path to the pickle file
            data: Data to return when file is loaded
            error: Exception to raise on load attempts
            delay_seconds: Artificial delay to simulate slow loading
            memory_limit_mb: Memory limit for testing memory constraints
            corrupted: Whether file should appear corrupted
            
        Returns:
            Path object for the added file
        """
        file_path = Path(file_path)
        path_str = str(file_path)
        
        self.mock_data[path_str] = data
        
        if error:
            self.error_scenarios[path_str] = error
        
        if delay_seconds > 0:
            self.load_delays[path_str] = delay_seconds
        
        if memory_limit_mb:
            self.memory_constraints[path_str] = memory_limit_mb
        
        if corrupted:
            self.corruption_scenarios.add(path_str)
            self.error_scenarios[path_str] = pickle.UnpicklingError("File appears to be corrupted")
        
        return file_path
    
    def add_yaml_file(self, file_path: Union[str, Path], 
                     data: Dict[str, Any],
                     error: Optional[Exception] = None,
                     malformed: bool = False) -> Path:
        """
        Add mock YAML file with malformed content simulation.
        
        Args:
            file_path: Path to the YAML file
            data: Data to return when file is loaded
            error: Exception to raise on load attempts
            malformed: Whether file should appear malformed
            
        Returns:
            Path object for the added file
        """
        file_path = Path(file_path)
        path_str = str(file_path)
        
        self.mock_data[path_str] = data
        
        if error:
            self.error_scenarios[path_str] = error
        elif malformed:
            if yaml:
                self.error_scenarios[path_str] = yaml.YAMLError("Invalid YAML syntax")
            else:
                self.error_scenarios[path_str] = ValueError("Invalid YAML syntax")
        
        return file_path
    
    def add_dataframe_file(self, file_path: Union[str, Path], 
                          dataframe: Optional[Any] = None,
                          shape: Tuple[int, int] = (100, 5),
                          columns: Optional[List[str]] = None) -> Path:
        """
        Add mock DataFrame pickle file with realistic structure.
        
        Args:
            file_path: Path to the DataFrame file
            dataframe: Specific DataFrame to return (or None for generated)
            shape: Shape of generated DataFrame if dataframe is None
            columns: Column names for generated DataFrame
            
        Returns:
            Path object for the added file
        """
        file_path = Path(file_path)
        path_str = str(file_path)
        
        if dataframe is not None:
            self.mock_data[path_str] = dataframe
        elif pd:
            # Generate realistic experimental DataFrame
            if columns is None:
                columns = ['t', 'x', 'y', 'signal'][:shape[1]]
            
            data_dict = {}
            for i, col in enumerate(columns):
                if col == 't':
                    data_dict[col] = np.linspace(0, shape[0]/60.0, shape[0]) if np else list(range(shape[0]))
                elif col in ['x', 'y']:
                    data_dict[col] = np.random.rand(shape[0]) * 100 if np else [50] * shape[0]
                else:
                    data_dict[col] = np.random.rand(shape[0]) if np else [0.5] * shape[0]
            
            self.mock_data[path_str] = pd.DataFrame(data_dict)
        else:
            # Fallback to dictionary if pandas not available
            self.mock_data[path_str] = {
                't': list(range(shape[0])),
                'x': [50] * shape[0],
                'y': [50] * shape[0]
            }
        
        return file_path
    
    def add_experimental_matrix(self, file_path: Union[str, Path],
                               n_timepoints: int = 1000,
                               include_signal: bool = True,
                               include_metadata: bool = True) -> Path:
        """
        Add mock experimental matrix file with realistic neuroscience data structure.
        
        Args:
            file_path: Path to the matrix file
            n_timepoints: Number of time points in the matrix
            include_signal: Whether to include signal data
            include_metadata: Whether to include metadata fields
            
        Returns:
            Path object for the added file
        """
        file_path = Path(file_path)
        path_str = str(file_path)
        
        # Generate realistic experimental matrix
        matrix = {}
        
        # Basic time series data
        if np:
            matrix['t'] = np.linspace(0, n_timepoints/60.0, n_timepoints)  # 60 Hz sampling
            matrix['x'] = np.random.rand(n_timepoints) * 120  # Arena coordinates
            matrix['y'] = np.random.rand(n_timepoints) * 120
        else:
            matrix['t'] = [i/60.0 for i in range(n_timepoints)]
            matrix['x'] = [60] * n_timepoints
            matrix['y'] = [60] * n_timepoints
        
        # Signal data
        if include_signal:
            if np:
                matrix['signal'] = np.random.rand(n_timepoints)
                matrix['signal_disp'] = np.random.rand(16, n_timepoints)  # Multi-channel
            else:
                matrix['signal'] = [0.5] * n_timepoints
                matrix['signal_disp'] = [[0.5] * n_timepoints for _ in range(16)]
        
        # Metadata
        if include_metadata:
            matrix.update({
                'date': '20241201',
                'exp_name': 'test_experiment',
                'rig': 'test_rig',
                'fly_id': 'test_fly_001'
            })
        
        self.mock_data[path_str] = matrix
        return file_path
    
    def add_network_delay(self, file_path: Union[str, Path], delay_seconds: float):
        """Add network delay simulation for remote file loading."""
        self.network_delays[str(file_path)] = delay_seconds
    
    def simulate_memory_pressure(self, file_path: Union[str, Path], limit_mb: float):
        """Simulate memory pressure during large file loading."""
        self.memory_constraints[str(file_path)] = limit_mb
    
    def load_file(self, file_path: Union[str, Path]) -> Any:
        """
        Simulate file loading with all configured behaviors.
        
        Args:
            file_path: Path to file to load
            
        Returns:
            Mock data for the file
            
        Raises:
            Various exceptions based on configuration
        """
        path_str = str(file_path)
        
        # Check for errors first
        if path_str in self.error_scenarios:
            raise self.error_scenarios[path_str]
        
        # Simulate loading delays (including network delays)
        total_delay = self.load_delays.get(path_str, 0) + self.network_delays.get(path_str, 0)
        if total_delay > 0:
            time.sleep(total_delay)
        
        # Check memory constraints
        if path_str in self.memory_constraints:
            if psutil:
                try:
                    available_mb = psutil.virtual_memory().available / 1024 / 1024
                    required_mb = self.memory_constraints[path_str]
                    
                    if available_mb < required_mb:
                        raise MemoryError(f"Insufficient memory: need {required_mb}MB, have {available_mb}MB")
                except Exception:
                    pass  # Skip memory check if psutil fails
        
        # Return mock data
        if path_str in self.mock_data:
            return copy.deepcopy(self.mock_data[path_str])
        
        raise FileNotFoundError(f"Mock data not configured for: {file_path}")
    
    def reset(self):
        """Reset all mock data loading state for clean test isolation."""
        self.mock_data.clear()
        self.error_scenarios.clear()
        self.load_delays.clear()
        self.memory_constraints.clear()
        self.corruption_scenarios.clear()
        self.network_delays.clear()


class MockConfigurationProvider:
    """
    Centralized mock configuration provider implementation for consistent
    configuration handling across all test layers.
    
    Features:
    - YAML configuration loading simulation
    - Configuration validation with error scenarios
    - Dataset and experiment information extraction
    - Pattern and filtering configuration management
    - Edge-case configuration scenarios
    """
    
    def __init__(self):
        """Initialize mock configuration provider."""
        self.configurations: Dict[str, Dict[str, Any]] = {}
        self.error_scenarios: Dict[str, Exception] = {}
        self.validation_errors: Set[str] = set()
        self._setup_default_configurations()
    
    def _setup_default_configurations(self):
        """Setup default test configurations."""
        # Basic test configuration
        self.configurations['default'] = {
            "project": {
                "directories": {
                    "major_data_directory": "/test/data",
                    "batchfile_directory": "/test/batch"
                },
                "ignore_substrings": ["backup", "temp", ".DS_Store"],
                "mandatory_substrings": ["experiment"],
                "file_extensions": [".pkl", ".csv"]
            },
            "rigs": {
                "test_rig": {
                    "sampling_frequency": 60,
                    "mm_per_px": 0.1,
                    "arena_diameter_mm": 120
                }
            },
            "datasets": {
                "test_dataset": {
                    "rig": "test_rig",
                    "patterns": ["*test*"],
                    "dates_vials": {
                        "20241201": ["test_001", "test_002"]
                    }
                }
            },
            "experiments": {
                "test_experiment": {
                    "datasets": ["test_dataset"],
                    "metadata": {
                        "study_type": "control"
                    }
                }
            }
        }
        
        # Comprehensive configuration for advanced testing
        self.configurations['comprehensive'] = {
            "project": {
                "directories": {
                    "major_data_directory": "/research/data/neuroscience",
                    "batchfile_directory": "/research/batch_definitions",
                    "backup_directory": "/research/backups"
                },
                "ignore_substrings": [
                    "static_horiz_ribbon", "._", ".DS_Store", "__pycache__",
                    ".tmp", "backup_", "test_calibration"
                ],
                "mandatory_substrings": ["experiment_", "data_"],
                "extraction_patterns": [
                    r".*_(?P<date>\d{8})_(?P<condition>\w+)_(?P<replicate>\d+)\.csv",
                    r"(?P<rig>\w+)_(?P<date>\d{8})_(?P<animal_id>\w+)_(?P<trial>\d+)"
                ],
                "file_extensions": [".csv", ".pkl", ".pickle", ".json"],
                "max_file_size_mb": 500,
                "parallel_processing": True
            },
            "rigs": {
                "old_opto": {
                    "sampling_frequency": 60, "mm_per_px": 0.154,
                    "camera_resolution": [1024, 768], "calibration_date": "2024-01-15"
                },
                "new_opto": {
                    "sampling_frequency": 60, "mm_per_px": 0.1818,
                    "camera_resolution": [1280, 1024], "calibration_date": "2024-06-01"
                }
            },
            "datasets": {
                "baseline_behavior": {
                    "rig": "old_opto", "patterns": ["*baseline*", "*control*"],
                    "dates_vials": {
                        "20241220": [1, 2, 3, 4, 5], "20241221": [1, 2, 3]
                    },
                    "metadata": {
                        "extraction_patterns": [r".*_(?P<dataset>\w+)_(?P<date>\d{8})_(?P<vial>\d+)\.csv"],
                        "required_fields": ["dataset", "date", "vial"],
                        "experiment_type": "baseline"
                    }
                },
                "optogenetic_stimulation": {
                    "rig": "new_opto", "patterns": ["*opto*", "*stim*"],
                    "dates_vials": {"20241218": [1, 2, 3, 4], "20241219": [1, 2, 3, 4, 5, 6]},
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
                        "study_type": "control",
                        "principal_investigator": "Dr. Test",
                        "grant_number": "TEST-123456"
                    }
                }
            }
        }
    
    def add_configuration(self, config_name: str, config_data: Dict[str, Any]):
        """Add a custom configuration for testing."""
        self.configurations[config_name] = copy.deepcopy(config_data)
    
    def add_malformed_configuration(self, config_name: str, error: Exception):
        """Add a configuration that should raise an error when loaded."""
        self.error_scenarios[config_name] = error
        self.validation_errors.add(config_name)
    
    def load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Mock configuration loading with error simulation."""
        path_str = str(config_path)
        config_name = Path(config_path).stem
        
        # Check for configured errors
        if config_name in self.error_scenarios:
            raise self.error_scenarios[config_name]
        
        # Return specific configuration or default
        if config_name in self.configurations:
            return copy.deepcopy(self.configurations[config_name])
        elif 'default' in self.configurations:
            return copy.deepcopy(self.configurations['default'])
        else:
            raise FileNotFoundError(f"Configuration not found: {config_path}")
    
    def get_ignore_patterns(self, config: Dict[str, Any], experiment: Optional[str] = None) -> List[str]:
        """Get ignore patterns from configuration with experiment-specific overrides."""
        patterns = []
        
        # Global ignore patterns
        if 'project' in config and 'ignore_substrings' in config['project']:
            patterns.extend(config['project']['ignore_substrings'])
        
        # Experiment-specific patterns
        if experiment and 'experiments' in config and experiment in config['experiments']:
            exp_config = config['experiments'][experiment]
            if 'ignore_substrings' in exp_config:
                patterns.extend(exp_config['ignore_substrings'])
        
        return patterns
    
    def get_mandatory_substrings(self, config: Dict[str, Any], experiment: Optional[str] = None) -> List[str]:
        """Get mandatory substrings from configuration with experiment-specific overrides."""
        patterns = []
        
        # Global mandatory patterns
        if 'project' in config and 'mandatory_substrings' in config['project']:
            patterns.extend(config['project']['mandatory_substrings'])
        
        # Experiment-specific patterns
        if experiment and 'experiments' in config and experiment in config['experiments']:
            exp_config = config['experiments'][experiment]
            if 'mandatory_substrings' in exp_config:
                patterns.extend(exp_config['mandatory_substrings'])
        
        return patterns
    
    def get_dataset_info(self, config: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
        """Get dataset information from configuration."""
        if 'datasets' not in config:
            raise ValueError(f"No datasets section in configuration")
        
        if dataset_name not in config['datasets']:
            raise ValueError(f"Dataset '{dataset_name}' not found in configuration")
        
        return copy.deepcopy(config['datasets'][dataset_name])
    
    def get_experiment_info(self, config: Dict[str, Any], experiment_name: str) -> Dict[str, Any]:
        """Get experiment information from configuration."""
        if 'experiments' not in config:
            raise ValueError(f"No experiments section in configuration")
        
        if experiment_name not in config['experiments']:
            raise ValueError(f"Experiment '{experiment_name}' not found in configuration")
        
        return copy.deepcopy(config['experiments'][experiment_name])
    
    def reset(self):
        """Reset all mock configuration state for clean test isolation."""
        self.configurations.clear()
        self.error_scenarios.clear()
        self.validation_errors.clear()
        self._setup_default_configurations()


# ============================================================================
# FACTORY FUNCTIONS FOR MOCK CREATION
# ============================================================================

def create_mock_filesystem(structure: Optional[Dict[str, Any]] = None,
                          unicode_files: bool = False,
                          corrupted_files: bool = False) -> MockFilesystem:
    """
    Factory function for creating consistent filesystem mocks across test modules.
    
    Args:
        structure: Dictionary defining the filesystem structure to create
        unicode_files: Whether to include Unicode filename test scenarios
        corrupted_files: Whether to include corrupted file test scenarios
        
    Returns:
        Configured MockFilesystem instance
        
    Example:
        # Basic filesystem mock
        fs_mock = create_mock_filesystem()
        
        # With custom structure
        structure = {
            'files': {
                '/test/data.pkl': {'size': 1024, 'content': b'test data'},
                '/test/config.yaml': {'size': 512, 'content': 'test: config'}
            },
            'directories': ['/test', '/test/subdir']
        }
        fs_mock = create_mock_filesystem(structure, unicode_files=True)
    """
    filesystem = MockFilesystem()
    
    # Create basic test structure if none provided
    if structure is None:
        structure = {
            'files': {
                '/test/data/experiment_001.pkl': {'size': 1024},
                '/test/data/experiment_002.pkl': {'size': 2048},
                '/test/config/test_config.yaml': {'size': 512}
            },
            'directories': ['/test', '/test/data', '/test/config']
        }
    
    # Create directories
    for directory in structure.get('directories', []):
        filesystem.add_directory(directory)
    
    # Create files
    for file_path, file_config in structure.get('files', {}).items():
        filesystem.add_file(file_path, **file_config)
    
    # Add Unicode files if requested
    if unicode_files:
        filesystem.add_unicode_files('/test/unicode', count=3)
    
    # Add corrupted files if requested
    if corrupted_files:
        filesystem.add_corrupted_scenarios('/test/corrupted')
    
    return filesystem


def create_mock_dataloader(scenarios: Optional[List[str]] = None,
                          include_experimental_data: bool = True) -> MockDataLoading:
    """
    Factory function for generating standardized data loading mocks for various test scenarios.
    
    Args:
        scenarios: List of scenario names to include ('basic', 'corrupted', 'network', 'memory')
        include_experimental_data: Whether to include realistic experimental data structures
        
    Returns:
        Configured MockDataLoading instance
        
    Example:
        # Basic data loader mock
        loader_mock = create_mock_dataloader()
        
        # With specific scenarios
        loader_mock = create_mock_dataloader(['corrupted', 'network'])
    """
    dataloader = MockDataLoading()
    
    if scenarios is None:
        scenarios = ['basic']
    
    # Basic scenario - standard files
    if 'basic' in scenarios:
        dataloader.add_experimental_matrix('/test/basic_experiment.pkl', n_timepoints=1000)
        dataloader.add_dataframe_file('/test/basic_dataframe.pkl', shape=(500, 4))
        dataloader.add_yaml_file('/test/basic_config.yaml', {
            'project': {'name': 'test'}, 
            'datasets': {'test': {'patterns': ['*test*']}}
        })
    
    # Corrupted files scenario
    if 'corrupted' in scenarios:
        dataloader.add_pickle_file(
            '/test/corrupted.pkl', 
            None, 
            error=pickle.UnpicklingError("Corrupted pickle file"),
            corrupted=True
        )
        dataloader.add_yaml_file(
            '/test/malformed.yaml',
            None,
            malformed=True
        )
    
    # Network delay scenario
    if 'network' in scenarios:
        dataloader.add_experimental_matrix('/test/remote_data.pkl', n_timepoints=2000)
        dataloader.add_network_delay('/test/remote_data.pkl', 0.1)  # 100ms delay
    
    # Memory constraint scenario
    if 'memory' in scenarios:
        dataloader.add_experimental_matrix('/test/large_data.pkl', n_timepoints=100000)
        dataloader.simulate_memory_pressure('/test/large_data.pkl', 1000)  # 1GB limit
    
    # Experimental data scenarios
    if include_experimental_data and np:
        # Baseline experiment
        dataloader.add_experimental_matrix(
            '/test/baseline_experiment.pkl',
            n_timepoints=18000,  # 5 minutes at 60 Hz
            include_signal=False,
            include_metadata=True
        )
        
        # Optogenetic experiment  
        dataloader.add_experimental_matrix(
            '/test/opto_experiment.pkl',
            n_timepoints=36000,  # 10 minutes at 60 Hz
            include_signal=True,
            include_metadata=True
        )
    
    return dataloader


def create_mock_config_provider(config_type: str = 'default',
                               include_errors: bool = False) -> MockConfigurationProvider:
    """
    Factory function for creating consistent configuration provider mocks.
    
    Args:
        config_type: Type of configuration to set up ('default', 'comprehensive', 'minimal')
        include_errors: Whether to include error scenarios for testing
        
    Returns:
        Configured MockConfigurationProvider instance
        
    Example:
        # Default configuration provider
        config_mock = create_mock_config_provider()
        
        # Comprehensive provider with error scenarios
        config_mock = create_mock_config_provider('comprehensive', include_errors=True)
    """
    provider = MockConfigurationProvider()
    
    # Add error scenarios if requested
    if include_errors:
        if yaml:
            provider.add_malformed_configuration(
                'malformed_config',
                yaml.YAMLError("Invalid YAML syntax in test")
            )
        provider.add_malformed_configuration(
            'missing_config',
            FileNotFoundError("Configuration file not found")
        )
        provider.add_malformed_configuration(
            'invalid_structure',
            ValueError("Invalid configuration structure")
        )
    
    # Add minimal configuration for edge-case testing
    if config_type in ['minimal', 'comprehensive']:
        provider.add_configuration('minimal', {
            'project': {'name': 'minimal_test'},
            'datasets': {}
        })
    
    return provider


# ============================================================================
# EDGE-CASE SCENARIO GENERATORS
# ============================================================================

class EdgeCaseScenarioGenerator:
    """
    Comprehensive edge-case scenario generator for boundary condition validation
    and error path testing across all test categories.
    """
    
    @staticmethod
    def generate_unicode_scenarios(count: int = 5) -> List[Dict[str, Any]]:
        """
        Generate Unicode path and filename scenarios for cross-platform testing.
        
        Args:
            count: Number of Unicode scenarios to generate
            
        Returns:
            List of Unicode test scenarios
        """
        unicode_patterns = [
            "tëst_fïlé_{}.csv",
            "dàtä_ñãmé_{}.pkl", 
            "ëxpérîmént_{}.yaml",
            "ūnïcōdė_tėst_{}.json",
            "rėsėärçh_dātā_{}.txt",
            "néūrōsçīéñçé_{}.csv",
            "bëhävîör_stūdy_{}.pkl"
        ]
        
        unicode_dirs = [
            "ūnïcōdė_dïr",
            "tëst_fōldėr", 
            "ëxpėrïmėnt_dāta",
            "rėsėārçh_fïlës",
            "dātā_ånålysīs"
        ]
        
        scenarios = []
        for i in range(min(count, len(unicode_patterns))):
            scenario = {
                'type': 'unicode_path',
                'filename': unicode_patterns[i].format(i),
                'directory': unicode_dirs[i % len(unicode_dirs)],
                'full_path': f"{unicode_dirs[i % len(unicode_dirs)]}/{unicode_patterns[i].format(i)}",
                'encoding': 'utf-8',
                'platform_specific': platform.system(),
                'expected_issues': []
            }
            
            # Add platform-specific considerations
            if platform.system() == "Windows":
                scenario['expected_issues'].append('path_length_limit')
            
            scenarios.append(scenario)
        
        return scenarios
    
    @staticmethod
    def generate_boundary_conditions(data_types: List[str] = None) -> Dict[str, List[Any]]:
        """
        Generate boundary condition test data for various data types.
        
        Args:
            data_types: List of data types to generate boundary conditions for
            
        Returns:
            Dictionary mapping data types to boundary condition values
        """
        if data_types is None:
            data_types = ['array_size', 'file_size', 'memory_size', 'time_duration']
        
        boundary_conditions = {}
        
        if 'array_size' in data_types:
            boundary_conditions['array_size'] = [
                0,      # Empty array
                1,      # Single element
                2,      # Minimal array
                10,     # Small array
                1000,   # Medium array
                100000, # Large array
                1000000 # Very large array
            ]
        
        if 'file_size' in data_types:
            boundary_conditions['file_size'] = [
                0,          # Empty file (0 bytes)
                1,          # 1 byte
                1024,       # 1 KB
                1024**2,    # 1 MB
                10*1024**2, # 10 MB
                100*1024**2,# 100 MB
                1024**3     # 1 GB
            ]
        
        if 'memory_size' in data_types:
            boundary_conditions['memory_size'] = [
                1,      # 1 MB
                10,     # 10 MB
                100,    # 100 MB
                1000,   # 1 GB
                4000,   # 4 GB
                8000    # 8 GB
            ]
        
        if 'time_duration' in data_types:
            boundary_conditions['time_duration'] = [
                0.001,  # 1 ms
                0.016,  # One frame at 60 Hz
                1.0,    # 1 second
                60.0,   # 1 minute
                300.0,  # 5 minutes
                3600.0, # 1 hour
                86400.0 # 1 day
            ]
        
        return boundary_conditions
    
    @staticmethod
    def generate_corrupted_data_scenarios() -> List[Dict[str, Any]]:
        """
        Generate various corrupted data scenarios for error handling testing.
        
        Returns:
            List of corrupted data test scenarios
        """
        scenarios = [
            {
                'type': 'truncated_pickle',
                'description': 'Pickle file truncated in middle',
                'data': b'truncated pickle data\x80\x03}',
                'expected_error': pickle.UnpicklingError
            },
            {
                'type': 'invalid_pickle_header',
                'description': 'Invalid pickle protocol header',
                'data': b'invalid pickle header data',
                'expected_error': pickle.UnpicklingError
            },
            {
                'type': 'malformed_yaml',
                'description': 'YAML with missing closing brackets',
                'data': 'project:\n  datasets: [\n    - name: test',
                'expected_error': yaml.YAMLError if yaml else ValueError
            },
            {
                'type': 'binary_in_text',
                'description': 'Binary data in text file',
                'data': 'valid text\x00\x01\x02binary data\xFF\xFE',
                'expected_error': UnicodeDecodeError
            },
            {
                'type': 'incomplete_csv',
                'description': 'CSV file with incomplete rows',
                'data': 't,x,y\n0.0,10.5,\n0.016,10.6',
                'expected_error': ValueError
            },
            {
                'type': 'empty_file',
                'description': 'Empty file with expected extension',
                'data': b'',
                'expected_error': EOFError
            }
        ]
        
        return scenarios
    
    @staticmethod
    def generate_memory_constraint_scenarios(include_large: bool = True) -> List[Dict[str, Any]]:
        """
        Generate memory constraint scenarios for resource limit testing.
        
        Args:
            include_large: Whether to include large memory scenarios
            
        Returns:
            List of memory constraint test scenarios
        """
        scenarios = [
            {
                'type': 'small_memory',
                'description': 'Small memory allocation',
                'target_size_mb': 1,
                'array_count': 1,
                'data_type': 'float64'
            },
            {
                'type': 'medium_memory',
                'description': 'Medium memory allocation',
                'target_size_mb': 10,
                'array_count': 5,
                'data_type': 'float32'
            },
            {
                'type': 'fragmented_memory',
                'description': 'Fragmented memory allocation',
                'target_size_mb': 5,
                'array_count': 100,
                'data_type': 'int32'
            }
        ]
        
        if include_large:
            scenarios.extend([
                {
                    'type': 'large_memory',
                    'description': 'Large memory allocation',
                    'target_size_mb': 100,
                    'array_count': 1,
                    'data_type': 'float64'
                },
                {
                    'type': 'very_large_memory',
                    'description': 'Very large memory allocation',
                    'target_size_mb': 500,
                    'array_count': 1,
                    'data_type': 'float64'
                }
            ])
        
        return scenarios
    
    @staticmethod
    def generate_concurrent_access_scenarios() -> List[Dict[str, Any]]:
        """
        Generate concurrent access scenarios for threading/multiprocessing tests.
        
        Returns:
            List of concurrent access test scenarios
        """
        scenarios = [
            {
                'type': 'simultaneous_read',
                'description': 'Multiple processes reading same file',
                'process_count': 4,
                'access_pattern': 'read_only',
                'duration_seconds': 1,
                'expected_conflicts': False
            },
            {
                'type': 'read_write_conflict',
                'description': 'Read while another process writes',
                'process_count': 2,
                'access_pattern': 'read_write',
                'duration_seconds': 2,
                'expected_conflicts': True
            },
            {
                'type': 'write_write_conflict',
                'description': 'Multiple processes writing same file',
                'process_count': 3,
                'access_pattern': 'write_only',
                'duration_seconds': 1,
                'expected_conflicts': True
            },
            {
                'type': 'file_locking',
                'description': 'File locking mechanism test',
                'process_count': 2,
                'access_pattern': 'locked_access',
                'duration_seconds': 3,
                'expected_conflicts': False
            }
        ]
        
        return scenarios


def generate_edge_case_scenarios(scenario_types: Optional[List[str]] = None,
                               include_platform_specific: bool = True) -> Dict[str, List[Dict[str, Any]]]:
    """
    Master function for generating comprehensive edge-case testing scenarios.
    
    Args:
        scenario_types: List of scenario types to generate
        include_platform_specific: Whether to include platform-specific scenarios
        
    Returns:
        Dictionary mapping scenario types to lists of test scenarios
    """
    if scenario_types is None:
        scenario_types = ['unicode', 'boundary', 'corrupted', 'memory', 'concurrent']
    
    generator = EdgeCaseScenarioGenerator()
    scenarios = {}
    
    if 'unicode' in scenario_types:
        scenarios['unicode'] = generator.generate_unicode_scenarios()
    
    if 'boundary' in scenario_types:
        boundary_data = generator.generate_boundary_conditions()
        scenarios['boundary'] = [
            {'type': 'boundary', 'data_type': dt, 'values': values}
            for dt, values in boundary_data.items()
        ]
    
    if 'corrupted' in scenario_types:
        scenarios['corrupted'] = generator.generate_corrupted_data_scenarios()
    
    if 'memory' in scenario_types:
        scenarios['memory'] = generator.generate_memory_constraint_scenarios()
    
    if 'concurrent' in scenario_types:
        scenarios['concurrent'] = generator.generate_concurrent_access_scenarios()
    
    # Add platform-specific scenarios
    if include_platform_specific:
        platform_scenarios = []
        
        if platform.system() == "Windows":
            platform_scenarios.extend([
                {
                    'type': 'windows_path_length',
                    'description': 'Windows long path limitation',
                    'path': 'C:\\' + 'very_long_directory_name\\' * 20 + 'file.pkl',
                    'expected_error': OSError
                },
                {
                    'type': 'windows_reserved_names',
                    'description': 'Windows reserved filename',
                    'path': 'CON.pkl',
                    'expected_error': OSError
                }
            ])
        
        if platform_scenarios:
            scenarios['platform_specific'] = platform_scenarios
    
    return scenarios


# ============================================================================
# TEST STRUCTURE VALIDATION UTILITIES
# ============================================================================

class TestStructureValidator:
    """
    Utilities for validating and enforcing AAA (Arrange-Act-Assert) patterns
    and consistent test structure across all test modules.
    """
    
    @staticmethod
    def validate_test_function_structure(test_func: Callable) -> Dict[str, Any]:
        """
        Validate that a test function follows AAA pattern and naming conventions.
        
        Args:
            test_func: Test function to validate
            
        Returns:
            Dictionary with validation results and recommendations
        """
        import inspect
        
        validation_result = {
            'function_name': test_func.__name__,
            'follows_naming_convention': False,
            'has_docstring': False,
            'aaa_pattern_detected': False,
            'issues': [],
            'recommendations': []
        }
        
        # Check naming convention
        if test_func.__name__.startswith('test_'):
            validation_result['follows_naming_convention'] = True
        else:
            validation_result['issues'].append('Function name does not start with "test_"')
            validation_result['recommendations'].append('Rename function to start with "test_"')
        
        # Check docstring
        if test_func.__doc__ and test_func.__doc__.strip():
            validation_result['has_docstring'] = True
        else:
            validation_result['issues'].append('Function lacks descriptive docstring')
            validation_result['recommendations'].append('Add docstring describing test purpose')
        
        # Analyze function source for AAA pattern
        try:
            source_lines = inspect.getsource(test_func).split('\n')
            aaa_keywords = {
                'arrange': ['setup', 'given', 'arrange', 'prepare'],
                'act': ['when', 'act', 'execute', 'call'],
                'assert': ['then', 'assert', 'expect', 'verify']
            }
            
            detected_sections = set()
            for line in source_lines:
                line_lower = line.lower().strip()
                for section, keywords in aaa_keywords.items():
                    if any(keyword in line_lower for keyword in keywords):
                        detected_sections.add(section)
            
            if len(detected_sections) >= 2:
                validation_result['aaa_pattern_detected'] = True
            else:
                validation_result['issues'].append('AAA pattern not clearly evident')
                validation_result['recommendations'].append('Structure test with clear Arrange-Act-Assert sections')
        
        except Exception as e:
            validation_result['issues'].append(f'Could not analyze source code: {e}')
        
        return validation_result
    
    @staticmethod
    def generate_aaa_template(test_name: str, 
                            test_type: str = 'unit',
                            includes_mocks: bool = False) -> str:
        """
        Generate AAA pattern template for consistent test structure.
        
        Args:
            test_name: Name of the test function
            test_type: Type of test (unit, integration, performance)
            includes_mocks: Whether test uses mock objects
            
        Returns:
            Template string for test function
        """
        mock_setup = """
    # Mock setup for dependency injection
    mock_dependency = create_mock_dependency()
    mock_dependency.setup_behavior()
""" if includes_mocks else ""
        
        template = f'''def test_{test_name}():
    """
    Test {test_name.replace('_', ' ')} functionality.
    
    Type: {test_type} test
    Validates: [Describe what is being validated]
    Edge cases: [List any edge cases covered]
    """
    # ARRANGE - Set up test data and dependencies{mock_setup}
    test_input = setup_test_input()
    expected_output = define_expected_output()
    
    # ACT - Execute the function under test
    actual_output = function_under_test(test_input)
    
    # ASSERT - Verify the results
    assert actual_output == expected_output
    assert additional_conditions_met(actual_output)
    
    # Optional: Verify mock interactions if applicable
    {f"mock_dependency.verify_interactions()" if includes_mocks else "# No mock verification needed"}
'''
        return template
    
    @staticmethod
    def check_fixture_naming_conventions(fixture_names: List[str]) -> Dict[str, List[str]]:
        """
        Check fixture names against standardized naming conventions.
        
        Args:
            fixture_names: List of fixture names to check
            
        Returns:
            Dictionary categorizing fixtures by naming convention compliance
        """
        conventions = {
            'mock_': 'Mock objects and simulated components',
            'temp_': 'Temporary resources requiring cleanup',
            'sample_': 'Test data and synthetic datasets',
            'fixture_': 'Complex setup scenarios and configurations',
            'generate_': 'Data generation and factory functions',
            'validate_': 'Test structure and pattern validation utilities',
            'create_': 'Factory functions for creating test objects',
            'scenario_': 'Edge-case and boundary condition test scenarios'
        }
        
        result = {
            'compliant': [],
            'non_compliant': [],
            'recommendations': {}
        }
        
        for fixture_name in fixture_names:
            is_compliant = False
            for prefix, description in conventions.items():
                if fixture_name.startswith(prefix):
                    result['compliant'].append(fixture_name)
                    is_compliant = True
                    break
            
            if not is_compliant:
                result['non_compliant'].append(fixture_name)
                # Suggest appropriate prefix based on fixture name
                if 'mock' in fixture_name.lower():
                    result['recommendations'][fixture_name] = 'Consider renaming with "mock_" prefix'
                elif 'temp' in fixture_name.lower() or 'tmp' in fixture_name.lower():
                    result['recommendations'][fixture_name] = 'Consider renaming with "temp_" prefix'
                elif 'sample' in fixture_name.lower() or 'data' in fixture_name.lower():
                    result['recommendations'][fixture_name] = 'Consider renaming with "sample_" prefix'
                else:
                    result['recommendations'][fixture_name] = 'Consider using standardized prefix'
        
        return result


def validate_test_structure(test_module: Any) -> Dict[str, Any]:
    """
    Comprehensive test module structure validation function.
    
    Args:
        test_module: Test module to validate
        
    Returns:
        Dictionary with comprehensive validation results
    """
    import inspect
    
    validator = TestStructureValidator()
    
    # Get all test functions from module
    test_functions = [
        obj for name, obj in inspect.getmembers(test_module)
        if inspect.isfunction(obj) and name.startswith('test_')
    ]
    
    # Get all fixtures from module
    fixtures = [
        obj for name, obj in inspect.getmembers(test_module)
        if hasattr(obj, '_pytestfixturefunction')
    ]
    
    validation_results = {
        'module_name': test_module.__name__ if hasattr(test_module, '__name__') else 'unknown',
        'test_function_count': len(test_functions),
        'fixture_count': len(fixtures),
        'test_function_validation': [],
        'fixture_naming_validation': {},
        'overall_compliance': False,
        'recommendations': []
    }
    
    # Validate each test function
    compliant_functions = 0
    for test_func in test_functions:
        func_validation = validator.validate_test_function_structure(test_func)
        validation_results['test_function_validation'].append(func_validation)
        
        if (func_validation['follows_naming_convention'] and 
            func_validation['has_docstring']):
            compliant_functions += 1
    
    # Validate fixture naming
    fixture_names = [getattr(fixture, '_pytestfixturefunction').name 
                    for fixture in fixtures if hasattr(fixture, '_pytestfixturefunction')]
    validation_results['fixture_naming_validation'] = validator.check_fixture_naming_conventions(fixture_names)
    
    # Calculate overall compliance
    function_compliance = compliant_functions / max(len(test_functions), 1)
    fixture_compliance = len(validation_results['fixture_naming_validation']['compliant']) / max(len(fixture_names), 1)
    overall_compliance = (function_compliance + fixture_compliance) / 2
    
    validation_results['overall_compliance'] = overall_compliance > 0.8
    validation_results['compliance_score'] = overall_compliance
    
    # Generate recommendations
    if function_compliance < 0.8:
        validation_results['recommendations'].append('Improve test function naming and documentation')
    if fixture_compliance < 0.8:
        validation_results['recommendations'].append('Standardize fixture naming conventions')
    
    return validation_results


# ============================================================================
# HYPOTHESIS STRATEGIES FOR DOMAIN-SPECIFIC TESTING
# ============================================================================

class FlyrigloaderStrategies:
    """
    Domain-specific Hypothesis strategies for flyrigloader property-based testing
    including neuroscience data patterns and configuration validation scenarios.
    """
    
    @staticmethod
    def experimental_file_paths():
        """Generate realistic experimental file path strategies."""
        animal_ids = st.text(
            alphabet="abcdefghijklmnopqrstuvwxyz0123456789_",
            min_size=3,
            max_size=15
        ).filter(lambda x: len(x.strip('_')) > 2)
        
        dates = st.dates(
            min_value=datetime(2020, 1, 1).date(),
            max_value=datetime(2024, 12, 31).date()
        ).map(lambda d: d.strftime("%Y%m%d"))
        
        conditions = st.sampled_from([
            "control", "treatment", "baseline", "stimulation",
            "drug_a", "drug_b", "sham", "recovery", "opto"
        ])
        
        extensions = st.sampled_from([".pkl", ".csv", ".json"])
        
        return st.builds(
            lambda animal, date, condition, rep, ext: f"{animal}_{date}_{condition}_rep{rep}{ext}",
            animal=animal_ids,
            date=dates,
            condition=conditions,
            rep=st.integers(min_value=1, max_value=10),
            ext=extensions
        )
    
    @staticmethod
    def neuroscience_time_series():
        """Generate realistic neuroscience time series data strategies."""
        if not np:
            return st.just({'t': [0, 1, 2], 'x': [0, 1, 2], 'y': [0, 1, 2]})
        
        def generate_time_series(n_points, sampling_freq, arena_size):
            """Generate synthetic neuroscience time series."""
            dt = 1.0 / sampling_freq
            t = np.arange(n_points) * dt
            
            # Generate realistic fly trajectory
            x = np.random.rand(n_points) * arena_size
            y = np.random.rand(n_points) * arena_size
            
            # Add some smoothing for realistic movement
            if n_points > 5:
                from scipy.ndimage import gaussian_filter1d
                try:
                    x = gaussian_filter1d(x, sigma=1.0)
                    y = gaussian_filter1d(y, sigma=1.0)
                except ImportError:
                    # Fallback if scipy not available
                    pass
            
            return {'t': t, 'x': x, 'y': y}
        
        return st.builds(
            generate_time_series,
            n_points=st.integers(min_value=10, max_value=10000),
            sampling_freq=st.sampled_from([30, 60, 120, 200]),
            arena_size=st.sampled_from([100, 120, 150, 200])
        )
    
    @staticmethod
    def experimental_configurations():
        """Generate realistic experimental configuration strategies."""
        project_names = st.text(
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Pc")),
            min_size=5,
            max_size=50
        ).filter(lambda x: x.strip() and not x.startswith('_'))
        
        ignore_patterns = st.lists(
            st.sampled_from([
                "backup", "temp", ".DS_Store", "__pycache__",
                "static_horiz_ribbon", "._", ".tmp"
            ]),
            min_size=0,
            max_size=5
        )
        
        file_extensions = st.lists(
            st.sampled_from([".pkl", ".csv", ".json", ".yaml", ".pickle"]),
            min_size=1,
            max_size=4
        )
        
        return st.fixed_dictionaries({
            'project': st.fixed_dictionaries({
                'name': project_names,
                'ignore_substrings': ignore_patterns,
                'file_extensions': file_extensions,
                'max_file_size_mb': st.integers(min_value=1, max_value=1000)
            }),
            'rigs': st.dictionaries(
                keys=st.text(min_size=3, max_size=20),
                values=st.fixed_dictionaries({
                    'sampling_frequency': st.sampled_from([30, 60, 120, 200]),
                    'mm_per_px': st.floats(min_value=0.05, max_value=0.5),
                    'arena_diameter_mm': st.integers(min_value=50, max_value=300)
                }),
                min_size=1,
                max_size=3
            )
        })
    
    @staticmethod
    def corrupted_data_patterns():
        """Generate patterns that should cause data loading failures."""
        corruption_types = st.sampled_from([
            'truncated_pickle',
            'invalid_header',
            'binary_in_text',
            'malformed_yaml',
            'empty_file',
            'permission_denied'
        ])
        
        corruption_positions = st.sampled_from([
            'beginning', 'middle', 'end', 'random'
        ])
        
        return st.fixed_dictionaries({
            'corruption_type': corruption_types,
            'position': corruption_positions,
            'severity': st.sampled_from(['minor', 'moderate', 'severe']),
            'recoverable': st.booleans()
        })
    
    @staticmethod
    def memory_usage_patterns():
        """Generate memory usage patterns for performance testing."""
        return st.fixed_dictionaries({
            'array_sizes': st.lists(
                st.integers(min_value=100, max_value=1000000),
                min_size=1,
                max_size=10
            ),
            'data_types': st.lists(
                st.sampled_from(['float32', 'float64', 'int32', 'int64']),
                min_size=1,
                max_size=4
            ),
            'allocation_pattern': st.sampled_from([
                'sequential', 'fragmented', 'burst', 'gradual'
            ]),
            'expected_peak_mb': st.integers(min_value=1, max_value=1000)
        })


def create_hypothesis_strategies() -> FlyrigloaderStrategies:
    """
    Factory function for creating domain-specific Hypothesis strategies.
    
    Returns:
        FlyrigloaderStrategies instance with all domain-specific strategies
        
    Example:
        strategies = create_hypothesis_strategies()
        
        @given(file_path=strategies.experimental_file_paths())
        def test_file_path_parsing(file_path):
            # Test with realistic experimental file paths
            pass
    """
    return FlyrigloaderStrategies()


# ============================================================================
# PERFORMANCE AND MEMORY TESTING UTILITIES
# ============================================================================

class PerformanceTestUtilities:
    """
    Utilities for performance testing and memory usage monitoring to support
    the testing strategy's performance test isolation requirements.
    """
    
    @staticmethod
    def create_memory_monitor(threshold_mb: float = 100) -> Callable:
        """
        Create a memory monitoring context manager for tests.
        
        Args:
            threshold_mb: Memory threshold in MB to trigger warnings
            
        Returns:
            Context manager for memory monitoring
        """
        @contextlib.contextmanager
        def memory_monitor():
            """Monitor memory usage during test execution."""
            if not psutil:
                logger.warning("psutil not available, skipping memory monitoring")
                yield
                return
            
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024
            
            logger.debug(f"Initial memory usage: {initial_memory:.2f} MB")
            
            try:
                yield
            finally:
                final_memory = process.memory_info().rss / 1024 / 1024
                memory_delta = final_memory - initial_memory
                
                logger.debug(f"Final memory usage: {final_memory:.2f} MB (delta: {memory_delta:+.2f} MB)")
                
                if memory_delta > threshold_mb:
                    logger.warning(f"Memory usage increased by {memory_delta:.2f} MB, exceeding threshold of {threshold_mb} MB")
        
        return memory_monitor
    
    @staticmethod
    def create_performance_timer(operation_name: str, 
                               expected_max_seconds: Optional[float] = None) -> Callable:
        """
        Create a performance timing context manager for tests.
        
        Args:
            operation_name: Name of the operation being timed
            expected_max_seconds: Maximum expected duration in seconds
            
        Returns:
            Context manager for performance timing
        """
        @contextlib.contextmanager
        def performance_timer():
            """Time operation execution and validate against expectations."""
            start_time = time.time()
            logger.debug(f"Starting performance timing for: {operation_name}")
            
            try:
                yield
            finally:
                end_time = time.time()
                duration = end_time - start_time
                
                logger.info(f"Performance timing for {operation_name}: {duration:.3f} seconds")
                
                if expected_max_seconds and duration > expected_max_seconds:
                    logger.warning(f"Performance SLA violation: {operation_name} took {duration:.3f}s, expected <= {expected_max_seconds:.3f}s")
                    raise AssertionError(f"Performance SLA violation: {operation_name} exceeded {expected_max_seconds}s limit")
        
        return performance_timer
    
    @staticmethod
    def simulate_large_dataset(size_mb: float, data_type: str = 'float64') -> Optional[Any]:
        """
        Generate large dataset for memory and performance testing.
        
        Args:
            size_mb: Target size in megabytes
            data_type: NumPy data type for the array
            
        Returns:
            Large array or None if NumPy not available
        """
        if not np:
            logger.warning("NumPy not available, cannot generate large dataset")
            return None
        
        # Calculate number of elements needed
        dtype = np.dtype(data_type)
        bytes_per_element = dtype.itemsize
        target_bytes = size_mb * 1024 * 1024
        num_elements = int(target_bytes / bytes_per_element)
        
        logger.debug(f"Generating {size_mb}MB dataset with {num_elements} {data_type} elements")
        
        # Generate random data
        if data_type.startswith('float'):
            return np.random.rand(num_elements).astype(data_type)
        elif data_type.startswith('int'):
            return np.random.randint(0, 1000, size=num_elements, dtype=data_type)
        else:
            return np.random.rand(num_elements).astype(data_type)
    
    @staticmethod
    def cleanup_memory():
        """Force garbage collection and memory cleanup."""
        gc.collect()
        if psutil:
            try:
                process = psutil.Process()
                logger.debug(f"Memory after cleanup: {process.memory_info().rss / 1024 / 1024:.2f} MB")
            except Exception:
                pass


def create_performance_test_utilities(enable_monitoring: bool = True) -> PerformanceTestUtilities:
    """
    Factory function for creating performance testing utilities.
    
    Args:
        enable_monitoring: Whether to enable memory and performance monitoring
        
    Returns:
        PerformanceTestUtilities instance
    """
    if not enable_monitoring:
        logger.warning("Performance monitoring disabled")
    
    return PerformanceTestUtilities()


# ============================================================================
# INTEGRATION TEST HELPERS
# ============================================================================

def create_integration_test_environment(config_type: str = 'comprehensive',
                                      include_filesystem: bool = True,
                                      include_corrupted_files: bool = True) -> Dict[str, Any]:
    """
    Create a comprehensive integration test environment with all necessary components.
    
    Args:
        config_type: Type of configuration to use ('default', 'comprehensive', 'minimal')
        include_filesystem: Whether to include filesystem mock
        include_corrupted_files: Whether to include corrupted file scenarios
        
    Returns:
        Dictionary containing all integration test components
    """
    environment = {}
    
    # Configuration provider
    environment['config_provider'] = create_mock_config_provider(
        config_type=config_type, 
        include_errors=True
    )
    
    # Data loader
    environment['data_loader'] = create_mock_dataloader(
        scenarios=['basic', 'corrupted', 'network', 'memory'],
        include_experimental_data=True
    )
    
    # Filesystem mock
    if include_filesystem:
        filesystem_structure = {
            'files': {
                '/test/data/baseline_001.pkl': {'size': 2048},
                '/test/data/opto_001.pkl': {'size': 4096},
                '/test/config/experiment.yaml': {'size': 1024}
            },
            'directories': ['/test', '/test/data', '/test/config']
        }
        
        environment['filesystem'] = create_mock_filesystem(
            structure=filesystem_structure,
            unicode_files=True,
            corrupted_files=include_corrupted_files
        )
    
    # Edge-case scenarios
    environment['edge_cases'] = generate_edge_case_scenarios(
        scenario_types=['unicode', 'boundary', 'corrupted', 'memory']
    )
    
    # Performance utilities
    environment['performance_utils'] = create_performance_test_utilities(
        enable_monitoring=psutil is not None
    )
    
    # Hypothesis strategies
    environment['strategies'] = create_hypothesis_strategies()
    
    return environment


# ============================================================================
# EXPORT ALL PUBLIC INTERFACES
# ============================================================================

__all__ = [
    # Protocol interfaces
    'MockConfigurationProvider',
    'MockFilesystemProvider', 
    'MockDataLoadingProvider',
    
    # Mock implementation classes
    'MockFilesystem',
    'MockDataLoading',
    'MockConfigurationProvider',
    
    # Factory functions
    'create_mock_filesystem',
    'create_mock_dataloader',
    'create_mock_config_provider',
    
    # Edge-case generators
    'EdgeCaseScenarioGenerator',
    'generate_edge_case_scenarios',
    
    # Test structure validation
    'TestStructureValidator',
    'validate_test_structure',
    
    # Hypothesis strategies
    'FlyrigloaderStrategies',
    'create_hypothesis_strategies',
    
    # Performance utilities
    'PerformanceTestUtilities',
    'create_performance_test_utilities',
    
    # Integration helpers
    'create_integration_test_environment'
]