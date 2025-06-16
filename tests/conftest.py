"""
Centralized pytest configuration file for flyrigloader test suite.

Provides comprehensive testing infrastructure with consolidated fixture management:
- Enhanced Loguru integration for test execution tracking
- Centralized fixture management eliminating duplication across test modules
- Standardized mocking patterns with protocol-based implementations
- Advanced synthetic experimental data generation for neuroscience workflows
- Property-based testing support with domain-specific strategies
- Edge-case testing capabilities including Unicode paths and corrupted files
- Performance benchmark fixtures with statistical analysis
- pytest-xdist parallel execution support optimized for 4-worker configuration

Centralized Test Infrastructure Features:
- TST-INF-001: Global fixture configuration with enhanced Loguru integration
- TST-INF-002: Test execution tracking and log capture with test isolation
- TST-MOD-003: Standardized mocking patterns across all test modules
- TST-INTEG-002: Comprehensive synthetic experimental data generation
- TST-COV-001: Edge-case testing fixtures for boundary condition validation
- TST-PERF-001: Performance benchmark fixtures with SLA validation
- TST-PAR-001: pytest-xdist parallel execution support with 4-worker optimization

Fixture Naming Conventions:
- mock_* : Mock objects and simulated components
- temp_* : Temporary resources requiring cleanup
- sample_* : Test data and synthetic datasets
- fixture_* : Complex setup scenarios and configurations
"""

import contextlib
import io
import logging
import os
import random
import sys
import tempfile
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Generator
from unittest.mock import MagicMock, Mock

# Add the src directory to the Python path
src_path = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src_path)

# Core testing imports
import pytest
from hypothesis import strategies as st
from hypothesis import settings

# Third-party testing utilities
try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

# Enhanced Loguru integration with fallback
try:
    from loguru import logger
except ModuleNotFoundError:  # pragma: no cover
    class _DummyLogger:
        """Fallback logger for environments without Loguru installed."""
        
        def __init__(self):
            self._logger = logging.getLogger("loguru")
            self._handlers = {}

        def debug(self, msg, *a, **k):
            self._logger.debug(msg, *a, **k)

        def info(self, msg, *a, **k):
            self._logger.info(msg, *a, **k)

        def warning(self, msg, *a, **k):
            self._logger.warning(msg, *a, **k)

        def error(self, msg, *a, **k):
            self._logger.error(msg, *a, **k)

        def add(self, handler, format=None, level=0):
            handler_id = id(handler)
            setattr(handler, "_id", handler_id)
            self._handlers[handler_id] = handler
            self._logger.setLevel(level)
            self._logger.addHandler(handler)
            return handler_id

        def remove(self, handler_id):
            if handler_id in self._handlers:
                handler = self._handlers.pop(handler_id)
                self._logger.removeHandler(handler)

    logger = _DummyLogger()


# ============================================================================
# ENHANCED LOGURU INTEGRATION FIXTURES
# ============================================================================

@pytest.fixture(autouse=True, scope="function")
def capture_loguru_logs_globally(caplog):
    """
    Enhanced autouse fixture to capture Loguru logs with improved error handling and test isolation.
    
    Features:
    - Robust Loguru-to-standard-logging bridge with comprehensive error handling
    - Complete test isolation preventing cross-test contamination
    - Enhanced log level mapping supporting all Loguru levels
    - Test execution tracking per TST-INF-001 requirements
    - pytest-xdist compatibility for parallel test execution
    
    Supports:
    - All Loguru log levels (TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL)
    - Structured log message preservation with full context
    - Exception traceback capture with variable diagnosis
    - Thread-safe operation for parallel test execution
    - Memory-efficient log capture preventing accumulation
    """
    class EnhancedPropagateHandler(logging.Handler):
        """Enhanced handler with robust error handling and thread safety."""
        
        def __init__(self):
            super().__init__()
            self._test_context = None
            
        def set_test_context(self, test_name):
            """Set current test context for enhanced logging."""
            self._test_context = test_name
        
        def emit(self, record):
            try:
                # Create or get logger in standard logging hierarchy with test context
                logger_name = f"flyrigloader.{self._test_context}" if self._test_context else "flyrigloader"
                std_logger = logging.getLogger(logger_name)
                
                # Enhanced log level mapping with precision
                level_mapping = {
                    'TRACE': logging.DEBUG - 5,
                    'DEBUG': logging.DEBUG,
                    'INFO': logging.INFO,
                    'SUCCESS': logging.INFO + 5,
                    'WARNING': logging.WARNING,
                    'ERROR': logging.ERROR,
                    'CRITICAL': logging.CRITICAL
                }
                
                # Map Loguru level to standard logging level with fallback
                original_levelno = getattr(record, 'levelno', logging.INFO)
                if hasattr(record, 'levelname') and record.levelname in level_mapping:
                    record.levelno = level_mapping[record.levelname]
                else:
                    record.levelno = original_levelno
                
                # Add test context to message if available
                if self._test_context and hasattr(record, 'msg'):
                    record.msg = f"[{self._test_context}] {record.msg}"
                
                # Handle record with enhanced error recovery
                try:
                    std_logger.handle(record)
                except (OSError, IOError, MemoryError):
                    # Critical error recovery - prevent test failure
                    pass
                    
            except Exception:
                # Comprehensive error suppression to prevent test contamination
                pass

    # Configure caplog for comprehensive log capture with memory limits
    caplog.set_level(logging.DEBUG)
    
    # Extract test name for enhanced context tracking
    test_name = "unknown_test"
    if hasattr(pytest, 'current_test') and pytest.current_test:
        test_name = pytest.current_test.nodeid.split("::")[-1]
    elif 'request' in locals():
        test_name = getattr(request.node, 'name', 'unknown_test')
    
    # Create enhanced handler with test context
    enhanced_handler = EnhancedPropagateHandler()
    enhanced_handler.set_test_context(test_name)
    
    # Add enhanced handler to Loguru with comprehensive configuration
    handler_id = logger.add(
        enhanced_handler,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        level="TRACE",
        catch=True,  # Catch exceptions in logging
        backtrace=True,  # Include backtrace in error logs
        diagnose=True,  # Include variable values in tracebacks
        enqueue=False,  # Synchronous logging for test predictability
        colorize=False,  # Disable colors for pytest compatibility
        serialize=False  # Disable serialization for performance
    )

    # Track test execution start with enhanced context
    logger.info(f"TEST_START: {test_name}")
    
    # Add memory usage tracking for large test scenarios
    initial_memory_mb = 0
    try:
        import psutil
        initial_memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        logger.debug(f"Initial memory usage: {initial_memory_mb:.2f} MB")
    except ImportError:
        pass

    yield

    # Track test execution completion with memory analysis
    try:
        if initial_memory_mb > 0:
            final_memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            memory_delta = final_memory_mb - initial_memory_mb
            logger.debug(f"Final memory usage: {final_memory_mb:.2f} MB (delta: {memory_delta:+.2f} MB)")
    except (ImportError, NameError):
        pass
    
    logger.info(f"TEST_END: {test_name}")

    # Enhanced cleanup with comprehensive error handling
    try:
        logger.remove(handler_id)
    except (ValueError, KeyError, RuntimeError):
        # Handle potential cleanup failures gracefully
        pass
    
    # Clear handler reference to prevent memory leaks
    enhanced_handler = None


@pytest.fixture(scope="session")
def test_execution_tracker():
    """
    Session-scoped fixture for tracking test execution metrics and providing
    comprehensive test session statistics per TST-INF-002 requirements.
    
    Provides:
    - Test execution timing and performance metrics
    - Coverage tracking hooks
    - Session-wide test statistics
    - Performance benchmark baseline data
    """
    class TestExecutionTracker:
        def __init__(self):
            self.session_start = datetime.now()
            self.test_count = 0
            self.test_timings = {}
            self.failed_tests = []
            self.performance_baselines = {
                'data_loading_per_mb': 1.0,  # seconds per MB
                'discovery_per_1k_files': 0.5,  # seconds per 1000 files
                'transformation_per_1m_rows': 0.5  # seconds per 1M rows
            }
    
        def start_test(self, test_name: str):
            self.test_count += 1
            self.test_timings[test_name] = datetime.now()
            logger.debug(f"Starting test {self.test_count}: {test_name}")
    
        def end_test(self, test_name: str, success: bool = True):
            if test_name in self.test_timings:
                duration = datetime.now() - self.test_timings[test_name]
                logger.debug(f"Test {test_name} completed in {duration.total_seconds():.3f}s")
                if not success:
                    self.failed_tests.append(test_name)
    
        def get_session_stats(self) -> Dict[str, Any]:
            return {
                'session_duration': datetime.now() - self.session_start,
                'total_tests': self.test_count,
                'failed_tests': len(self.failed_tests),
                'success_rate': (self.test_count - len(self.failed_tests)) / max(self.test_count, 1),
                'performance_baselines': self.performance_baselines
            }
    
    tracker = TestExecutionTracker()
    logger.info("TEST_SESSION_START: Global test execution tracking initialized")
    
    yield tracker
    
    stats = tracker.get_session_stats()
    logger.info(f"TEST_SESSION_END: {stats}")


# ============================================================================
# CENTRALIZED CONFIGURATION FIXTURES (Consolidated from flyrigloader/conftest.py)
# ============================================================================

@pytest.fixture(scope="module")
def sample_comprehensive_config_dict():
    """
    Module-scoped comprehensive sample configuration dictionary consolidating all supported features.
    
    Provides full range of configuration options for comprehensive testing scenarios:
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
                "static_horiz_ribbon", "._", ".DS_Store", "__pycache__",
                ".tmp", "backup_", "test_calibration"
            ],
            "mandatory_substrings": ["experiment_", "data_"],
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
                "sampling_frequency": 60, "mm_per_px": 0.154,
                "camera_resolution": [1024, 768], "calibration_date": "2024-01-15",
                "arena_diameter_mm": 120, "led_wavelength_nm": 470
            },
            "new_opto": {
                "sampling_frequency": 60, "mm_per_px": 0.1818,
                "camera_resolution": [1280, 1024], "calibration_date": "2024-06-01",
                "arena_diameter_mm": 150, "led_wavelength_nm": 470
            },
            "high_speed_rig": {
                "sampling_frequency": 200, "mm_per_px": 0.05,
                "camera_resolution": [2048, 2048], "calibration_date": "2024-08-15",
                "arena_diameter_mm": 200, "led_wavelength_nm": 590
            }
        },
        "datasets": {
            "baseline_behavior": {
                "rig": "old_opto", "patterns": ["*baseline*", "*control*"],
                "dates_vials": {
                    "2024-12-20": [1, 2, 3, 4, 5], "2024-12-21": [1, 2, 3], "2024-12-22": [1, 2]
                },
                "metadata": {
                    "extraction_patterns": [r".*_(?P<dataset>\w+)_(?P<date>\d{8})_(?P<vial>\d+)\.csv"],
                    "required_fields": ["dataset", "date", "vial"], "experiment_type": "baseline"
                },
                "filters": {
                    "min_duration_seconds": 300, "max_duration_seconds": 3600,
                    "required_columns": ["t", "x", "y"]
                }
            },
            "optogenetic_stimulation": {
                "rig": "new_opto", "patterns": ["*opto*", "*stim*"],
                "dates_vials": {"2024-12-18": [1, 2, 3, 4], "2024-12-19": [1, 2, 3, 4, 5, 6]},
                "metadata": {
                    "extraction_patterns": [r".*_(?P<dataset>\w+)_(?P<stimulation_type>\w+)_(?P<date>\d{8})\.csv"],
                    "required_fields": ["dataset", "stimulation_type", "date"], "experiment_type": "optogenetic"
                },
                "filters": {"ignore_substrings": ["failed", "aborted"], "min_file_size_bytes": 10000}
            }
        },
        "experiments": {
            "baseline_control_study": {
                "datasets": ["baseline_behavior"],
                "metadata": {
                    "extraction_patterns": [r".*_(?P<experiment>baseline)_(?P<date>\d{8})\.csv"],
                    "required_fields": ["experiment", "date"], "study_type": "control",
                    "principal_investigator": "Dr. Research", "grant_number": "NSF-123456"
                },
                "analysis_parameters": {
                    "velocity_threshold": 2.0, "smoothing_window": 5, "edge_exclusion_mm": 10
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

@pytest.fixture(scope="function")
def sample_config_dict(sample_comprehensive_config_dict):
    """
    Function-scoped simplified configuration dictionary for basic testing scenarios.
    
    Provides subset of comprehensive configuration for tests that don't need full complexity.
    
    Returns:
        Dict[str, Any]: Simplified sample configuration dictionary
    """
    return {
        "project": sample_comprehensive_config_dict["project"],
        "rigs": {"old_opto": sample_comprehensive_config_dict["rigs"]["old_opto"]},
        "datasets": {"test_dataset": sample_comprehensive_config_dict["datasets"]["baseline_behavior"]},
        "experiments": {"test_experiment": sample_comprehensive_config_dict["experiments"]["baseline_control_study"]}
    }

@pytest.fixture(scope="function")
def sample_config_file(sample_comprehensive_config_dict, temp_cross_platform_dir):
    """
    Function-scoped temporary config file with comprehensive sample configuration.
    
    Args:
        sample_comprehensive_config_dict: The comprehensive configuration data
        temp_cross_platform_dir: Cross-platform temporary directory fixture
    
    Returns:
        str: Path to the temporary config file
    """
    config_path = temp_cross_platform_dir / "config.yaml"
    
    # Write the config to the file with proper YAML formatting
    with open(config_path, 'w') as f:
        yaml.dump(sample_comprehensive_config_dict, f, default_flow_style=False, sort_keys=False)
    
    return str(config_path)

# ============================================================================
# CENTRALIZED TEMPORARY FILESYSTEM FIXTURES
# ============================================================================

@pytest.fixture(scope="function")
def temp_cross_platform_dir():
    """
    Function-scoped cross-platform temporary directory with proper cleanup and enhanced error handling.
    
    Handles platform-specific considerations for Windows, Linux, and macOS:
    - Uses appropriate temporary directory locations per OS
    - Ensures proper permissions across platforms
    - Handles long path limitations on Windows
    - Provides comprehensive cleanup even on test failures
    - Enhanced Unicode path support for edge-case testing
    
    Returns:
        Path: Cross-platform temporary directory path with guaranteed cleanup
    """
    import shutil
    import stat
    
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
        # Ensure the directory is writable with enhanced validation
        test_file = temp_path / "write_test.tmp"
        test_file.write_text("test", encoding='utf-8')
        test_file.unlink()
        
        # Test Unicode path support for edge-case scenarios
        unicode_test_dir = temp_path / "tëst_ùnïcødé"
        unicode_test_dir.mkdir(exist_ok=True)
        unicode_test_file = unicode_test_dir / "tëst_fïlé.txt"
        unicode_test_file.write_text("unicode test", encoding='utf-8')
        unicode_test_file.unlink()
        unicode_test_dir.rmdir()
        
        yield temp_path
        
    finally:
        # Comprehensive cleanup with enhanced error handling
        def handle_remove_readonly(func, path, exc):
            """Enhanced readonly file handler for Windows compatibility."""
            try:
                os.chmod(path, stat.S_IWRITE)
                func(path)
            except (OSError, PermissionError):
                pass  # Best effort cleanup
        
        try:
            if temp_path.exists():
                if platform.system() == "Windows":
                    shutil.rmtree(temp_path, onerror=handle_remove_readonly)
                else:
                    shutil.rmtree(temp_path, ignore_errors=True)
        except Exception:
            # Final fallback - log but don't fail test
            try:
                logger.warning(f"Failed to cleanup temporary directory: {temp_path}")
            except:
                pass

@pytest.fixture(scope="function")
def temp_filesystem_structure(temp_cross_platform_dir):
    """
    Function-scoped realistic temporary filesystem structure for comprehensive integration testing.
    
    Creates directory structure that mimics real research data organization:
    - Multiple experiments, datasets, and file types
    - Realistic experimental naming patterns
    - Cross-platform path compatibility
    - Support for Unicode filenames in edge-case testing
    
    Returns:
        Dict[str, Path]: Dictionary mapping logical names to filesystem paths
    """
    base_dir = temp_cross_platform_dir
    
    # Create comprehensive directory structure
    structure = {
        "data_root": base_dir / "research_data",
        "experiments": base_dir / "research_data" / "experiments",
        "baselines": base_dir / "research_data" / "experiments" / "baseline",
        "optogenetics": base_dir / "research_data" / "experiments" / "optogenetics", 
        "navigation": base_dir / "research_data" / "experiments" / "navigation",
        "batch_files": base_dir / "batch_definitions",
        "configs": base_dir / "configs",
        "processed": base_dir / "processed_data",
        # Edge-case testing directories
        "unicode_test": base_dir / "tëst_ùnïcødé_dïr",
        "special_chars": base_dir / "test-dir_with.special@chars"
    }
    
    # Create all directories
    for dir_path in structure.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create sample data files with realistic experimental naming
    sample_files = {
        # Standard experimental files
        "baseline_file_1": structure["baselines"] / "baseline_20241220_control_1.csv",
        "baseline_file_2": structure["baselines"] / "baseline_20241221_control_2.csv",
        "opto_file_1": structure["optogenetics"] / "opto_stim_20241218_treatment_1.csv",
        "opto_file_2": structure["optogenetics"] / "opto_stim_20241219_treatment_2.csv",
        "nav_file_1": structure["navigation"] / "plume_navigation_20241025_trial_1.csv",
        "nav_file_2": structure["navigation"] / "plume_navigation_20241025_trial_2.csv",
        
        # Edge-case testing files
        "unicode_file": structure["unicode_test"] / "dätä_fïlé_tëst.csv",
        "special_chars_file": structure["special_chars"] / "data-file_test@2024.csv",
        
        # Ignore pattern test files (should be filtered out)
        "ignored_file_1": structure["baselines"] / "static_horiz_ribbon_calibration.csv",
        "ignored_file_2": structure["optogenetics"] / "._temp_file.csv",
        "ignored_file_3": structure["navigation"] / ".DS_Store",
        
        # Configuration file
        "config_file": structure["configs"] / "experiment_config.yaml"
    }
    
    # Create realistic CSV content for experimental files
    sample_csv_content = """t,x,y,signal
0.0,10.5,20.3,0.1
0.016,10.6,20.2,0.2
0.032,10.7,20.1,0.3
0.048,10.8,20.0,0.4
"""
    
    # Create sample YAML configuration content
    sample_yaml_content = """
project:
  name: test_experiment
  mandatory_experiment_strings: []
  ignore_substrings: ["backup", "temp"]

datasets:
  neural_data:
    dates_vials:
      "20240101": ["mouse_001", "mouse_002"]
      "20240102": ["mouse_003", "mouse_004"]
"""
    
    # Write sample files with proper encoding
    for file_key, file_path in sample_files.items():
        try:
            if file_path.suffix == ".csv":
                file_path.write_text(sample_csv_content, encoding='utf-8')
            elif file_path.suffix == ".yaml":
                file_path.write_text(sample_yaml_content, encoding='utf-8')
            else:
                file_path.write_text("# Sample file", encoding='utf-8')
        except (OSError, UnicodeError) as e:
            # Handle edge cases where Unicode paths may not be supported
            logger.debug(f"Could not create edge-case file {file_path}: {e}")
            continue
    
    return {**structure, **sample_files}

# ============================================================================
# ADVANCED PYTEST FIXTURE SYSTEM
# ============================================================================

@pytest.fixture(scope="session")
def test_data_generator():
    """
    Session-scoped fixture providing comprehensive synthetic experimental data generation
    supporting NumPy and Pandas data structures per Section 2.1.11 requirements.
    
    Features:
    - Realistic experimental data patterns (neural recordings, behavioral data)
    - Configurable data sizes and complexity levels
    - Cross-platform file system simulation
    - Memory-efficient data generation strategies
    """
    class TestDataGenerator:
        def __init__(self):
            self.random_seed = 42
            random.seed(self.random_seed)
            if np:
                np.random.seed(self.random_seed)
        
        def generate_experimental_matrix(self, 
                                       rows: int = 1000, 
                                       cols: int = 50,
                                       data_type: str = "neural") -> Optional[np.ndarray]:
            """Generate synthetic experimental data matrices."""
            if not np:
                return None
                
            if data_type == "neural":
                # Simulate neural recording data with realistic noise patterns
                base_signal = np.sin(np.linspace(0, 100, rows * cols)).reshape(rows, cols)
                noise = np.random.normal(0, 0.1, (rows, cols))
                return base_signal + noise
            elif data_type == "behavioral":
                # Simulate behavioral tracking data
                return np.random.gamma(2, 2, (rows, cols))
            else:
                return np.random.normal(0, 1, (rows, cols))
        
        def generate_experiment_metadata(self, 
                                       animal_ids: Optional[List[str]] = None,
                                       date_range: int = 30) -> Dict[str, Any]:
            """Generate realistic experimental metadata."""
            if not animal_ids:
                animal_ids = [f"mouse_{i:03d}" for i in range(1, 6)]
            
            conditions = ["control", "treatment_a", "treatment_b"]
            base_date = datetime.now() - timedelta(days=date_range)
            
            return {
                "animal_id": random.choice(animal_ids),
                "condition": random.choice(conditions),
                "experiment_date": base_date + timedelta(days=random.randint(0, date_range)),
                "replicate": random.randint(1, 5),
                "session_duration_minutes": random.randint(30, 120),
                "experimenter": random.choice(["researcher_a", "researcher_b", "researcher_c"])
            }
        
        def generate_dataframe(self, 
                             rows: int = 1000,
                             include_metadata: bool = True) -> Optional[pd.DataFrame]:
            """Generate synthetic pandas DataFrame with experimental structure."""
            if not pd or not np:
                return None
            
            # Base experimental data
            data = {
                'timestamp': pd.date_range('2024-01-01', periods=rows, freq='1S'),
                'signal_1': np.random.normal(0, 1, rows),
                'signal_2': np.random.normal(5, 2, rows),
                'behavior_score': np.random.exponential(2, rows)
            }
            
            if include_metadata:
                metadata = self.generate_experiment_metadata()
                for key, value in metadata.items():
                    if key != "experiment_date":  # Skip datetime columns for simplicity
                        data[key] = [value] * rows
            
            return pd.DataFrame(data)
        
        def create_test_files(self, 
                            temp_dir: Path,
                            file_count: int = 10,
                            file_patterns: Optional[List[str]] = None) -> List[Path]:
            """Create temporary test files with realistic naming patterns."""
            if not file_patterns:
                file_patterns = [
                    "mouse_{animal_id}_{date}_{condition}_rep{replicate}.pkl",
                    "experiment_{exp_id}_{animal_id}_{condition}.pkl",
                    "behavioral_data_{date}_{animal_id}.pkl"
                ]
            
            created_files = []
            for i in range(file_count):
                metadata = self.generate_experiment_metadata()
                pattern = random.choice(file_patterns)
                
                filename = pattern.format(
                    animal_id=metadata["animal_id"],
                    date=metadata["experiment_date"].strftime("%Y%m%d"),
                    condition=metadata["condition"],
                    replicate=metadata["replicate"],
                    exp_id=f"EXP{i:03d}"
                )
                
                file_path = temp_dir / filename
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Create empty file for testing
                file_path.touch()
                created_files.append(file_path)
            
            return created_files
    
    return TestDataGenerator()

# ============================================================================
# ENHANCED SYNTHETIC EXPERIMENTAL DATA GENERATION (Consolidated)
# ============================================================================

@pytest.fixture(scope="module")
def fixture_realistic_time_series_params():
    """
    Module-scoped parameters for generating realistic experimental time series data.
    
    Provides consistent parameters across test modules for synthetic data generation
    supporting neuroscience research workflows and experimental data patterns.
    
    Returns:
        Dict: Parameters for synthetic data generation with research-relevant defaults
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

@pytest.fixture(scope="session")
def fixture_synthetic_trajectory_generator():
    """
    Session-scoped factory function for generating realistic synthetic fly trajectories.
    
    Generates biologically plausible movement patterns including:
    - Brownian motion with drift toward center
    - Realistic velocity profiles
    - Arena boundary constraints with reflection
    - Temporally correlated movement patterns
    
    Returns:
        Callable: Function that generates trajectory data with configurable parameters
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
        Generate a realistic fly trajectory with biologically plausible movement.
        
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
                # Reflect off boundary with slight inward bias
                reflection_factor = arena_radius / new_radius
                new_x *= reflection_factor * 0.95
                new_y *= reflection_factor * 0.95
            
            x_pos[i] = new_x
            y_pos[i] = new_y
        
        return time, x_pos, y_pos
    
    return generate_trajectory

@pytest.fixture(scope="session") 
def fixture_synthetic_signal_generator():
    """
    Session-scoped factory function for generating realistic multi-channel signal data.
    
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
            
            # Add harmonic components for realism
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

@pytest.fixture(scope="function")
def sample_exp_matrix_comprehensive(fixture_synthetic_trajectory_generator, 
                                   fixture_synthetic_signal_generator,
                                   fixture_realistic_time_series_params):
    """
    Function-scoped comprehensive experimental data matrix with all possible columns.
    
    Consolidates experimental data generation providing complete dataset for testing
    all column handling and transformation functionality.
    
    Returns:
        Dict[str, np.ndarray]: Comprehensive experimental data matrix
    """
    params = fixture_realistic_time_series_params
    n_points = int(params["duration_seconds"] * params["sampling_frequency"])
    
    # Generate base trajectory data
    time, x_pos, y_pos = fixture_synthetic_trajectory_generator(
        n_timepoints=n_points,
        sampling_freq=params["sampling_frequency"],
        arena_diameter=params["arena_diameter_mm"],
        seed=42  # Reproducible for tests
    )
    
    # Generate multi-channel signal data
    signal_data = fixture_synthetic_signal_generator(
        n_timepoints=n_points,
        n_channels=15,  # 15 channels as in original
        seed=42
    )
    
    # Generate single-channel signal
    single_signal = fixture_synthetic_signal_generator(
        n_timepoints=n_points,
        n_channels=1,
        seed=43
    )[0, :]
    
    # Calculate derived kinematic measures
    x_diff = np.diff(x_pos, prepend=x_pos[0])
    y_diff = np.diff(y_pos, prepend=y_pos[0])
    dt = np.diff(time, prepend=time[1] - time[0])
    
    # Construct comprehensive matrix
    matrix = {
        # Base experimental data
        't': time,
        'x': x_pos,
        'y': y_pos,
        
        # Signal data
        'signal': single_signal,
        'signal_disp': signal_data,
        
        # Kinematic measures
        'vx': x_diff / dt,
        'vy': y_diff / dt,
        'speed': np.sqrt((x_diff / dt)**2 + (y_diff / dt)**2),
        
        # Angular measures
        'dtheta': np.arctan2(y_diff, x_diff),
        'dtheta_smooth': np.arctan2(y_diff, x_diff) + 0.1 * np.random.normal(0, 1, len(time)),
        
        # Derived spatial measures
        'distance_from_center': np.sqrt(x_pos**2 + y_pos**2),
        'cumulative_distance': np.cumsum(np.sqrt(x_diff**2 + y_diff**2))
    }
    
    return matrix

@pytest.fixture(scope="function")
def sample_experimental_metadata():
    """
    Function-scoped sample metadata dictionary for experimental data testing.
    
    Provides realistic metadata structure for neuroscience experiments
    supporting comprehensive test scenarios.
    
    Returns:
        Dict[str, str]: Sample metadata with research-relevant fields
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
        'humidity_percent': '45.2',
        'arena_type': 'circular',
        'session_duration_minutes': '300'
    }

@pytest.fixture(scope="function")
def sample_pandas_dataframe_comprehensive(sample_exp_matrix_comprehensive, sample_experimental_metadata):
    """
    Function-scoped comprehensive pandas DataFrame with experimental data and metadata.
    
    Demonstrates expected output format after complete data processing pipeline.
    Consolidates matrix data with metadata for integration testing scenarios.
    
    Returns:
        pd.DataFrame: Complete DataFrame with experimental data and metadata
    """
    if not pd:
        pytest.skip("pandas not available")
    
    # Convert matrix to DataFrame
    df_data = {}
    
    # Add time series data
    for col, data in sample_exp_matrix_comprehensive.items():
        if data.ndim == 1:
            df_data[col] = data
        elif data.ndim == 2:
            # For 2D data like signal_disp, add as columns with channel indices
            for ch in range(data.shape[0]):
                df_data[f"{col}_ch{ch:02d}"] = data[ch, :]
    
    df = pd.DataFrame(df_data)
    
    # Add metadata as constant columns
    for key, value in sample_experimental_metadata.items():
        df[key] = value
    
    return df


@pytest.fixture(scope="function")
def temp_experiment_directory(tmp_path, test_data_generator):
    """
    Function-scoped fixture providing realistic temporary experimental directory
    structure with synthetic data files for comprehensive testing scenarios.
    """
    # Create realistic directory structure
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir()
    
    # Create subdirectories
    subdirs = ["raw_data", "processed", "configs", "metadata"]
    for subdir in subdirs:
        (exp_dir / subdir).mkdir()
    
    # Generate test files with realistic patterns
    raw_files = test_data_generator.create_test_files(
        exp_dir / "raw_data", 
        file_count=15,
        file_patterns=[
            "mouse_{animal_id}_{date}_{condition}_rep{replicate}.pkl",
            "rat_{animal_id}_{date}_{condition}_session{replicate}.pkl"
        ]
    )
    
    # Create configuration files
    config_content = """
project:
  name: test_experiment
  mandatory_experiment_strings: []
  ignore_substrings: ["backup", "temp"]

datasets:
  neural_data:
    dates_vials:
      "20240101": ["mouse_001", "mouse_002"]
      "20240102": ["mouse_003", "mouse_004"]

experiments:
  exp_001:
    filters:
      mandatory_experiment_strings: ["mouse"]
      ignore_substrings: ["old"]
"""
    
    config_file = exp_dir / "configs" / "experiment_config.yaml"
    config_file.write_text(config_content)
    
    return {
        "directory": exp_dir,
        "raw_files": raw_files,
        "config_file": config_file,
        "subdirs": subdirs
    }


# ============================================================================
# ENHANCED EDGE-CASE TESTING FIXTURES
# ============================================================================

@pytest.fixture(scope="function")
def fixture_unicode_path_generator(temp_cross_platform_dir):
    """
    Function-scoped Unicode path generator for cross-platform edge-case testing.
    
    Generates various Unicode path scenarios to test file handling robustness:
    - Unicode characters in directory names
    - Unicode characters in file names
    - Mixed ASCII and Unicode combinations
    - Platform-specific Unicode edge cases
    
    Returns:
        Callable: Function that generates Unicode paths for testing
    """
    def generate_unicode_paths(num_paths: int = 5) -> List[Path]:
        """Generate a variety of Unicode paths for edge-case testing."""
        unicode_patterns = [
            "tëst_fïlé_{}.csv",
            "dàtä_ñãmé_{}.pkl", 
            "ëxpérîmént_{}.yaml",
            "ūnïcōdė_tėst_{}.json",
            "rėsėärçh_dātā_{}.txt"
        ]
        
        unicode_dirs = [
            "ūnïcōdė_dïr",
            "tëst_fōldėr", 
            "ëxpėrïmėnt_dāta",
            "rėsėārçh_fïlës"
        ]
        
        generated_paths = []
        
        for i in range(num_paths):
            try:
                # Create Unicode directory
                unicode_dir = temp_cross_platform_dir / unicode_dirs[i % len(unicode_dirs)]
                unicode_dir.mkdir(exist_ok=True)
                
                # Create Unicode file path
                filename = unicode_patterns[i % len(unicode_patterns)].format(i)
                unicode_path = unicode_dir / filename
                
                # Test path creation (but don't create file yet)
                generated_paths.append(unicode_path)
                
            except (OSError, UnicodeError):
                # Some platforms may not support certain Unicode patterns
                logger.debug(f"Unicode path generation failed for pattern {i}")
                continue
        
        return generated_paths
    
    return generate_unicode_paths

@pytest.fixture(scope="function")
def fixture_corrupted_file_scenarios(temp_cross_platform_dir):
    """
    Function-scoped corrupted file scenario generator for error handling testing.
    
    Creates various corrupted file scenarios to test error handling robustness:
    - Corrupted pickle files
    - Malformed YAML configurations  
    - Truncated CSV files
    - Binary data in text files
    - Empty files with expected extensions
    
    Returns:
        Dict[str, Path]: Dictionary mapping scenario names to corrupted file paths
    """
    corrupted_dir = temp_cross_platform_dir / "corrupted_files"
    corrupted_dir.mkdir(exist_ok=True)
    
    scenarios = {}
    
    # Corrupted pickle file
    corrupted_pickle = corrupted_dir / "corrupted_data.pkl"
    corrupted_pickle.write_bytes(b"not a pickle file - corrupted data")
    scenarios["corrupted_pickle"] = corrupted_pickle
    
    # Malformed YAML file
    malformed_yaml = corrupted_dir / "malformed_config.yaml"
    malformed_yaml.write_text("invalid: yaml: content: [\n  - missing closing", encoding='utf-8')
    scenarios["malformed_yaml"] = malformed_yaml
    
    # Truncated CSV file
    truncated_csv = corrupted_dir / "truncated_data.csv"
    truncated_csv.write_text("t,x,y\n0.0,10.5,", encoding='utf-8')  # Incomplete line
    scenarios["truncated_csv"] = truncated_csv
    
    # Binary data in text file
    binary_in_text = corrupted_dir / "binary_contaminated.csv" 
    binary_in_text.write_bytes(b"t,x,y\n0.0,10.5,20.3\n\x00\x01\x02invalid binary data")
    scenarios["binary_contaminated"] = binary_in_text
    
    # Empty file with expected extension
    empty_pickle = corrupted_dir / "empty_file.pkl"
    empty_pickle.write_bytes(b"")
    scenarios["empty_pickle"] = empty_pickle
    
    # Oversized fake file (for memory testing)
    fake_large_file = corrupted_dir / "fake_large_file.pkl"
    fake_large_file.write_text("fake large file content" * 1000, encoding='utf-8')
    scenarios["fake_large_file"] = fake_large_file
    
    return scenarios

@pytest.fixture(scope="function")
def fixture_boundary_condition_data():
    """
    Function-scoped boundary condition data generator for edge-case validation.
    
    Generates data at various boundary conditions to test robustness:
    - Minimum/maximum array sizes
    - Extreme numerical values
    - Edge cases for time series processing
    - Memory constraint scenarios
    
    Returns:
        Dict[str, Any]: Dictionary containing boundary condition test data
    """
    if not np:
        pytest.skip("numpy not available for boundary condition testing")
    
    boundary_data = {}
    
    # Minimum array sizes
    boundary_data["minimal_time_series"] = {
        't': np.array([0.0]),
        'x': np.array([0.0]),
        'y': np.array([0.0])
    }
    
    # Single element arrays
    boundary_data["single_element"] = {
        't': np.array([0.0]),
        'x': np.array([10.5]),
        'y': np.array([20.3]),
        'signal': np.array([0.1])
    }
    
    # Very small arrays (edge of processing limits)
    boundary_data["tiny_arrays"] = {
        't': np.array([0.0, 0.016]),
        'x': np.array([10.5, 10.6]),
        'y': np.array([20.3, 20.2])
    }
    
    # Extreme numerical values
    boundary_data["extreme_values"] = {
        't': np.array([0.0, 1e-10, 1e10]),
        'x': np.array([-1e6, 0.0, 1e6]),
        'y': np.array([-1e6, 0.0, 1e6]),
        'signal': np.array([np.finfo(float).min, 0.0, np.finfo(float).max])
    }
    
    # Arrays with special float values
    boundary_data["special_float_values"] = {
        't': np.array([0.0, 1.0, 2.0]),
        'x': np.array([np.inf, -np.inf, np.nan]),
        'y': np.array([np.nan, np.inf, -np.inf]),
        'signal': np.array([np.nan, np.nan, np.nan])
    }
    
    # Large array (memory testing)
    large_size = 1_000_000  # 1M elements
    boundary_data["large_arrays"] = {
        't': np.linspace(0, 16666.67, large_size),  # ~4.6 hours at 60Hz
        'x': np.random.rand(large_size) * 100,
        'y': np.random.rand(large_size) * 100
    }
    
    # Empty arrays
    boundary_data["empty_arrays"] = {
        't': np.array([]),
        'x': np.array([]),
        'y': np.array([])
    }
    
    return boundary_data

@pytest.fixture(scope="function")
def fixture_memory_constraint_scenarios():
    """
    Function-scoped memory constraint scenarios for resource limit testing.
    
    Generates scenarios that test memory handling under various constraints:
    - Large dataset simulation
    - Memory allocation patterns
    - Garbage collection scenarios
    - Memory leak detection support
    
    Returns:
        Dict[str, Callable]: Dictionary of memory scenario generators
    """
    def generate_large_dataset(size_mb: float = 100) -> Dict[str, Any]:
        """Generate large dataset of specified size in MB."""
        if not np:
            return {}
        
        # Calculate array size for target memory usage
        # Assuming float64 (8 bytes per element)
        target_elements = int((size_mb * 1024 * 1024) / 8)
        
        return {
            'large_time_series': np.random.rand(target_elements),
            'large_positions_x': np.random.rand(target_elements),
            'large_positions_y': np.random.rand(target_elements),
            'size_mb': size_mb,
            'element_count': target_elements
        }
    
    def simulate_memory_pressure(iterations: int = 10) -> Generator[Dict[str, Any], None, None]:
        """Simulate memory pressure through repeated allocation/deallocation."""
        for i in range(iterations):
            yield generate_large_dataset(size_mb=10 * (i + 1))
    
    def create_fragmented_data(num_fragments: int = 100) -> List[np.ndarray]:
        """Create fragmented data to test memory management."""
        if not np:
            return []
        
        fragments = []
        for i in range(num_fragments):
            # Create arrays of varying sizes to fragment memory
            size = np.random.randint(1000, 10000)
            fragment = np.random.rand(size)
            fragments.append(fragment)
        
        return fragments
    
    return {
        'large_dataset_generator': generate_large_dataset,
        'memory_pressure_simulator': simulate_memory_pressure,
        'fragmented_data_creator': create_fragmented_data
    }

# ============================================================================
# CENTRALIZED PYTEST-MOCK INTEGRATION AND STANDARDIZED MOCKING
# ============================================================================

@pytest.fixture(scope="function")
def mock_filesystem_enhanced(mocker):
    """
    Enhanced filesystem mocking fixture with comprehensive edge-case support.
    
    Features:
    - Cross-platform path mocking with Unicode support
    - File existence and permission simulation including access errors
    - Directory traversal mocking with realistic metadata
    - Corrupted file scenario simulation
    - Memory constraint simulation for large files
    - Concurrent access simulation
    """
    class MockFilesystemEnhanced:
        def __init__(self, mocker):
            self.mocker = mocker
            self.mock_files = {}
            self.mock_dirs = set()
            self.access_errors = {}  # Paths that should raise access errors
            self.corrupted_files = set()  # Files that should appear corrupted
            self._lock_simulation = {}  # Simulate file locks
        
        def add_file(self, path: Union[str, Path], size: int = 1024, 
                    mtime: Optional[datetime] = None, 
                    corrupted: bool = False,
                    access_error: Optional[Exception] = None) -> Path:
            """Add a mock file with comprehensive properties."""
            path = Path(path)
            path_str = str(path)
            
            self.mock_files[path_str] = {
                'size': size,
                'mtime': mtime or datetime.now(),
                'exists': True,
                'corrupted': corrupted,
                'content': b"mock file content" if not corrupted else b"corrupted\x00\x01\x02"
            }
            
            # Handle special file scenarios
            if corrupted:
                self.corrupted_files.add(path_str)
            
            if access_error:
                self.access_errors[path_str] = access_error
            
            # Ensure parent directories exist
            parent = path.parent
            while parent != Path('.') and str(parent) != '.':
                self.mock_dirs.add(str(parent))
                parent = parent.parent
            
            return path
        
        def add_directory(self, path: Union[str, Path], 
                         access_error: Optional[Exception] = None) -> Path:
            """Add a mock directory with optional access restrictions."""
            path = Path(path)
            path_str = str(path)
            self.mock_dirs.add(path_str)
            
            if access_error:
                self.access_errors[path_str] = access_error
            
            return path
        
        def add_unicode_file(self, unicode_path: str, **kwargs) -> Optional[Path]:
            """Add Unicode file path, handling platform limitations gracefully."""
            try:
                return self.add_file(unicode_path, **kwargs)
            except (UnicodeError, OSError):
                logger.debug(f"Unicode path not supported: {unicode_path}")
                return None
        
        def simulate_file_lock(self, path: Union[str, Path], locked: bool = True):
            """Simulate file lock for concurrent access testing."""
            self._lock_simulation[str(path)] = locked
        
        def mock_path_exists(self, path):
            """Enhanced mock pathlib.Path.exists() with error simulation."""
            path_str = str(path)
            
            # Check for access errors first
            if path_str in self.access_errors:
                raise self.access_errors[path_str]
            
            return path_str in self.mock_files or path_str in self.mock_dirs
        
        def mock_path_is_file(self, path):
            """Enhanced mock pathlib.Path.is_file() with error handling."""
            path_str = str(path)
            
            if path_str in self.access_errors:
                raise self.access_errors[path_str]
            
            return path_str in self.mock_files
        
        def mock_path_is_dir(self, path):
            """Enhanced mock pathlib.Path.is_dir() with access control."""
            path_str = str(path)
            
            if path_str in self.access_errors:
                raise self.access_errors[path_str]
            
            return path_str in self.mock_dirs
        
        def mock_path_stat(self, path):
            """Enhanced mock pathlib.Path.stat() with comprehensive metadata."""
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
                mock_stat.st_mode = 0o644  # Regular file permissions
                mock_stat.st_uid = 1000
                mock_stat.st_gid = 1000
                return mock_stat
            
            raise FileNotFoundError(f"Mocked file not found: {path}")
        
        def mock_open_file(self, path, mode='r', **kwargs):
            """Mock file opening with corruption and lock simulation."""
            path_str = str(path)
            
            # Check for access errors
            if path_str in self.access_errors:
                raise self.access_errors[path_str]
            
            # Check for file locks
            if path_str in self._lock_simulation and self._lock_simulation[path_str]:
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
                if 'b' in mode:
                    return io.BytesIO(file_info['content'])
                else:
                    return io.StringIO(file_info['content'].decode('utf-8', errors='replace'))
            
            raise FileNotFoundError(f"Mocked file not found: {path}")
        
        def activate(self):
            """Activate all enhanced filesystem mocks."""
            self.mocker.patch('pathlib.Path.exists', side_effect=self.mock_path_exists)
            self.mocker.patch('pathlib.Path.is_file', side_effect=self.mock_path_is_file)
            self.mocker.patch('pathlib.Path.is_dir', side_effect=self.mock_path_is_dir)
            self.mocker.patch('pathlib.Path.stat', side_effect=self.mock_path_stat)
            self.mocker.patch('builtins.open', side_effect=self.mock_open_file)
        
        def reset(self):
            """Reset all mock state for clean test isolation."""
            self.mock_files.clear()
            self.mock_dirs.clear()
            self.access_errors.clear()
            self.corrupted_files.clear()
            self._lock_simulation.clear()
    
    return MockFilesystemEnhanced(mocker)

# Maintain backward compatibility with original fixture name
@pytest.fixture(scope="function")
def mock_filesystem(mock_filesystem_enhanced):
    """Backward compatibility alias for enhanced filesystem mock."""
    return mock_filesystem_enhanced


@pytest.fixture(scope="function")
def mock_data_loading_comprehensive(mocker):
    """
    Comprehensive data loading mocking with edge-case and error scenario support.
    
    Provides enhanced mocking interfaces for:
    - Pickle file loading (standard, gzipped, pandas-specific, corrupted)
    - YAML configuration loading with malformed file handling
    - DataFrame operations with memory constraints
    - Network-dependent data loading simulation
    - Concurrent access and file locking scenarios
    - Error simulation for all data loading paths
    """
    class MockDataLoadingComprehensive:
        def __init__(self, mocker):
            self.mocker = mocker
            self.mock_data = {}
            self.error_scenarios = {}
            self.load_delays = {}  # Simulate slow loading
            self.memory_constraints = {}
            self.corruption_scenarios = set()
        
        def mock_pickle_load(self, file_path: str, return_data: Any, 
                           error: Optional[Exception] = None,
                           delay_seconds: float = 0.0,
                           memory_limit_mb: Optional[float] = None):
            """Enhanced mock pickle.load with error and performance simulation."""
            self.mock_data[file_path] = return_data
            
            if error:
                self.error_scenarios[file_path] = error
            
            if delay_seconds > 0:
                self.load_delays[file_path] = delay_seconds
            
            if memory_limit_mb:
                self.memory_constraints[file_path] = memory_limit_mb
        
        def mock_yaml_load(self, file_path: str, return_data: Dict[str, Any],
                          error: Optional[Exception] = None):
            """Enhanced mock yaml.safe_load with malformed content simulation."""
            self.mock_data[file_path] = return_data
            
            if error:
                self.error_scenarios[file_path] = error
        
        def add_corrupted_pickle(self, file_path: str):
            """Mark a pickle file as corrupted for error testing."""
            self.corruption_scenarios.add(file_path)
            self.error_scenarios[file_path] = pickle.UnpicklingError("File appears to be corrupted")
        
        def add_network_delay(self, file_path: str, delay_seconds: float):
            """Add network delay simulation for remote file loading."""
            self.load_delays[file_path] = delay_seconds
        
        def simulate_memory_pressure(self, file_path: str, limit_mb: float):
            """Simulate memory pressure during large file loading."""
            self.memory_constraints[file_path] = limit_mb
        
        def activate_pickle_mocks(self):
            """Activate enhanced pickle loading mocks."""
            def mock_pickle_load_func(file_obj):
                import time
                
                # Extract file path from file object
                file_path = getattr(file_obj, 'name', str(file_obj))
                
                # Check for errors first
                if file_path in self.error_scenarios:
                    raise self.error_scenarios[file_path]
                
                # Simulate loading delays
                if file_path in self.load_delays:
                    time.sleep(self.load_delays[file_path])
                
                # Check memory constraints
                if file_path in self.memory_constraints:
                    try:
                        import psutil
                        available_mb = psutil.virtual_memory().available / 1024 / 1024
                        required_mb = self.memory_constraints[file_path]
                        
                        if available_mb < required_mb:
                            raise MemoryError(f"Insufficient memory: need {required_mb}MB, have {available_mb}MB")
                    except ImportError:
                        pass  # Skip memory check if psutil not available
                
                # Return mock data
                if file_path in self.mock_data:
                    return self.mock_data[file_path]
                
                raise FileNotFoundError(f"Mock data not configured for: {file_path}")
            
            self.mocker.patch('pickle.load', side_effect=mock_pickle_load_func)
        
        def activate_yaml_mocks(self):
            """Activate enhanced YAML loading mocks."""
            def mock_yaml_load_func(file_obj):
                file_path = getattr(file_obj, 'name', str(file_obj))
                
                # Check for errors (malformed YAML, etc.)
                if file_path in self.error_scenarios:
                    raise self.error_scenarios[file_path]
                
                # Return mock data
                if file_path in self.mock_data:
                    return self.mock_data[file_path]
                
                raise FileNotFoundError(f"Mock YAML data not configured for: {file_path}")
            
            self.mocker.patch('yaml.safe_load', side_effect=mock_yaml_load_func)
        
        def activate_pandas_mocks(self):
            """Activate pandas-specific loading mocks."""
            def mock_pandas_read_pickle(file_path, **kwargs):
                if not pd:
                    raise ImportError("pandas not available")
                
                # Convert file path to string for lookup
                file_path_str = str(file_path)
                
                # Check for errors
                if file_path_str in self.error_scenarios:
                    raise self.error_scenarios[file_path_str]
                
                # Return mock DataFrame if configured
                if file_path_str in self.mock_data:
                    data = self.mock_data[file_path_str]
                    if isinstance(data, dict):
                        return pd.DataFrame(data)
                    return data
                
                raise FileNotFoundError(f"Mock pandas data not configured for: {file_path}")
            
            if pd:
                self.mocker.patch('pandas.read_pickle', side_effect=mock_pandas_read_pickle)
        
        def activate_gzip_mocks(self):
            """Activate gzip file handling mocks."""
            def mock_gzip_open(filename, mode='rb', **kwargs):
                file_path = str(filename)
                
                if file_path in self.error_scenarios:
                    raise self.error_scenarios[file_path]
                
                # Return mock file-like object
                if file_path in self.mock_data:
                    content = self.mock_data[file_path]
                    if isinstance(content, (str, bytes)):
                        if 'b' in mode:
                            return io.BytesIO(content if isinstance(content, bytes) else content.encode())
                        else:
                            return io.StringIO(content if isinstance(content, str) else content.decode())
                
                raise FileNotFoundError(f"Mock gzip file not found: {file_path}")
            
            self.mocker.patch('gzip.open', side_effect=mock_gzip_open)
        
        def activate_all_mocks(self):
            """Activate all data loading mocks for comprehensive testing."""
            self.activate_pickle_mocks()
            self.activate_yaml_mocks()
            self.activate_pandas_mocks()
            self.activate_gzip_mocks()
        
        def reset(self):
            """Reset all mock data for clean test isolation."""
            self.mock_data.clear()
            self.error_scenarios.clear()
            self.load_delays.clear()
            self.memory_constraints.clear()
            self.corruption_scenarios.clear()
    
    return MockDataLoadingComprehensive(mocker)

# Maintain backward compatibility with original fixture name
@pytest.fixture(scope="function")
def mock_data_loading(mock_data_loading_comprehensive):
    """Backward compatibility alias for comprehensive data loading mock."""
    return mock_data_loading_comprehensive


# ============================================================================
# HYPOTHESIS PROPERTY-BASED TESTING SUPPORT
# ============================================================================

@pytest.fixture(scope="session")
def hypothesis_settings():
    """
    Configure Hypothesis for property-based testing with appropriate settings
    for robust edge case discovery per Section 3.6.3 requirements.
    """
    # Configure Hypothesis for test environment
    settings.register_profile("test", 
                            max_examples=100,
                            deadline=None,
                            suppress_health_check=[],
                            phases=None)
    settings.load_profile("test")
    
    return settings


@pytest.fixture(scope="function")
def hypothesis_strategies():
    """
    Provide domain-specific Hypothesis strategies for flyrigloader testing scenarios.
    
    Strategies include:
    - File path generation with realistic experimental naming patterns
    - Experimental metadata generation
    - Configuration dictionary generation
    - NumPy array generation with experimental data characteristics
    """
    class ExperimentalStrategies:
        @staticmethod
        def animal_ids():
            """Generate realistic animal ID strings."""
            return st.text(
                alphabet="abcdefghijklmnopqrstuvwxyz0123456789_",
                min_size=3,
                max_size=15
            ).filter(lambda x: len(x.strip('_')) > 2)
        
        @staticmethod
        def experiment_dates():
            """Generate experiment date strings."""
            return st.dates(
                min_value=datetime(2020, 1, 1).date(),
                max_value=datetime(2024, 12, 31).date()
            ).map(lambda d: d.strftime("%Y%m%d"))
        
        @staticmethod
        def experimental_conditions():
            """Generate experimental condition labels."""
            return st.sampled_from([
                "control", "treatment", "baseline", "stimulation",
                "drug_a", "drug_b", "sham", "recovery"
            ])
        
        @staticmethod
        def file_patterns():
            """Generate realistic experimental file naming patterns."""
            return st.builds(
                lambda animal, date, condition, rep: f"{animal}_{date}_{condition}_rep{rep}.pkl",
                animal=ExperimentalStrategies.animal_ids(),
                date=ExperimentalStrategies.experiment_dates(),
                condition=ExperimentalStrategies.experimental_conditions(),
                rep=st.integers(min_value=1, max_value=10)
            )
        
        @staticmethod
        def config_dicts():
            """Generate configuration dictionaries with experimental structure."""
            return st.fixed_dictionaries({
                "project": st.fixed_dictionaries({
                    "name": st.text(min_size=1, max_size=50),
                    "mandatory_experiment_strings": st.lists(st.text(min_size=1, max_size=20), max_size=5),
                    "ignore_substrings": st.lists(st.text(min_size=1, max_size=20), max_size=5)
                }),
                "datasets": st.dictionaries(
                    keys=st.text(min_size=1, max_size=20),
                    values=st.fixed_dictionaries({
                        "dates_vials": st.dictionaries(
                            keys=ExperimentalStrategies.experiment_dates(),
                            values=st.lists(ExperimentalStrategies.animal_ids(), min_size=1, max_size=10)
                        )
                    }),
                    max_size=3
                )
            })
        
        @staticmethod 
        def numpy_arrays():
            """Generate NumPy arrays with experimental data characteristics."""
            if not np:
                return st.none()
            return st.builds(
                np.random.normal,
                loc=st.floats(min_value=-10, max_value=10),
                scale=st.floats(min_value=0.1, max_value=5.0),
                size=st.tuples(
                    st.integers(min_value=10, max_value=1000),
                    st.integers(min_value=1, max_value=100)
                )
            )
    
    return ExperimentalStrategies()

# ============================================================================
# COMPREHENSIVE INTEGRATION TEST MOCKING (Consolidated from flyrigloader/conftest.py)
# ============================================================================

@pytest.fixture(scope="function")
def mock_config_and_discovery_comprehensive(mocker, sample_comprehensive_config_dict, 
                                           temp_filesystem_structure):
    """
    Comprehensive integration mocking for config loading and file discovery workflows.
    
    Provides sophisticated mocking of the entire configuration and discovery pipeline
    with realistic return values supporting complex integration testing scenarios,
    edge-case handling, and error simulation.
    
    Args:
        mocker: pytest-mock fixture for enhanced mocking capabilities
        sample_comprehensive_config_dict: Comprehensive configuration data
        temp_filesystem_structure: Realistic filesystem structure
        
    Returns:
        Dict: Dictionary containing all mock objects and helper functions
    """
    # Mock the config loading function with comprehensive return values
    mock_load_config = mocker.patch("flyrigloader.api.load_config")
    mock_load_config.return_value = sample_comprehensive_config_dict
    
    # Create realistic file discovery return values with metadata
    discovered_files = {}
    
    # Generate discovered files based on filesystem structure
    if "baseline_file_1" in temp_filesystem_structure:
        discovered_files[str(temp_filesystem_structure["baseline_file_1"])] = {
            "date": "20241220", "condition": "control", "replicate": "1",
            "dataset": "baseline", "file_size": 1024,
            "modification_time": datetime.now().isoformat()
        }
    
    if "baseline_file_2" in temp_filesystem_structure:
        discovered_files[str(temp_filesystem_structure["baseline_file_2"])] = {
            "date": "20241221", "condition": "control", "replicate": "2", 
            "dataset": "baseline", "file_size": 1536,
            "modification_time": datetime.now().isoformat()
        }
    
    if "opto_file_1" in temp_filesystem_structure:
        discovered_files[str(temp_filesystem_structure["opto_file_1"])] = {
            "date": "20241218", "condition": "treatment", "replicate": "1",
            "dataset": "optogenetic", "stimulation_type": "stim", "file_size": 2048,
            "modification_time": datetime.now().isoformat()
        }
    
    if "nav_file_1" in temp_filesystem_structure:
        discovered_files[str(temp_filesystem_structure["nav_file_1"])] = {
            "date": "20241025", "condition": "navigation", "replicate": "1",
            "dataset": "plume", "plume_type": "plume", "trial": "1", "file_size": 3072,
            "modification_time": datetime.now().isoformat()
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
    mock_yaml_load_config.return_value = sample_comprehensive_config_dict
    
    # Enhanced pickle loading mock with realistic experimental data
    mock_pickle_loader = mocker.patch("flyrigloader.io.pickle.read_pickle_any_format")
    
    def pickle_loader_side_effect(path):
        """Dynamic side effect for pickle loading based on file path with realistic data."""
        path_str = str(path)
        
        if "baseline" in path_str:
            return {
                't': np.linspace(0, 300, 18000),  # 5 minutes at 60 Hz
                'x': np.random.rand(18000) * 100,
                'y': np.random.rand(18000) * 100
            }
        elif "opto" in path_str:
            return {
                't': np.linspace(0, 600, 36000),  # 10 minutes at 60 Hz  
                'x': np.random.rand(36000) * 100,
                'y': np.random.rand(36000) * 100,
                'signal': np.random.rand(36000)
            }
        elif "nav" in path_str:
            return {
                't': np.linspace(0, 180, 10800),  # 3 minutes at 60 Hz
                'x': np.random.rand(10800) * 120,
                'y': np.random.rand(10800) * 120,
                'signal_disp': np.random.rand(16, 10800)
            }
        elif "unicode" in path_str or "tëst" in path_str:
            # Edge-case testing with Unicode paths
            return {
                't': np.array([0, 1, 2]), 
                'x': np.array([0, 1, 2]), 
                'y': np.array([0, 1, 2])
            }
        else:
            # Default fallback data
            return {
                't': np.array([0, 1, 2]), 
                'x': np.array([0, 1, 2]), 
                'y': np.array([0, 1, 2])
            }
    
    mock_pickle_loader.side_effect = pickle_loader_side_effect
    
    # Enhanced column configuration loading mock
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
    
    # Add error simulation capabilities
    def simulate_config_loading_error():
        """Simulate configuration loading errors for error handling tests."""
        mock_load_config.side_effect = yaml.YAMLError("Malformed configuration file")
    
    def simulate_discovery_failure():
        """Simulate file discovery failures for resilience testing."""
        mock_discover_experiment_files.side_effect = FileNotFoundError("Discovery failed")
    
    def reset_mocks():
        """Reset all mocks to default successful behavior."""
        mock_load_config.side_effect = None
        mock_load_config.return_value = sample_comprehensive_config_dict
        mock_discover_experiment_files.side_effect = None
        mock_discover_experiment_files.return_value = discovered_files
    
    return {
        "load_config": mock_load_config,
        "discover_experiment_files": mock_discover_experiment_files,
        "discover_dataset_files": mock_discover_dataset_files,
        "file_discoverer": mock_file_discoverer,
        "yaml_load_config": mock_yaml_load_config,
        "pickle_loader": mock_pickle_loader,
        "column_config": mock_column_config,
        "discovered_files": discovered_files,
        # Error simulation helpers
        "simulate_config_loading_error": simulate_config_loading_error,
        "simulate_discovery_failure": simulate_discovery_failure,
        "reset_mocks": reset_mocks
    }

@pytest.fixture(scope="function")
def mock_external_dependencies_comprehensive(mocker):
    """
    Comprehensive external library dependency mocking for isolated unit testing.
    
    Mocks external dependencies like NumPy, Pandas, YAML operations, and system
    utilities to enable fast unit testing without full library overhead while
    maintaining realistic behavior simulation.
    
    Args:
        mocker: pytest-mock fixture
        
    Returns:
        Dict: Dictionary containing external dependency mocks with control functions
    """
    # Enhanced numpy operations mocking
    mock_numpy_array = mocker.patch("numpy.array")
    mock_numpy_array.side_effect = lambda x: np.array(x) if np else x
    
    mock_numpy_linspace = mocker.patch("numpy.linspace")
    mock_numpy_linspace.side_effect = lambda start, stop, num: (
        np.linspace(start, stop, num) if np else list(range(int(num)))
    )
    
    mock_numpy_random = mocker.patch("numpy.random.rand")
    mock_numpy_random.side_effect = lambda *args: (
        np.random.rand(*args) if np else [0.5] * (args[0] if args else 1)
    )
    
    # Enhanced pandas operations mocking
    mock_pandas_dataframe = mocker.patch("pandas.DataFrame")
    mock_pandas_dataframe.side_effect = lambda data: pd.DataFrame(data) if pd else data
    
    mock_pandas_read_pickle = mocker.patch("pandas.read_pickle")
    mock_pandas_read_pickle.side_effect = lambda path: (
        pd.DataFrame({'t': [0, 1], 'x': [0, 1], 'y': [0, 1]}) if pd else {}
    )
    
    # Enhanced YAML operations mocking
    mock_yaml_safe_load = mocker.patch("yaml.safe_load")
    mock_yaml_safe_load.return_value = {"test": "config"}
    
    mock_yaml_dump = mocker.patch("yaml.dump")
    mock_yaml_dump.return_value = "test: config\n"
    
    # System utilities mocking for edge-case testing
    mock_psutil = None
    try:
        import psutil
        mock_psutil = mocker.patch("psutil.Process")
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 100 * 1024 * 1024  # 100MB
        mock_psutil.return_value = mock_process
    except ImportError:
        pass
    
    # Path operations mocking for cross-platform testing
    mock_pathlib_path = mocker.patch("pathlib.Path")
    mock_path_instance = MagicMock()
    mock_path_instance.exists.return_value = True
    mock_path_instance.is_file.return_value = True
    mock_path_instance.is_dir.return_value = False
    mock_pathlib_path.return_value = mock_path_instance
    
    # Error simulation capabilities
    def simulate_numpy_error():
        """Simulate NumPy operation errors."""
        mock_numpy_array.side_effect = ImportError("NumPy not available")
    
    def simulate_pandas_error():
        """Simulate Pandas operation errors."""
        mock_pandas_dataframe.side_effect = ImportError("Pandas not available")
    
    def simulate_yaml_error():
        """Simulate YAML parsing errors."""
        mock_yaml_safe_load.side_effect = yaml.YAMLError("Invalid YAML syntax")
    
    def reset_all_mocks():
        """Reset all external dependency mocks to default behavior."""
        mock_numpy_array.side_effect = lambda x: np.array(x) if np else x
        mock_pandas_dataframe.side_effect = lambda data: pd.DataFrame(data) if pd else data
        mock_yaml_safe_load.side_effect = None
        mock_yaml_safe_load.return_value = {"test": "config"}
    
    return {
        "numpy_array": mock_numpy_array,
        "numpy_linspace": mock_numpy_linspace,
        "numpy_random": mock_numpy_random,
        "pandas_dataframe": mock_pandas_dataframe,
        "pandas_read_pickle": mock_pandas_read_pickle,
        "yaml_safe_load": mock_yaml_safe_load,
        "yaml_dump": mock_yaml_dump,
        "psutil_process": mock_psutil,
        "pathlib_path": mock_pathlib_path,
        # Error simulation helpers
        "simulate_numpy_error": simulate_numpy_error,
        "simulate_pandas_error": simulate_pandas_error,
        "simulate_yaml_error": simulate_yaml_error,
        "reset_all_mocks": reset_all_mocks
    }


# ============================================================================
# PERFORMANCE BENCHMARK FIXTURES
# ============================================================================

@pytest.fixture(scope="function")
def performance_benchmarks(test_execution_tracker):
    """
    Performance benchmark fixtures for SLA validation per Section 3.6.3 requirements.
    
    Provides:
    - Data loading performance baselines
    - File discovery timing validation
    - DataFrame transformation benchmarks
    - Memory usage monitoring hooks
    """
    class PerformanceBenchmarks:
        def __init__(self, tracker):
            self.tracker = tracker
            self.benchmarks = {}
        
        def benchmark_data_loading(self, data_size_mb: float) -> float:
            """Return expected maximum time for data loading based on size."""
            return data_size_mb * self.tracker.performance_baselines['data_loading_per_mb']
        
        def benchmark_file_discovery(self, file_count: int) -> float:
            """Return expected maximum time for file discovery based on count."""
            return (file_count / 1000) * self.tracker.performance_baselines['discovery_per_1k_files']
        
        def benchmark_dataframe_transform(self, row_count: int) -> float:
            """Return expected maximum time for DataFrame transformation."""
            return (row_count / 1_000_000) * self.tracker.performance_baselines['transformation_per_1m_rows']
        
        def assert_performance_sla(self, operation_name: str, actual_time: float, expected_time: float):
            """Assert that operation meets performance SLA."""
            self.benchmarks[operation_name] = {
                'actual_time': actual_time,
                'expected_time': expected_time,
                'meets_sla': actual_time <= expected_time
            }
            logger.info(f"Performance SLA for {operation_name}: {actual_time:.3f}s <= {expected_time:.3f}s")
            assert actual_time <= expected_time, f"Performance SLA violation: {operation_name} took {actual_time:.3f}s, expected <= {expected_time:.3f}s"
    
    return PerformanceBenchmarks(test_execution_tracker)

# ============================================================================
# ENHANCED HYPOTHESIS STRATEGIES FOR EDGE-CASE TESTING
# ============================================================================

@pytest.fixture(scope="function")
def hypothesis_edge_case_strategies():
    """
    Enhanced Hypothesis strategies for comprehensive edge-case discovery.
    
    Provides domain-specific strategies for flyrigloader edge-case testing
    including boundary conditions, Unicode handling, corrupted data scenarios,
    and performance constraint testing.
    
    Returns:
        Object: Enhanced strategies with edge-case focus
    """
    class EdgeCaseStrategies:
        @staticmethod
        def unicode_paths():
            """Generate Unicode file paths for cross-platform testing."""
            unicode_chars = st.sampled_from([
                'ë', 'ü', 'ñ', 'ä', 'ö', 'ç', 'å', 'ø', 'æ', 'ß',
                'é', 'è', 'à', 'ù', 'î', 'ô', 'â', 'ê', 'ï', 'ÿ'
            ])
            
            return st.builds(
                lambda base, unicode_part, ext: f"{base}_{unicode_part}{ext}",
                base=st.text(alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")), 
                           min_size=3, max_size=10),
                unicode_part=st.text(alphabet=unicode_chars, min_size=1, max_size=5),
                ext=st.sampled_from(['.csv', '.pkl', '.yaml'])
            )
        
        @staticmethod
        def boundary_arrays():
            """Generate arrays at boundary conditions."""
            if not np:
                return st.just([])
            
            return st.one_of([
                # Empty arrays
                st.just(np.array([])),
                # Single element arrays
                st.builds(np.array, st.lists(st.floats(allow_nan=False, allow_infinity=False), 
                                           min_size=1, max_size=1)),
                # Very small arrays
                st.builds(np.array, st.lists(st.floats(allow_nan=False, allow_infinity=False), 
                                           min_size=2, max_size=3)),
                # Arrays with extreme values
                st.builds(np.array, st.lists(st.floats(min_value=-1e6, max_value=1e6), 
                                           min_size=10, max_size=100))
            ])
        
        @staticmethod
        def special_float_arrays():
            """Generate arrays with special float values (nan, inf, -inf)."""
            if not np:
                return st.just([])
            
            special_values = [np.nan, np.inf, -np.inf, 0.0, -0.0]
            
            return st.builds(
                np.array,
                st.lists(
                    st.one_of([
                        st.sampled_from(special_values),
                        st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False)
                    ]),
                    min_size=1,
                    max_size=50
                )
            )
        
        @staticmethod
        def corrupted_config_dicts():
            """Generate configuration dictionaries with corruption scenarios."""
            return st.one_of([
                # Missing required keys
                st.fixed_dictionaries({
                    "datasets": st.dictionaries(st.text(), st.text())
                    # Missing "project" key
                }),
                # Invalid data types
                st.fixed_dictionaries({
                    "project": st.text(),  # Should be dict
                    "datasets": st.lists(st.text())  # Should be dict
                }),
                # Circular references (simulated with duplicate keys)
                st.fixed_dictionaries({
                    "project": st.fixed_dictionaries({
                        "ref": st.just("datasets.test")
                    }),
                    "datasets": st.fixed_dictionaries({
                        "test": st.fixed_dictionaries({
                            "ref": st.just("project.ref")
                        })
                    })
                })
            ])
        
        @staticmethod
        def memory_constraint_scenarios():
            """Generate scenarios for memory constraint testing."""
            return st.fixed_dictionaries({
                "array_size": st.integers(min_value=1000, max_value=1_000_000),
                "num_arrays": st.integers(min_value=1, max_value=10),
                "data_type": st.sampled_from(['float64', 'float32', 'int64', 'int32']),
                "memory_limit_mb": st.integers(min_value=10, max_value=1000)
            })
        
        @staticmethod
        def file_corruption_scenarios():
            """Generate file corruption scenarios for error handling testing."""
            return st.fixed_dictionaries({
                "corruption_type": st.sampled_from([
                    "truncated", "binary_in_text", "invalid_pickle", 
                    "malformed_yaml", "encoding_error", "empty_file"
                ]),
                "corruption_position": st.one_of([
                    st.just("beginning"),
                    st.just("middle"), 
                    st.just("end"),
                    st.just("random")
                ]),
                "severity": st.sampled_from(["minor", "moderate", "severe"])
            })
        
        @staticmethod
        def concurrent_access_scenarios():
            """Generate concurrent access scenarios for threading/multiprocessing tests."""
            return st.fixed_dictionaries({
                "num_processes": st.integers(min_value=2, max_value=8),
                "access_pattern": st.sampled_from([
                    "simultaneous_read", "read_write_conflict", 
                    "write_write_conflict", "sequential_access"
                ]),
                "file_size_mb": st.integers(min_value=1, max_value=100),
                "duration_seconds": st.integers(min_value=1, max_value=10)
            })
    
    return EdgeCaseStrategies()

# ============================================================================
# PYTEST-XDIST PARALLEL EXECUTION SUPPORT
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def configure_pytest_xdist_parallel_execution():
    """
    Session-scoped autouse fixture for pytest-xdist parallel execution optimization.
    
    Configures pytest-xdist with 4-worker optimization for both local development
    and CI environments, ensuring optimal performance while maintaining test isolation.
    
    Features:
    - Automatic worker detection and configuration
    - Test isolation guarantees for parallel execution
    - Shared fixture management across workers
    - Load balancing optimization
    - Memory usage monitoring per worker
    """
    # Check if running under pytest-xdist
    worker_id = os.environ.get('PYTEST_XDIST_WORKER')
    
    if worker_id:
        # Configure worker-specific settings
        logger.info(f"Initializing pytest-xdist worker: {worker_id}")
        
        # Set worker-specific random seed for reproducible tests
        worker_num = int(worker_id.replace('gw', '')) if 'gw' in worker_id else 0
        random.seed(42 + worker_num)
        if np:
            np.random.seed(42 + worker_num)
        
        # Configure worker-specific temporary directory prefix
        original_tempdir = tempfile.gettempdir()
        worker_tempdir = Path(original_tempdir) / f"pytest_worker_{worker_id}"
        worker_tempdir.mkdir(exist_ok=True)
        
        # Monitor worker memory usage
        try:
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024
            logger.debug(f"Worker {worker_id} initial memory: {initial_memory:.2f} MB")
        except ImportError:
            pass
    
    yield
    
    # Cleanup worker-specific resources
    if worker_id:
        logger.info(f"Cleaning up pytest-xdist worker: {worker_id}")
        
        # Final memory usage report
        try:
            import psutil
            process = psutil.Process()
            final_memory = process.memory_info().rss / 1024 / 1024
            logger.debug(f"Worker {worker_id} final memory: {final_memory:.2f} MB")
        except ImportError:
            pass

@pytest.fixture(scope="function")
def pytest_xdist_worker_info():
    """
    Function-scoped fixture providing pytest-xdist worker information.
    
    Provides information about the current worker for tests that need to
    adjust behavior based on parallel execution context.
    
    Returns:
        Dict: Worker information including ID, process info, and capabilities
    """
    worker_id = os.environ.get('PYTEST_XDIST_WORKER', 'master')
    
    worker_info = {
        'worker_id': worker_id,
        'is_worker': worker_id != 'master',
        'worker_number': int(worker_id.replace('gw', '')) if 'gw' in worker_id else 0,
        'process_id': os.getpid(),
        'parallel_execution': worker_id != 'master'
    }
    
    # Add memory information if available
    try:
        import psutil
        process = psutil.Process()
        worker_info.update({
            'memory_mb': process.memory_info().rss / 1024 / 1024,
            'cpu_percent': process.cpu_percent(),
            'thread_count': process.num_threads()
        })
    except ImportError:
        worker_info.update({
            'memory_mb': None,
            'cpu_percent': None,
            'thread_count': None
        })
    
    return worker_info

# ============================================================================
# COLUMN CONFIGURATION FIXTURES (Consolidated from flyrigloader/conftest.py)
# ============================================================================

@pytest.fixture(scope="function")
def sample_column_config_file(temp_cross_platform_dir):
    """
    Function-scoped temporary column config file for testing column validation.
    
    Creates comprehensive column configuration supporting all flyrigloader
    column types and validation scenarios.
    
    Returns:
        str: Path to the temporary column config file
    """
    config_path = temp_cross_platform_dir / "column_config.yaml"
    
    # Define comprehensive test configuration
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
    
    # Write YAML configuration
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(test_config, f, default_flow_style=False, sort_keys=False)
    
    return str(config_path)

# ============================================================================
# INTEGRATION TEST SCENARIO FIXTURES (Consolidated)
# ============================================================================

@pytest.fixture(scope="function")
def fixture_integration_test_scenario_comprehensive(
    sample_comprehensive_config_dict,
    temp_filesystem_structure,
    fixture_corrupted_file_scenarios,
    temp_cross_platform_dir
):
    """
    Function-scoped comprehensive integration test scenario combining all test elements.
    
    Creates complete integration test environment including:
    - Realistic configuration and filesystem structure
    - Corrupted file scenarios for error handling testing
    - Edge-case data generation capabilities
    - Complete metadata extraction scenarios
    - Performance testing support
    
    Supports end-to-end integration testing per TST-INTEG-002 requirements.
    
    Returns:
        Dict[str, Any]: Complete integration test scenario data
    """
    scenario = {
        "config": sample_comprehensive_config_dict,
        "filesystem": temp_filesystem_structure,
        "corrupted_files": fixture_corrupted_file_scenarios,
        "temp_dir": temp_cross_platform_dir,
        
        # Expected file discovery results
        "expected_files": {
            "baseline_experiments": [
                temp_filesystem_structure.get("baseline_file_1"),
                temp_filesystem_structure.get("baseline_file_2")
            ],
            "optogenetic_experiments": [
                temp_filesystem_structure.get("opto_file_1"), 
                temp_filesystem_structure.get("opto_file_2")
            ],
            "navigation_experiments": [
                temp_filesystem_structure.get("nav_file_1"),
                temp_filesystem_structure.get("nav_file_2") 
            ],
            "edge_case_files": [
                temp_filesystem_structure.get("unicode_file"),
                temp_filesystem_structure.get("special_chars_file")
            ]
        },
        
        # Expected metadata extraction results
        "expected_metadata_extractions": {},
        
        # Performance testing data
        "performance_test_data": {
            "small_dataset_files": [],
            "medium_dataset_files": [],
            "large_dataset_files": []
        }
    }
    
    # Populate expected metadata extractions safely
    if temp_filesystem_structure.get("baseline_file_1"):
        scenario["expected_metadata_extractions"][str(temp_filesystem_structure["baseline_file_1"])] = {
            "dataset": "baseline", "date": "20241220", "condition": "control", "replicate": "1"
        }
    
    if temp_filesystem_structure.get("opto_file_1"):
        scenario["expected_metadata_extractions"][str(temp_filesystem_structure["opto_file_1"])] = {
            "dataset": "opto", "stimulation_type": "stim", "date": "20241218", 
            "condition": "treatment", "replicate": "1"
        }
    
    # Populate performance test file lists
    for key, path in temp_filesystem_structure.items():
        if key.endswith('_file_1') or key.endswith('_file_2'):
            if 'baseline' in key:
                scenario["performance_test_data"]["small_dataset_files"].append(path)
            elif 'opto' in key:
                scenario["performance_test_data"]["medium_dataset_files"].append(path)
            elif 'nav' in key:
                scenario["performance_test_data"]["large_dataset_files"].append(path)
    
    return scenario
