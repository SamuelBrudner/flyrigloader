"""
Pytest configuration file for flyrigloader test suite.

Provides comprehensive testing infrastructure including:
- Enhanced Loguru integration for test execution tracking
- Advanced pytest fixture system with session-scoped data generation
- Standardized mocking patterns via pytest-mock integration
- Comprehensive test environment setup for NumPy/Pandas data generation
- Property-based testing support using Hypothesis
- Performance benchmark fixtures for SLA validation

Test Infrastructure Features:
- TST-INF-001: Global fixture configuration with Loguru integration
- TST-INF-002: Test execution tracking and log capture
- TST-MOD-003: Standardized mocking patterns across modules
- TST-INTEG-002: Synthetic experimental data generation support
"""

import contextlib
import importlib.util
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
import types

# Add the src directory to the Python path
src_path = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src_path)

# Core testing imports
import pytest

HYPOTHESIS_AVAILABLE = importlib.util.find_spec("hypothesis") is not None
YAML_AVAILABLE = importlib.util.find_spec("yaml") is not None

if HYPOTHESIS_AVAILABLE:
    from hypothesis import strategies as st
    from hypothesis import settings
else:  # pragma: no cover - exercised when Hypothesis is absent
    st = None  # type: ignore[assignment]
    settings = None  # type: ignore[assignment]

if not YAML_AVAILABLE:
    yaml_stub = types.ModuleType("yaml")

    def _unavailable(*_: object, **__: object) -> None:
        raise ModuleNotFoundError("PyYAML is required for YAML-based tests.")

    yaml_stub.safe_load = _unavailable  # type: ignore[attr-defined]
    yaml_stub.safe_dump = _unavailable  # type: ignore[attr-defined]
    yaml_stub.load = _unavailable  # type: ignore[attr-defined]
    yaml_stub.dump = _unavailable  # type: ignore[attr-defined]

    class _MissingYamlError(ModuleNotFoundError):
        """Raised when YAML functionality is unavailable in test environments."""

    yaml_stub.YAMLError = _MissingYamlError  # type: ignore[attr-defined]
    sys.modules.setdefault("yaml", yaml_stub)

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


collect_ignore: List[str] = []

if not HYPOTHESIS_AVAILABLE:
    hypothesis_only_modules = [
        "flyrigloader/test_registry_threading.py",
        "flyrigloader/discovery/test_patterns.py",
        "flyrigloader/discovery/test_stats.py",
        "flyrigloader/discovery/test_files.py",
        "flyrigloader/test_api_metadata.py",
        "flyrigloader/test_api.py",
        "flyrigloader/config/test_models.py",
        "flyrigloader/config/test_discovery.py",
        "flyrigloader/config/test_yaml_config.py",
        "flyrigloader/integration/test_end_to_end_workflows.py",
        "flyrigloader/io/test_pickle.py",
        "flyrigloader/io/test_pydantic_features.py",
        "flyrigloader/io/test_column_config.py",
        "flyrigloader/test_config_builders.py",
        "flyrigloader/test_kedro_integration.py",
        "flyrigloader/benchmarks/test_benchmark_data_loading.py",
        "flyrigloader/benchmarks/test_benchmark_transformations.py",
    ]
    collect_ignore.extend(hypothesis_only_modules)
    logger.warning(
        "Hypothesis is unavailable; skipping %d property-based test modules.",
        len(hypothesis_only_modules),
    )

if not YAML_AVAILABLE:
    yaml_only_modules = [
        "flyrigloader/io/test_column_models.py",
        "flyrigloader/integration/test_api_facade_integration.py",
        "flyrigloader/integration/test_configuration_driven_workflows.py",
        "flyrigloader/integration/test_cross_module_integration.py",
        "flyrigloader/integration/test_realistic_experimental_scenarios.py",
        "flyrigloader/benchmarks/test_benchmark_config.py",
    ]
    collect_ignore.extend(yaml_only_modules)
    logger.warning(
        "PyYAML is unavailable; skipping %d YAML-dependent test modules.",
        len(yaml_only_modules),
    )


# ============================================================================
# ENHANCED LOGURU INTEGRATION FIXTURES
# ============================================================================

@pytest.fixture(autouse=True, scope="function")
def capture_loguru_logs_globally(caplog):
    """
    Enhanced fixture to capture Loguru logs into pytest's caplog for comprehensive test execution tracking.
    
    Features:
    - Proper Loguru-to-standard-logging bridge with enhanced error handling
    - Automatic cleanup to prevent cross-test contamination
    - Configurable log levels with DEBUG default for thorough testing
    - Session-scoped test execution tracking per TST-INF-001 requirements
    
    Supports:
    - All Loguru log levels (TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - Structured log message preservation
    - Exception traceback capture
    - Test-specific log isolation
    """
    class EnhancedPropagateHandler(logging.Handler):
        """Enhanced handler for bridging Loguru to standard logging with better error handling."""
        
        def emit(self, record):
            try:
                # Create or get logger in standard logging hierarchy
                std_logger = logging.getLogger(record.name or "flyrigloader")
                
                # Preserve original log level mapping
                level_mapping = {
                    'TRACE': logging.DEBUG - 5,
                    'DEBUG': logging.DEBUG,
                    'INFO': logging.INFO,
                    'SUCCESS': logging.INFO + 5,
                    'WARNING': logging.WARNING,
                    'ERROR': logging.ERROR,
                    'CRITICAL': logging.CRITICAL
                }
                
                # Map Loguru level to standard logging level
                if hasattr(record, 'levelname'):
                    record.levelno = level_mapping.get(record.levelname, logging.INFO)
                
                # Ensure proper record handling
                std_logger.handle(record)
                
            except Exception:
                # Prevent logging failures from breaking tests
                pass

    # Configure caplog for comprehensive log capture
    caplog.set_level(logging.DEBUG)
    
    # Add enhanced handler to Loguru with comprehensive formatting
    handler_id = logger.add(
        EnhancedPropagateHandler(),
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        level="TRACE",
        catch=True,  # Catch exceptions in logging
        backtrace=True,  # Include backtrace in error logs
        diagnose=True,  # Include variable values in tracebacks
        enqueue=False  # Synchronous logging for test predictability
    )

    # Track test execution start
    test_name = getattr(pytest.current_test, "nodeid", "unknown_test") if hasattr(pytest, 'current_test') else "unknown_test"
    logger.info(f"TEST_START: {test_name}")

    yield

    # Track test execution completion
    logger.info(f"TEST_END: {test_name}")

    # Clean up Loguru handler to prevent cross-test contamination
    with contextlib.suppress(ValueError, KeyError):
        logger.remove(handler_id)


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
# PYTEST-MOCK INTEGRATION AND STANDARDIZED MOCKING
# ============================================================================

@pytest.fixture(scope="function")
def mock_filesystem(mocker):
    """
    Enhanced filesystem mocking fixture providing standardized patterns
    for file system operations per TST-MOD-003 requirements.
    
    Features:
    - Cross-platform path mocking
    - File existence and permission simulation
    - Directory traversal mocking
    - Realistic file metadata generation
    """
    class MockFilesystem:
        def __init__(self, mocker):
            self.mocker = mocker
            self.mock_files = {}
            self.mock_dirs = set()
        
        def add_file(self, path: Union[str, Path], size: int = 1024, 
                    mtime: Optional[datetime] = None) -> Path:
            """Add a mock file with specified properties."""
            path = Path(path)
            self.mock_files[str(path)] = {
                'size': size,
                'mtime': mtime or datetime.now(),
                'exists': True
            }
            # Ensure parent directories exist
            parent = path.parent
            while parent != Path('.'):
                self.mock_dirs.add(str(parent))
                parent = parent.parent
            return path
        
        def add_directory(self, path: Union[str, Path]) -> Path:
            """Add a mock directory."""
            path = Path(path)
            self.mock_dirs.add(str(path))
            return path
        
        def mock_path_exists(self, path):
            """Mock pathlib.Path.exists() method."""
            path_str = str(path)
            return path_str in self.mock_files or path_str in self.mock_dirs
        
        def mock_path_is_file(self, path):
            """Mock pathlib.Path.is_file() method."""
            return str(path) in self.mock_files
        
        def mock_path_is_dir(self, path):
            """Mock pathlib.Path.is_dir() method."""
            return str(path) in self.mock_dirs
        
        def mock_path_stat(self, path):
            """Mock pathlib.Path.stat() method."""
            path_str = str(path)
            if path_str in self.mock_files:
                file_info = self.mock_files[path_str]
                mock_stat = MagicMock()
                mock_stat.st_size = file_info['size']
                mock_stat.st_mtime = file_info['mtime'].timestamp()
                return mock_stat
            raise FileNotFoundError(f"Mocked file not found: {path}")
        
        def activate(self):
            """Activate all filesystem mocks."""
            self.mocker.patch('pathlib.Path.exists', side_effect=self.mock_path_exists)
            self.mocker.patch('pathlib.Path.is_file', side_effect=self.mock_path_is_file)
            self.mocker.patch('pathlib.Path.is_dir', side_effect=self.mock_path_is_dir)
            self.mocker.patch('pathlib.Path.stat', side_effect=self.mock_path_stat)
    
    return MockFilesystem(mocker)


@pytest.fixture(scope="function")
def mock_data_loading(mocker):
    """
    Standardized mocking patterns for data loading operations across all test modules.
    
    Provides consistent mocking interfaces for:
    - Pickle file loading (standard, gzipped, pandas-specific)
    - YAML configuration loading
    - DataFrame operations
    - Error simulation scenarios
    """
    class MockDataLoading:
        def __init__(self, mocker):
            self.mocker = mocker
            self.mock_data = {}
        
        def mock_pickle_load(self, file_path: str, return_data: Any):
            """Mock pickle.load for specific file path."""
            self.mock_data[file_path] = return_data
        
        def mock_yaml_load(self, file_path: str, return_data: Dict[str, Any]):
            """Mock yaml.safe_load for specific file path."""
            self.mock_data[file_path] = return_data
        
        def activate_pickle_mocks(self):
            """Activate pickle loading mocks."""
            def mock_pickle_load_func(file_obj):
                # Extract file path from file object
                file_path = getattr(file_obj, 'name', str(file_obj))
                if file_path in self.mock_data:
                    return self.mock_data[file_path]
                raise FileNotFoundError(f"Mock data not configured for: {file_path}")
            
            self.mocker.patch('pickle.load', side_effect=mock_pickle_load_func)
        
        def activate_yaml_mocks(self):
            """Activate YAML loading mocks."""
            def mock_yaml_load_func(file_obj):
                file_path = getattr(file_obj, 'name', str(file_obj))
                if file_path in self.mock_data:
                    return self.mock_data[file_path]
                raise FileNotFoundError(f"Mock YAML data not configured for: {file_path}")
            
            self.mocker.patch('yaml.safe_load', side_effect=mock_yaml_load_func)
    
    return MockDataLoading(mocker)


# ============================================================================
# HYPOTHESIS PROPERTY-BASED TESTING SUPPORT
# ============================================================================

@pytest.fixture(scope="session")
def hypothesis_settings():
    """
    Configure Hypothesis for property-based testing with appropriate settings
    for robust edge case discovery per Section 3.6.3 requirements.
    """
    if not HYPOTHESIS_AVAILABLE:
        pytest.skip("Hypothesis is required for property-based testing fixtures")

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
    if not HYPOTHESIS_AVAILABLE:
        pytest.skip("Hypothesis is required for property-based testing strategies")

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
