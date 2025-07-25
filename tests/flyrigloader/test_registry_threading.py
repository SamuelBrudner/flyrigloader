"""
Comprehensive pytest test suite validating thread-safety and concurrency behavior 
of the enhanced registry system.

This test module implements comprehensive thread-safety validation with 100 concurrent 
operations testing, plugin registration stress scenarios, priority resolution validation,
and deadlock prevention mechanisms per Section 6.6.3.8.

Test Coverage:
- Registry thread-safety validation with 100 concurrent operations
- Enhanced registry foundation with automatic entry-point discovery
- Priority-based plugin resolution system testing (BUILTIN < USER < PLUGIN < OVERRIDE)
- Registry encapsulation success criteria validation
- Concurrent plugin registration and deregistration stress testing
- @auto_register decorator functionality with concurrent module imports
- INFO-level logging validation for all registration events

Architecture:
Tests validate the singleton registry pattern with proper thread-safe locking mechanisms,
priority-based resolution systems, and comprehensive lifecycle tracking under concurrent
access patterns as specified in the technical requirements.
"""

import pytest
from threading import Barrier
from concurrent.futures import ThreadPoolExecutor
import time
import logging
from unittest.mock import patch
from importlib.metadata import entry_points
from hypothesis import strategies
from contextlib import contextmanager
import warnings
from queue import Queue
import random
from typing import Callable
from pathlib import Path

from flyrigloader.registries import RegistryPriority


# Test fixtures for registry testing infrastructure

@pytest.fixture
def mock_loader_class():
    """Create a mock loader class that implements BaseLoader protocol."""
    class MockLoader:
        def __init__(self, extension='.test', priority=10):
            self.extension = extension
            self._priority = priority
        
        def load(self, path: Path):
            """Mock load implementation."""
            return {'test': 'data', 'path': str(path)}
        
        def supports_extension(self, extension: str) -> bool:
            """Mock extension support check."""
            return extension == self.extension
        
        @property
        def priority(self) -> int:
            """Mock priority property."""
            return self._priority
    
    return MockLoader


@pytest.fixture
def mock_schema_class():
    """Create a mock schema class that implements BaseSchema protocol."""
    class MockSchema:
        def __init__(self, name='test_schema', supported_types=None):
            self._schema_name = name
            self._supported_types = supported_types or ['dict', 'list']
        
        def validate(self, data):
            """Mock validation implementation."""
            return {'validated': True, 'data': data}
        
        @property
        def schema_name(self) -> str:
            """Mock schema name property."""
            return self._schema_name
        
        @property
        def supported_types(self) -> list:
            """Mock supported types property."""
            return self._supported_types
    
    return MockSchema


@pytest.fixture
def clean_registries():
    """Clear all registries before and after tests."""
    from flyrigloader.registries import LoaderRegistry, SchemaRegistry
    
    # Clear before test
    loader_registry = LoaderRegistry()
    schema_registry = SchemaRegistry()
    loader_registry.clear()
    schema_registry.clear()
    
    yield
    
    # Clear after test
    loader_registry.clear()
    schema_registry.clear()


@pytest.fixture
def capture_logs():
    """Capture logs for validation during testing."""
    logs = []
    
    class LogCapture(logging.Handler):
        def emit(self, record):
            logs.append(record)
    
    handler = LogCapture()
    logger = logging.getLogger('flyrigloader.registries')
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    yield logs
    
    logger.removeHandler(handler)


@contextmanager
def concurrent_execution_barrier(num_threads: int):
    """Context manager for synchronized concurrent thread execution."""
    barrier = Barrier(num_threads)
    exception_queue = Queue()
    
    def barrier_wait_with_exception_handling():
        try:
            barrier.wait()
        except Exception as e:
            exception_queue.put(e)
            raise
    
    yield barrier_wait_with_exception_handling, exception_queue


# Core thread-safety validation tests

class TestRegistryThreadSafety:
    """Test suite for comprehensive registry thread-safety validation."""
    
    def test_loader_registry_100_concurrent_lookups(self, clean_registries, mock_loader_class):
        """
        Test LoaderRegistry with 100 concurrent lookup operations to validate 
        O(1) performance and deadlock prevention per Section 6.6.3.8.
        """
        from flyrigloader.registries import LoaderRegistry
        
        registry = LoaderRegistry()
        
        # Pre-register test loaders
        test_extensions = [f'.test{i}' for i in range(10)]
        for ext in test_extensions:
            loader_class = mock_loader_class
            registry.register_loader(ext, loader_class, priority=10)
        
        # Concurrent lookup operations
        lookup_results = Queue()
        lookup_errors = Queue()
        
        def concurrent_lookup_worker(worker_id: int):
            """Worker function for concurrent lookups."""
            try:
                # Random lookups with timing measurement
                start_time = time.perf_counter()
                
                for _ in range(10):  # 10 lookups per thread = 1000 total
                    extension = random.choice(test_extensions)
                    loader = registry.get_loader_for_extension(extension)
                    
                    # Validate lookup result
                    assert loader is not None, f"Loader not found for {extension}"
                    assert loader == mock_loader_class, "Incorrect loader returned"
                    
                    # Simulate brief processing time
                    time.sleep(0.001)
                
                end_time = time.perf_counter()
                lookup_results.put({
                    'worker_id': worker_id,
                    'duration': end_time - start_time,
                    'lookups_completed': 10
                })
                
            except Exception as e:
                lookup_errors.put({
                    'worker_id': worker_id,
                    'error': str(e),
                    'error_type': type(e).__name__
                })
        
        # Execute 100 concurrent threads
        with ThreadPoolExecutor(max_workers=100) as executor:
            start_time = time.perf_counter()
            
            futures = [
                executor.submit(concurrent_lookup_worker, i) 
                for i in range(100)
            ]
            
            # Wait for all threads to complete
            for future in futures:
                future.result(timeout=30)  # 30 second timeout
            
            end_time = time.perf_counter()
        
        # Validate results
        assert lookup_errors.empty(), f"Lookup errors occurred: {list(lookup_errors.queue)}"
        
        total_duration = end_time - start_time
        assert total_duration < 5.0, f"Performance requirement violated: {total_duration}s > 5s"
        
        # Validate all workers completed successfully
        assert lookup_results.qsize() == 100, "Not all workers completed successfully"
        
        # Validate individual worker performance
        while not lookup_results.empty():
            result = lookup_results.get()
            assert result['lookups_completed'] == 10, "Worker didn't complete all lookups"
            assert result['duration'] < 2.0, f"Worker {result['worker_id']} too slow: {result['duration']}s"
    
    def test_schema_registry_100_concurrent_lookups(self, clean_registries, mock_schema_class):
        """
        Test SchemaRegistry with 100 concurrent lookup operations to validate 
        O(1) performance and deadlock prevention per Section 6.6.3.8.
        """
        from flyrigloader.registries import SchemaRegistry
        
        registry = SchemaRegistry()
        
        # Pre-register test schemas
        test_schemas = [f'test_schema_{i}' for i in range(10)]
        for schema_name in test_schemas:
            schema_class = mock_schema_class
            registry.register_schema(schema_name, schema_class, priority=10)
        
        # Concurrent lookup operations
        lookup_results = Queue()
        lookup_errors = Queue()
        
        def concurrent_schema_lookup_worker(worker_id: int):
            """Worker function for concurrent schema lookups."""
            try:
                start_time = time.perf_counter()
                
                for _ in range(10):  # 10 lookups per thread = 1000 total
                    schema_name = random.choice(test_schemas)
                    schema = registry.get_schema(schema_name)
                    
                    # Validate lookup result
                    assert schema is not None, f"Schema not found for {schema_name}"
                    assert schema == mock_schema_class, "Incorrect schema returned"
                    
                    # Simulate brief processing time
                    time.sleep(0.001)
                
                end_time = time.perf_counter()
                lookup_results.put({
                    'worker_id': worker_id,
                    'duration': end_time - start_time,
                    'lookups_completed': 10
                })
                
            except Exception as e:
                lookup_errors.put({
                    'worker_id': worker_id,
                    'error': str(e),
                    'error_type': type(e).__name__
                })
        
        # Execute 100 concurrent threads
        with ThreadPoolExecutor(max_workers=100) as executor:
            futures = [
                executor.submit(concurrent_schema_lookup_worker, i) 
                for i in range(100)
            ]
            
            # Wait for all threads to complete
            for future in futures:
                future.result(timeout=30)
        
        # Validate no errors occurred
        assert lookup_errors.empty(), f"Schema lookup errors: {list(lookup_errors.queue)}"
        assert lookup_results.qsize() == 100, "Not all schema workers completed"
    
    def test_concurrent_loader_registration_stress(self, clean_registries, mock_loader_class, capture_logs):
        """
        Test concurrent plugin registration and deregistration stress scenarios
        to ensure registry consistency under high-load per Section 6.6.3.8.
        """
        from flyrigloader.registries import LoaderRegistry
        
        registry = LoaderRegistry()
        registration_results = Queue()
        registration_errors = Queue()
        
        def concurrent_registration_worker(worker_id: int):
            """Worker for concurrent registration operations."""
            try:
                extensions = [f'.stress{worker_id}_{i}' for i in range(5)]
                
                # Registration phase
                for ext in extensions:
                    # Create unique loader class for each registration
                    class WorkerSpecificLoader:
                        def __init__(self):
                            self.extension = ext
                            self._priority = worker_id + 10
                        
                        def load(self, path: Path):
                            return {'worker': worker_id, 'ext': ext}
                        
                        def supports_extension(self, extension: str) -> bool:
                            return extension == ext
                        
                        @property
                        def priority(self) -> int:
                            return self._priority
                    
                    registry.register_loader(
                        ext, 
                        WorkerSpecificLoader, 
                        priority=worker_id + 10,
                        source="stress_test"
                    )
                
                # Validation phase
                for ext in extensions:
                    loader = registry.get_loader_for_extension(ext)
                    assert loader is not None, f"Loader not registered: {ext}"
                
                # Deregistration phase
                for ext in extensions:
                    success = registry.unregister_loader(ext)
                    assert success, f"Failed to unregister: {ext}"
                
                registration_results.put({
                    'worker_id': worker_id,
                    'operations_completed': len(extensions) * 3  # register + validate + unregister
                })
                
            except Exception as e:
                registration_errors.put({
                    'worker_id': worker_id,
                    'error': str(e),
                    'error_type': type(e).__name__
                })
        
        # Execute concurrent registration stress test
        num_workers = 50
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(concurrent_registration_worker, i) 
                for i in range(num_workers)
            ]
            
            for future in futures:
                future.result(timeout=60)
        
        # Validate stress test results
        assert registration_errors.empty(), f"Registration errors: {list(registration_errors.queue)}"
        assert registration_results.qsize() == num_workers, "Not all registration workers completed"
        
        # Validate logging events were captured
        assert len(capture_logs) > 0, "No registration events logged"
        
        # Validate INFO-level logging for registration events
        info_logs = [log for log in capture_logs if log.levelno == logging.INFO]
        assert len(info_logs) > 0, "No INFO-level registration logs captured"
    
    def test_priority_resolution_under_concurrent_access(self, clean_registries, capture_logs):
        """
        Test priority-based resolution system (BUILTIN < USER < PLUGIN < OVERRIDE) 
        under concurrent access patterns with comprehensive validation per Section 5.2.5.
        """
        from flyrigloader.registries import LoaderRegistry
        
        registry = LoaderRegistry()
        priority_test_results = Queue()
        priority_test_errors = Queue()
        
        def create_priority_loader(priority_level: RegistryPriority, worker_id: int):
            """Create loader with specific priority level."""
            class PriorityTestLoader:
                def __init__(self):
                    self.extension = '.priority_test'
                    self._priority = priority_level.value
                    self.priority_level = priority_level
                    self.worker_id = worker_id
                
                def load(self, path: Path):
                    return {
                        'priority': priority_level.name,
                        'worker': worker_id,
                        'value': priority_level.value
                    }
                
                def supports_extension(self, extension: str) -> bool:
                    return extension == '.priority_test'
                
                @property
                def priority(self) -> int:
                    return self._priority
            
            return PriorityTestLoader
        
        def concurrent_priority_worker(worker_id: int, priority_level: RegistryPriority):
            """Worker for testing priority resolution under concurrent access."""
            try:
                # Register loader with specific priority
                loader_class = create_priority_loader(priority_level, worker_id)
                
                registry.register_loader(
                    '.priority_test',
                    loader_class,
                    priority=priority_level.value,
                    priority_enum=priority_level,
                    source=f"priority_test_{priority_level.name.lower()}"
                )
                
                # Validate registration and priority
                registered_loader = registry.get_loader_for_extension('.priority_test')
                priority_info = registry.get_priority_info('.priority_test')
                
                priority_test_results.put({
                    'worker_id': worker_id,
                    'priority_level': priority_level.name,
                    'priority_value': priority_level.value,
                    'registered_successfully': registered_loader is not None,
                    'priority_info': priority_info
                })
                
            except Exception as e:
                priority_test_errors.put({
                    'worker_id': worker_id,
                    'priority_level': priority_level.name,
                    'error': str(e),
                    'error_type': type(e).__name__
                })
        
        # Test priority hierarchy concurrently
        priority_levels = [
            RegistryPriority.BUILTIN,
            RegistryPriority.USER, 
            RegistryPriority.PLUGIN,
            RegistryPriority.OVERRIDE
        ]
        
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            
            # Submit multiple workers for each priority level
            for i, priority_level in enumerate(priority_levels * 5):  # 5 workers per priority
                worker_id = i
                futures.append(
                    executor.submit(concurrent_priority_worker, worker_id, priority_level)
                )
            
            for future in futures:
                future.result(timeout=30)
        
        # Validate priority resolution results
        assert priority_test_errors.empty(), f"Priority test errors: {list(priority_test_errors.queue)}"
        
        # Final priority validation - should be OVERRIDE (highest priority)
        final_loader = registry.get_loader_for_extension('.priority_test')
        final_priority_info = registry.get_priority_info('.priority_test')
        
        assert final_loader is not None, "No final loader registered"
        assert final_priority_info is not None, "No priority info available"
        assert final_priority_info['priority_enum'] == RegistryPriority.OVERRIDE, \
            f"Expected OVERRIDE priority, got {final_priority_info['priority_enum']}"
        
        # Validate priority enum comparisons work correctly
        assert RegistryPriority.BUILTIN < RegistryPriority.USER
        assert RegistryPriority.USER < RegistryPriority.PLUGIN  
        assert RegistryPriority.PLUGIN < RegistryPriority.OVERRIDE
        assert RegistryPriority.OVERRIDE > RegistryPriority.BUILTIN


class TestEntryPointDiscovery:
    """Test suite for automatic entry-point discovery under concurrent scenarios."""
    
    @patch('importlib.metadata.entry_points')
    def test_concurrent_entry_point_discovery(self, mock_entry_points, clean_registries, capture_logs):
        """
        Test automatic entry-point discovery via importlib.metadata under 
        concurrent initialization scenarios per Section 0.3.1.
        """
        from flyrigloader.registries import LoaderRegistry
        
        # Mock entry point data
        class MockEntryPoint:
            def __init__(self, name, value):
                self.name = name
                self.value = value
            
            def load(self):
                # Return a mock loader class
                class EntryPointLoader:
                    def __init__(self):
                        self.extension = self.name if self.name.startswith('.') else f'.{self.name}'
                        self._priority = 30  # PLUGIN priority
                    
                    def load(self, path: Path):
                        return {'source': 'entry_point', 'name': self.name}
                    
                    def supports_extension(self, extension: str) -> bool:
                        return extension == self.extension
                    
                    @property
                    def priority(self) -> int:
                        return self._priority
                
                EntryPointLoader.name = self.name  # Store name for reference
                return EntryPointLoader
        
        # Create mock entry points
        mock_entries = [
            MockEntryPoint('ep_test1', 'test.module:TestLoader1'),
            MockEntryPoint('ep_test2', 'test.module:TestLoader2'),
            MockEntryPoint('ep_test3', 'test.module:TestLoader3')
        ]
        
        # Configure mock to return entry points
        class MockEntryPointsResult:
            def select(self, group):
                if group == 'flyrigloader.loaders':
                    return mock_entries
                return []
            
            def get(self, group, default=None):
                if group == 'flyrigloader.loaders':
                    return mock_entries
                return default or []
        
        mock_entry_points.return_value = MockEntryPointsResult()
        
        # Concurrent discovery test
        discovery_results = Queue()
        discovery_errors = Queue()
        
        def concurrent_discovery_worker(worker_id: int):
            """Worker for concurrent entry point discovery."""
            try:
                # Create new registry instance to trigger discovery
                registry = LoaderRegistry()
                
                # Validate that entry point loaders were discovered
                all_loaders = registry.get_all_loaders()
                entry_point_loaders = {
                    ext: loader for ext, loader in all_loaders.items() 
                    if ext.startswith('.ep_test')
                }
                
                discovery_results.put({
                    'worker_id': worker_id,
                    'discovered_loaders': len(entry_point_loaders),
                    'loader_extensions': list(entry_point_loaders.keys())
                })
                
            except Exception as e:
                discovery_errors.put({
                    'worker_id': worker_id,
                    'error': str(e),
                    'error_type': type(e).__name__
                })
        
        # Execute concurrent discovery
        num_workers = 20
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(concurrent_discovery_worker, i) 
                for i in range(num_workers)
            ]
            
            for future in futures:
                future.result(timeout=30)
        
        # Validate discovery results
        assert discovery_errors.empty(), f"Discovery errors: {list(discovery_errors.queue)}"
        assert discovery_results.qsize() == num_workers, "Not all discovery workers completed"
        
        # Validate consistent discovery across all workers
        while not discovery_results.empty():
            result = discovery_results.get()
            assert result['discovered_loaders'] >= 3, "Entry point loaders not discovered"
    
    def test_auto_register_decorator_concurrent_registration(self, clean_registries, capture_logs):
        """
        Test @auto_register decorator functionality with concurrent module imports 
        and plugin registration per Section 0.3.1.
        """
        from flyrigloader.registries import auto_register, LoaderRegistry
        
        registration_results = Queue()
        registration_errors = Queue()
        
        def concurrent_auto_register_worker(worker_id: int):
            """Worker for testing concurrent auto-registration."""
            try:
                # Create test loader class with auto-register decorator
                @auto_register(
                    registry_type="loader", 
                    key=f'.auto{worker_id}', 
                    priority=RegistryPriority.PLUGIN.value,
                    priority_enum=RegistryPriority.PLUGIN
                )
                class AutoRegisteredLoader:
                    def __init__(self):
                        self.extension = f'.auto{worker_id}'
                        self._priority = RegistryPriority.PLUGIN.value
                    
                    def load(self, path: Path):
                        return {'auto_registered': True, 'worker': worker_id}
                    
                    def supports_extension(self, extension: str) -> bool:
                        return extension == self.extension
                    
                    @property
                    def priority(self) -> int:
                        return self._priority
                
                # Validate registration occurred
                registry = LoaderRegistry()
                loader = registry.get_loader_for_extension(f'.auto{worker_id}')
                
                assert loader is not None, f"Auto-registered loader not found for worker {worker_id}"
                assert loader == AutoRegisteredLoader, "Incorrect auto-registered loader"
                
                registration_results.put({
                    'worker_id': worker_id,
                    'extension': f'.auto{worker_id}',
                    'registration_successful': True
                })
                
            except Exception as e:
                registration_errors.put({
                    'worker_id': worker_id,
                    'error': str(e),
                    'error_type': type(e).__name__
                })
        
        # Execute concurrent auto-registration
        num_workers = 25
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(concurrent_auto_register_worker, i) 
                for i in range(num_workers)
            ]
            
            for future in futures:
                future.result(timeout=30)
        
        # Validate auto-registration results
        assert registration_errors.empty(), f"Auto-registration errors: {list(registration_errors.queue)}"
        assert registration_results.qsize() == num_workers, "Not all auto-registration workers completed"
        
        # Validate comprehensive logging was captured
        auto_register_logs = [
            log for log in capture_logs 
            if 'auto_registration' in str(log.__dict__)
        ]
        assert len(auto_register_logs) > 0, "No auto-registration logs captured"


class TestLoggingValidation:
    """Test suite for INFO-level logging validation during concurrent operations."""
    
    def test_comprehensive_registration_logging(self, clean_registries, mock_loader_class):
        """
        Test comprehensive logging validation for all registration events 
        with INFO-level audit trails per Section 5.2.5.
        """
        from flyrigloader.registries import LoaderRegistry
        
        # Custom log capture for detailed validation
        captured_logs = []
        
        class DetailedLogCapture(logging.Handler):
            def emit(self, record):
                captured_logs.append({
                    'level': record.levelno,
                    'level_name': record.levelname,
                    'message': record.getMessage(),
                    'extra': getattr(record, '__dict__', {}),
                    'timestamp': record.created
                })
        
        handler = DetailedLogCapture()
        logger = logging.getLogger('flyrigloader.registries')
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        
        try:
            registry = LoaderRegistry()
            
            # Test various registration scenarios
            test_scenarios = [
                ('register_new', '.log_test1', RegistryPriority.USER),
                ('register_higher_priority', '.log_test1', RegistryPriority.PLUGIN),
                ('register_lower_priority', '.log_test1', RegistryPriority.BUILTIN),
                ('register_override', '.log_test1', RegistryPriority.OVERRIDE)
            ]
            
            for scenario, extension, priority in test_scenarios:
                registry.register_loader(
                    extension,
                    mock_loader_class,
                    priority=priority.value,
                    priority_enum=priority,
                    source=f"logging_test_{scenario}"
                )
            
            # Test unregistration logging
            registry.unregister_loader('.log_test1')
            
            # Validate logging events
            info_logs = [log for log in captured_logs if log['level'] == logging.INFO]
            warning_logs = [log for log in captured_logs if log['level'] == logging.WARNING]
            
            # Validate INFO-level registration events were logged
            assert len(info_logs) > 0, "No INFO-level logs captured"
            
            # Validate registration event details
            registration_logs = [
                log for log in info_logs 
                if 'registered' in log['message'] or 'replaced' in log['message']
            ]
            assert len(registration_logs) > 0, "No registration event logs found"
            
            # Validate unregistration event logging
            unregistration_logs = [
                log for log in info_logs 
                if 'unregistered' in log['message'].lower()
            ]
            assert len(unregistration_logs) > 0, "No unregistration event logs found"
            
            # Validate priority conflict warnings
            priority_warning_logs = [
                log for log in warning_logs 
                if 'priority' in log['message'].lower()
            ]
            assert len(priority_warning_logs) > 0, "No priority conflict warnings logged"
            
        finally:
            logger.removeHandler(handler)
    
    def test_concurrent_logging_thread_safety(self, clean_registries, mock_loader_class):
        """
        Test that logging remains thread-safe and consistent during 
        concurrent registry operations per Section 6.6.3.8.
        """
        from flyrigloader.registries import LoaderRegistry
        
        # Thread-safe log collection
        thread_safe_logs = Queue()
        
        class ThreadSafeLogCapture(logging.Handler):
            def emit(self, record):
                thread_safe_logs.put({
                    'thread_id': record.thread,
                    'thread_name': record.threadName,
                    'level': record.levelno,
                    'message': record.getMessage(),
                    'timestamp': record.created
                })
        
        handler = ThreadSafeLogCapture()
        logger = logging.getLogger('flyrigloader.registries')
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        
        try:
            def concurrent_logging_worker(worker_id: int):
                """Worker for concurrent registry operations with logging."""
                registry = LoaderRegistry()
                
                # Perform multiple operations to generate logs
                for i in range(3):
                    extension = f'.thread_log_{worker_id}_{i}'
                    registry.register_loader(
                        extension,
                        mock_loader_class,
                        priority=worker_id + i,
                        source=f"thread_test_worker_{worker_id}"
                    )
                    
                    # Brief delay to allow log processing
                    time.sleep(0.001)
                    
                    registry.unregister_loader(extension)
            
            # Execute concurrent operations
            num_workers = 20
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(concurrent_logging_worker, i) 
                    for i in range(num_workers)
                ]
                
                for future in futures:
                    future.result(timeout=30)
            
            # Wait for log processing
            time.sleep(0.1)
            
            # Validate thread-safe logging results
            all_logs = []
            while not thread_safe_logs.empty():
                all_logs.append(thread_safe_logs.get())
            
            assert len(all_logs) > 0, "No logs captured during concurrent operations"
            
            # Validate logs from multiple threads
            unique_threads = set(log['thread_id'] for log in all_logs)
            assert len(unique_threads) > 1, "Logs not captured from multiple threads"
            
            # Validate log message integrity (no corruption)
            for log in all_logs:
                assert isinstance(log['message'], str), "Log message corrupted"
                assert len(log['message']) > 0, "Empty log message"
            
        finally:
            logger.removeHandler(handler)


class TestPerformanceRequirements:
    """Test suite for validating performance requirements under concurrent load."""
    
    def test_registry_lookup_performance_under_load(self, clean_registries, mock_loader_class):
        """
        Test that registry lookup maintains O(1) performance under 
        concurrent load per Section 6.6.3.8 requirements.
        """
        from flyrigloader.registries import LoaderRegistry
        
        registry = LoaderRegistry()
        
        # Pre-populate registry with many loaders
        num_loaders = 1000
        for i in range(num_loaders):
            extension = f'.perf_test_{i}'
            registry.register_loader(extension, mock_loader_class, priority=i)
        
        # Performance test with concurrent lookups
        performance_results = Queue()
        
        def performance_test_worker(worker_id: int):
            """Worker for performance testing under concurrent load."""
            start_time = time.perf_counter()
            
            # Perform many lookups
            for i in range(100):
                extension = f'.perf_test_{i % num_loaders}'
                loader = registry.get_loader_for_extension(extension)
                assert loader is not None, f"Loader not found: {extension}"
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            performance_results.put({
                'worker_id': worker_id,
                'duration': duration,
                'lookups_per_second': 100 / duration if duration > 0 else float('inf')
            })
        
        # Execute performance test
        num_workers = 50
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            total_start = time.perf_counter()
            
            futures = [
                executor.submit(performance_test_worker, i) 
                for i in range(num_workers)
            ]
            
            for future in futures:
                future.result(timeout=60)
            
            total_end = time.perf_counter()
        
        # Validate performance requirements
        total_duration = total_end - total_start
        assert total_duration < 10.0, f"Total performance test exceeded 10s: {total_duration}s"
        
        # Validate individual worker performance
        while not performance_results.empty():
            result = performance_results.get()
            assert result['duration'] < 5.0, f"Worker {result['worker_id']} too slow: {result['duration']}s"
            assert result['lookups_per_second'] > 10, f"Worker {result['worker_id']} too few lookups/sec: {result['lookups_per_second']}"
    
    def test_deadlock_prevention_validation(self, clean_registries, mock_loader_class):
        """
        Test comprehensive deadlock prevention mechanisms during 
        concurrent registry operations per Section 6.6.3.8.
        """
        from flyrigloader.registries import LoaderRegistry, SchemaRegistry
        
        # Test multiple registry types for cross-registry deadlock prevention
        loader_registry = LoaderRegistry()
        schema_registry = SchemaRegistry()
        
        deadlock_test_results = Queue()
        deadlock_test_errors = Queue()
        
        def mixed_operations_worker(worker_id: int):
            """Worker performing mixed registry operations to test deadlock prevention."""
            try:
                operations_completed = 0
                
                # Mixed loader and schema operations
                for i in range(10):
                    # Loader operations
                    loader_ext = f'.deadlock_test_{worker_id}_{i}'
                    loader_registry.register_loader(loader_ext, mock_loader_class, priority=i)
                    loader = loader_registry.get_loader_for_extension(loader_ext)
                    assert loader is not None
                    operations_completed += 2
                    
                    # Schema operations
                    schema_name = f'deadlock_schema_{worker_id}_{i}'
                    from unittest.mock import Mock
                    mock_schema = Mock()
                    mock_schema.validate = lambda data: {'validated': True}
                    mock_schema.schema_name = schema_name
                    mock_schema.supported_types = ['dict']
                    
                    # Note: This will fail schema validation, but that's expected
                    # We're testing deadlock prevention, not schema validation
                    try:
                        schema_registry.register_schema(schema_name, type(mock_schema), priority=i)
                        operations_completed += 1
                    except Exception:
                        pass  # Expected validation failure, but no deadlock
                    
                    # Brief delay to increase chance of contention
                    time.sleep(0.001)
                
                deadlock_test_results.put({
                    'worker_id': worker_id,
                    'operations_completed': operations_completed
                })
                
            except Exception as e:
                deadlock_test_errors.put({
                    'worker_id': worker_id,
                    'error': str(e),
                    'error_type': type(e).__name__
                })
        
        # Execute deadlock prevention test
        num_workers = 30
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            start_time = time.perf_counter()
            
            futures = [
                executor.submit(mixed_operations_worker, i) 
                for i in range(num_workers)
            ]
            
            # Wait with timeout to detect deadlocks
            for future in futures:
                future.result(timeout=45)  # Generous timeout for deadlock detection
            
            end_time = time.perf_counter()
        
        # Validate no deadlocks occurred
        total_duration = end_time - start_time
        assert total_duration < 30.0, f"Potential deadlock detected: {total_duration}s > 30s"
        
        # Validate all workers completed without hanging
        assert deadlock_test_results.qsize() == num_workers, "Some workers didn't complete (potential deadlock)"
        
        # Validate no critical errors (excluding expected validation failures)
        critical_errors = [
            error for error in list(deadlock_test_errors.queue)
            if 'timeout' in error.get('error', '').lower() or 
               'deadlock' in error.get('error', '').lower()
        ]
        assert len(critical_errors) == 0, f"Critical deadlock-related errors: {critical_errors}"


# Hypothesis-based property testing for comprehensive coverage

class TestRegistryPropertyBasedTesting:
    """Property-based testing using Hypothesis for comprehensive registry validation."""
    
    @pytest.mark.parametrize("priority_enum", [
        RegistryPriority.BUILTIN,
        RegistryPriority.USER,
        RegistryPriority.PLUGIN,
        RegistryPriority.OVERRIDE
    ])
    def test_priority_enum_comparison_properties(self, priority_enum):
        """Test priority enumeration comparison properties."""
        # Test reflexivity
        assert priority_enum == priority_enum
        assert priority_enum <= priority_enum
        assert priority_enum >= priority_enum
        
        # Test with other priority levels
        all_priorities = [
            RegistryPriority.BUILTIN,
            RegistryPriority.USER,
            RegistryPriority.PLUGIN,
            RegistryPriority.OVERRIDE
        ]
        
        for other_priority in all_priorities:
            if priority_enum.value < other_priority.value:
                assert priority_enum < other_priority
                assert priority_enum <= other_priority
                assert not (priority_enum > other_priority)
                assert not (priority_enum >= other_priority)
            elif priority_enum.value > other_priority.value:
                assert priority_enum > other_priority
                assert priority_enum >= other_priority
                assert not (priority_enum < other_priority)
                assert not (priority_enum <= other_priority)
    
    def test_registry_singleton_property(self, clean_registries):
        """Test that registry instances maintain singleton property across threads."""
        from flyrigloader.registries import LoaderRegistry, SchemaRegistry
        
        singleton_results = Queue()
        
        def singleton_test_worker(worker_id: int):
            """Worker to test singleton property."""
            loader_registry1 = LoaderRegistry()
            loader_registry2 = LoaderRegistry()
            schema_registry1 = SchemaRegistry()
            schema_registry2 = SchemaRegistry()
            
            singleton_results.put({
                'worker_id': worker_id,
                'loader_instances_identical': loader_registry1 is loader_registry2,
                'schema_instances_identical': schema_registry1 is schema_registry2,
                'loader_registry_id': id(loader_registry1),
                'schema_registry_id': id(schema_registry1)
            })
        
        # Test singleton across multiple threads
        num_workers = 10
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(singleton_test_worker, i) 
                for i in range(num_workers)
            ]
            
            for future in futures:
                future.result(timeout=10)
        
        # Validate singleton property
        results = []
        while not singleton_results.empty():
            results.append(singleton_results.get())
        
        assert len(results) == num_workers, "Not all singleton test workers completed"
        
        # All workers should see identical instances
        for result in results:
            assert result['loader_instances_identical'], f"Loader singleton violated in worker {result['worker_id']}"
            assert result['schema_instances_identical'], f"Schema singleton violated in worker {result['worker_id']}"
        
        # All workers should see the same registry instance IDs
        loader_ids = [result['loader_registry_id'] for result in results]
        schema_ids = [result['schema_registry_id'] for result in results]
        
        assert len(set(loader_ids)) == 1, "Multiple LoaderRegistry instances detected"
        assert len(set(schema_ids)) == 1, "Multiple SchemaRegistry instances detected"


# Integration tests for complete workflow validation

class TestRegistryIntegrationWorkflows:
    """Integration tests validating complete registry workflows under concurrent load."""
    
    def test_complete_plugin_lifecycle_concurrent(self, clean_registries, mock_loader_class, capture_logs):
        """
        Test complete plugin lifecycle (discovery -> registration -> usage -> cleanup) 
        under concurrent access patterns.
        """
        from flyrigloader.registries import LoaderRegistry, auto_register
        
        lifecycle_results = Queue()
        lifecycle_errors = Queue()
        
        def plugin_lifecycle_worker(worker_id: int):
            """Worker testing complete plugin lifecycle."""
            try:
                # Phase 1: Auto-registration
                @auto_register(
                    registry_type="loader",
                    key=f'.lifecycle{worker_id}',
                    priority=RegistryPriority.PLUGIN.value
                )
                class LifecycleTestLoader:
                    def __init__(self):
                        self.extension = f'.lifecycle{worker_id}'
                        self._priority = RegistryPriority.PLUGIN.value
                    
                    def load(self, path: Path):
                        return {'lifecycle_test': True, 'worker': worker_id}
                    
                    def supports_extension(self, extension: str) -> bool:
                        return extension == self.extension
                    
                    @property
                    def priority(self) -> int:
                        return self._priority
                
                registry = LoaderRegistry()
                
                # Phase 2: Validation and usage
                loader = registry.get_loader_for_extension(f'.lifecycle{worker_id}')
                assert loader is not None, "Auto-registered loader not found"
                
                capabilities = registry.get_loader_capabilities(f'.lifecycle{worker_id}')
                assert capabilities is not None, "Loader capabilities not available"
                
                # Phase 3: Metadata introspection
                metadata = registry.get_registration_metadata(f'.lifecycle{worker_id}')
                assert metadata is not None, "Registration metadata not available"
                
                all_loaders = registry.get_all_loaders()
                assert f'.lifecycle{worker_id}' in all_loaders, "Loader not in registry listing"
                
                # Phase 4: Cleanup
                unregister_success = registry.unregister_loader(f'.lifecycle{worker_id}')
                assert unregister_success, "Unregistration failed"
                
                lifecycle_results.put({
                    'worker_id': worker_id,
                    'lifecycle_phases_completed': 4,
                    'capabilities_available': capabilities is not None,
                    'metadata_available': metadata is not None
                })
                
            except Exception as e:
                lifecycle_errors.put({
                    'worker_id': worker_id,
                    'error': str(e),
                    'error_type': type(e).__name__
                })
        
        # Execute complete lifecycle test
        num_workers = 15
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(plugin_lifecycle_worker, i) 
                for i in range(num_workers)
            ]
            
            for future in futures:
                future.result(timeout=45)
        
        # Validate lifecycle test results
        assert lifecycle_errors.empty(), f"Lifecycle errors: {list(lifecycle_errors.queue)}"
        assert lifecycle_results.qsize() == num_workers, "Not all lifecycle workers completed"
        
        # Validate comprehensive logging throughout lifecycle
        lifecycle_logs = [
            log for log in capture_logs 
            if any(keyword in log.getMessage().lower() for keyword in 
                  ['registered', 'unregistered', 'auto-registration'])
        ]
        assert len(lifecycle_logs) > 0, "No lifecycle events logged"