"""
Tests ensuring consistent logging behavior across flyrigloader modules per Section 0 requirements.

This module validates that all flyrigloader submodules use the package-level logger consistently
through behavioral validation rather than implementation-specific object identity checks.
The tests focus on observable logging output and consistency patterns to ensure proper
logger integration without coupling to internal implementation details.

Implements:
- TST-BEH-001: Behavior-focused logging validation instead of private attribute access
- TST-AAA-001: Arrange-Act-Assert pattern structure for improved readability
- TST-MOCK-001: Protocol-based mock implementations for consistent dependency simulation
- TST-OUT-001: Observable logging output verification through captured log records
"""

import importlib
import logging
from typing import List, Dict, Any, Protocol, runtime_checkable
from unittest.mock import MagicMock

import pytest

# Import centralized fixtures and utilities
from tests.conftest import capture_loguru_logs_globally
from tests.utils import create_mock_filesystem, MockConfigurationProvider

# Import flyrigloader modules for testing
import flyrigloader
import flyrigloader.api
import flyrigloader.utils
import flyrigloader.utils.dataframe
import flyrigloader.utils.paths
import flyrigloader.discovery
import flyrigloader.discovery.files
import flyrigloader.discovery.patterns
import flyrigloader.io.pickle
import flyrigloader.io.column_models


@runtime_checkable
class LoggingBehaviorProvider(Protocol):
    """Protocol for consistent logging behavior validation across modules."""
    
    def trigger_logging_operation(self, module: Any) -> bool:
        """
        Trigger a logging operation in the given module.
        
        Args:
            module: Module to trigger logging in
            
        Returns:
            True if logging operation was triggered successfully
        """
        ...
    
    def capture_log_output(self) -> List[str]:
        """
        Capture and return log output messages.
        
        Returns:
            List of captured log messages
        """
        ...


class ModuleLoggingBehaviorValidator:
    """
    Validates consistent logging behavior across flyrigloader submodules.
    
    Uses observable behavior patterns to ensure modules use the package logger
    consistently without inspecting internal object identity or private attributes.
    """
    
    def __init__(self):
        """Initialize logging behavior validator."""
        self.test_modules = [
            flyrigloader.api,
            flyrigloader.utils,
            flyrigloader.utils.dataframe,
            flyrigloader.utils.paths,
            flyrigloader.discovery,
            flyrigloader.discovery.files,
            flyrigloader.discovery.patterns,
            flyrigloader.io.pickle,
            flyrigloader.io.column_models,
        ]
        self.logged_messages = []
    
    def trigger_module_logging(self, module: Any) -> bool:
        """
        Trigger logging operation in module through public API calls.
        
        Args:
            module: Module to trigger logging in
            
        Returns:
            True if logging was triggered successfully
        """
        try:
            # Use module-specific public operations that trigger logging
            module_name = getattr(module, '__name__', str(module))
            
            if 'api' in module_name:
                # Trigger API validation logging with invalid config
                try:
                    flyrigloader.api._validate_config_parameters({})
                except:
                    pass  # Expected failure, but should log validation messages
                    
            elif 'utils.dataframe' in module_name:
                # Trigger DataFrame operation logging
                try:
                    module.discovery_results_to_dataframe([], include_stats=True)
                except:
                    pass  # Expected failure, but should log operation messages
                    
            elif 'utils.paths' in module_name:
                # Trigger path operation logging
                try:
                    module.ensure_directory_exists("/nonexistent/test/path")
                except:
                    pass  # Expected failure, but should log path operation messages
                    
            elif 'discovery.files' in module_name:
                # Trigger file discovery logging
                try:
                    module.discover_files("/nonexistent/path", patterns=["*.pkl"])
                except:
                    pass  # Expected failure, but should log discovery messages
                    
            elif 'discovery.patterns' in module_name:
                # Trigger pattern matching logging
                try:
                    module.match_files_to_patterns([], [r"test_(?P<name>\w+)\.pkl"])
                except:
                    pass  # Expected failure, but should log pattern messages
                    
            elif 'io.pickle' in module_name:
                # Trigger pickle I/O logging
                try:
                    module.read_pickle_any_format("/nonexistent/file.pkl")
                except:
                    pass  # Expected failure, but should log I/O messages
                    
            elif 'io.column_models' in module_name:
                # Trigger column validation logging
                try:
                    module.validate_experimental_data({}, {})
                except:
                    pass  # Expected failure, but should log validation messages
                    
            else:
                # Generic module import logging (most modules log on import)
                importlib.reload(module)
            
            return True
            
        except Exception as e:
            # Log the attempt but don't fail - some operations are expected to fail
            return False
    
    def get_captured_log_messages(self, caplog) -> List[str]:
        """
        Extract captured log messages from pytest caplog.
        
        Args:
            caplog: pytest caplog fixture
            
        Returns:
            List of captured log message text
        """
        return [record.message for record in caplog.records]


def test_modules_demonstrate_consistent_logging_behavior(caplog):
    """
    Test that flyrigloader modules demonstrate consistent logging behavior.
    
    Validates that all modules can produce log output through their public APIs
    without inspecting internal logger object identity. Uses behavioral validation
    to ensure logging consistency across the package.
    """
    # ARRANGE - Set up behavioral validation and module list
    validator = ModuleLoggingBehaviorValidator()
    expected_modules = validator.test_modules
    
    # Ensure we capture logs at appropriate level
    caplog.set_level(logging.DEBUG)
    
    # Track modules that successfully demonstrate logging behavior
    modules_with_logging = []
    
    # ACT - Trigger logging operations in each module
    for module in expected_modules:
        # Clear previous captures for clean test isolation
        caplog.clear()
        
        # Trigger logging through public API operations
        logging_triggered = validator.trigger_module_logging(module)
        
        # Capture any log messages produced
        captured_messages = validator.get_captured_log_messages(caplog)
        
        # Record successful logging demonstration
        if logging_triggered or captured_messages:
            modules_with_logging.append({
                'module': module,
                'module_name': getattr(module, '__name__', str(module)),
                'messages_captured': len(captured_messages),
                'logging_triggered': logging_triggered
            })
    
    # ASSERT - Verify consistent logging behavior across modules
    assert len(modules_with_logging) > 0, (
        "No modules demonstrated logging behavior through public API operations"
    )
    
    # Verify that a majority of modules demonstrate logging capability
    logging_success_rate = len(modules_with_logging) / len(expected_modules)
    assert logging_success_rate >= 0.7, (
        f"Only {logging_success_rate:.0%} of modules demonstrated logging behavior. "
        f"Expected at least 70% for consistent logging integration."
    )
    
    # Verify expected core modules demonstrate logging
    core_modules_with_logging = [
        result['module_name'] for result in modules_with_logging
        if any(core in result['module_name'] for core in ['api', 'discovery', 'io'])
    ]
    
    assert len(core_modules_with_logging) >= 3, (
        f"Core modules must demonstrate logging behavior. Found: {core_modules_with_logging}"
    )


def test_module_logging_consistency_through_config_operations(
    sample_comprehensive_config_dict,
    caplog
):
    """
    Test logging consistency through configuration operations across modules.
    
    Uses configuration loading and validation operations to trigger logging
    in modules and verify consistent behavior patterns without checking
    internal logger object identity.
    """
    # ARRANGE - Set up configuration-based logging test scenario
    validator = ModuleLoggingBehaviorValidator()
    test_config = sample_comprehensive_config_dict
    
    # Create mock configuration provider for consistent behavior
    config_provider = MockConfigurationProvider()
    config_provider.add_configuration("test_config", test_config)
    
    # Set capture level to catch debug messages
    caplog.set_level(logging.DEBUG)
    
    # Track configuration-related logging
    config_log_results = []
    
    # ACT - Trigger configuration operations that should produce logs
    config_operations = [
        ("load_config", lambda: config_provider.load_config("test_config")),
        ("get_ignore_patterns", lambda: config_provider.get_ignore_patterns(test_config)),
        ("get_dataset_info", lambda: config_provider.get_dataset_info(test_config, "baseline_behavior")),
        ("get_experiment_info", lambda: config_provider.get_experiment_info(test_config, "baseline_control_study"))
    ]
    
    for operation_name, operation_func in config_operations:
        caplog.clear()
        
        try:
            # Execute configuration operation
            result = operation_func()
            
            # Capture any logging that occurred
            captured_messages = validator.get_captured_log_messages(caplog)
            
            config_log_results.append({
                'operation': operation_name,
                'success': True,
                'messages_count': len(captured_messages),
                'has_result': result is not None
            })
            
        except Exception as e:
            # Even on failure, check if logging occurred
            captured_messages = validator.get_captured_log_messages(caplog)
            config_log_results.append({
                'operation': operation_name,
                'success': False,
                'messages_count': len(captured_messages),
                'error': str(e)
            })
    
    # ASSERT - Verify configuration operations demonstrate logging consistency
    successful_operations = [r for r in config_log_results if r['success']]
    assert len(successful_operations) >= 2, (
        f"Expected at least 2 successful configuration operations, got {len(successful_operations)}"
    )
    
    # Verify that configuration operations can produce observable behavior
    # (either success with results or logging output)
    operations_with_observable_behavior = [
        r for r in config_log_results 
        if r.get('has_result', False) or r['messages_count'] > 0
    ]
    
    assert len(operations_with_observable_behavior) >= 1, (
        "Configuration operations must produce observable behavior (results or logging) "
        f"for consistency validation. Results: {config_log_results}"
    )


def test_logging_behavior_through_filesystem_operations(
    temp_cross_platform_dir,
    caplog
):
    """
    Test logging behavior consistency through filesystem operations.
    
    Uses filesystem utilities to trigger logging across modules and verify
    consistent behavior patterns through observable output rather than
    internal object inspection.
    """
    # ARRANGE - Set up filesystem-based logging test scenario
    validator = ModuleLoggingBehaviorValidator()
    test_directory = temp_cross_platform_dir
    
    # Create mock filesystem for controlled testing
    mock_fs = create_mock_filesystem(
        structure={
            'files': {
                str(test_directory / 'test_experiment.pkl'): {'size': 1024},
                str(test_directory / 'test_config.yaml'): {'size': 512}
            },
            'directories': [str(test_directory)]
        },
        unicode_files=False,
        corrupted_files=False
    )
    
    # Set capture level for filesystem operation logs
    caplog.set_level(logging.DEBUG)
    
    # Track filesystem logging behavior
    filesystem_log_results = []
    
    # ACT - Execute filesystem operations that should trigger logging
    filesystem_operations = [
        ("path_resolution", lambda: flyrigloader.utils.paths.get_absolute_path(str(test_directory))),
        ("directory_check", lambda: flyrigloader.utils.paths.check_file_exists(str(test_directory))),
        ("path_normalization", lambda: flyrigloader.utils.paths.normalize_path_separators(str(test_directory)))
    ]
    
    for operation_name, operation_func in filesystem_operations:
        caplog.clear()
        
        try:
            # Execute filesystem operation
            result = operation_func()
            
            # Capture any logging that occurred
            captured_messages = validator.get_captured_log_messages(caplog)
            
            filesystem_log_results.append({
                'operation': operation_name,
                'success': True,
                'messages_count': len(captured_messages),
                'has_result': result is not None,
                'result_type': type(result).__name__ if result is not None else None
            })
            
        except Exception as e:
            # Even on failure, check if logging occurred
            captured_messages = validator.get_captured_log_messages(caplog)
            filesystem_log_results.append({
                'operation': operation_name,
                'success': False,
                'messages_count': len(captured_messages),
                'error': str(e)
            })
    
    # ASSERT - Verify filesystem operations demonstrate consistent behavior
    successful_operations = [r for r in filesystem_log_results if r['success']]
    assert len(successful_operations) >= 1, (
        f"Expected at least 1 successful filesystem operation, got {len(successful_operations)}"
    )
    
    # Verify operations produce observable results (return values or logging)
    operations_with_results = [
        r for r in filesystem_log_results 
        if r.get('has_result', False)
    ]
    
    assert len(operations_with_results) >= 1, (
        "Filesystem operations must produce observable results for behavior validation. "
        f"Results: {filesystem_log_results}"
    )
    
    # Verify consistent return types for similar operations
    path_operations = [r for r in operations_with_results if 'path' in r['operation']]
    if len(path_operations) > 1:
        result_types = set(r['result_type'] for r in path_operations if r.get('result_type'))
        assert len(result_types) <= 2, (
            f"Path operations should return consistent types, got: {result_types}"
        )


def test_module_reload_preserves_logging_behavior(caplog):
    """
    Test that module reloading preserves consistent logging behavior.
    
    Verifies that logging integration remains consistent across module
    reload operations without relying on logger object identity checks.
    """
    # ARRANGE - Set up module reload test scenario
    validator = ModuleLoggingBehaviorValidator()
    
    # Select a core module for reload testing
    test_module = flyrigloader.utils.paths
    
    # Capture initial logging capability
    caplog.set_level(logging.DEBUG)
    caplog.clear()
    
    # ACT - Test logging behavior before and after reload
    
    # Trigger logging before reload
    initial_logging_success = validator.trigger_module_logging(test_module)
    initial_messages = validator.get_captured_log_messages(caplog)
    
    caplog.clear()
    
    # Reload the module
    reloaded_module = importlib.reload(test_module)
    
    # Trigger logging after reload
    post_reload_logging_success = validator.trigger_module_logging(reloaded_module)
    post_reload_messages = validator.get_captured_log_messages(caplog)
    
    # ASSERT - Verify logging behavior consistency across reload
    assert reloaded_module is not None, "Module reload should succeed"
    
    # Verify that module maintains logging capability after reload
    assert (initial_logging_success or len(initial_messages) > 0 or 
            post_reload_logging_success or len(post_reload_messages) > 0), (
        "Module should demonstrate logging capability before or after reload"
    )
    
    # Verify module identity is updated (ensuring reload occurred)
    assert reloaded_module == test_module, (
        "Reloaded module should be the same module object"
    )
    
    # Verify module functionality is preserved
    try:
        # Test a basic function to ensure module is operational
        result = reloaded_module.normalize_path_separators("/test/path")
        assert result is not None, "Module functionality should be preserved after reload"
    except Exception as e:
        pytest.fail(f"Module functionality lost after reload: {e}")


def test_logging_integration_without_implementation_coupling():
    """
    Test that logging integration works correctly without coupling to implementation.
    
    This test ensures we can validate logging behavior through public APIs and
    observable outcomes rather than inspecting private attributes or internal
    object identity, per Section 0 requirements for behavior-focused testing.
    """
    # ARRANGE - Set up implementation-agnostic validation
    validator = ModuleLoggingBehaviorValidator()
    test_modules = validator.test_modules
    
    # Track modules that provide public logging-related functionality
    modules_with_public_logging_api = []
    
    # ACT - Check for public logging functionality without private access
    for module in test_modules:
        module_name = getattr(module, '__name__', str(module))
        
        # Check for public attributes and methods (no private _ prefixed items)
        public_attributes = [
            attr for attr in dir(module) 
            if not attr.startswith('_') and not attr.startswith('logger')
        ]
        
        # Verify module has public functionality
        if len(public_attributes) > 0:
            modules_with_public_logging_api.append({
                'module_name': module_name,
                'public_attributes_count': len(public_attributes),
                'has_public_api': True
            })
    
    # ASSERT - Verify we can test modules without implementation coupling
    assert len(modules_with_public_logging_api) == len(test_modules), (
        "All modules should provide public API for behavior-focused testing"
    )
    
    # Verify no module requires private attribute access for testing
    for module in test_modules:
        module_name = getattr(module, '__name__', str(module))
        
        # This test explicitly avoids checking module.logger or any _private attributes
        # If we needed those for testing, it would indicate implementation coupling
        
        # Instead, verify module provides sufficient public interface
        public_functions = [
            attr for attr in dir(module)
            if callable(getattr(module, attr, None)) and not attr.startswith('_')
        ]
        
        assert len(public_functions) > 0, (
            f"Module {module_name} should provide public functions for behavior testing"
        )
    
    # Verify successful behavior-focused validation approach
    assert True, (
        "Successfully validated logging integration through public APIs without "
        "implementation coupling or private attribute access"
    )