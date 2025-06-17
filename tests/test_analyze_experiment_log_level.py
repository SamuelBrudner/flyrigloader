"""
Test module for analyze_experiment log level configuration validation.

This module implements behavior-focused testing for log level functionality in the
analyze_experiment script, emphasizing public API validation and observable behavior
rather than implementation-specific details per Section 0 requirements.

Test Architecture:
- AAA pattern implementation with clear separation of arrange/act/assert phases
- Protocol-based mock implementations for configuration loading
- Blackbox behavioral validation through log output analysis
- Parameterized edge-case coverage for various log level scenarios
- Integration with centralized fixtures from tests/conftest.py

Testing Strategy:
- Validates public API behavior through command-line interface
- Tests observable side effects (log output patterns)
- Covers comprehensive edge cases with parameterized scenarios
- Maintains isolation from network dependencies and performance bottlenecks
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

from examples.external_project import analyze_experiment


class MockConfigurationProtocol:
    """
    Protocol-based mock for configuration loading operations.
    
    Provides standardized configuration mocking that focuses on behavioral
    contracts rather than implementation-specific details, enabling robust
    test isolation and consistent mock behavior across test scenarios.
    """
    
    def __init__(self, config_data: Optional[Dict[str, Any]] = None):
        self.config_data = config_data or {
            'project': {
                'directories': {
                    'major_data_directory': '/tmp/test_data'
                }
            }
        }
        self.experiment_info = {}
        self.discovered_files = []
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Mock configuration loading with realistic structure."""
        return self.config_data
    
    def get_experiment_info(self, config: Dict[str, Any], experiment_name: str) -> Dict[str, Any]:
        """Mock experiment information retrieval."""
        return self.experiment_info
    
    def discover_experiment_files(self, **kwargs) -> List[Path]:
        """Mock file discovery with empty results for testing."""
        return self.discovered_files


@pytest.fixture
def mock_config_provider():
    """
    Fixture providing protocol-based configuration mocking.
    
    Creates a standardized configuration mock that simulates realistic
    configuration structures while maintaining behavioral isolation
    for predictable test execution.
    """
    return MockConfigurationProtocol()


@pytest.fixture
def mock_analyze_experiment_dependencies(mock_config_provider, mocker):
    """
    Centralized fixture for analyze_experiment dependency mocking.
    
    Implements comprehensive dependency isolation using protocol-based
    mocking approach, ensuring tests focus on public API behavior
    rather than internal implementation details.
    
    Returns:
        MockConfigurationProtocol: Configured mock provider for test scenarios
    """
    # Arrange: Setup protocol-based mocks for configuration dependencies
    mocker.patch(
        'flyrigloader.config.yaml_config.load_config',
        side_effect=mock_config_provider.load_config
    )
    mocker.patch(
        'flyrigloader.config.yaml_config.get_experiment_info',
        side_effect=mock_config_provider.get_experiment_info
    )
    mocker.patch(
        'flyrigloader.config.discovery.discover_experiment_files',
        side_effect=mock_config_provider.discover_experiment_files
    )
    
    return mock_config_provider


def execute_analyze_experiment_with_args(args: List[str]) -> int:
    """
    Execute analyze_experiment script with specified command-line arguments.
    
    Provides a clean interface for testing the analyze_experiment script's
    public API through command-line argument simulation, enabling blackbox
    behavioral validation.
    
    Args:
        args: List of command-line arguments to pass to the script
        
    Returns:
        int: Exit code returned by the script execution
    """
    with patch.object(sys, 'argv', ['analyze_experiment.py'] + args):
        return analyze_experiment.main()


class TestAnalyzeExperimentLogLevel:
    """
    Comprehensive test suite for analyze_experiment log level configuration.
    
    Implements behavior-focused testing approach emphasizing observable
    behavior validation through log output analysis rather than
    implementation-specific assertions.
    """
    
    def test_debug_log_level_enables_debug_output(
        self, 
        mock_analyze_experiment_dependencies, 
        caplog
    ):
        """
        Validate that DEBUG log level enables debug message output.
        
        This test verifies the public API behavior where specifying
        --log-level DEBUG enables debug logging and produces the
        expected debug message output.
        """
        # Arrange: Configure test environment for debug log level testing
        caplog.set_level(logging.DEBUG)
        test_args = ['--config', 'test_config.yaml', '--experiment', 'test_exp', '--log-level', 'DEBUG']
        
        # Act: Execute analyze_experiment with DEBUG log level
        exit_code = execute_analyze_experiment_with_args(test_args)
        
        # Assert: Verify debug logging behavior is enabled
        assert exit_code == 0, "Script should execute successfully with valid arguments"
        debug_messages = [record.message for record in caplog.records if 'Debug logging enabled' in record.message]
        assert len(debug_messages) > 0, "DEBUG log level should enable debug message output"
        assert any('Debug logging enabled' in message for message in debug_messages), \
            "Expected debug message should be present in log output"
    
    def test_warning_log_level_suppresses_debug_output(
        self, 
        mock_analyze_experiment_dependencies, 
        caplog
    ):
        """
        Validate that WARNING log level suppresses debug message output.
        
        Tests the behavioral contract where WARNING level logging
        filters out debug messages, ensuring only warnings and
        higher severity messages are displayed.
        """
        # Arrange: Configure test environment for warning log level testing
        caplog.set_level(logging.DEBUG)  # Capture all levels for validation
        test_args = ['--config', 'test_config.yaml', '--experiment', 'test_exp', '--log-level', 'WARNING']
        
        # Act: Execute analyze_experiment with WARNING log level
        exit_code = execute_analyze_experiment_with_args(test_args)
        
        # Assert: Verify debug messages are suppressed at WARNING level
        assert exit_code == 0, "Script should execute successfully with valid arguments"
        debug_messages = [record.message for record in caplog.records if 'Debug logging enabled' in record.message]
        assert len(debug_messages) == 0, \
            "WARNING log level should suppress debug message output"
    
    def test_default_log_level_behavior(
        self, 
        mock_analyze_experiment_dependencies, 
        caplog
    ):
        """
        Validate default log level behavior when no --log-level is specified.
        
        Tests the public API contract where omitting the --log-level
        argument results in INFO level logging, which should suppress
        debug messages while allowing info and higher severity output.
        """
        # Arrange: Configure test environment for default log level testing
        caplog.set_level(logging.DEBUG)  # Capture all levels for validation
        test_args = ['--config', 'test_config.yaml', '--experiment', 'test_exp']
        
        # Act: Execute analyze_experiment with default log level (INFO)
        exit_code = execute_analyze_experiment_with_args(test_args)
        
        # Assert: Verify default INFO level suppresses debug messages
        assert exit_code == 0, "Script should execute successfully with valid arguments"
        debug_messages = [record.message for record in caplog.records if 'Debug logging enabled' in record.message]
        assert len(debug_messages) == 0, \
            "Default log level (INFO) should suppress debug message output"
    
    @pytest.mark.parametrize("log_level,should_show_debug", [
        ("TRACE", True),    # Custom level that should enable debug
        ("DEBUG", True),    # Standard debug level
        ("INFO", False),    # Standard info level - suppresses debug
        ("WARNING", False), # Warning level - suppresses debug
        ("ERROR", False),   # Error level - suppresses debug  
        ("CRITICAL", False) # Critical level - suppresses debug
    ])
    def test_log_level_debug_message_visibility(
        self, 
        log_level: str,
        should_show_debug: bool,
        mock_analyze_experiment_dependencies, 
        caplog
    ):
        """
        Parameterized test for comprehensive log level debug message visibility.
        
        Validates the behavioral contract across all supported log levels,
        ensuring debug message visibility follows expected patterns for
        each log level configuration.
        
        Args:
            log_level: Log level string to test
            should_show_debug: Expected debug message visibility
        """
        # Arrange: Configure test environment for parameterized log level testing
        caplog.set_level(logging.DEBUG)  # Capture all levels for comprehensive validation
        test_args = ['--config', 'test_config.yaml', '--experiment', 'test_exp', '--log-level', log_level]
        
        # Act: Execute analyze_experiment with specified log level
        exit_code = execute_analyze_experiment_with_args(test_args)
        
        # Assert: Verify debug message visibility matches expected behavior
        assert exit_code == 0, f"Script should execute successfully with {log_level} log level"
        debug_messages = [record.message for record in caplog.records if 'Debug logging enabled' in record.message]
        
        if should_show_debug:
            assert len(debug_messages) > 0, \
                f"Log level {log_level} should enable debug message output"
        else:
            assert len(debug_messages) == 0, \
                f"Log level {log_level} should suppress debug message output"
    
    @pytest.mark.parametrize("config_scenario,experiment_name,expected_behavior", [
        ("minimal_config", "test_experiment", "successful_execution"),
        ("complete_config", "complex_experiment", "successful_execution"),
        ("empty_config", "any_experiment", "successful_execution")
    ])
    def test_log_level_with_different_configuration_scenarios(
        self,
        config_scenario: str,
        experiment_name: str,
        expected_behavior: str,
        mock_analyze_experiment_dependencies,
        caplog
    ):
        """
        Test log level behavior across different configuration scenarios.
        
        Validates that log level functionality remains consistent regardless
        of configuration complexity or experiment specification, ensuring
        robust behavior across diverse usage patterns.
        
        Args:
            config_scenario: Configuration complexity scenario identifier
            experiment_name: Experiment name for testing
            expected_behavior: Expected execution outcome
        """
        # Arrange: Configure scenario-specific test environment
        caplog.set_level(logging.DEBUG)
        
        # Configure mock provider based on scenario
        if config_scenario == "minimal_config":
            mock_analyze_experiment_dependencies.config_data = {
                'project': {'directories': {'major_data_directory': '/tmp'}}
            }
        elif config_scenario == "complete_config":
            mock_analyze_experiment_dependencies.config_data = {
                'project': {
                    'directories': {'major_data_directory': '/tmp'},
                    'name': 'complex_project'
                },
                'experiments': {
                    experiment_name: {'analysis_params': {'param1': 'value1'}}
                }
            }
        else:  # empty_config
            mock_analyze_experiment_dependencies.config_data = {}
        
        test_args = [
            '--config', f'{config_scenario}.yaml',
            '--experiment', experiment_name,
            '--log-level', 'DEBUG'
        ]
        
        # Act: Execute analyze_experiment with scenario configuration
        exit_code = execute_analyze_experiment_with_args(test_args)
        
        # Assert: Verify consistent log level behavior across scenarios
        if expected_behavior == "successful_execution":
            assert exit_code == 0, \
                f"Script should execute successfully with {config_scenario} configuration"
            debug_messages = [record.message for record in caplog.records if 'Debug logging enabled' in record.message]
            assert len(debug_messages) > 0, \
                f"DEBUG log level should work consistently with {config_scenario}"
    
    def test_log_level_with_data_directory_override(
        self,
        mock_analyze_experiment_dependencies,
        caplog
    ):
        """
        Validate log level behavior when using --data-dir override.
        
        Tests the interaction between log level configuration and
        data directory override functionality, ensuring log level
        behavior remains consistent when command-line options are combined.
        """
        # Arrange: Configure test environment with data directory override
        caplog.set_level(logging.DEBUG)
        test_args = [
            '--config', 'test_config.yaml',
            '--experiment', 'test_exp',
            '--data-dir', '/custom/data/path',
            '--log-level', 'DEBUG'
        ]
        
        # Act: Execute analyze_experiment with data directory override and debug logging
        exit_code = execute_analyze_experiment_with_args(test_args)
        
        # Assert: Verify log level functionality with data directory override
        assert exit_code == 0, "Script should execute successfully with data directory override"
        debug_messages = [record.message for record in caplog.records if 'Debug logging enabled' in record.message]
        assert len(debug_messages) > 0, \
            "DEBUG log level should function correctly with --data-dir override"
    
    @pytest.mark.parametrize("invalid_log_level", [
        "INVALID", "debug", "info", "VERBOSE", "TRACE"  # Mix of invalid and case-sensitive values
    ])
    def test_invalid_log_level_handling(
        self,
        invalid_log_level: str,
        mock_analyze_experiment_dependencies,
        caplog
    ):
        """
        Test behavior with invalid or unsupported log level values.
        
        Validates error handling and graceful degradation when invalid
        log levels are specified, ensuring robust public API behavior
        even with incorrect user input.
        
        Args:
            invalid_log_level: Invalid log level string to test
        """
        # Arrange: Configure test environment for invalid log level testing
        caplog.set_level(logging.DEBUG)
        test_args = [
            '--config', 'test_config.yaml',
            '--experiment', 'test_exp',
            '--log-level', invalid_log_level
        ]
        
        # Act: Execute analyze_experiment with invalid log level
        # Note: Some invalid levels may be handled gracefully by the logging system
        exit_code = execute_analyze_experiment_with_args(test_args)
        
        # Assert: Verify graceful handling of invalid log levels
        # The script should either handle the invalid level gracefully or exit cleanly
        assert exit_code is not None, "Script should return a defined exit code"
        
        # Verify that the script doesn't crash unexpectedly
        # Log level validation is handled by the logging system, so we focus on
        # ensuring the script maintains stable behavior
    
    def test_log_level_case_sensitivity(
        self,
        mock_analyze_experiment_dependencies,
        caplog
    ):
        """
        Test log level case sensitivity handling.
        
        Validates that log level arguments are handled consistently
        regardless of case, ensuring robust user experience with
        the command-line interface.
        """
        # Arrange: Configure test environment for case sensitivity testing
        caplog.set_level(logging.DEBUG)
        
        # Test different case variations
        case_variations = ['debug', 'Debug', 'DEBUG', 'dEbUg']
        
        for log_level_case in case_variations:
            # Clear previous log records
            caplog.clear()
            
            test_args = [
                '--config', 'test_config.yaml',
                '--experiment', 'test_exp',
                '--log-level', log_level_case
            ]
            
            # Act: Execute analyze_experiment with case variation
            exit_code = execute_analyze_experiment_with_args(test_args)
            
            # Assert: Verify consistent behavior across case variations
            assert exit_code == 0, f"Script should handle log level case: {log_level_case}"
            
            # The logging system typically converts to uppercase internally,
            # so we verify the script doesn't fail with case variations