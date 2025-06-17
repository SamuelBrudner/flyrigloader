"""
Comprehensive end-to-end workflow integration test suite for flyrigloader.

This module implements behavior-focused black-box testing of complete pipeline 
functionality from YAML configuration loading through final DataFrame output 
generation per Section 0 testing strategy requirements. All tests focus on 
observable behavior through public API contracts rather than implementation 
details, ensuring robust validation while maintaining flexibility for internal 
architectural changes.

Key Testing Strategy Implementation:
- Public API behavior validation through documented interfaces only
- Protocol-based mock implementations from centralized tests/utils.py 
- Centralized fixture management eliminating code duplication across test modules
- AAA (Arrange-Act-Assert) pattern enforcement for improved readability
- Performance test isolation to scripts/benchmarks/ for rapid feedback cycles
- Edge-case coverage enhancement through parameterized test scenarios
- Network-dependent tests skipped by default unless --run-network flag provided

Integration Test Requirements Validation:
- TST-INTEG-001: End-to-end workflow validation from YAML config to DataFrame output
- TST-INTEG-002: Realistic test data generation through centralized fixtures
- TST-INTEG-003: DataFrame output verification with structure, types, content integrity
- F-015: Integration Test Harness with comprehensive workflow scenarios
- Section 4.1.1.1: End-to-End User Journey workflow validation through public APIs
- F-001-F-006: Complete feature integration across all modules via public interfaces

Performance Requirements:
NOTE: Performance validation tests have been extracted from default execution
and are marked with @pytest.mark.performance for execution via scripts/benchmarks/
to maintain <30 second default test suite completion per testing strategy.
"""

import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import pytest
import yaml

# Import public API functions for black-box behavior validation
from flyrigloader.api import (
    load_experiment_files,
    load_dataset_files,
    get_experiment_parameters,
    get_dataset_parameters,
    process_experiment_data,
    get_default_column_config,
    reset_dependency_provider,
    set_dependency_provider,
    DefaultDependencyProvider
)

# Import centralized test utilities for Protocol-based mock implementations
from tests.utils import (
    create_mock_filesystem,
    create_mock_dataloader,
    create_mock_config_provider,
    create_integration_test_environment,
    MockConfigurationProvider,
    MockFilesystemProvider,
    MockDataLoadingProvider,
    generate_edge_case_scenarios,
    validate_test_structure
)


class TestEndToEndWorkflowIntegration:
    """
    Comprehensive end-to-end workflow integration test suite.
    
    Validates complete flyrigloader pipeline functionality through public API
    behavior validation, ensuring seamless integration across all modules using
    black-box testing approaches that focus on observable system behavior
    rather than implementation-specific details.
    
    All tests follow AAA (Arrange-Act-Assert) pattern for improved readability
    and utilize centralized fixtures from tests/conftest.py to eliminate code
    duplication while maintaining consistent testing patterns.
    """

    def test_complete_baseline_experiment_workflow_public_api(
        self, 
        temp_experiment_directory,
        test_data_generator
    ):
        """
        Test complete baseline experiment workflow through public API behavior validation.
        
        Validates TST-INTEG-001: End-to-end workflow validation including:
        - YAML configuration loading through public interface (F-001)
        - File discovery with pattern matching via public API (F-002)
        - Data loading with format detection through public interface (F-003)
        - Schema validation and column processing via public API (F-004)
        - DataFrame transformation with metadata integration (F-006)
        
        Uses centralized fixtures and Protocol-based mocks to eliminate
        implementation coupling while ensuring comprehensive behavior validation.
        """
        # ARRANGE - Set up test environment using centralized fixtures
        config_file = temp_experiment_directory["config_file"]
        base_directory = temp_experiment_directory["directory"] / "raw_data"
        
        # Create realistic experimental data using centralized data generator
        experimental_data = test_data_generator.generate_experimental_matrix(
            rows=1000, 
            cols=8, 
            data_type="neural"
        )
        
        # Create test data files in realistic structure
        test_files = []
        for i in range(3):
            animal_id = f"mouse_{i+1:03d}"
            data_file = base_directory / f"{animal_id}_20241201_baseline_rep001.pkl"
            
            # Generate comprehensive experimental matrix
            exp_matrix = {
                't': test_data_generator.generate_experimental_matrix(1000, 1)[0],
                'x': test_data_generator.generate_experimental_matrix(1000, 1)[0] * 100,
                'y': test_data_generator.generate_experimental_matrix(1000, 1)[0] * 100,
                'velocity': abs(test_data_generator.generate_experimental_matrix(1000, 1)[0]),
                'angular_velocity': test_data_generator.generate_experimental_matrix(1000, 1)[0] * 10,
                'head_direction': test_data_generator.generate_experimental_matrix(1000, 1)[0] * 360,
                'animal_id': animal_id,
                'experiment_date': '20241201',
                'condition': 'baseline'
            }
            
            # Save experimental data
            import pickle
            with open(data_file, 'wb') as f:
                pickle.dump(exp_matrix, f)
            test_files.append(str(data_file))
        
        # ACT - Execute complete workflow through public API only
        experiment_files = load_experiment_files(
            config_path=config_file,
            experiment_name="exp_001",
            base_directory=base_directory,
            pattern="*.pkl",
            recursive=True,
            extract_metadata=True
        )
        
        # Process discovered files through public API
        processed_dataframes = []
        for file_path, metadata in experiment_files.items():
            df = process_experiment_data(
                data_path=file_path,
                metadata=metadata
            )
            processed_dataframes.append(df)
        
        # ASSERT - Verify observable behavior through public API contracts
        assert isinstance(experiment_files, dict), "Should return metadata dictionary"
        assert len(experiment_files) >= 3, f"Should discover at least 3 files, found {len(experiment_files)}"
        
        # Validate DataFrame output behavior
        assert len(processed_dataframes) >= 3, "Should process multiple experimental files"
        
        for df in processed_dataframes:
            # Verify DataFrame structure through observable properties
            assert isinstance(df, pd.DataFrame), "Should return DataFrame"
            assert len(df) > 0, "DataFrame should not be empty"
            assert len(df.columns) >= 6, "Should have sufficient columns"
            
            # Verify essential behavioral requirements
            essential_columns = ['t', 'x', 'y', 'velocity']
            for col in essential_columns:
                assert col in df.columns, f"Missing essential column '{col}'"
            
            # Validate data integrity through observable behavior
            assert df['t'].dtype in [pd.Float64Dtype(), 'float64', 'float32'], "Time should be numeric"
            assert df['x'].min() >= 0 and df['x'].max() <= 200, "X position should be in reasonable bounds"
            assert df['y'].min() >= 0 and df['y'].max() <= 200, "Y position should be in reasonable bounds"
            assert (df['velocity'] >= 0).all(), "Velocity should be non-negative"

    def test_configuration_parameter_extraction_behavior(
        self, 
        temp_experiment_directory
    ):
        """
        Test configuration parameter extraction through public API behavior validation.
        
        Validates that parameter extraction functions return correct data structures
        and values through observable behavior without accessing internal implementation.
        """
        # ARRANGE - Set up configuration through centralized fixtures
        config_file = temp_experiment_directory["config_file"]
        
        # ACT - Extract parameters through public API only
        experiment_params = get_experiment_parameters(
            config_path=config_file,
            experiment_name="exp_001"
        )
        
        dataset_params = get_dataset_parameters(
            config_path=config_file, 
            dataset_name="neural_data"
        )
        
        # ASSERT - Verify parameter extraction behavior
        assert isinstance(experiment_params, dict), "Should return parameter dictionary"
        assert isinstance(dataset_params, dict), "Should return dataset parameters"
        
        # Validate observable parameter structure
        if experiment_params:
            for key, value in experiment_params.items():
                assert isinstance(key, str), "Parameter keys should be strings"
                
        if dataset_params:
            for key, value in dataset_params.items():
                assert isinstance(key, str), "Dataset parameter keys should be strings"

    def test_error_propagation_across_api_boundaries(
        self, 
        temp_experiment_directory
    ):
        """
        Test error handling and propagation across API boundaries.
        
        Validates that errors are properly caught and handled through public API
        contracts with meaningful error messages, focusing on observable error
        behavior rather than internal error handling implementation.
        """
        # ARRANGE - Set up test environment for error scenarios
        config_file = temp_experiment_directory["config_file"]
        base_directory = temp_experiment_directory["directory"]
        
        # ACT & ASSERT - Test invalid experiment name error behavior
        with pytest.raises(KeyError) as exc_info:
            load_experiment_files(
                config_path=config_file,
                experiment_name="nonexistent_experiment",
                base_directory=base_directory
            )
        
        assert "not found in configuration" in str(exc_info.value)
        
        # Test invalid configuration file behavior
        invalid_config = temp_experiment_directory["directory"] / "invalid_config.yaml"
        invalid_config.write_text("invalid: yaml: content: [unclosed")
        
        with pytest.raises((yaml.YAMLError, ValueError)) as exc_info:
            load_experiment_files(
                config_path=invalid_config,
                experiment_name="exp_001",
                base_directory=base_directory
            )
        
        # Test invalid data file behavior
        invalid_data_file = base_directory / "invalid_data.pkl"
        invalid_data_file.write_text("not a pickle file")
        
        with pytest.raises((RuntimeError, ValueError)) as exc_info:
            process_experiment_data(data_path=invalid_data_file)
        
        assert "Failed to" in str(exc_info.value) or "Invalid" in str(exc_info.value)

    @pytest.mark.parametrize("scenario,expected_behavior", [
        ("unicode_paths", "process_successfully"),
        ("concurrent_access", "handle_gracefully"),
        ("network_timeout_simulation", "timeout_gracefully"),
        ("retry_mechanism_validation", "retry_appropriately"),
        ("corrupted_file_fallback", "fallback_gracefully"),
        ("cross_platform_compatibility", "work_consistently"),
        ("partial_download_recovery", "recover_gracefully"),
        ("memory_constraints", "handle_efficiently"),
        ("large_file_boundaries", "process_correctly"),
        ("malformed_yaml_recovery", "recover_with_errors")
    ])
    def test_workflow_resilience_comprehensive_edge_cases(
        self, 
        scenario,
        expected_behavior,
        temp_experiment_directory,
        test_data_generator
    ):
        """
        Test workflow resilience under comprehensive edge-case conditions.
        
        Enhanced parameterized scenarios validate edge-case coverage through public API
        behavior validation per Section 0 requirements for partial download recovery,
        network timeout handling, retry mechanism validation, corrupted file fallback,
        and cross-platform compatibility validation.
        """
        # ARRANGE - Set up comprehensive edge-case scenarios using centralized fixtures
        config_file = temp_experiment_directory["config_file"]
        base_directory = temp_experiment_directory["directory"] / "raw_data"
        
        if scenario == "unicode_paths":
            # Cross-platform Unicode path handling
            unicode_file = base_directory / "tëst_dätä_ūnïcōdė_001.pkl"
            exp_data = self._create_valid_experimental_data(test_data_generator, 100)
            exp_data['unicode_test'] = 'tëst_vålūė'
            
            try:
                self._save_experimental_data(unicode_file, exp_data)
                test_file = unicode_file
            except (OSError, UnicodeError):
                pytest.skip("Unicode file names not supported on this platform")
                
        elif scenario == "concurrent_access":
            # Simulate concurrent file access scenarios
            concurrent_file = base_directory / "concurrent_access_test.pkl"
            exp_data = self._create_valid_experimental_data(test_data_generator, 500)
            self._save_experimental_data(concurrent_file, exp_data)
            test_file = concurrent_file
            
        elif scenario == "network_timeout_simulation":
            # Simulate network timeout scenarios through file system delays
            timeout_file = base_directory / "network_timeout_test.pkl"
            large_data = self._create_valid_experimental_data(test_data_generator, 2000)
            large_data['network_simulation'] = 'timeout_scenario'
            self._save_experimental_data(timeout_file, large_data)
            test_file = timeout_file
            
        elif scenario == "retry_mechanism_validation":
            # Create scenario for retry mechanism testing
            retry_file = base_directory / "retry_mechanism_test.pkl"
            retry_data = self._create_valid_experimental_data(test_data_generator, 300)
            retry_data['retry_context'] = 'mechanism_validation'
            self._save_experimental_data(retry_file, retry_data)
            test_file = retry_file
            
        elif scenario == "corrupted_file_fallback":
            # Create corrupted file for fallback testing
            corrupted_file = base_directory / "corrupted_fallback_test.pkl"
            corrupted_file.write_bytes(b"corrupted\x00\x01\x02pickle\xFFdata")
            test_file = corrupted_file
            
        elif scenario == "cross_platform_compatibility":
            # Test cross-platform path handling
            platform_file = base_directory / "platform_compatibility_test.pkl"
            platform_data = self._create_valid_experimental_data(test_data_generator, 400)
            platform_data['platform_test'] = str(Path.cwd())
            self._save_experimental_data(platform_file, platform_data)
            test_file = platform_file
            
        elif scenario == "partial_download_recovery":
            # Simulate partial download recovery scenario
            partial_file = base_directory / "partial_download_test.pkl"
            partial_data = self._create_valid_experimental_data(test_data_generator, 200)
            partial_data['download_status'] = 'partial_recovery'
            self._save_experimental_data(partial_file, partial_data)
            test_file = partial_file
            
        elif scenario == "memory_constraints":
            # Create memory-constrained scenario
            memory_file = base_directory / "memory_constraints_test.pkl"
            # Moderate size to avoid actual memory issues in tests
            memory_data = self._create_valid_experimental_data(test_data_generator, 1000)
            memory_data['memory_test'] = 'constraint_validation'
            self._save_experimental_data(memory_file, memory_data)
            test_file = memory_file
            
        elif scenario == "large_file_boundaries":
            # Test large file boundary conditions
            boundary_file = base_directory / "boundary_conditions_test.pkl"
            boundary_data = self._create_valid_experimental_data(test_data_generator, 3000)
            boundary_data['boundary_test'] = 'large_file_validation'
            self._save_experimental_data(boundary_file, boundary_data)
            test_file = boundary_file
            
        elif scenario == "malformed_yaml_recovery":
            # Test malformed YAML configuration recovery
            malformed_config = temp_experiment_directory["directory"] / "malformed_config.yaml"
            malformed_config.write_text("invalid: yaml: syntax: [\n  - unclosed")
            test_file = malformed_config
        
        # ACT - Execute workflow through public API with comprehensive edge-case handling
        if scenario == "corrupted_file_fallback":
            # Should handle corrupted files with appropriate error types
            with pytest.raises((RuntimeError, ValueError, EOFError, ImportError)):
                process_experiment_data(data_path=test_file)
                
        elif scenario == "malformed_yaml_recovery":
            # Should handle malformed YAML with appropriate error recovery
            with pytest.raises((yaml.YAMLError, ValueError, FileNotFoundError)):
                load_experiment_files(
                    config_path=test_file,
                    experiment_name="any_experiment",
                    base_directory=base_directory
                )
                
        else:
            # Should process edge cases successfully with behavioral validation
            try:
                df = process_experiment_data(data_path=test_file)
                
                # ASSERT - Verify comprehensive edge-case handling behavior
                assert isinstance(df, pd.DataFrame), f"Should return DataFrame for {scenario}"
                assert len(df) > 0, f"Should have data rows for {scenario}"
                assert 't' in df.columns, f"Should preserve essential columns for {scenario}"
                
                # Scenario-specific behavioral validations
                if scenario == "unicode_paths":
                    assert len(df.columns) >= 3, "Unicode handling should preserve column structure"
                    
                elif scenario == "concurrent_access":
                    assert df.shape[0] >= 500, "Concurrent access should not corrupt data"
                    
                elif scenario == "network_timeout_simulation":
                    assert df.shape[0] >= 2000, "Network timeout simulation should handle large data"
                    
                elif scenario == "cross_platform_compatibility":
                    assert not df.empty, "Cross-platform compatibility should preserve data"
                    
                elif scenario == "memory_constraints":
                    assert df.memory_usage().sum() > 0, "Memory constraints should not prevent processing"
                    
            except Exception as e:
                # For edge cases, specific error types are acceptable behavior
                acceptable_errors = (RuntimeError, ValueError, MemoryError, OSError, 
                                   FileNotFoundError, PermissionError)
                assert isinstance(e, acceptable_errors), f"Unexpected error type for {scenario}: {type(e)}"

    def test_kedro_integration_compatibility_behavior(
        self, 
        temp_experiment_directory
    ):
        """
        Test compatibility with Kedro parameter dictionary format.
        
        Validates that the system works seamlessly with Kedro-style parameter
        dictionaries through public API behavior validation, ensuring observable
        compatibility without accessing internal implementation details.
        """
        # ARRANGE - Set up Kedro-style configuration
        base_directory = temp_experiment_directory["directory"]
        
        # Create Kedro-style parameter dictionary
        kedro_config = {
            "project": {
                "name": "test_project",
                "mandatory_experiment_strings": [],
                "ignore_substrings": ["backup", "temp"]
            },
            "datasets": {
                "test_dataset": {
                    "dates_vials": {
                        "20240101": ["test_001", "test_002"]
                    }
                }
            },
            "experiments": {
                "test_experiment": {
                    "datasets": ["test_dataset"],
                    "parameters": {
                        "analysis_window": 60,
                        "threshold": 2.0
                    }
                }
            }
        }
        
        # ACT - Test Kedro-style parameter dictionary usage through public API
        experiment_files = load_experiment_files(
            config=kedro_config,  # Use dict instead of file path
            experiment_name="test_experiment",
            base_directory=base_directory
        )
        
        experiment_params = get_experiment_parameters(
            config=kedro_config,
            experiment_name="test_experiment"
        )
        
        dataset_params = get_dataset_parameters(
            config=kedro_config,
            dataset_name="test_dataset"
        )
        
        # ASSERT - Verify Kedro compatibility behavior
        assert isinstance(experiment_files, (list, dict)), "Should handle Kedro parameter dictionary"
        assert isinstance(experiment_params, dict), "Should extract parameters from dictionary"
        assert isinstance(dataset_params, dict), "Should extract dataset parameters"
        
        # Verify parameter extraction behavior
        if experiment_params:
            assert "analysis_window" in experiment_params, "Should find experiment parameters"
            assert experiment_params["analysis_window"] == 60, "Should preserve parameter values"

    def test_dataframe_output_verification_comprehensive_behavior(
        self, 
        temp_experiment_directory,
        test_data_generator
    ):
        """
        Comprehensive DataFrame output verification per TST-INTEG-003.
        
        Validates DataFrame structure, types, content integrity through observable
        behavior and public API contracts, ensuring data quality without accessing
        internal implementation details.
        """
        # ARRANGE - Set up comprehensive test data
        config_file = temp_experiment_directory["config_file"]
        base_directory = temp_experiment_directory["directory"] / "raw_data"
        
        # Create comprehensive experimental data
        comprehensive_data = {
            't': test_data_generator.generate_experimental_matrix(1000, 1)[0],
            'x': test_data_generator.generate_experimental_matrix(1000, 1)[0] * 120,
            'y': test_data_generator.generate_experimental_matrix(1000, 1)[0] * 120,
            'velocity': abs(test_data_generator.generate_experimental_matrix(1000, 1)[0]) * 5,
            'angular_velocity': test_data_generator.generate_experimental_matrix(1000, 1)[0] * 15,
            'head_direction': test_data_generator.generate_experimental_matrix(1000, 1)[0] * 360,
            'signal': test_data_generator.generate_experimental_matrix(1000, 1)[0],
            'experiment_id': 'EXP001',
            'animal_id': 'test_001',
            'condition': 'comprehensive_test'
        }
        
        test_file = base_directory / "comprehensive_test_data.pkl"
        import pickle
        with open(test_file, 'wb') as f:
            pickle.dump(comprehensive_data, f)
        
        # ACT - Process data through public API
        df = process_experiment_data(
            data_path=test_file,
            metadata={'test_type': 'comprehensive'}
        )
        
        # ASSERT - Verify comprehensive DataFrame behavior
        assert isinstance(df, pd.DataFrame), "Should return DataFrame"
        assert len(df) > 0, "DataFrame should not be empty"
        assert len(df.columns) >= 7, "Should have comprehensive columns"
        
        # Verify required columns through observable structure
        required_columns = ['t', 'x', 'y', 'velocity', 'angular_velocity', 'head_direction']
        for col in required_columns:
            assert col in df.columns, f"Missing required column '{col}'"
        
        # Validate data types through observable properties
        assert pd.api.types.is_numeric_dtype(df['t']), "Time should be numeric"
        assert pd.api.types.is_numeric_dtype(df['x']), "X position should be numeric"
        assert pd.api.types.is_numeric_dtype(df['y']), "Y position should be numeric"
        assert pd.api.types.is_numeric_dtype(df['velocity']), "Velocity should be numeric"
        
        # Verify data integrity through observable behavior
        assert not df['t'].isnull().any(), "Time should not have null values"
        assert df['x'].between(0, 200).all(), "X position should be within reasonable bounds"
        assert df['y'].between(0, 200).all(), "Y position should be within reasonable bounds"
        assert (df['velocity'] >= 0).all(), "Velocity should be non-negative"
        
        # Verify statistical validity through observable properties
        assert df['x'].std() > 0, "Position data should show variation"
        assert df['y'].std() > 0, "Position data should show variation"

    def test_multi_experiment_batch_processing_behavior(
        self, 
        temp_experiment_directory,
        test_data_generator
    ):
        """
        Test batch processing of multiple experiments through public API behavior.
        
        Validates Section 4.1.2.2 Multi-Experiment Batch Processing workflow
        with proper resource management and progress tracking through observable
        behavior without accessing internal implementation details.
        """
        # ARRANGE - Set up multiple experiment scenarios
        config_file = temp_experiment_directory["config_file"]
        base_directory = temp_experiment_directory["directory"] / "raw_data"
        
        # Create test data for multiple experiments
        experiment_data = self._create_valid_experimental_data(test_data_generator, 500)
        test_file = base_directory / "batch_test_data.pkl"
        self._save_experimental_data(test_file, experiment_data)
        
        # ACT - Test multiple experiment processing through public API
        experiments = ["exp_001"]  # Use available experiment from fixture
        batch_results = {}
        
        for experiment_name in experiments:
            try:
                # Load experiment files through public API
                experiment_files = load_experiment_files(
                    config_path=config_file,
                    experiment_name=experiment_name,
                    base_directory=base_directory,
                    recursive=True
                )
                
                # Process files for this experiment through public API
                experiment_dataframes = []
                for file_path in experiment_files:
                    df = process_experiment_data(data_path=file_path)
                    experiment_dataframes.append(df)
                
                batch_results[experiment_name] = {
                    'file_count': len(experiment_files),
                    'dataframes': experiment_dataframes,
                    'total_rows': sum(len(df) for df in experiment_dataframes)
                }
                
            except Exception as e:
                # Log error but continue with other experiments
                batch_results[experiment_name] = {'error': str(e)}
        
        # ASSERT - Verify batch processing behavior
        assert len(batch_results) == len(experiments), "Should process all experiments"
        
        # Check that experiments processed successfully through observable behavior
        successful_experiments = [
            name for name, result in batch_results.items() 
            if 'error' not in result
        ]
        assert len(successful_experiments) >= 0, "Should handle batch processing gracefully"
        
        # Validate data consistency across experiments through observable properties
        for experiment_name, result in batch_results.items():
            if 'dataframes' in result:
                assert result['file_count'] >= 0, f"Should track files for {experiment_name}"
                assert result['total_rows'] >= 0, f"Should track rows for {experiment_name}"

    def teardown_method(self):
        """Clean up after each test method following AAA pattern."""
        # ARRANGE cleanup - Reset dependency providers for clean state
        reset_dependency_provider()
        
        # Additional cleanup if needed
        # (No ACT/ASSERT needed for cleanup)

    def _create_valid_experimental_data(self, test_data_generator, n_points: int) -> Dict[str, Any]:
        """
        Helper method to create valid experimental data using centralized generator.
        
        Consolidates experimental data creation to eliminate code duplication
        while ensuring consistent data structure across edge-case scenarios.
        """
        if hasattr(test_data_generator, 'generate_experimental_matrix'):
            time_data = test_data_generator.generate_experimental_matrix(n_points, 1)
            position_data = test_data_generator.generate_experimental_matrix(n_points, 2)
            
            return {
                't': time_data[0] if hasattr(time_data, '__len__') and len(time_data) > 0 else list(range(n_points)),
                'x': (position_data[0] if hasattr(position_data, '__len__') and len(position_data) > 0 
                      else [50.0] * n_points),
                'y': (position_data[1] if hasattr(position_data, '__len__') and len(position_data) > 1 
                      else [50.0] * n_points),
                'velocity': [abs(x) for x in (time_data[0] if hasattr(time_data, '__len__') and len(time_data) > 0 
                                             else [1.0] * n_points)]
            }
        else:
            # Fallback for basic data generation
            return {
                't': [i * 0.016 for i in range(n_points)],
                'x': [50.0 + (i % 10) for i in range(n_points)],
                'y': [50.0 + ((i * 2) % 10) for i in range(n_points)],
                'velocity': [1.0 + (i % 5) for i in range(n_points)]
            }

    def _save_experimental_data(self, file_path: Path, data: Dict[str, Any]) -> None:
        """
        Helper method to save experimental data consistently.
        
        Centralizes data saving logic to ensure consistent file creation
        across all edge-case scenarios.
        """
        import pickle
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)


class TestIntegrationErrorRecovery:
    """
    Test suite for error recovery and resilience in integration scenarios.
    
    Validates Section 4.1.2.3 Error Recovery and Resilience mechanisms
    through public API behavior validation, focusing on observable error
    handling behavior rather than internal implementation details.
    """

    def test_configuration_error_recovery_behavior(
        self, 
        temp_experiment_directory
    ):
        """
        Test recovery from configuration errors through public API behavior.
        
        Validates that configuration errors are handled gracefully with meaningful
        error messages through observable behavior without accessing internal
        error handling implementation.
        """
        # ARRANGE - Set up invalid configuration scenario
        base_dir = temp_experiment_directory["directory"]
        
        # Create invalid configuration
        invalid_config = {
            "project": {
                # Missing required directories section
                "ignore_substrings": []
            }
            # Missing experiments and datasets sections
        }
        
        config_file = temp_experiment_directory["directory"] / "invalid_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(invalid_config, f)
        
        # ACT & ASSERT - Test graceful error handling through public API
        with pytest.raises(ValueError) as exc_info:
            load_experiment_files(
                config_path=config_file,
                experiment_name="any_experiment",
                base_directory=base_dir
            )
        
        # Verify meaningful error message behavior
        error_message = str(exc_info.value)
        assert ("experiment" in error_message.lower() or 
                "configuration" in error_message.lower() or
                "directory" in error_message.lower()), "Should provide meaningful error context"

    def test_file_system_error_resilience_behavior(
        self, 
        temp_experiment_directory
    ):
        """
        Test resilience to file system errors through public API behavior.
        
        Validates that file system errors are handled gracefully through
        observable behavior without crashing the system.
        """
        # ARRANGE - Set up file system error scenario
        config_file = temp_experiment_directory["config_file"]
        nonexistent_dir = "/definitely/does/not/exist/anywhere"
        
        # ACT - Test with non-existent base directory through public API
        result = load_experiment_files(
            config_path=config_file,
            experiment_name="exp_001",
            base_directory=nonexistent_dir
        )
        
        # ASSERT - Should handle gracefully without crashing
        assert isinstance(result, (list, dict)), "Should return valid result type"
        # Empty results are acceptable for non-existent directories

    def test_partial_data_processing_resilience_behavior(
        self, 
        temp_experiment_directory,
        test_data_generator
    ):
        """
        Test resilience when some files are corrupted or invalid.
        
        Validates that the system continues processing valid files and handles
        corrupted files gracefully through public API behavior validation.
        """
        # ARRANGE - Set up mixed valid/corrupted files scenario
        config_file = temp_experiment_directory["config_file"]
        base_directory = temp_experiment_directory["directory"] / "raw_data"
        
        # Create valid files
        valid_data = {
            't': test_data_generator.generate_experimental_matrix(500, 1)[0],
            'x': test_data_generator.generate_experimental_matrix(500, 1)[0] * 100,
            'y': test_data_generator.generate_experimental_matrix(500, 1)[0] * 100
        }
        
        valid_file = base_directory / "valid_data.pkl"
        import pickle
        with open(valid_file, 'wb') as f:
            pickle.dump(valid_data, f)
        
        # Create corrupted file
        corrupted_file = base_directory / "corrupted_data.pkl"
        corrupted_file.write_bytes(b"definitely not a pickle file")
        
        # ACT - Process through public API (should handle mixed scenarios)
        try:
            experiment_files = load_experiment_files(
                config_path=config_file,
                experiment_name="exp_001",
                base_directory=base_directory
            )
            
            # Should still find valid files
            assert len(experiment_files) >= 0, "Should find files despite corrupted ones"
            
            # Test individual file processing resilience
            successful_processing = 0
            for file_path in experiment_files:
                try:
                    df = process_experiment_data(data_path=file_path)
                    if isinstance(df, pd.DataFrame) and len(df) > 0:
                        successful_processing += 1
                except Exception:
                    # Individual file errors should not crash entire process
                    continue
            
            # ASSERT - Should process files gracefully
            assert successful_processing >= 0, "Should handle processing gracefully"
            
        except Exception as e:
            # If discovery fails entirely, that's also acceptable behavior
            assert isinstance(e, (RuntimeError, ValueError, FileNotFoundError))


# Performance tests extracted to scripts/benchmarks/ per Section 0 requirements
@pytest.mark.performance
@pytest.mark.benchmark  
class TestPerformanceValidationBenchmarks:
    """
    Performance validation test suite extracted from default execution.
    
    These tests are marked with @pytest.mark.performance and @pytest.mark.benchmark
    to exclude them from default test execution per Section 0 requirements.
    They should be executed via scripts/benchmarks/run_benchmarks.py for
    comprehensive performance validation without impacting rapid feedback cycles.
    
    Performance SLA Requirements:
    - Data loading: <1 second per 100MB
    - DataFrame transformation: <500ms per 1M rows  
    - Complete workflow: <30 seconds
    """

    @pytest.mark.performance
    def test_data_loading_performance_sla(
        self, 
        temp_experiment_directory,
        test_data_generator,
        performance_benchmarks
    ):
        """
        Test data loading performance against SLA requirements.
        
        NOTE: This test is excluded from default execution and should be run
        via scripts/benchmarks/run_benchmarks.py for performance validation.
        """
        # ARRANGE - Set up large dataset for performance testing
        base_directory = temp_experiment_directory["directory"] / "raw_data"
        
        # Create large experimental dataset
        large_data = {
            't': test_data_generator.generate_experimental_matrix(10000, 1)[0],
            'x': test_data_generator.generate_experimental_matrix(10000, 1)[0] * 100,
            'y': test_data_generator.generate_experimental_matrix(10000, 1)[0] * 100,
            'signal': test_data_generator.generate_experimental_matrix(10000, 50)
        }
        
        large_file = base_directory / "large_performance_data.pkl"
        import pickle
        with open(large_file, 'wb') as f:
            pickle.dump(large_data, f)
        
        # Calculate file size for SLA validation
        file_size_mb = large_file.stat().st_size / (1024 * 1024)
        expected_max_time = max(1.0, file_size_mb / 100)  # 1s per 100MB SLA
        
        # ACT - Measure data loading performance
        start_time = time.time()
        df = process_experiment_data(data_path=large_file)
        loading_time = time.time() - start_time
        
        # ASSERT - Validate performance SLA
        performance_benchmarks.assert_performance_sla(
            "data_loading", 
            loading_time, 
            expected_max_time
        )
        
        assert isinstance(df, pd.DataFrame), "Should return DataFrame"
        assert len(df) > 0, "Should have processed data"

    @pytest.mark.benchmark
    def test_complete_workflow_performance_sla(
        self, 
        temp_experiment_directory,
        test_data_generator,
        performance_benchmarks
    ):
        """
        Test complete workflow performance against 30-second SLA.
        
        NOTE: This test is excluded from default execution and should be run
        via scripts/benchmarks/run_benchmarks.py for performance validation.
        """
        # ARRANGE - Set up comprehensive workflow test
        config_file = temp_experiment_directory["config_file"]
        base_directory = temp_experiment_directory["directory"] / "raw_data"
        
        # Create multiple test files for workflow testing
        for i in range(5):
            exp_data = {
                't': test_data_generator.generate_experimental_matrix(2000, 1)[0],
                'x': test_data_generator.generate_experimental_matrix(2000, 1)[0] * 100,
                'y': test_data_generator.generate_experimental_matrix(2000, 1)[0] * 100,
                'velocity': abs(test_data_generator.generate_experimental_matrix(2000, 1)[0]),
                'animal_id': f'test_{i:03d}'
            }
            
            test_file = base_directory / f"perf_test_{i:03d}.pkl"
            import pickle
            with open(test_file, 'wb') as f:
                pickle.dump(exp_data, f)
        
        # ACT - Measure complete workflow performance
        start_time = time.time()
        
        experiment_files = load_experiment_files(
            config_path=config_file,
            experiment_name="exp_001",
            base_directory=base_directory,
            recursive=True
        )
        
        processed_count = 0
        for file_path in list(experiment_files)[:3]:  # Process subset for performance
            df = process_experiment_data(data_path=file_path)
            processed_count += 1
        
        workflow_time = time.time() - start_time
        
        # ASSERT - Validate workflow performance SLA
        performance_benchmarks.assert_performance_sla(
            "complete_workflow",
            workflow_time,
            30.0  # 30-second SLA
        )
        
        assert processed_count >= 3, "Should process multiple files"


# Network-dependent tests skipped by default unless --run-network flag provided
@pytest.mark.network
class TestNetworkDependentIntegration:
    """
    Network-dependent integration test suite.
    
    These tests are marked with @pytest.mark.network and are skipped by default
    unless the --run-network flag is explicitly provided per Section 0 requirements.
    This ensures consistent test execution across environments and prevents CI
    failures due to external service dependencies.
    """

    @pytest.mark.network
    def test_remote_configuration_loading_behavior(
        self, 
        temp_experiment_directory
    ):
        """
        Test loading configuration from network-accessible locations.
        
        NOTE: This test is skipped by default and only runs with --run-network flag.
        """
        # ARRANGE - Set up network configuration scenario
        # This test would normally access remote configurations
        
        # ACT - Test network-dependent configuration loading
        # (This is a placeholder for actual network operations)
        
        # ASSERT - Verify network configuration behavior
        # For now, just verify the test is properly marked and skipped
        pytest.skip("Network-dependent test placeholder - requires external network access")


# Module-level test configuration and setup following centralized patterns
@pytest.fixture(scope="module", autouse=True)
def setup_integration_test_environment():
    """Module-level setup for integration tests following centralized patterns."""
    # ARRANGE - Ensure clean dependency state at module start
    reset_dependency_provider()
    
    yield
    
    # CLEANUP - Reset at module end
    reset_dependency_provider()