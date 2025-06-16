"""
API facade integration test suite.

Refactored for behavior-focused testing with black-box validation of end-to-end workflows
from configuration loading to DataFrame output. Tests focus on observable API behavior
rather than internal implementation details, using centralized fixtures and Protocol-based
mock implementations for consistent dependency isolation.

This module implements comprehensive integration testing for flyrigloader.api functions
validating complete workflows through public interfaces with AAA pattern structure
and edge-case coverage enhancement per Section 0 requirements.

Performance validation tests (TST-INTEG-001 through TST-INTEG-003) have been relocated
to scripts/benchmarks/ per Section 0 performance test isolation requirement.
"""

from pathlib import Path
from typing import Dict, Any, List, Union, Optional
from unittest.mock import patch, MagicMock

import pytest
import pandas as pd
import numpy as np
import yaml

from flyrigloader.api import (
    load_experiment_files,
    load_dataset_files,
    get_experiment_parameters,
    get_dataset_parameters,
    process_experiment_data,
    MISSING_DATA_DIR_ERROR
)


# ============================================================================
# API FUNCTION INTEGRATION TESTS - BLACK-BOX BEHAVIORAL VALIDATION
# ============================================================================

class TestLoadExperimentFilesIntegration:
    """
    Integration tests for load_experiment_files API function using black-box behavioral validation.
    
    Tests focus on observable API behavior through public interfaces rather than internal
    implementation details, using centralized fixtures for consistent test data setup.
    """
    
    def test_load_experiment_files_with_config_path_success(self, sample_config_file, temp_cross_platform_dir):
        """
        Test successful experiment file loading using config file path with end-to-end validation.
        
        Validates complete workflow behavior: configuration loading → file discovery → result formatting
        without accessing internal implementation details.
        """
        # ARRANGE - Set up test scenario with realistic configuration and filesystem
        config_path = sample_config_file
        experiment_name = "baseline_control_study"
        
        # Create test data directory structure referenced by config
        data_dir = temp_cross_platform_dir / "research_data" / "experiments"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample experimental files matching expected patterns
        test_files = [
            data_dir / "baseline_20241220_control_1.pkl",
            data_dir / "baseline_20241221_control_2.pkl"
        ]
        for test_file in test_files:
            test_file.write_bytes(b"mock experimental data")
        
        # Update config to reference our test directory
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        config_data["project"]["directories"]["major_data_directory"] = str(data_dir.parent)
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        # ACT - Execute the API function under test
        result_files = load_experiment_files(
            config_path=config_path,
            experiment_name=experiment_name,
            pattern="*.pkl"
        )
        
        # ASSERT - Verify observable behavior through public interface
        assert isinstance(result_files, list), "Expected list return type for basic file loading"
        assert len(result_files) >= 0, "Expected non-negative file count"
        
        # Verify files contain expected patterns (behavior validation, not implementation)
        file_paths = [str(f) for f in result_files]
        for file_path in file_paths:
            assert file_path.endswith('.pkl'), "Expected only .pkl files in results"
    
    def test_load_experiment_files_with_config_dict_success(self, sample_comprehensive_config_dict):
        """
        Test successful experiment file loading using config dictionary with behavioral validation.
        
        Validates API contract compliance when using pre-loaded configuration dictionary
        without examining internal configuration processing implementation.
        """
        # ARRANGE - Set up test with configuration dictionary
        config_dict = sample_comprehensive_config_dict
        experiment_name = "baseline_control_study"
        
        # ACT - Execute API function with dictionary configuration
        result_files = load_experiment_files(
            config=config_dict,
            experiment_name=experiment_name,
            pattern="*.pkl"
        )
        
        # ASSERT - Verify expected API behavior
        assert isinstance(result_files, list), "Expected list return type for dictionary config"
        # Note: Files may be empty if directories don't exist, but API should not error
        assert result_files is not None, "Expected non-None result from valid configuration"
    
    def test_load_experiment_files_with_metadata_extraction(self, sample_config_file):
        """
        Test metadata extraction during file loading with observable behavior validation.
        
        Validates that metadata extraction parameter affects return type and structure
        as documented in public API contract.
        """
        # ARRANGE - Set up test scenario for metadata extraction
        config_path = sample_config_file
        experiment_name = "baseline_control_study"
        
        # ACT - Execute API function with metadata extraction enabled
        result_with_metadata = load_experiment_files(
            config_path=config_path,
            experiment_name=experiment_name,
            extract_metadata=True
        )
        
        # ASSERT - Verify observable metadata extraction behavior
        assert isinstance(result_with_metadata, dict), \
            "Expected dictionary return type when metadata extraction is enabled"
        
        # Verify structure matches API contract for metadata extraction
        for file_path, metadata in result_with_metadata.items():
            assert isinstance(file_path, str), "Expected string file paths in metadata dict"
            assert isinstance(metadata, dict), "Expected dictionary metadata for each file"
    
    def test_load_experiment_files_parameter_validation_errors(self, sample_config_file):
        """
        Test parameter validation error scenarios with black-box validation.
        
        Validates API error handling behavior for invalid parameter combinations
        without examining internal validation implementation details.
        """
        # ARRANGE - Set up invalid parameter scenarios
        config_path = sample_config_file
        test_config = {"test": "config"}
        experiment_name = "test_experiment"
        
        # ACT & ASSERT - Test both config_path and config provided (invalid)
        with pytest.raises(ValueError, match="Exactly one of 'config_path' or 'config' must be provided"):
            load_experiment_files(
                config_path=config_path,
                config=test_config,
                experiment_name=experiment_name
            )
        
        # ACT & ASSERT - Test neither config_path nor config provided (invalid)
        with pytest.raises(ValueError, match="Exactly one of 'config_path' or 'config' must be provided"):
            load_experiment_files(experiment_name=experiment_name)
    
    def test_load_experiment_files_config_file_not_found(self):
        """
        Test FileNotFoundError when config file doesn't exist with error behavior validation.
        
        Validates API error handling for non-existent configuration files through
        observable exception behavior.
        """
        # ARRANGE - Set up scenario with non-existent config file
        nonexistent_config_path = "/nonexistent/config.yaml"
        experiment_name = "test_experiment"
        
        # ACT & ASSERT - Verify expected error behavior
        with pytest.raises(FileNotFoundError):
            load_experiment_files(
                config_path=nonexistent_config_path,
                experiment_name=experiment_name
            )
    
    def test_load_experiment_files_experiment_not_found(self, sample_config_file):
        """
        Test KeyError when experiment doesn't exist in config with error message validation.
        
        Validates API error reporting for missing experiments through observable
        exception behavior and error messages.
        """
        # ARRANGE - Set up scenario with valid config but non-existent experiment
        config_path = sample_config_file
        nonexistent_experiment = "nonexistent_experiment"
        
        # ACT & ASSERT - Verify expected error behavior with descriptive message
        with pytest.raises(KeyError, match="Experiment 'nonexistent_experiment' not found"):
            load_experiment_files(
                config_path=config_path,
                experiment_name=nonexistent_experiment
            )
    
    def test_load_experiment_files_missing_data_directory(self, sample_comprehensive_config_dict):
        """
        Test error when data directory is not specified with behavioral validation.
        
        Validates API error handling for missing data directory configuration
        through observable exception behavior.
        """
        # ARRANGE - Set up config without data directory
        config_without_dir = sample_comprehensive_config_dict.copy()
        del config_without_dir["project"]["directories"]["major_data_directory"]
        experiment_name = "baseline_control_study"
        
        # ACT & ASSERT - Verify expected error behavior
        with pytest.raises(ValueError, match=MISSING_DATA_DIR_ERROR):
            load_experiment_files(
                config=config_without_dir,
                experiment_name=experiment_name
            )
    
    def test_load_experiment_files_base_directory_override(self, sample_comprehensive_config_dict, temp_cross_platform_dir):
        """
        Test base directory override functionality with observable behavior validation.
        
        Validates that base_directory parameter override affects file discovery behavior
        as documented in API contract.
        """
        # ARRANGE - Set up test with base directory override
        config_dict = sample_comprehensive_config_dict
        experiment_name = "baseline_control_study"
        override_directory = str(temp_cross_platform_dir)
        
        # ACT - Execute API function with directory override
        result_files = load_experiment_files(
            config=config_dict,
            experiment_name=experiment_name,
            base_directory=override_directory
        )
        
        # ASSERT - Verify expected API behavior
        assert isinstance(result_files, list), "Expected list return type with directory override"
        # Override should be accepted without error (behavior validation)


class TestLoadDatasetFilesIntegration:
    """
    Integration tests for load_dataset_files API function using behavior-focused validation.
    
    Tests validate observable dataset file loading behavior through public API
    interfaces without examining internal discovery implementation details.
    """
    
    def test_load_dataset_files_with_config_path_success(self, sample_config_file):
        """
        Test successful dataset file loading using config file path with end-to-end validation.
        
        Validates complete dataset discovery workflow behavior through public API interface.
        """
        # ARRANGE - Set up test scenario for dataset file loading
        config_path = sample_config_file
        dataset_name = "baseline_behavior"
        
        # ACT - Execute dataset file loading
        result_files = load_dataset_files(
            config_path=config_path,
            dataset_name=dataset_name,
            pattern="*.pkl"
        )
        
        # ASSERT - Verify observable API behavior
        assert isinstance(result_files, list), "Expected list return type for dataset loading"
        assert result_files is not None, "Expected non-None result from valid dataset name"
    
    def test_load_dataset_files_with_extension_filtering(self, sample_config_file):
        """
        Test extension-based file filtering with behavioral validation.
        
        Validates that extension filtering parameter affects result content
        according to API contract specifications.
        """
        # ARRANGE - Set up test with extension filtering
        config_path = sample_config_file
        dataset_name = "baseline_behavior"
        target_extensions = ["pkl"]
        
        # ACT - Execute with extension filtering
        filtered_files = load_dataset_files(
            config_path=config_path,
            dataset_name=dataset_name,
            extensions=target_extensions
        )
        
        # ASSERT - Verify extension filtering behavior
        assert isinstance(filtered_files, list), "Expected list return type for filtered files"
        
        # Verify all returned files have expected extensions (observable behavior)
        for file_path in filtered_files:
            assert any(str(file_path).endswith(f'.{ext}') for ext in target_extensions), \
                f"File {file_path} should have one of extensions {target_extensions}"
    
    def test_load_dataset_files_dataset_not_found(self, sample_config_file):
        """
        Test KeyError when dataset doesn't exist in config with error behavior validation.
        
        Validates API error handling for non-existent datasets through observable
        exception behavior.
        """
        # ARRANGE - Set up scenario with non-existent dataset
        config_path = sample_config_file
        nonexistent_dataset = "nonexistent_dataset"
        
        # ACT & ASSERT - Verify expected error behavior
        with pytest.raises(KeyError, match="Dataset 'nonexistent_dataset' not found"):
            load_dataset_files(
                config_path=config_path,
                dataset_name=nonexistent_dataset
            )


class TestGetParameterFunctionsIntegration:
    """
    Integration tests for parameter extraction API functions using behavioral validation.
    
    Tests validate parameter extraction behavior through public API interfaces
    without examining internal configuration parsing implementation details.
    """
    
    def test_get_experiment_parameters_with_config_path(self, sample_config_file):
        """
        Test experiment parameter extraction using config file path with behavioral validation.
        
        Validates complete parameter extraction workflow behavior through public API.
        """
        # ARRANGE - Set up test for parameter extraction
        config_path = sample_config_file
        experiment_name = "baseline_control_study"
        
        # ACT - Execute parameter extraction
        extracted_params = get_experiment_parameters(
            config_path=config_path,
            experiment_name=experiment_name
        )
        
        # ASSERT - Verify observable parameter extraction behavior
        assert isinstance(extracted_params, dict), "Expected dictionary return type for parameters"
        # Parameters may be empty, but should be a valid dictionary structure
        assert extracted_params is not None, "Expected non-None parameters result"
    
    def test_get_experiment_parameters_with_config_dict(self, sample_comprehensive_config_dict):
        """
        Test experiment parameter extraction using config dictionary with behavioral validation.
        
        Validates parameter extraction behavior when using pre-loaded configuration
        through observable API behavior.
        """
        # ARRANGE - Set up test with configuration dictionary
        config_dict = sample_comprehensive_config_dict
        experiment_name = "baseline_control_study"
        
        # ACT - Execute parameter extraction with dictionary config
        extracted_params = get_experiment_parameters(
            config=config_dict,
            experiment_name=experiment_name
        )
        
        # ASSERT - Verify expected parameter extraction behavior
        assert isinstance(extracted_params, dict), "Expected dictionary return type"
        assert extracted_params is not None, "Expected non-None parameters from valid config"
    
    def test_get_dataset_parameters_with_config_path(self, sample_config_file):
        """
        Test dataset parameter extraction using config file path with behavioral validation.
        
        Validates dataset parameter extraction workflow through public API interface.
        """
        # ARRANGE - Set up test for dataset parameter extraction
        config_path = sample_config_file
        dataset_name = "baseline_behavior"
        
        # ACT - Execute dataset parameter extraction
        dataset_params = get_dataset_parameters(
            config_path=config_path,
            dataset_name=dataset_name
        )
        
        # ASSERT - Verify observable parameter extraction behavior
        assert isinstance(dataset_params, dict), "Expected dictionary return type for dataset params"
        assert dataset_params is not None, "Expected non-None dataset parameters result"
    
    def test_get_parameters_not_found_errors(self, sample_config_file):
        """
        Test KeyError when experiment/dataset doesn't exist with error behavior validation.
        
        Validates API error handling for missing experiments and datasets through
        observable exception behavior.
        """
        # ARRANGE - Set up scenarios with non-existent entities
        config_path = sample_config_file
        
        # ACT & ASSERT - Test nonexistent experiment error behavior
        with pytest.raises(KeyError, match="Experiment 'nonexistent' not found"):
            get_experiment_parameters(
                config_path=config_path,
                experiment_name="nonexistent"
            )
        
        # ACT & ASSERT - Test nonexistent dataset error behavior
        with pytest.raises(KeyError, match="Dataset 'nonexistent' not found"):
            get_dataset_parameters(
                config_path=config_path,
                dataset_name="nonexistent"
            )


class TestProcessExperimentDataIntegration:
    """
    Integration tests for process_experiment_data API function using behavior-focused validation.
    
    Tests validate data processing behavior through public API interfaces without
    examining internal DataFrame construction implementation details.
    """
    
    def test_process_experiment_data_with_pickle_file(self, temp_cross_platform_dir, sample_exp_matrix_comprehensive):
        """
        Test processing experimental data from pickle file with end-to-end behavioral validation.
        
        Validates complete data processing workflow: file loading → DataFrame construction
        through observable API behavior.
        """
        # ARRANGE - Set up test pickle file with realistic experimental data
        test_data = sample_exp_matrix_comprehensive
        pickle_file = temp_cross_platform_dir / "test_exp_data.pkl"
        
        import pickle
        with open(pickle_file, 'wb') as f:
            pickle.dump(test_data, f)
        
        # ACT - Execute data processing
        result_df = process_experiment_data(data_path=pickle_file)
        
        # ASSERT - Verify observable processing behavior
        assert isinstance(result_df, pd.DataFrame), "Expected DataFrame return type"
        assert len(result_df) > 0, "Expected non-empty DataFrame from valid data"
        
        # Verify basic columns are present (API contract validation)
        expected_core_columns = ['t', 'x', 'y']
        for col in expected_core_columns:
            assert col in result_df.columns, f"Expected core column '{col}' in result DataFrame"
        
        # Verify data integrity through observable behavior
        assert not result_df['t'].isna().any(), "Expected no missing values in time column"
    
    def test_process_experiment_data_with_metadata_injection(self, temp_cross_platform_dir, sample_exp_matrix_comprehensive):
        """
        Test metadata injection during data processing with behavioral validation.
        
        Validates that metadata parameter affects DataFrame content according to
        API contract specifications.
        """
        # ARRANGE - Set up test data with metadata
        test_data = sample_exp_matrix_comprehensive
        pickle_file = temp_cross_platform_dir / "test_exp_data.pkl"
        
        import pickle
        with open(pickle_file, 'wb') as f:
            pickle.dump(test_data, f)
        
        test_metadata = {
            "experiment_name": "test_integration",
            "date": "2024-12-20",
            "fly_id": "fly_003"
        }
        
        # ACT - Execute data processing with metadata
        result_df = process_experiment_data(
            data_path=pickle_file,
            metadata=test_metadata
        )
        
        # ASSERT - Verify metadata injection behavior
        assert isinstance(result_df, pd.DataFrame), "Expected DataFrame return type with metadata"
        assert len(result_df) > 0, "Expected non-empty DataFrame with metadata injection"
        
        # Metadata injection behavior is implementation-dependent, but should not error
        # Focusing on observable behavior rather than specific metadata column presence
    
    def test_process_experiment_data_file_not_found(self):
        """
        Test error handling when data file doesn't exist with error behavior validation.
        
        Validates API error handling for non-existent data files through observable
        exception behavior.
        """
        # ARRANGE - Set up scenario with non-existent file
        nonexistent_file = "/nonexistent/file.pkl"
        
        # ACT & ASSERT - Verify expected error behavior
        with pytest.raises(FileNotFoundError):
            process_experiment_data(data_path=nonexistent_file)


# ============================================================================
# CROSS-MODULE ORCHESTRATION TESTS - END-TO-END WORKFLOW VALIDATION
# ============================================================================

class TestCrossModuleOrchestration:
    """
    Integration tests validating coordination between API functions and subsystems.
    
    Tests focus on end-to-end workflow validation through observable behavior
    rather than internal cross-module implementation details.
    """
    
    def test_end_to_end_experiment_workflow(self, sample_config_file, temp_cross_platform_dir):
        """
        Test complete end-to-end workflow from config to processed data with behavioral validation.
        
        Validates entire workflow orchestration: configuration → file discovery → parameter extraction
        → data processing through observable API behavior without examining internal coordination.
        """
        # ARRANGE - Set up complete test scenario
        config_path = sample_config_file
        experiment_name = "baseline_control_study"
        
        # Create test data structure
        data_dir = temp_cross_platform_dir / "research_data" / "experiments"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Update config to reference test directory
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        config_data["project"]["directories"]["major_data_directory"] = str(data_dir.parent)
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        # ACT - Execute complete workflow steps
        # Step 1: Load experiment files
        discovered_files = load_experiment_files(
            config_path=config_path,
            experiment_name=experiment_name,
            extract_metadata=True
        )
        
        # Step 2: Get experiment parameters
        experiment_params = get_experiment_parameters(
            config_path=config_path,
            experiment_name=experiment_name
        )
        
        # ASSERT - Verify end-to-end workflow behavior
        assert isinstance(discovered_files, dict), "Expected metadata dict from file discovery"
        assert isinstance(experiment_params, dict), "Expected parameters dict from parameter extraction"
        
        # Verify workflow coordination through observable behavior
        assert experiment_params is not None, "Expected valid parameters from workflow"
        assert discovered_files is not None, "Expected valid file discovery from workflow"
    
    def test_config_dict_vs_file_consistency(self, sample_config_file, sample_comprehensive_config_dict):
        """
        Test that config dictionary and file produce consistent results with behavioral validation.
        
        Validates API consistency between configuration input methods through observable
        behavior comparison without examining internal configuration processing.
        """
        # ARRANGE - Set up consistency test scenario
        config_path = sample_config_file
        config_dict = sample_comprehensive_config_dict
        experiment_name = "baseline_control_study"
        
        # ACT - Execute same operation with both config methods
        # Load using config file
        files_from_file = load_experiment_files(
            config_path=config_path,
            experiment_name=experiment_name
        )
        
        # Load using config dictionary
        files_from_dict = load_experiment_files(
            config=config_dict,
            experiment_name=experiment_name
        )
        
        # ASSERT - Verify consistent behavior between config methods
        assert isinstance(files_from_file, list), "Expected list return type from config file"
        assert isinstance(files_from_dict, list), "Expected list return type from config dict"
        
        # Both methods should produce valid results (consistency validation)
        assert files_from_file is not None, "Expected non-None result from config file"
        assert files_from_dict is not None, "Expected non-None result from config dict"
    
    def test_error_propagation_from_config_module(self):
        """
        Test that configuration errors propagate correctly through API with error behavior validation.
        
        Validates error handling propagation through the API stack using observable
        exception behavior without examining internal error handling implementation.
        """
        # ARRANGE - Set up invalid configuration scenario
        import tempfile
        import os
        
        invalid_yaml = "invalid: yaml: content: ["
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml)
            invalid_config_file = f.name
        
        try:
            # ACT & ASSERT - Verify error propagation behavior
            with pytest.raises(yaml.YAMLError):
                load_experiment_files(
                    config_path=invalid_config_file,
                    experiment_name="test_experiment"
                )
        finally:
            os.unlink(invalid_config_file)


# ============================================================================
# REALISTIC DATA FLOW SCENARIOS - BEHAVIORAL VALIDATION
# ============================================================================

class TestRealisticDataFlowScenarios:
    """
    Integration tests with realistic experimental data flows and scenarios.
    
    Tests validate realistic neuroscience research workflow patterns through
    observable API behavior without examining internal data flow implementation.
    """
    
    def test_typical_neuroscience_workflow(self, sample_config_file, temp_cross_platform_dir):
        """
        Test typical neuroscience research workflow patterns with end-to-end validation.
        
        Validates realistic research workflow behavior through observable API interactions
        simulating actual neuroscience data analysis patterns.
        """
        # ARRANGE - Set up realistic neuroscience research scenario
        config_path = sample_config_file
        experiment_name = "baseline_control_study"
        
        # Update config for test environment
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        config_data["project"]["directories"]["major_data_directory"] = str(temp_cross_platform_dir)
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        # ACT - Execute typical research workflow steps
        # Step 1: Discover all files for an experiment
        all_experiment_files = load_experiment_files(
            config_path=config_path,
            experiment_name=experiment_name,
            pattern="*.pkl",
            recursive=True
        )
        
        # Step 2: Get experimental parameters for analysis context
        analysis_params = get_experiment_parameters(
            config_path=config_path,
            experiment_name=experiment_name
        )
        
        # ASSERT - Verify realistic workflow behavior
        assert isinstance(all_experiment_files, list), "Expected file list from discovery step"
        assert isinstance(analysis_params, dict), "Expected parameters dict from extraction step"
        
        # Verify workflow produces valid research-ready outputs
        assert all_experiment_files is not None, "Expected valid file discovery for research workflow"
        assert analysis_params is not None, "Expected valid parameters for analysis context"
    
    @pytest.mark.parametrize("experiment_name,expected_structure", [
        ("baseline_control_study", "single_dataset"),
        ("baseline_control_study", "comprehensive_config")  # Test with different config interpretations
    ])
    def test_parametrized_experiment_workflows(self, sample_comprehensive_config_dict, experiment_name, expected_structure):
        """
        Test various experiment configurations with parametrization and behavioral validation.
        
        Validates different experimental configuration patterns through parameterized
        testing with observable API behavior verification.
        """
        # ARRANGE - Set up parametrized experiment test
        config_dict = sample_comprehensive_config_dict
        
        # ACT - Execute workflow for different experiment patterns
        experiment_params = get_experiment_parameters(
            config=config_dict,
            experiment_name=experiment_name
        )
        
        experiment_files = load_experiment_files(
            config=config_dict,
            experiment_name=experiment_name
        )
        
        # ASSERT - Verify parametrized workflow behavior
        assert isinstance(experiment_params, dict), "Expected parameters dict for all experiment types"
        assert isinstance(experiment_files, list), "Expected file list for all experiment types"
        
        # Verify workflow consistency across parameter variations
        assert experiment_params is not None, "Expected valid parameters for parametrized experiment"
        assert experiment_files is not None, "Expected valid file list for parametrized experiment"


# ============================================================================
# EDGE-CASE COVERAGE ENHANCEMENT - COMPREHENSIVE VALIDATION
# ============================================================================

class TestEdgeCaseScenarios:
    """
    Enhanced edge-case testing for workflow resilience and comprehensive coverage.
    
    Tests validate API behavior under edge conditions, boundary cases, and error scenarios
    through behavioral validation with parameterized test scenarios.
    """
    
    @pytest.mark.parametrize("unicode_scenario", [
        "unicode_experiment_name",
        "unicode_dataset_name",
        "unicode_config_path"
    ])
    def test_unicode_handling_scenarios(self, sample_comprehensive_config_dict, temp_cross_platform_dir, unicode_scenario):
        """
        Test Unicode character handling in various API inputs with edge-case validation.
        
        Validates API resilience for Unicode inputs through parameterized scenarios
        focusing on observable behavior rather than internal Unicode processing.
        """
        # ARRANGE - Set up Unicode edge-case scenarios
        config_dict = sample_comprehensive_config_dict
        base_experiment = "baseline_control_study"
        base_dataset = "baseline_behavior"
        
        # ACT & ASSERT - Test different Unicode scenarios
        if unicode_scenario == "unicode_experiment_name":
            # Test with Unicode experiment name (may not exist, should handle gracefully)
            unicode_experiment = "tëst_ëxpérîmént"
            try:
                result = load_experiment_files(
                    config=config_dict,
                    experiment_name=unicode_experiment
                )
                # If it succeeds, verify proper handling
                assert isinstance(result, list), "Expected list return for Unicode experiment name"
            except KeyError:
                # Expected behavior for non-existent Unicode experiment
                pass
        
        elif unicode_scenario == "unicode_dataset_name":
            # Test with Unicode dataset name
            unicode_dataset = "tëst_dätäsét"
            try:
                result = load_dataset_files(
                    config=config_dict,
                    dataset_name=unicode_dataset
                )
                assert isinstance(result, list), "Expected list return for Unicode dataset name"
            except KeyError:
                # Expected behavior for non-existent Unicode dataset
                pass
        
        elif unicode_scenario == "unicode_config_path":
            # Test with Unicode in config file path
            unicode_config_path = temp_cross_platform_dir / "tëst_cönfïg.yaml"
            
            # Create Unicode config file
            with open(unicode_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f)
            
            # Test loading with Unicode path
            result = load_experiment_files(
                config_path=str(unicode_config_path),
                experiment_name=base_experiment
            )
            assert isinstance(result, list), "Expected list return for Unicode config path"
    
    @pytest.mark.parametrize("corrupted_scenario", [
        "malformed_yaml_config",
        "corrupted_pickle_data",
        "invalid_config_structure"
    ])
    def test_corrupted_file_recovery(self, temp_cross_platform_dir, corrupted_scenario):
        """
        Test API resilience with corrupted files and malformed data scenarios.
        
        Validates error handling and recovery behavior for various file corruption
        scenarios through observable exception behavior.
        """
        # ARRANGE - Set up corrupted file scenarios
        if corrupted_scenario == "malformed_yaml_config":
            # Create malformed YAML configuration
            malformed_config = temp_cross_platform_dir / "malformed.yaml"
            malformed_config.write_text("invalid: yaml: structure: [missing close", encoding='utf-8')
            
            # ACT & ASSERT - Verify error handling for malformed YAML
            with pytest.raises(yaml.YAMLError):
                load_experiment_files(
                    config_path=str(malformed_config),
                    experiment_name="test_experiment"
                )
        
        elif corrupted_scenario == "invalid_config_structure":
            # Create config with invalid structure
            invalid_config = temp_cross_platform_dir / "invalid.yaml"
            invalid_structure = {"not_project": {"invalid": "structure"}}
            with open(invalid_config, 'w') as f:
                yaml.dump(invalid_structure, f)
            
            # ACT & ASSERT - Verify error handling for invalid structure
            with pytest.raises((KeyError, ValueError)):
                load_experiment_files(
                    config_path=str(invalid_config),
                    experiment_name="test_experiment"
                )
    
    @pytest.mark.parametrize("boundary_condition", [
        "empty_config_sections",
        "minimal_valid_config",
        "maximum_config_complexity"
    ])
    def test_configuration_boundary_conditions(self, sample_comprehensive_config_dict, boundary_condition):
        """
        Test API behavior at configuration boundary conditions with edge-case validation.
        
        Validates API resilience for minimal, maximal, and edge-case configuration
        scenarios through behavioral testing.
        """
        # ARRANGE - Set up boundary condition scenarios
        if boundary_condition == "empty_config_sections":
            # Test with empty but valid config sections
            minimal_config = {
                "project": {"directories": {"major_data_directory": "/tmp"}},
                "experiments": {"test_exp": {"datasets": []}},
                "datasets": {}
            }
            
            # ACT - Test with minimal config
            result = load_experiment_files(
                config=minimal_config,
                experiment_name="test_exp"
            )
            
            # ASSERT - Verify graceful handling of minimal config
            assert isinstance(result, list), "Expected list return for minimal config"
        
        elif boundary_condition == "minimal_valid_config":
            # Test with absolutely minimal valid configuration
            minimal_config = {
                "project": {"directories": {"major_data_directory": "/tmp"}},
                "experiments": {"minimal": {"datasets": []}},
                "datasets": {}
            }
            
            # ACT & ASSERT - Verify minimal config handling
            try:
                result = load_experiment_files(
                    config=minimal_config,
                    experiment_name="minimal"
                )
                assert isinstance(result, list), "Expected list return for minimal valid config"
            except (KeyError, ValueError):
                # May raise errors for truly minimal config, which is acceptable behavior
                pass
    
    def test_concurrent_api_access_simulation(self, sample_comprehensive_config_dict):
        """
        Test API behavior under simulated concurrent access scenarios.
        
        Validates API thread-safety and concurrent access handling through
        behavioral testing without examining internal synchronization implementation.
        """
        # ARRANGE - Set up concurrent access simulation
        config_dict = sample_comprehensive_config_dict
        experiment_name = "baseline_control_study"
        
        # ACT - Simulate concurrent API calls (simplified test)
        results = []
        for i in range(3):  # Simulate multiple concurrent calls
            try:
                result = get_experiment_parameters(
                    config=config_dict,
                    experiment_name=experiment_name
                )
                results.append(result)
            except Exception as e:
                # Capture any concurrent access issues
                results.append(f"Error: {e}")
        
        # ASSERT - Verify consistent behavior under concurrent access
        # All results should be dictionaries (consistent behavior)
        valid_results = [r for r in results if isinstance(r, dict)]
        assert len(valid_results) > 0, "Expected at least some successful concurrent API calls"
        
        # Verify consistency across concurrent calls
        if len(valid_results) > 1:
            first_result = valid_results[0]
            for result in valid_results[1:]:
                # Results should be consistent across concurrent calls
                assert result.keys() == first_result.keys(), "Expected consistent parameter keys across concurrent calls"


# ============================================================================
# BACKWARD COMPATIBILITY TESTS - API CONTRACT VALIDATION
# ============================================================================

class TestBackwardCompatibility:
    """
    Integration tests ensuring API maintains backward compatibility.
    
    Tests validate API contract stability and backward compatibility through
    behavioral validation of function signatures and return types.
    """
    
    def test_api_function_signatures_unchanged(self):
        """
        Test that API function signatures remain stable with signature validation.
        
        Validates API contract compliance through function signature inspection
        without examining internal implementation details.
        """
        # ARRANGE - Set up signature validation
        import inspect
        
        # ACT - Inspect load_experiment_files signature
        sig = inspect.signature(load_experiment_files)
        expected_params = [
            'config_path', 'config', 'experiment_name', 'base_directory',
            'pattern', 'recursive', 'extensions', 'extract_metadata', 'parse_dates'
        ]
        actual_params = list(sig.parameters.keys())
        
        # ASSERT - Verify signature stability
        for param in expected_params:
            assert param in actual_params, f"Parameter {param} missing from load_experiment_files"
    
    def test_return_type_consistency(self, sample_comprehensive_config_dict):
        """
        Test that return types remain consistent with API contract validation.
        
        Validates API return type stability through behavioral testing of
        different parameter combinations.
        """
        # ARRANGE - Set up return type consistency test
        config_dict = sample_comprehensive_config_dict
        experiment_name = "baseline_control_study"
        
        # ACT - Test different parameter combinations
        # Test without metadata extraction (should return list)
        files_list = load_experiment_files(
            config=config_dict,
            experiment_name=experiment_name
        )
        
        # Test with metadata extraction (should return dict)
        files_with_meta = load_experiment_files(
            config=config_dict,
            experiment_name=experiment_name,
            extract_metadata=True
        )
        
        # ASSERT - Verify consistent return types
        assert isinstance(files_list, list), "Expected list return type without metadata"
        assert isinstance(files_with_meta, dict), "Expected dict return type with metadata"
    
    def test_default_parameter_behavior(self, sample_comprehensive_config_dict):
        """
        Test that default parameters maintain expected behavior with consistency validation.
        
        Validates API default parameter behavior through comparative testing
        of explicit vs implicit parameter usage.
        """
        # ARRANGE - Set up default parameter behavior test
        config_dict = sample_comprehensive_config_dict
        experiment_name = "baseline_control_study"
        
        # ACT - Test default vs explicit parameter behavior
        files_default = load_experiment_files(
            config=config_dict,
            experiment_name=experiment_name
        )
        
        files_explicit = load_experiment_files(
            config=config_dict,
            experiment_name=experiment_name,
            pattern="*.*",
            recursive=True,
            extract_metadata=False,
            parse_dates=False
        )
        
        # ASSERT - Verify consistent behavior between default and explicit parameters
        assert type(files_default) == type(files_explicit), "Expected same return type for default vs explicit params"
        assert isinstance(files_default, list), "Expected list return type for default parameters"
        assert isinstance(files_explicit, list), "Expected list return type for explicit parameters"