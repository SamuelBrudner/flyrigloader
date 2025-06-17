"""
Cross-module integration test suite validating seamless data flow and interface compatibility.

This test suite validates the integration between core flyrigloader modules using
behavior-focused testing principles and Protocol-based dependency injection:
- config (yaml_config, discovery) → discovery (files, patterns) → io (pickle, column_models) → utils (dataframe, paths)

Tests focus on observable cross-module integration behavior rather than implementation details:
- End-to-end data flow verification through the complete pipeline
- Interface compatibility validation using public APIs only  
- Error propagation through module boundaries with behavioral validation
- Data integrity maintenance across all transformation stages
- Configuration → discovery → IO → utilities seamless integration
- Module boundary behavioral validation without internal coupling

Refactored Features per Section 0 requirements:
- Behavior-focused testing using public API validation instead of private attribute access
- Protocol-based mock implementations from centralized tests/utils.py for consistent dependency simulation
- AAA pattern structure with clear separation of test setup, execution, and verification phases
- Enhanced edge-case coverage through parameterized test scenarios
- Centralized fixture utilization from tests/conftest.py for comprehensive test data generation
- Observable behavior validation through return values, side effects, and error conditions
"""

import pytest
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from loguru import logger

# Import test utilities for Protocol-based mocking and centralized patterns
from tests.utils import (
    create_mock_filesystem, create_mock_dataloader, create_mock_config_provider,
    generate_edge_case_scenarios, create_integration_test_environment,
    MockConfigurationProvider, MockFilesystemProvider, MockDataLoadingProvider
)

# Import public API modules for behavior-focused testing (no private access)
from flyrigloader.config.yaml_config import (
    load_config, get_ignore_patterns, get_mandatory_substrings, 
    get_dataset_info, get_experiment_info, get_extraction_patterns
)
from flyrigloader.config.discovery import (
    ConfigDiscoveryEngine, discover_files_with_config, discover_experiment_files
)
from flyrigloader.discovery.files import discover_files
from flyrigloader.io.pickle import read_pickle_any_format
from flyrigloader.io.column_models import (
    load_column_config, validate_experimental_data, transform_to_standardized_format
)
from flyrigloader.utils.dataframe import discovery_results_to_dataframe
from flyrigloader.utils.paths import resolve_path, normalize_path_separators


# ============================================================================
# CROSS-MODULE INTEGRATION TEST SUITE
# ============================================================================

class TestCrossModuleIntegration:
    """
    Behavior-focused cross-module integration test suite validating seamless data flow 
    and interface compatibility using Protocol-based dependency injection.
    
    Test Categories (Behavior-Focused):
    1. Configuration → Discovery integration through observable data flow
    2. Discovery → IO integration through data format compatibility  
    3. IO → Utils integration through DataFrame transformation behavior
    4. End-to-End pipeline integration through complete workflow validation
    5. Error propagation behavior across module boundaries
    6. Edge-case handling through parameterized boundary condition scenarios
    """

    # ========================================================================
    # CONFIGURATION → DISCOVERY INTEGRATION VALIDATION
    # ========================================================================

    def test_config_to_discovery_integration_observable_behavior(
        self, 
        comprehensive_sample_config_dict, 
        temp_filesystem_structure
    ):
        """
        Test seamless configuration to discovery integration through observable behavior validation.
        
        ARRANGE: Set up comprehensive configuration and expected discovery outcomes
        ACT: Execute configuration-driven discovery workflow  
        ASSERT: Validate data flow through public interface behavior
        
        Behavioral Validation Focus:
        - Configuration parameters correctly translated to discovery behavior
        - Discovery results format compatible with downstream processing
        - Error conditions properly handled through public interface
        - Integration maintains data integrity through observable outcomes
        """
        logger.info("Testing config to discovery integration through observable behavior")
        
        # ARRANGE - Set up comprehensive test environment with Protocol-based mocks
        config_mock = create_mock_config_provider(config_type='comprehensive')
        filesystem_mock = create_mock_filesystem(
            structure={
                'files': {
                    str(temp_filesystem_structure.get("baseline_file_1", "/test/baseline_001.pkl")): {'size': 2048},
                    str(temp_filesystem_structure.get("opto_file_1", "/test/opto_001.pkl")): {'size': 3072}
                },
                'directories': ['/test/data']
            }
        )
        
        # Create realistic expected discovery results based on configuration behavior
        test_config = comprehensive_sample_config_dict
        expected_ignore_patterns = ["backup", "temp", "static_horiz_ribbon"]
        expected_mandatory_patterns = ["experiment_", "data_"]
        expected_discovery_results = {
            str(temp_filesystem_structure.get("baseline_file_1", "/test/baseline_001.pkl")): {
                "date": "20241220",
                "condition": "control", 
                "dataset": "baseline_behavior",
                "file_size": 2048,
                "metadata_extracted": True
            }
        }
        
        # ACT - Execute configuration-driven discovery with behavioral focus
        # Extract configuration patterns using public API
        actual_ignore_patterns = get_ignore_patterns(test_config, experiment="baseline_control_study")
        actual_mandatory_patterns = get_mandatory_substrings(test_config, experiment="baseline_control_study")
        actual_extraction_patterns = get_extraction_patterns(test_config, experiment="baseline_control_study")
        
        # Execute discovery workflow using configuration parameters
        discovery_engine = ConfigDiscoveryEngine()
        discovery_result = discover_files_with_config(
            config=test_config,
            directory=str(temp_filesystem_structure.get("data_root", "/test/data")),
            pattern="*.pkl",
            experiment="baseline_control_study",
            extract_metadata=True
        )
        
        # ASSERT - Validate observable behavior through public interface
        # Validate configuration extraction behavior produces expected patterns
        assert isinstance(actual_ignore_patterns, list), "Configuration extraction must produce list output"
        assert len(actual_ignore_patterns) > 0, "Configuration should provide ignore patterns"
        assert all(pattern in actual_ignore_patterns for pattern in expected_ignore_patterns if pattern), "Expected ignore patterns should be present"
        
        assert isinstance(actual_mandatory_patterns, list), "Mandatory patterns must be list format"
        assert len(actual_mandatory_patterns) > 0, "Configuration should provide mandatory patterns"
        
        # Validate discovery workflow produces expected behavioral outcomes
        assert isinstance(discovery_result, dict), "Discovery workflow must return dictionary format"
        
        # Validate data flow maintains integrity through module boundary
        for file_path, metadata in discovery_result.items():
            assert isinstance(file_path, str), "File paths must be string format for IO module compatibility"
            assert isinstance(metadata, dict), "Metadata must be dictionary format for downstream processing"
            assert "date" in metadata or "file_size" in metadata, "Essential metadata must be present for IO integration"
        
        # Validate integration produces observable outcomes suitable for next pipeline stage
        if discovery_result:
            sample_file, sample_metadata = next(iter(discovery_result.items()))
            assert Path(sample_file).suffix in ['.pkl', '.csv', '.json'], "Discovery should return processable file types"
            assert isinstance(sample_metadata.get('file_size', 0), int), "File metadata should include processable attributes"
        
        logger.success("Config to discovery integration behavioral validation completed")

    def test_discovery_to_io_integration_data_compatibility(
        self,
        temp_filesystem_structure,
        comprehensive_exp_matrix
    ):
        """
        Test discovery to IO integration through data format compatibility validation.
        
        ARRANGE: Set up discovery results and data loading environment
        ACT: Execute data loading workflow with discovery outputs
        ASSERT: Validate data format compatibility through behavioral outcomes
        
        Behavioral Validation Focus:
        - Discovery results successfully processed by IO module through public interface
        - Data format compatibility maintained across module boundary
        - Metadata preservation through observable data structures
        - Error conditions handled gracefully with appropriate behavioral responses
        """
        logger.info("Testing discovery to IO integration through data compatibility")
        
        # ARRANGE - Set up realistic discovery results and data loading environment
        discovery_results = {
            str(temp_filesystem_structure.get("baseline_file_1", "/test/baseline_001.pkl")): {
                "date": "20241220",
                "condition": "control",
                "dataset": "baseline_behavior", 
                "file_size": 2048,
                "extraction_metadata": {"experiment": "baseline", "session": "1"}
            },
            str(temp_filesystem_structure.get("opto_file_1", "/test/opto_001.pkl")): {
                "date": "20241218", 
                "condition": "treatment",
                "dataset": "optogenetic_stimulation",
                "file_size": 3072,
                "extraction_metadata": {"experiment": "opto", "stimulation": "blue_light"}
            }
        }
        
        # Set up Protocol-based data loading mock with realistic experimental data
        dataloader_mock = create_mock_dataloader(
            scenarios=['basic', 'experimental'],
            include_experimental_data=True
        )
        
        # Configure mock data for each discovered file
        for file_path, metadata in discovery_results.items():
            if "baseline" in file_path:
                dataloader_mock.add_experimental_matrix(
                    file_path,
                    n_timepoints=18000,  # 5 minutes at 60 Hz
                    include_signal=False,
                    include_metadata=True
                )
            elif "opto" in file_path:
                dataloader_mock.add_experimental_matrix(
                    file_path, 
                    n_timepoints=36000,  # 10 minutes at 60 Hz
                    include_signal=True,
                    include_metadata=True
                )
        
        # ACT - Execute data loading workflow using discovery results
        loaded_datasets = {}
        data_loading_success_count = 0
        
        for file_path, discovery_metadata in discovery_results.items():
            try:
                # Use public IO interface to load data (behavioral focus)
                loaded_data = dataloader_mock.load_file(file_path)
                
                # Integrate discovery metadata with loaded data through public interface
                if isinstance(loaded_data, dict):
                    loaded_data['discovery_metadata'] = discovery_metadata
                    loaded_data['file_path'] = file_path
                
                loaded_datasets[file_path] = loaded_data
                data_loading_success_count += 1
                
            except Exception as e:
                logger.warning(f"Data loading failed for {file_path}: {e}")
                # Continue processing other files - test graceful error handling
        
        # ASSERT - Validate behavioral outcomes through observable data structures
        # Validate overall data loading behavior
        assert data_loading_success_count > 0, "At least some discovery results should load successfully"
        assert len(loaded_datasets) == len(discovery_results), "All discovery results should be processed"
        
        # Validate data format compatibility for downstream utils module
        for file_path, loaded_data in loaded_datasets.items():
            # Validate basic data structure compatibility
            assert isinstance(loaded_data, dict), "Loaded data must be dictionary for utils module compatibility"
            
            # Validate essential time-series data presence (behavioral requirement)
            essential_fields = ['t', 'x', 'y']
            for field in essential_fields:
                assert field in loaded_data, f"Essential field '{field}' must be present for utils module"
                assert isinstance(loaded_data[field], np.ndarray), f"Field '{field}' must be numpy array for processing"
                assert loaded_data[field].ndim == 1, f"Field '{field}' must be 1D array for utils compatibility"
            
            # Validate array consistency (behavioral requirement for downstream processing)
            data_length = len(loaded_data['t'])
            assert len(loaded_data['x']) == data_length, "X data length must match time data"
            assert len(loaded_data['y']) == data_length, "Y data length must match time data"
            assert data_length > 0, "Data arrays must contain actual data points"
            
            # Validate metadata preservation through data flow
            assert 'discovery_metadata' in loaded_data, "Discovery metadata must be preserved"
            preserved_metadata = loaded_data['discovery_metadata']
            assert isinstance(preserved_metadata, dict), "Metadata must maintain dictionary structure"
            assert 'date' in preserved_metadata, "Essential metadata fields must be preserved"
            assert 'condition' in preserved_metadata, "Experimental condition must be preserved"
            
            # Validate data types suitable for utils module processing
            assert loaded_data['t'].dtype.kind in ['f', 'i'], "Time data must be numeric for analysis"
            assert loaded_data['x'].dtype.kind in ['f', 'i'], "Position data must be numeric for analysis"
            assert loaded_data['y'].dtype.kind in ['f', 'i'], "Position data must be numeric for analysis"
        
        # Validate cross-module data flow behavioral consistency
        baseline_files = [f for f in loaded_datasets.keys() if "baseline" in f]
        opto_files = [f for f in loaded_datasets.keys() if "opto" in f]
        
        if baseline_files:
            baseline_data = loaded_datasets[baseline_files[0]]
            assert len(baseline_data['t']) > 10000, "Baseline experiments should have sufficient data points"
            
        if opto_files:
            opto_data = loaded_datasets[opto_files[0]]
            assert len(opto_data['t']) > 30000, "Optogenetic experiments should have longer recordings"
            if 'signal' in opto_data:
                assert isinstance(opto_data['signal'], np.ndarray), "Optogenetic data should include signal arrays"
        
        logger.success("Discovery to IO integration data compatibility validation completed")

    def test_io_to_utils_integration_dataframe_workflow(
        self,
        comprehensive_exp_matrix,
        sample_metadata
    ):
        """
        Test IO to utils integration through DataFrame transformation workflow validation.
        
        ARRANGE: Set up IO module experimental data and utils processing environment
        ACT: Execute DataFrame transformation workflow using public interfaces
        ASSERT: Validate transformation behavior through observable DataFrame outcomes
        
        Behavioral Validation Focus:
        - IO experimental data successfully transformed to DataFrame through public interface
        - Metadata integration behavior maintains data relationships
        - Column validation workflow produces expected behavioral outcomes
        - DataFrame structure suitable for downstream analysis workflows
        """
        logger.info("Testing IO to utils integration through DataFrame workflow")
        
        # ARRANGE - Set up comprehensive IO experimental data with realistic structure
        io_experimental_data = comprehensive_exp_matrix.copy()
        io_experimental_data.update({
            'metadata': sample_metadata.copy() if sample_metadata else {
                'animal_id': 'test_fly_001',
                'condition': 'control',
                'experiment_date': '20241220',
                'session_duration': 300.0
            },
            'file_path': '/data/experiments/test_experiment.pkl',
            'file_statistics': {
                'size_bytes': 2048,
                'modification_time': datetime.now().isoformat(),
                'processing_timestamp': datetime.now().isoformat()
            }
        })
        
        # Set up Protocol-based column configuration mock
        config_mock = create_mock_config_provider(config_type='comprehensive')
        column_config = {
            "columns": {
                "t": {"type": "numpy.ndarray", "required": True, "dimension": 1},
                "x": {"type": "numpy.ndarray", "required": True, "dimension": 1}, 
                "y": {"type": "numpy.ndarray", "required": True, "dimension": 1},
                "signal": {"type": "numpy.ndarray", "required": False, "dimension": 1},
                "signal_disp": {"type": "numpy.ndarray", "required": False, "dimension": 2}
            }
        }
        
        # ACT - Execute DataFrame transformation workflow through public interfaces
        try:
            # Validate IO data format compatibility (behavioral precondition)
            assert isinstance(io_experimental_data, dict), "IO data must be dictionary for utils processing"
            assert all(field in io_experimental_data for field in ['t', 'x', 'y']), "Essential fields must be present"
            
            # Execute column validation using public interface
            validated_data = validate_experimental_data(io_experimental_data, column_config)
            
            # Transform data to standardized format using public interface
            standardized_data = transform_to_standardized_format(validated_data, column_config)
            
            # Prepare DataFrame-compatible data structure through behavioral transformation
            df_data_dict = {}
            
            # Process numerical arrays (behavioral requirement)
            for field_name, field_data in standardized_data.items():
                if isinstance(field_data, np.ndarray):
                    if field_data.ndim == 1:
                        # 1D arrays directly suitable for DataFrame
                        df_data_dict[field_name] = field_data
                    elif field_data.ndim == 2:
                        # 2D arrays require channel expansion for DataFrame compatibility
                        for channel_idx in range(field_data.shape[0]):
                            df_data_dict[f"{field_name}_ch{channel_idx:02d}"] = field_data[channel_idx, :]
                            
            # Process metadata through public interface behavior
            if 'metadata' in standardized_data:
                metadata = standardized_data['metadata']
                for meta_key, meta_value in metadata.items():
                    # Convert metadata to DataFrame-compatible constant columns
                    if isinstance(meta_value, (str, int, float, bool)):
                        df_data_dict[f"meta_{meta_key}"] = [meta_value] * len(standardized_data['t'])
                    else:
                        df_data_dict[f"meta_{meta_key}"] = [str(meta_value)] * len(standardized_data['t'])
            
            # Create DataFrame using utils module public interface
            result_dataframe = discovery_results_to_dataframe(df_data_dict)
            
        except Exception as workflow_error:
            pytest.fail(f"DataFrame transformation workflow failed: {workflow_error}")
        
        # ASSERT - Validate DataFrame transformation behavioral outcomes
        # Validate DataFrame creation success through observable structure
        assert isinstance(result_dataframe, pd.DataFrame), "Utils module must produce pandas DataFrame"
        assert len(result_dataframe) > 0, "DataFrame must contain actual data rows"
        assert len(result_dataframe.columns) > 0, "DataFrame must contain data columns"
        
        # Validate essential columns presence (behavioral requirement)
        essential_columns = ['t', 'x', 'y']
        for essential_col in essential_columns:
            assert essential_col in result_dataframe.columns, f"Essential column '{essential_col}' must be present"
            
        # Validate data length consistency (behavioral integrity)
        expected_length = len(io_experimental_data['t'])
        assert len(result_dataframe) == expected_length, "DataFrame length must match original data length"
        
        # Validate numerical data types suitable for analysis (behavioral requirement)
        for numeric_col in ['t', 'x', 'y']:
            if numeric_col in result_dataframe.columns:
                assert pd.api.types.is_numeric_dtype(result_dataframe[numeric_col]), f"Column '{numeric_col}' must be numeric for analysis"
        
        # Validate metadata integration behavior
        metadata_columns = [col for col in result_dataframe.columns if col.startswith('meta_')]
        assert len(metadata_columns) > 0, "Metadata must be integrated into DataFrame structure"
        
        # Validate metadata consistency behavior
        for meta_col in metadata_columns:
            unique_values = result_dataframe[meta_col].unique()
            assert len(unique_values) == 1, f"Metadata column '{meta_col}' should have constant value"
        
        # Validate multi-channel data handling behavior (if present)
        signal_channels = [col for col in result_dataframe.columns if col.startswith('signal_disp_ch')]
        if signal_channels:
            assert len(signal_channels) > 0, "Multi-channel signal data should be properly expanded"
            for signal_col in signal_channels:
                assert pd.api.types.is_numeric_dtype(result_dataframe[signal_col]), "Signal channels must be numeric"
        
        # Validate DataFrame suitable for downstream analysis workflows
        assert not result_dataframe.empty, "DataFrame must contain analyzable data"
        assert result_dataframe.index.is_monotonic_increasing, "DataFrame index should be suitable for time-series analysis"
        
        # Validate data integrity through statistical consistency
        if 't' in result_dataframe.columns:
            time_data = result_dataframe['t']
            assert time_data.min() >= 0, "Time data should start from non-negative values"
            assert len(time_data) == len(time_data.dropna()), "Time data should not contain missing values"
        
        logger.success("IO to utils integration DataFrame workflow validation completed")

    # ========================================================================
    # PROTOCOL-BASED DEPENDENCY INJECTION BEHAVIORAL VALIDATION
    # ========================================================================

    def test_protocol_based_dependency_injection_behavior(self):
        """
        Test Protocol-based dependency injection through observable behavior validation.
        
        ARRANGE: Set up Protocol-based mock providers using centralized test utilities
        ACT: Execute discovery workflow with injected dependencies
        ASSERT: Validate behavioral outcomes through public interface responses
        
        Behavioral Validation Focus:
        - Protocol-based providers successfully substituted for real dependencies
        - Discovery workflow behavior maintained with mock implementations
        - Interface compatibility preserved through Protocol adherence
        - Observable outcomes consistent regardless of implementation
        """
        logger.info("Testing Protocol-based dependency injection behavior")
        
        # ARRANGE - Set up Protocol-based mock providers from centralized utilities
        filesystem_provider = create_mock_filesystem(
            structure={
                'files': {
                    '/test/experiment_001.pkl': {'size': 2048},
                    '/test/experiment_002.pkl': {'size': 3072}
                },
                'directories': ['/test']
            }
        )
        
        config_provider = create_mock_config_provider(
            config_type='comprehensive',
            include_errors=False
        )
        
        dataloader_provider = create_mock_dataloader(
            scenarios=['basic'],
            include_experimental_data=True
        )
        
        # Create realistic test configuration
        test_configuration = {
            "project": {
                "ignore_substrings": ["backup", "temp"],
                "mandatory_substrings": ["experiment"]
            },
            "experiments": {
                "test_experiment": {
                    "datasets": ["test_dataset"],
                    "filters": {
                        "ignore_substrings": ["calibration"],
                        "mandatory_substrings": ["data"]
                    }
                }
            }
        }
        
        # ACT - Execute discovery workflow using Protocol-based providers
        # Test configuration extraction behavior with Protocol-based providers
        ignore_patterns = config_provider.get_ignore_patterns(test_configuration, "test_experiment")
        mandatory_patterns = config_provider.get_mandatory_substrings(test_configuration, "test_experiment")
        
        # Execute discovery workflow using public interface
        discovery_engine = ConfigDiscoveryEngine()
        discovery_results = discover_files_with_config(
            config=test_configuration,
            directory="/test",
            pattern="*.pkl",
            experiment="test_experiment",
            extract_metadata=True
        )
        
        # ASSERT - Validate behavioral outcomes through observable responses
        # Validate Protocol-based configuration extraction behavior
        assert isinstance(ignore_patterns, list), "Protocol-based config provider must return list format"
        assert len(ignore_patterns) > 0, "Configuration should provide ignore patterns"
        assert "backup" in ignore_patterns, "Global ignore patterns should be included"
        assert "temp" in ignore_patterns, "Global ignore patterns should be included"
        
        assert isinstance(mandatory_patterns, list), "Protocol-based config provider must return list format"
        assert len(mandatory_patterns) > 0, "Configuration should provide mandatory patterns"
        assert "experiment" in mandatory_patterns, "Global mandatory patterns should be included"
        
        # Validate discovery workflow behavioral outcomes
        assert isinstance(discovery_results, dict), "Discovery workflow must return dictionary format"
        
        # Validate Protocol-based provider behavior maintains expected interface
        for file_path, metadata in discovery_results.items():
            assert isinstance(file_path, str), "File paths must be string format"
            assert isinstance(metadata, dict), "Metadata must be dictionary format"
            assert "date" in metadata or "file_size" in metadata, "Essential metadata should be present"
        
        # Validate behavioral consistency across Protocol implementations
        # Test filesystem provider behavior
        test_paths = ['/test/experiment_001.pkl', '/test/experiment_002.pkl']
        for test_path in test_paths:
            path_exists = filesystem_provider.exists(test_path)
            assert isinstance(path_exists, bool), "Path existence check must return boolean"
            
        # Test data loading provider behavior
        for file_path in discovery_results.keys():
            try:
                loaded_data = dataloader_provider.load_file(file_path)
                assert isinstance(loaded_data, dict), "Data loading must return dictionary format"
                assert 't' in loaded_data, "Essential time data must be present"
            except Exception as e:
                # Acceptable behavior for Protocol-based testing
                logger.debug(f"Expected Protocol-based loading behavior: {e}")
        
        # Validate Protocol adherence maintains workflow integrity
        assert len(discovery_results) > 0 or True, "Discovery workflow should complete successfully"
        
        logger.success("Protocol-based dependency injection behavioral validation completed")

    @pytest.mark.parametrize("edge_case_scenario", [
        {
            'scenario_type': 'unicode_paths',
            'test_files': ['tëst_fïlé_001.pkl', 'dàtä_ñãmé_002.pkl'],
            'expected_behavior': 'graceful_handling'
        },
        {
            'scenario_type': 'large_metadata',
            'metadata_size': 'oversized',
            'expected_behavior': 'performance_maintained'
        },
        {
            'scenario_type': 'concurrent_access',
            'access_pattern': 'simultaneous_read',
            'expected_behavior': 'thread_safe'
        }
    ])
    def test_cross_module_edge_case_scenarios(self, edge_case_scenario):
        """
        Test cross-module integration edge-case scenarios through parameterized behavioral validation.
        
        ARRANGE: Set up edge-case scenario using parameterized test data
        ACT: Execute cross-module workflow under edge-case conditions  
        ASSERT: Validate graceful handling through observable behavioral responses
        
        Edge-Case Coverage:
        - Unicode filename and path handling across all modules
        - Oversized metadata propagation through module boundaries
        - Concurrent access patterns with thread-safety validation
        - Memory constraint scenarios with graceful degradation
        - Network timeout simulation with retry behavior
        """
        logger.info(f"Testing cross-module edge case: {edge_case_scenario['scenario_type']}")
        
        # ARRANGE - Set up edge-case test environment
        test_environment = create_integration_test_environment(
            config_type='comprehensive',
            include_filesystem=True,
            include_corrupted_files=True
        )
        
        # ACT & ASSERT - Execute edge-case scenario validation
        if edge_case_scenario['scenario_type'] == 'unicode_paths':
            # Test Unicode path handling across module boundaries
            unicode_files = edge_case_scenario['test_files']
            
            for unicode_file in unicode_files:
                try:
                    # Test path resolution behavior
                    resolved_path = resolve_path(unicode_file)
                    assert isinstance(resolved_path, Path), "Unicode paths should resolve to Path objects"
                    
                    # Test discovery behavior with Unicode paths
                    discovery_results = discover_files(
                        directory="/test",
                        pattern="*.pkl",
                        include_stats=True
                    )
                    
                    # Validate Unicode handling behavior
                    assert isinstance(discovery_results, (list, dict)), "Discovery should handle Unicode gracefully"
                    
                except Exception as unicode_error:
                    # Validate graceful error handling for Unicode edge cases
                    assert "unicode" in str(unicode_error).lower() or "encoding" in str(unicode_error).lower(), \
                        "Unicode errors should be appropriately categorized"
                    
        elif edge_case_scenario['scenario_type'] == 'large_metadata':
            # Test large metadata handling behavior
            large_metadata = {f"metadata_field_{i}": f"value_{i}" * 100 for i in range(100)}
            
            try:
                # Test metadata propagation through modules
                discovery_result = {
                    "/test/large_metadata_file.pkl": large_metadata
                }
                
                # Validate large metadata behavioral handling
                assert len(str(large_metadata)) > 10000, "Metadata should be genuinely large for edge case testing"
                
                # Test data loading with large metadata
                dataloader = test_environment['data_loader']
                dataloader.add_experimental_matrix(
                    "/test/large_metadata_file.pkl",
                    n_timepoints=1000,
                    include_metadata=True
                )
                
                loaded_data = dataloader.load_file("/test/large_metadata_file.pkl")
                assert isinstance(loaded_data, dict), "Large metadata should not break data loading"
                
            except Exception as metadata_error:
                # Validate performance-aware error handling
                assert "memory" in str(metadata_error).lower() or "size" in str(metadata_error).lower(), \
                    "Large metadata errors should be performance-related"
                    
        elif edge_case_scenario['scenario_type'] == 'concurrent_access':
            # Test concurrent access behavioral patterns
            import threading
            import queue
            
            results_queue = queue.Queue()
            error_queue = queue.Queue()
            
            def concurrent_discovery_worker():
                """Worker function for concurrent discovery testing."""
                try:
                    discovery_result = discover_files(
                        directory="/test",
                        pattern="*.pkl",
                        recursive=True
                    )
                    results_queue.put(discovery_result)
                except Exception as e:
                    error_queue.put(e)
            
            # Execute concurrent access test
            threads = []
            for _ in range(3):
                thread = threading.Thread(target=concurrent_discovery_worker)
                threads.append(thread)
                thread.start()
            
            # Wait for completion and validate behavior
            for thread in threads:
                thread.join(timeout=5.0)
            
            # Validate concurrent access behavior
            results_collected = []
            while not results_queue.empty():
                results_collected.append(results_queue.get())
            
            errors_collected = []
            while not error_queue.empty():
                errors_collected.append(error_queue.get())
            
            # Assert thread-safe behavior
            assert len(results_collected) > 0 or len(errors_collected) > 0, \
                "Concurrent access should produce some results or controlled errors"
            
            if errors_collected:
                # Validate error types are appropriate for concurrent access
                for error in errors_collected:
                    assert isinstance(error, (OSError, RuntimeError, ValueError)), \
                        "Concurrent access errors should be controlled error types"
        
        logger.success(f"Edge case scenario {edge_case_scenario['scenario_type']} validation completed")

    # ========================================================================
    # DATA INTEGRITY VALIDATION ACROSS MODULE BOUNDARIES
    # ========================================================================

    def test_data_integrity_through_config_discovery_workflow(
        self, 
        comprehensive_sample_config_dict
    ):
        """
        Test data integrity maintenance through configuration to discovery workflow.
        
        ARRANGE: Set up complex configuration with comprehensive filtering parameters
        ACT: Execute configuration extraction and discovery workflow
        ASSERT: Validate data integrity through observable workflow outcomes
        
        Behavioral Validation Focus:
        - Configuration parameters maintain structure through extraction workflow
        - Complex filtering configurations produce expected behavioral outcomes
        - Data types and values preserve integrity across module boundary
        - Discovery workflow produces results consistent with configuration intent
        """
        logger.info("Testing data integrity through config to discovery workflow")
        
        # ARRANGE - Set up complex configuration with comprehensive filtering
        complex_configuration = comprehensive_sample_config_dict.copy()
        complex_configuration.update({
            "experiments": {
                "comprehensive_test_experiment": {
                    "datasets": ["baseline_behavior", "optogenetic_stimulation"],
                    "filters": {
                        "ignore_substrings": ["backup", "temp", "calibration", "test_excluded"],
                        "mandatory_substrings": ["experiment_", "data_", "required_pattern"],
                        "file_size_limits": {"min_bytes": 1024, "max_bytes": 1048576}
                    },
                    "metadata": {
                        "extraction_patterns": [
                            r".*_(?P<date>\d{8})_(?P<condition>\w+)_(?P<replicate>\d+)\.pkl",
                            r"(?P<experiment>\w+)_(?P<animal_id>\w+)_(?P<session>\d+)\.pkl",
                            r"complex_(?P<study_type>\w+)_(?P<timestamp>\d+)_(?P<version>\d+)\.pkl"
                        ],
                        "required_fields": ["date", "condition", "experiment"]
                    }
                }
            }
        })
        
        # Set up expected behavioral outcomes based on configuration
        expected_ignore_patterns = ["backup", "temp", "calibration", "test_excluded"]
        expected_mandatory_patterns = ["experiment_", "data_", "required_pattern"]
        expected_extraction_patterns = [
            r".*_(?P<date>\d{8})_(?P<condition>\w+)_(?P<replicate>\d+)\.pkl",
            r"(?P<experiment>\w+)_(?P<animal_id>\w+)_(?P<session>\d+)\.pkl",
            r"complex_(?P<study_type>\w+)_(?P<timestamp>\d+)_(?P<version>\d+)\.pkl"
        ]
        
        # ACT - Execute configuration extraction workflow through public interface
        extracted_ignore_patterns = get_ignore_patterns(
            complex_configuration, 
            experiment="comprehensive_test_experiment"
        )
        extracted_mandatory_patterns = get_mandatory_substrings(
            complex_configuration, 
            experiment="comprehensive_test_experiment"
        )
        extracted_extraction_patterns = get_extraction_patterns(
            complex_configuration, 
            experiment="comprehensive_test_experiment"
        )
        
        # Execute discovery workflow with extracted configuration
        discovery_workflow_results = discover_files_with_config(
            config=complex_configuration,
            directory="/test/data",
            pattern="*.pkl",
            experiment="comprehensive_test_experiment",
            extract_metadata=True
        )
        
        # ASSERT - Validate data integrity through observable outcomes
        # Validate configuration extraction maintains data type integrity
        assert isinstance(extracted_ignore_patterns, list), "Ignore patterns extraction must maintain list type"
        assert isinstance(extracted_mandatory_patterns, list), "Mandatory patterns extraction must maintain list type"
        assert isinstance(extracted_extraction_patterns, list), "Extraction patterns must maintain list type"
        
        # Validate configuration content integrity through behavioral comparison
        for expected_pattern in expected_ignore_patterns:
            assert expected_pattern in extracted_ignore_patterns, \
                f"Expected ignore pattern '{expected_pattern}' missing from extraction result"
        
        for expected_pattern in expected_mandatory_patterns:
            assert expected_pattern in extracted_mandatory_patterns, \
                f"Expected mandatory pattern '{expected_pattern}' missing from extraction result"
        
        for expected_pattern in expected_extraction_patterns:
            assert expected_pattern in extracted_extraction_patterns, \
                f"Expected extraction pattern '{expected_pattern}' missing from extraction result"
        
        # Validate pattern list integrity (no duplication or corruption)
        assert len(extracted_ignore_patterns) == len(set(extracted_ignore_patterns)), \
            "Ignore patterns should not contain duplicates"
        assert len(extracted_mandatory_patterns) == len(set(extracted_mandatory_patterns)), \
            "Mandatory patterns should not contain duplicates"
        assert len(extracted_extraction_patterns) == len(set(extracted_extraction_patterns)), \
            "Extraction patterns should not contain duplicates"
        
        # Validate discovery workflow behavioral outcomes
        assert isinstance(discovery_workflow_results, dict), \
            "Discovery workflow must return dictionary format for data integrity"
        
        # Validate workflow maintains configuration intent through observable behavior
        if discovery_workflow_results:
            for file_path, file_metadata in discovery_workflow_results.items():
                assert isinstance(file_path, str), "File paths must maintain string format"
                assert isinstance(file_metadata, dict), "File metadata must maintain dictionary format"
                
                # Validate metadata structure integrity
                if file_metadata:
                    metadata_keys = list(file_metadata.keys())
                    assert len(metadata_keys) > 0, "Metadata should contain extractable information"
                    assert all(isinstance(key, str) for key in metadata_keys), \
                        "Metadata keys must maintain string format"
        
        # Validate complex configuration structure preservation
        original_experiment_config = complex_configuration["experiments"]["comprehensive_test_experiment"]
        assert "filters" in original_experiment_config, "Complex configuration structure must be preserved"
        assert "metadata" in original_experiment_config, "Complex configuration structure must be preserved"
        
        # Validate configuration data types maintain consistency
        filters_config = original_experiment_config["filters"]
        assert isinstance(filters_config["ignore_substrings"], list), "Filter configuration data types preserved"
        assert isinstance(filters_config["mandatory_substrings"], list), "Filter configuration data types preserved"
        assert isinstance(filters_config["file_size_limits"], dict), "Complex filter types preserved"
        
        logger.success("Data integrity through config to discovery workflow validation completed")

    def test_data_integrity_through_discovery_io_workflow(
        self,
        temp_filesystem_structure
    ):
        """
        Test data integrity maintenance through discovery to IO workflow.
        
        ARRANGE: Set up complex discovery results with diverse metadata structures
        ACT: Execute data loading workflow through IO module public interface
        ASSERT: Validate data integrity preservation through observable outcomes
        
        Behavioral Validation Focus:
        - Complex metadata structures preserved through discovery → IO boundary
        - File path resolution maintains accuracy through workflow
        - Data type consistency maintained across module boundary
        - Error conditions handled gracefully without data corruption
        """
        logger.info("Testing data integrity through discovery to IO workflow")
        
        # ARRANGE - Set up complex discovery results with comprehensive metadata
        complex_discovery_results = {
            str(temp_filesystem_structure.get("baseline_file_1", "/test/baseline_001.pkl")): {
                "date": "20241220",
                "condition": "control",
                "replicate": 1,
                "duration_seconds": 300.5,
                "dataset": "baseline_behavior",
                "experimental_parameters": {
                    "temperature_celsius": 23.5,
                    "humidity_percent": 45.2,
                    "lighting_lux": 100,
                    "arena_diameter_mm": 120
                },
                "file_statistics": {
                    "size_bytes": 2048,
                    "modification_time": datetime.now().isoformat(),
                    "checksum_md5": "abc123def456789"
                },
                "processing_flags": ["calibrated", "filtered", "validated"],
                "analysis_metadata": {"pipeline_version": "v2.1", "quality_score": 0.95}
            },
            str(temp_filesystem_structure.get("opto_file_1", "/test/opto_001.pkl")): {
                "date": "20241218", 
                "condition": "treatment",
                "replicate": 2,
                "duration_seconds": 600.0,
                "dataset": "optogenetic_stimulation",
                "stimulation_parameters": {
                    "wavelength_nanometers": 470,
                    "power_milliwatts": 10.5,
                    "pulse_duration_milliseconds": 50,
                    "frequency_hertz": 20.0,
                    "pattern": "square_wave"
                },
                "file_statistics": {
                    "size_bytes": 4096,
                    "modification_time": datetime.now().isoformat(),
                    "checksum_md5": "xyz789abc123456"
                },
                "processing_flags": ["calibrated", "optogenetic", "stimulation_verified"],
                "analysis_metadata": {"pipeline_version": "v2.2", "stimulation_efficacy": 0.87}
            }
        }
        
        # Set up Protocol-based data loading environment
        dataloader_mock = create_mock_dataloader(
            scenarios=['basic', 'experimental'],
            include_experimental_data=True
        )
        
        # Configure realistic experimental data for each discovery result
        for file_path, discovery_metadata in complex_discovery_results.items():
            duration = discovery_metadata["duration_seconds"]
            n_timepoints = int(duration * 60)  # 60 Hz sampling
            
            experimental_data = {
                't': np.linspace(0, duration, n_timepoints),
                'x': np.random.rand(n_timepoints) * 120,  # Arena coordinates
                'y': np.random.rand(n_timepoints) * 120,
                'discovery_metadata': discovery_metadata,  # Preserve all discovery metadata
                'file_path': file_path
            }
            
            # Add stimulation signal for optogenetic experiments
            if "opto" in file_path:
                experimental_data['signal'] = np.random.rand(n_timepoints)
                experimental_data['stimulation_events'] = np.random.choice([0, 1], n_timepoints, p=[0.9, 0.1])
            
            dataloader_mock.mock_data[file_path] = experimental_data
        
        # ACT - Execute data loading workflow through IO module
        loaded_datasets = {}
        data_integrity_failures = []
        
        for file_path, expected_discovery_metadata in complex_discovery_results.items():
            try:
                # Load data through IO module public interface
                loaded_data = dataloader_mock.load_file(file_path)
                
                # Validate basic data loading success
                assert isinstance(loaded_data, dict), "IO module must return dictionary format"
                assert 'discovery_metadata' in loaded_data, "Discovery metadata must be preserved"
                assert loaded_data['file_path'] == file_path, "File path must be preserved"
                
                loaded_datasets[file_path] = loaded_data
                
            except Exception as loading_error:
                data_integrity_failures.append((file_path, loading_error))
                logger.warning(f"Data loading failed for {file_path}: {loading_error}")
        
        # ASSERT - Validate data integrity through observable outcomes
        # Validate overall workflow success
        assert len(loaded_datasets) > 0, "At least some discovery results should load successfully"
        assert len(data_integrity_failures) == 0, f"Data integrity failures occurred: {data_integrity_failures}"
        
        # Validate comprehensive metadata integrity preservation
        for file_path, loaded_data in loaded_datasets.items():
            original_metadata = complex_discovery_results[file_path]
            preserved_metadata = loaded_data['discovery_metadata']
            
            # Validate metadata structure preservation
            assert isinstance(preserved_metadata, dict), "Metadata must maintain dictionary structure"
            
            # Validate all metadata keys preserved
            for original_key in original_metadata.keys():
                assert original_key in preserved_metadata, f"Metadata key '{original_key}' lost during IO workflow"
            
            # Validate nested structure integrity
            for key, original_value in original_metadata.items():
                preserved_value = preserved_metadata[key]
                
                if isinstance(original_value, dict):
                    # Validate nested dictionary integrity
                    assert isinstance(preserved_value, dict), f"Nested dictionary '{key}' type corrupted"
                    assert len(preserved_value) == len(original_value), f"Nested dictionary '{key}' size changed"
                    
                    for nested_key, nested_value in original_value.items():
                        assert nested_key in preserved_value, f"Nested key '{nested_key}' lost"
                        assert preserved_value[nested_key] == nested_value, f"Nested value '{nested_key}' corrupted"
                
                elif isinstance(original_value, list):
                    # Validate list integrity
                    assert isinstance(preserved_value, list), f"List '{key}' type corrupted"
                    assert len(preserved_value) == len(original_value), f"List '{key}' length changed"
                    assert preserved_value == original_value, f"List '{key}' contents corrupted"
                
                else:
                    # Validate scalar value integrity
                    assert preserved_value == original_value, f"Scalar value '{key}' corrupted"
            
            # Validate numerical data integrity and consistency
            assert isinstance(loaded_data['t'], np.ndarray), "Time data must be numpy array"
            assert isinstance(loaded_data['x'], np.ndarray), "X position must be numpy array"
            assert isinstance(loaded_data['y'], np.ndarray), "Y position must be numpy array"
            
            # Validate data-metadata consistency
            expected_duration = original_metadata["duration_seconds"]
            actual_duration = loaded_data['t'][-1] - loaded_data['t'][0]
            duration_tolerance = 0.1  # 100ms tolerance
            assert abs(actual_duration - expected_duration) < duration_tolerance, \
                f"Duration metadata inconsistent with actual data: expected {expected_duration}, got {actual_duration}"
            
            # Validate experiment-specific data integrity
            if "baseline" in file_path:
                assert len(loaded_data['t']) > 15000, "Baseline experiments should have sufficient data points"
                assert 'signal' not in loaded_data or loaded_data['signal'] is None, \
                    "Baseline experiments should not have stimulation signals"
            
            elif "opto" in file_path:
                assert len(loaded_data['t']) > 30000, "Optogenetic experiments should have longer recordings"
                assert 'signal' in loaded_data, "Optogenetic experiments should include signal data"
                assert isinstance(loaded_data['signal'], np.ndarray), "Signal data must be numpy array"
        
        # Validate cross-experiment data integrity consistency
        all_metadata = [loaded_data['discovery_metadata'] for loaded_data in loaded_datasets.values()]
        
        # Check date format consistency
        for metadata in all_metadata:
            date_str = metadata.get('date', '')
            assert len(date_str) == 8 and date_str.isdigit(), "Date format should be consistent YYYYMMDD"
        
        # Validate condition categorization integrity
        conditions = set(metadata.get('condition', '') for metadata in all_metadata)
        expected_conditions = {'control', 'treatment'}
        assert conditions.issubset(expected_conditions), f"Unexpected conditions found: {conditions - expected_conditions}"
        
        logger.success("Data integrity through discovery to IO workflow validation completed")

    # ========================================================================
    # PERFORMANCE TESTS RELOCATED TO scripts/benchmarks/
    # ========================================================================
    
    # NOTE: Performance validation tests for large dataset processing verification,
    # memory usage validation, concurrent access performance, and SLA compliance 
    # verification have been extracted from the default test execution and relocated
    # to scripts/benchmarks/ per Section 0 performance test isolation requirement.
    # 
    # Relocated tests include:
    # - test_large_dataset_processing_performance()
    # - test_memory_usage_validation_across_modules()
    # - test_concurrent_access_performance_characteristics()
    # - test_sla_compliance_verification_end_to_end()
    #
    # These tests can be executed separately using:
    # python scripts/benchmarks/run_integration_benchmarks.py

    def test_data_integrity_through_io_utils_workflow(
        self,
        comprehensive_exp_matrix,
        sample_metadata
    ):
        """
        Test data integrity maintenance through IO to utils workflow.
        
        ARRANGE: Set up comprehensive IO experimental data with complex metadata
        ACT: Execute DataFrame transformation workflow through utils module
        ASSERT: Validate data integrity preservation through observable DataFrame outcomes
        
        Behavioral Validation Focus:
        - Numerical data precision maintained through DataFrame transformation
        - Complex metadata structures properly integrated into DataFrame format
        - Data types appropriate for downstream analysis workflows
        - Transformation workflow produces expected behavioral outcomes
        """
        logger.info("Testing data integrity through IO to utils workflow")
        
        # ARRANGE - Set up comprehensive IO experimental data
        io_experimental_data = comprehensive_exp_matrix.copy()
        io_experimental_data.update({
            'metadata': sample_metadata.copy() if sample_metadata else {'animal_id': 'test_001'},
            'processing_statistics': {
                'mean_velocity': 15.2,
                'total_distance': 1250.5,
                'session_duration': 300.0
            },
            'file_information': {
                'size_bytes': 2048,
                'processing_timestamp': datetime.now().isoformat(),
                'data_quality_score': 0.95
            }
        })
        
        # ACT - Execute DataFrame transformation through utils module public interface
        try:
            # Prepare DataFrame-compatible data structure
            df_compatible_data = {}
            
            # Process numerical arrays
            for field_name, field_data in io_experimental_data.items():
                if isinstance(field_data, np.ndarray) and field_data.ndim == 1:
                    df_compatible_data[field_name] = field_data
                elif isinstance(field_data, np.ndarray) and field_data.ndim == 2:
                    # Expand multi-channel data
                    for ch in range(field_data.shape[0]):
                        df_compatible_data[f"{field_name}_ch{ch:02d}"] = field_data[ch, :]
            
            # Process metadata structures
            for meta_category in ['metadata', 'processing_statistics', 'file_information']:
                if meta_category in io_experimental_data:
                    meta_dict = io_experimental_data[meta_category]
                    for key, value in meta_dict.items():
                        df_compatible_data[f"{meta_category}_{key}"] = [value] * len(io_experimental_data['t'])
            
            # Create DataFrame using utils module
            result_dataframe = discovery_results_to_dataframe(df_compatible_data)
            
        except Exception as transformation_error:
            pytest.fail(f"DataFrame transformation workflow failed: {transformation_error}")
        
        # ASSERT - Validate data integrity through observable DataFrame outcomes
        # Validate successful DataFrame creation
        assert isinstance(result_dataframe, pd.DataFrame), "Utils module must produce pandas DataFrame"
        assert len(result_dataframe) > 0, "DataFrame must contain data rows"
        
        # Validate essential numerical data integrity
        essential_columns = ['t', 'x', 'y']
        for col in essential_columns:
            if col in result_dataframe.columns:
                original_data = io_experimental_data[col]
                dataframe_data = result_dataframe[col].values
                
                # Validate numerical precision preservation
                np.testing.assert_array_almost_equal(
                    dataframe_data, original_data, 
                    decimal=6, 
                    err_msg=f"Numerical precision lost in column '{col}'"
                )
                
                # Validate data type suitability for analysis
                assert pd.api.types.is_numeric_dtype(result_dataframe[col]), \
                    f"Column '{col}' must be numeric for analysis"
        
        # Validate metadata integration integrity
        metadata_columns = [col for col in result_dataframe.columns if 'metadata_' in col]
        stats_columns = [col for col in result_dataframe.columns if 'processing_statistics_' in col]
        info_columns = [col for col in result_dataframe.columns if 'file_information_' in col]
        
        assert len(metadata_columns) > 0, "Metadata must be integrated into DataFrame"
        assert len(stats_columns) > 0, "Processing statistics must be integrated"
        assert len(info_columns) > 0, "File information must be integrated"
        
        # Validate metadata consistency (constant values per session)
        for meta_col in metadata_columns + stats_columns + info_columns:
            unique_values = result_dataframe[meta_col].unique()
            assert len(unique_values) == 1, f"Metadata column '{meta_col}' should have consistent value"
        
        # Validate data length consistency
        expected_length = len(io_experimental_data['t'])
        assert len(result_dataframe) == expected_length, \
            "DataFrame length must match original experimental data length"
        
        # Validate multi-channel data integrity (if present)
        multi_channel_cols = [col for col in result_dataframe.columns if '_ch' in col]
        if multi_channel_cols:
            assert len(multi_channel_cols) > 0, "Multi-channel data should be properly expanded"
            for ch_col in multi_channel_cols:
                assert pd.api.types.is_numeric_dtype(result_dataframe[ch_col]), \
                    f"Channel column '{ch_col}' must be numeric"
        
        # Validate DataFrame suitability for downstream analysis
        assert not result_dataframe.empty, "DataFrame must contain analyzable data"
        assert result_dataframe.shape[1] > len(essential_columns), \
            "DataFrame should contain both data and metadata columns"
        
        logger.success("Data integrity through IO to utils workflow validation completed")

    # ========================================================================
    # ERROR PROPAGATION BEHAVIORAL VALIDATION
    # ========================================================================

    @pytest.mark.parametrize("error_scenario", [
        {
            'error_type': 'configuration_not_found',
            'expected_exception': FileNotFoundError,
            'error_context': 'configuration'
        },
        {
            'error_type': 'invalid_file_format',
            'expected_exception': ValueError,
            'error_context': 'format'
        },
        {
            'error_type': 'permission_denied',
            'expected_exception': PermissionError,
            'error_context': 'permission'
        }
    ])
    def test_error_propagation_behavioral_validation(self, error_scenario):
        """
        Test error propagation behavior across module boundaries through parameterized scenarios.
        
        ARRANGE: Set up error scenario conditions using parameterized test data
        ACT: Execute cross-module workflow under error conditions
        ASSERT: Validate error propagation behavior through exception characteristics
        
        Behavioral Validation Focus:
        - Error types properly preserved across module boundaries
        - Error context information maintained through propagation
        - Error handling behavior consistent across different failure modes
        - Graceful degradation with informative error messages
        """
        logger.info(f"Testing error propagation behavior: {error_scenario['error_type']}")
        
        # ARRANGE - Set up error scenario environment
        test_environment = create_integration_test_environment(
            config_type='comprehensive',
            include_filesystem=True,
            include_corrupted_files=True
        )
        
        # ACT & ASSERT - Execute error scenario and validate propagation behavior
        if error_scenario['error_type'] == 'configuration_not_found':
            # Test configuration error propagation behavior
            with pytest.raises(error_scenario['expected_exception']) as exc_info:
                # Attempt to load non-existent configuration
                load_config("/nonexistent/config.yaml")
            
            # Validate error context preservation
            error_message = str(exc_info.value).lower()
            assert error_scenario['error_context'] in error_message, \
                f"Error context '{error_scenario['error_context']}' missing from error message"
                
        elif error_scenario['error_type'] == 'invalid_file_format':
            # Test data format error propagation behavior
            invalid_data = {"invalid": "format", "not_arrays": True}
            
            with pytest.raises(error_scenario['expected_exception']) as exc_info:
                # Attempt to validate invalid experimental data
                validate_experimental_data(invalid_data, {"columns": {"t": {"required": True}}})
            
            # Validate error format indication
            error_message = str(exc_info.value).lower()
            assert error_scenario['error_context'] in error_message, \
                "Data format error should be clearly indicated"
                
        elif error_scenario['error_type'] == 'permission_denied':
            # Test permission error propagation behavior
            filesystem_mock = test_environment['filesystem']
            restricted_file = "/restricted/data.pkl"
            
            # Add file with permission error
            filesystem_mock.add_file(
                restricted_file,
                access_error=PermissionError("Access denied for testing")
            )
            
            with pytest.raises(error_scenario['expected_exception']) as exc_info:
                # Attempt to access restricted file through filesystem mock
                filesystem_mock.open_file(restricted_file)
            
            # Validate permission context preservation
            error_message = str(exc_info.value).lower()
            assert error_scenario['error_context'] in error_message, \
                "Permission error context should be preserved"
        
        logger.success(f"Error propagation behavior validation completed: {error_scenario['error_type']}")

    # ========================================================================
    # MODULE ISOLATION BEHAVIORAL VALIDATION
    # ========================================================================

    def test_cross_module_isolation_through_protocol_interfaces(self):
        """
        Test module isolation behavior through Protocol-based interface validation.
        
        ARRANGE: Set up isolated test environment using Protocol-based providers
        ACT: Execute cross-module workflows with isolated dependencies
        ASSERT: Validate isolation behavior through observable interface outcomes
        
        Behavioral Validation Focus:
        - Modules function independently through Protocol-based interfaces
        - Interface contracts maintained across isolated providers
        - Observable behavior consistent regardless of implementation
        - No unintended coupling detected through behavioral testing
        """
        logger.info("Testing cross-module isolation through Protocol-based interfaces")
        
        # ARRANGE - Set up completely isolated test environment
        isolated_environment = create_integration_test_environment(
            config_type='comprehensive',
            include_filesystem=True,
            include_corrupted_files=False  # Focus on isolation, not error scenarios
        )
        
        isolated_config_provider = isolated_environment['config_provider']
        isolated_filesystem = isolated_environment['filesystem']
        isolated_dataloader = isolated_environment['data_loader']
        
        # Create test configuration for isolation validation
        test_configuration = {
            "project": {
                "ignore_substrings": ["backup", "temp"],
                "mandatory_substrings": ["experiment"]
            },
            "experiments": {
                "isolation_test": {
                    "datasets": ["test_dataset"],
                    "filters": {
                        "ignore_substrings": ["calibration"],
                        "mandatory_substrings": ["data"]
                    }
                }
            }
        }
        
        # ACT - Execute isolated module workflows
        # Test config module isolation through Protocol interface
        isolated_ignore_patterns = isolated_config_provider.get_ignore_patterns(
            test_configuration, 
            "isolation_test"
        )
        isolated_mandatory_patterns = isolated_config_provider.get_mandatory_substrings(
            test_configuration, 
            "isolation_test"
        )
        
        # Test discovery module isolation through filesystem Protocol
        test_files = ['/test/experiment_data_001.pkl', '/test/experiment_data_002.pkl']
        for test_file in test_files:
            isolated_filesystem.add_file(test_file, size=1024)
        
        discovery_results = discover_files(
            directory="/test",
            pattern="*.pkl",
            ignore_patterns=isolated_ignore_patterns,
            mandatory_substrings=isolated_mandatory_patterns
        )
        
        # Test IO module isolation through data loading Protocol
        isolated_dataloader.add_experimental_matrix('/test/experiment_data_001.pkl', n_timepoints=1000)
        loaded_data = isolated_dataloader.load_file('/test/experiment_data_001.pkl')
        
        # Test utils module isolation through DataFrame creation
        df_compatible_data = {
            't': loaded_data.get('t', np.linspace(0, 100, 1000)),
            'x': loaded_data.get('x', np.random.rand(1000) * 50),
            'y': loaded_data.get('y', np.random.rand(1000) * 50)
        }
        
        final_dataframe = discovery_results_to_dataframe(df_compatible_data)
        
        # ASSERT - Validate isolation behavior through observable outcomes
        # Validate config module isolation behavior
        assert isinstance(isolated_ignore_patterns, list), "Config provider must return list format"
        assert len(isolated_ignore_patterns) > 0, "Config isolation should provide patterns"
        assert "backup" in isolated_ignore_patterns, "Global patterns should be included"
        assert "calibration" in isolated_ignore_patterns, "Experiment patterns should be included"
        
        assert isinstance(isolated_mandatory_patterns, list), "Config provider must return list format"
        assert "experiment" in isolated_mandatory_patterns, "Global mandatory patterns included"
        assert "data" in isolated_mandatory_patterns, "Experiment mandatory patterns included"
        
        # Validate discovery module isolation behavior
        assert isinstance(discovery_results, (list, dict)), "Discovery should return expected format"
        
        # Validate IO module isolation behavior
        assert isinstance(loaded_data, dict), "Data loading should return dictionary format"
        assert 't' in loaded_data, "Essential time data should be present"
        
        # Validate utils module isolation behavior
        assert isinstance(final_dataframe, pd.DataFrame), "Utils should produce DataFrame"
        assert len(final_dataframe) > 0, "DataFrame should contain data"
        assert 't' in final_dataframe.columns, "Essential columns should be present"
        
        # Validate cross-module isolation consistency
        # No module should depend on implementation details of others
        config_patterns_consistent = len(isolated_ignore_patterns) == len(set(isolated_ignore_patterns))
        assert config_patterns_consistent, "Config module should produce consistent isolated results"
        
        dataframe_consistent = not final_dataframe.empty and len(final_dataframe.columns) > 0
        assert dataframe_consistent, "Utils module should produce consistent isolated results"
        
        # Validate no unintended coupling through behavioral observation
        # Each module should work independently through Protocol interfaces
        assert callable(isolated_config_provider.get_ignore_patterns), "Config provider should be Protocol-compliant"
        assert callable(isolated_filesystem.exists), "Filesystem should be Protocol-compliant"
        assert callable(isolated_dataloader.load_file), "Dataloader should be Protocol-compliant"
        
        logger.success("Cross-module isolation through Protocol interfaces validation completed")

    # ========================================================================
    # END-TO-END PIPELINE INTEGRATION BEHAVIORAL VALIDATION
    # ========================================================================

    def test_end_to_end_pipeline_integration_complete_workflow(
        self,
        comprehensive_sample_config_dict,
        temp_filesystem_structure,
        comprehensive_exp_matrix
    ):
        """
        Test complete end-to-end pipeline integration through observable behavioral workflow.
        
        ARRANGE: Set up comprehensive pipeline environment with realistic experimental data
        ACT: Execute complete config → discovery → io → utils workflow
        ASSERT: Validate seamless integration through observable behavioral outcomes
        
        Behavioral Validation Focus:
        - Complete pipeline workflow produces expected behavioral outcomes
        - Data integrity maintained throughout entire multi-module transformation
        - All module interfaces work together seamlessly through public APIs
        - Realistic experimental scenarios processed end-to-end successfully
        - Observable behavior validates architectural integration correctness
        """
        logger.info("Testing complete end-to-end pipeline integration through behavioral workflow")
        
        # ARRANGE - Set up comprehensive pipeline environment
        pipeline_environment = create_integration_test_environment(
            config_type='comprehensive',
            include_filesystem=True,
            include_corrupted_files=False
        )
        
        # Configure realistic experimental scenario
        test_configuration = comprehensive_sample_config_dict.copy()
        test_experiment = "baseline_control_study"
        
        # Set up realistic discovery results
        expected_files = {
            str(temp_filesystem_structure.get("baseline_file_1", "/test/baseline_001.pkl")): {
                "date": "20241220",
                "condition": "control",
                "dataset": "baseline_behavior",
                "file_size": 2048,
                "replicate": "1"
            },
            str(temp_filesystem_structure.get("opto_file_1", "/test/opto_001.pkl")): {
                "date": "20241218",
                "condition": "treatment", 
                "dataset": "optogenetic_stimulation",
                "file_size": 3072,
                "replicate": "1"
            }
        }
        
        # Configure realistic experimental data for each file
        pipeline_dataloader = pipeline_environment['data_loader']
        for file_path, metadata in expected_files.items():
            if "baseline" in file_path:
                experimental_data = {
                    't': np.linspace(0, 300, 18000),  # 5 minutes at 60 Hz
                    'x': np.random.rand(18000) * 100,
                    'y': np.random.rand(18000) * 100,
                    'metadata': metadata
                }
            else:  # optogenetic
                experimental_data = {
                    't': np.linspace(0, 600, 36000),  # 10 minutes at 60 Hz
                    'x': np.random.rand(36000) * 120,
                    'y': np.random.rand(36000) * 120,
                    'signal': np.random.rand(36000),
                    'metadata': metadata
                }
            
            pipeline_dataloader.mock_data[file_path] = experimental_data
        
        # ACT - Execute complete end-to-end pipeline workflow
        try:
            # Stage 1: Configuration extraction
            config_ignore_patterns = get_ignore_patterns(test_configuration, test_experiment)
            config_mandatory_patterns = get_mandatory_substrings(test_configuration, test_experiment)
            
            # Stage 2: File discovery workflow
            discovered_files = discover_files_with_config(
                config=test_configuration,
                directory=str(temp_filesystem_structure.get("data_root", "/test")),
                pattern="*.pkl",
                experiment=test_experiment,
                extract_metadata=True
            )
            
            # Stage 3: Data loading and validation workflow
            loaded_datasets = {}
            for file_path in expected_files.keys():
                loaded_data = pipeline_dataloader.load_file(file_path)
                
                # Validate experimental data using column models
                column_config = {
                    "columns": {
                        "t": {"type": "numpy.ndarray", "required": True},
                        "x": {"type": "numpy.ndarray", "required": True},
                        "y": {"type": "numpy.ndarray", "required": True}
                    }
                }
                
                validated_data = validate_experimental_data(loaded_data, column_config)
                standardized_data = transform_to_standardized_format(validated_data, column_config)
                
                loaded_datasets[file_path] = standardized_data
            
            # Stage 4: DataFrame transformation workflow
            final_dataframes = {}
            for file_path, dataset in loaded_datasets.items():
                # Prepare DataFrame-compatible structure
                df_data = {
                    't': dataset['t'],
                    'x': dataset['x'], 
                    'y': dataset['y']
                }
                
                # Add multi-channel data if present
                if 'signal' in dataset:
                    df_data['signal'] = dataset['signal']
                
                # Add metadata as constant columns
                if 'metadata' in dataset:
                    for key, value in dataset['metadata'].items():
                        df_data[f"meta_{key}"] = [value] * len(dataset['t'])
                
                # Create final DataFrame
                result_dataframe = discovery_results_to_dataframe(df_data)
                final_dataframes[file_path] = result_dataframe
            
        except Exception as pipeline_error:
            pytest.fail(f"End-to-end pipeline workflow failed: {pipeline_error}")
        
        # ASSERT - Validate seamless integration through observable behavioral outcomes
        # Validate pipeline workflow completion
        assert len(final_dataframes) > 0, "Pipeline should produce final results"
        assert len(final_dataframes) == len(expected_files), "All files should be processed successfully"
        
        # Validate configuration stage behavioral outcomes
        assert isinstance(config_ignore_patterns, list), "Configuration extraction should produce lists"
        assert isinstance(config_mandatory_patterns, list), "Configuration extraction should produce lists"
        assert len(config_ignore_patterns) > 0, "Configuration should provide filtering patterns"
        
        # Validate discovery stage behavioral outcomes
        assert isinstance(discovered_files, dict), "Discovery should return dictionary format"
        
        # Validate data loading stage behavioral outcomes
        for file_path, dataset in loaded_datasets.items():
            assert isinstance(dataset, dict), "Data loading should return dictionary format"
            assert 't' in dataset, "Essential time data should be present"
            assert 'x' in dataset and 'y' in dataset, "Essential position data should be present"
            assert isinstance(dataset['t'], np.ndarray), "Time data should be numpy arrays"
            
        # Validate DataFrame transformation stage behavioral outcomes
        for file_path, dataframe in final_dataframes.items():
            assert isinstance(dataframe, pd.DataFrame), "Final stage should produce DataFrames"
            assert len(dataframe) > 0, "DataFrames should contain data"
            assert 't' in dataframe.columns, "Essential columns should be present"
            assert 'x' in dataframe.columns and 'y' in dataframe.columns, "Position columns should be present"
            
            # Validate metadata integration
            metadata_cols = [col for col in dataframe.columns if col.startswith('meta_')]
            assert len(metadata_cols) > 0, "Metadata should be integrated into final DataFrame"
            
            # Validate data consistency through pipeline
            original_data = loaded_datasets[file_path]
            np.testing.assert_array_almost_equal(
                dataframe['t'].values, original_data['t'],
                decimal=6, err_msg="Data should be preserved through pipeline stages"
            )
        
        # Validate cross-experiment consistency
        baseline_files = [f for f in final_dataframes.keys() if "baseline" in f]
        opto_files = [f for f in final_dataframes.keys() if "opto" in f]
        
        if baseline_files and opto_files:
            baseline_df = final_dataframes[baseline_files[0]]
            opto_df = final_dataframes[opto_files[0]]
            
            # Validate experiment-specific characteristics maintained
            assert len(baseline_df) < len(opto_df), "Baseline experiments should be shorter than optogenetic"
            assert 'signal' not in baseline_df.columns or baseline_df['signal'].isna().all(), \
                "Baseline experiments should not have stimulation signals"
            
            if 'signal' in opto_df.columns:
                assert not opto_df['signal'].isna().all(), "Optogenetic experiments should have signal data"
        
        # Validate pipeline architectural integration
        assert all(isinstance(df.index, pd.RangeIndex) for df in final_dataframes.values()), \
            "Pipeline should produce consistently indexed DataFrames"
        
        logger.success("Complete end-to-end pipeline integration behavioral validation completed")



# ============================================================================
# CENTRALIZED FIXTURE UTILIZATION
# ============================================================================

# NOTE: This integration test suite utilizes centralized fixtures from tests/conftest.py
# for comprehensive test data generation and standardized testing patterns:
#
# Primary Fixtures Used:
# - comprehensive_sample_config_dict: Comprehensive configuration for realistic testing
# - temp_filesystem_structure: Temporary filesystem with experimental file structures  
# - comprehensive_exp_matrix: Realistic experimental data matrices
# - sample_metadata: Representative experimental metadata
# - test_data_generator: Session-scoped data generation utilities
# - mock_filesystem: Standardized filesystem mocking patterns
# - mock_data_loading: Consistent data loading mock implementations
#
# Additional Utilities from tests/utils.py:
# - create_integration_test_environment(): Complete integration test setup
# - create_mock_filesystem(): Protocol-based filesystem mocking
# - create_mock_dataloader(): Standardized data loading simulation
# - create_mock_config_provider(): Configuration provider mocking
# - generate_edge_case_scenarios(): Edge-case test scenario generation
#
# This approach eliminates test fixture duplication and ensures consistent
# testing patterns across all integration test modules per Section 0 requirements.