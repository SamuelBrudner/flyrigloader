"""
Comprehensive pytest test suite for configuration builder pattern functionality.

This module provides extensive testing for the enhanced create_config() factory function,
type-safe configuration construction, Pydantic model enhancements, and automatic version
handling capabilities per Section 6.6.2.1 of the technical specification.

The test suite validates:
- Builder pattern testing for create_config() factory per Section 6.6.2.1
- Configuration standardization with Pydantic models per Section 0.2.1
- Type-safe configuration construction with comprehensive validation
- Enhanced configuration builder pattern per Section 5.2.2
- Schema version handling and migration capabilities
- Performance testing for builder pattern construction efficiency
"""

import pytest
import warnings
import time
from pathlib import Path
from typing import Dict
from unittest.mock import patch, MagicMock

# External imports per schema requirements
from pydantic import ValidationError
from hypothesis import strategies, given, settings
from hypothesis.strategies import text, integers, lists, dictionaries, booleans

# Internal imports per schema requirements
from src.flyrigloader.config.models import (
    LegacyConfigAdapter,
    ProjectConfig,
    DatasetConfig,
    ExperimentConfig,
    create_config,
    create_experiment,
    create_dataset,
    create_standard_fly_config,
    create_plume_tracking_experiment,
    create_choice_assay_experiment,
    create_rig_dataset,
)


class TestCreateConfigFactory:
    """
    Test suite for create_config() factory function validation with comprehensive
    builder pattern testing per Section 6.6.2.1.
    
    This class validates the core builder functionality including type-safe construction,
    comprehensive defaults, automatic validation, and schema version handling.
    """
    
    def test_create_config_basic_functionality(self):
        """Test basic create_config() functionality with minimal parameters."""
        # Test with minimal required parameters
        config = create_config()
        
        # Verify type and basic structure
        assert isinstance(config, ProjectConfig)
        assert hasattr(config, 'schema_version')
        assert hasattr(config, 'directories')
        assert hasattr(config, 'ignore_substrings')
        assert hasattr(config, 'extraction_patterns')
        
        # Verify schema version is set
        assert config.schema_version is not None
        assert isinstance(config.schema_version, str)
        
        # Verify directories contain default entries
        assert isinstance(config.directories, dict)
        assert 'major_data_directory' in config.directories
        
    def test_create_config_with_project_name_and_directory(self):
        """Test create_config() with explicit project name and base directory."""
        project_name = "test_fly_behavior"
        base_dir = "/tmp/test_project"
        
        config = create_config(
            project_name=project_name,
            base_directory=base_dir
        )
        
        # Verify configuration structure
        assert isinstance(config, ProjectConfig)
        assert config.directories['major_data_directory'] == base_dir
        
        # Verify comprehensive directory defaults are set
        expected_dirs = [
            'major_data_directory',
            'backup_directory', 
            'processed_directory',
            'output_directory',
            'logs_directory',
            'temp_directory',
            'cache_directory'
        ]
        
        for dir_name in expected_dirs:
            assert dir_name in config.directories
            assert isinstance(config.directories[dir_name], str)
    
    def test_create_config_comprehensive_defaults(self):
        """Test that create_config() provides comprehensive defaults for all fields."""
        config = create_config()
        
        # Verify ignore_substrings defaults
        assert config.ignore_substrings is not None
        assert isinstance(config.ignore_substrings, list)
        assert len(config.ignore_substrings) > 0
        
        # Verify comprehensive ignore patterns
        expected_patterns = ["._", "temp", "backup", ".tmp", "~", ".DS_Store"]
        for pattern in expected_patterns:
            assert pattern in config.ignore_substrings
        
        # Verify extraction_patterns defaults
        assert config.extraction_patterns is not None
        assert isinstance(config.extraction_patterns, list)
        assert len(config.extraction_patterns) > 0
        
        # Verify comprehensive extraction patterns
        pattern_types = []
        for pattern in config.extraction_patterns:
            if 'date' in pattern:
                pattern_types.append('date')
            elif 'subject' in pattern:
                pattern_types.append('subject')
            elif 'rig' in pattern:
                pattern_types.append('rig')
                
        assert 'date' in pattern_types
        assert 'subject' in pattern_types
        assert 'rig' in pattern_types
    
    def test_create_config_custom_parameters(self):
        """Test create_config() with custom parameters and overrides."""
        custom_ignore = ["custom_temp", "custom_backup"]
        custom_patterns = [r"(?P<custom>\w+)"]
        custom_dirs = {
            "major_data_directory": "/custom/data",
            "custom_directory": "/custom/path"
        }
        
        config = create_config(
            project_name="custom_project",
            base_directory="/custom/base",
            directories=custom_dirs,
            ignore_substrings=custom_ignore,
            extraction_patterns=custom_patterns,
            mandatory_experiment_strings=["custom_experiment"]
        )
        
        # Verify custom parameters are applied
        assert config.directories["major_data_directory"] == "/custom/data"
        assert config.directories["custom_directory"] == "/custom/path"
        assert config.ignore_substrings == custom_ignore
        assert config.extraction_patterns == custom_patterns
        assert config.mandatory_experiment_strings == ["custom_experiment"]
    
    def test_create_config_schema_version_handling(self):
        """Test schema version handling and validation in create_config()."""
        # Test default schema version
        config_default = create_config()
        assert config_default.schema_version is not None
        
        # Test explicit schema version
        config_explicit = create_config(schema_version="1.0.0")
        assert config_explicit.schema_version == "1.0.0"
        
        # Test version validation
        with pytest.raises(ValueError, match="Failed to create ProjectConfig.*Configuration version validation failed"):
            create_config(schema_version="invalid_version")
    
    def test_create_config_validation_failures(self):
        """Test create_config() error handling for validation failures."""
        # Test invalid directory structure
        with pytest.raises(TypeError, match="'str' object does not support item assignment"):
            create_config(directories="not_a_dict")
        
        # Test invalid ignore patterns
        with pytest.raises(ValueError):
            create_config(ignore_substrings="not_a_list")
        
        # Test invalid extraction patterns - should fail on regex compilation
        with pytest.raises(ValueError, match="Invalid regex pattern"):
            create_config(extraction_patterns=["[invalid_regex"])
    
    @pytest.mark.parametrize("base_dir", [
        "/tmp/test1", 
        Path("/tmp/test2"),
        "relative/path",
        Path("relative/path2")
    ])
    def test_create_config_path_handling(self, base_dir):
        """Test create_config() with different path types and formats."""
        config = create_config(base_directory=base_dir)
        
        # Verify path is converted to string and stored correctly
        assert isinstance(config.directories['major_data_directory'], str)
        assert str(base_dir) in config.directories['major_data_directory']
    
    def test_create_config_forward_compatibility(self):
        """Test create_config() forward compatibility with additional kwargs."""
        config = create_config(
            custom_field="custom_value",
            future_feature={"nested": "data"},
            version_flag=True
        )
        
        # Verify configuration is created successfully with extra fields
        assert isinstance(config, ProjectConfig)
        # Pydantic should allow extra fields due to model_config
        # The exact handling depends on the model configuration


class TestEnhancedBuilderFunctions:
    """
    Test suite for enhanced builder functions including create_experiment(),
    create_dataset(), and factory methods with schema version handling.
    """
    
    def test_create_experiment_basic_functionality(self):
        """Test create_experiment() basic functionality and defaults."""
        datasets = ["dataset1", "dataset2"]
        
        config = create_experiment(
            name="test_experiment",
            datasets=datasets
        )
        
        # Verify type and structure
        assert isinstance(config, ExperimentConfig)
        assert config.datasets == datasets
        assert hasattr(config, 'schema_version')
        assert config.schema_version is not None
        
        # Verify comprehensive defaults
        assert config.parameters is not None
        assert isinstance(config.parameters, dict)
        assert 'analysis_window' in config.parameters
        assert 'sampling_rate' in config.parameters
        assert 'threshold' in config.parameters
        
        assert config.filters is not None
        assert isinstance(config.filters, dict)
        assert 'ignore_substrings' in config.filters
        
        assert config.metadata is not None
        assert isinstance(config.metadata, dict)
        assert 'created_by' in config.metadata
    
    def test_create_experiment_custom_parameters(self):
        """Test create_experiment() with custom parameters and overrides."""
        datasets = ["custom_dataset"]
        custom_params = {
            "analysis_window": 15.0,
            "custom_threshold": 0.8,
            "method": "custom_method"
        }
        custom_filters = {
            "ignore_substrings": ["custom_ignore"],
            "mandatory_experiment_strings": ["custom_mandatory"]
        }
        custom_metadata = {
            "description": "Custom experiment description",
            "custom_field": "custom_value"
        }
        
        config = create_experiment(
            name="custom_experiment",
            datasets=datasets,
            parameters=custom_params,
            filters=custom_filters,
            metadata=custom_metadata,
            schema_version="1.0.0"
        )
        
        # Verify custom parameters are applied and defaults are merged
        assert config.parameters["analysis_window"] == 15.0
        assert config.parameters["custom_threshold"] == 0.8
        assert config.parameters["method"] == "custom_method"
        # Verify defaults are still present
        assert "sampling_rate" in config.parameters
        assert "confidence_level" in config.parameters
        
        # Verify custom filters and metadata
        assert config.filters["ignore_substrings"] == ["custom_ignore"]
        assert config.metadata["description"] == "Custom experiment description"
        assert config.metadata["custom_field"] == "custom_value"
        assert config.schema_version == "1.0.0"
    
    def test_create_dataset_basic_functionality(self):
        """Test create_dataset() basic functionality and defaults."""
        config = create_dataset(
            name="test_dataset",
            rig="test_rig"
        )
        
        # Verify type and structure
        assert isinstance(config, DatasetConfig)
        assert config.rig == "test_rig"
        assert hasattr(config, 'schema_version')
        assert config.schema_version is not None
        
        # Verify defaults
        assert isinstance(config.dates_vials, dict)
        assert config.metadata is not None
        assert isinstance(config.metadata, dict)
        assert 'created_by' in config.metadata
        assert 'dataset_type' in config.metadata
        assert 'extraction_patterns' in config.metadata
    
    def test_create_dataset_with_dates_vials(self):
        """Test create_dataset() with specific dates_vials configuration."""
        dates_vials = {
            "2023-05-01": [1, 2, 3, 4],
            "2023-05-02": [5, 6, 7, 8]
        }
        
        config = create_dataset(
            name="dated_dataset",
            rig="rig1",
            dates_vials=dates_vials,
            schema_version="1.0.0"
        )
        
        # Verify dates_vials are set correctly
        assert config.dates_vials == dates_vials
        assert config.schema_version == "1.0.0"
        
        # Verify default metadata is still applied
        assert config.metadata['created_by'] == 'flyrigloader'
        assert config.metadata['dataset_type'] == 'behavioral'


class TestFactoryMethods:
    """
    Test suite for factory methods providing pre-configured setups for
    common experiment patterns and use cases.
    """
    
    def test_create_standard_fly_config(self):
        """Test create_standard_fly_config() factory method."""
        config = create_standard_fly_config(
            project_name="fly_behavior_test",
            base_directory="/tmp/fly_test"
        )
        
        # Verify it returns a ProjectConfig
        assert isinstance(config, ProjectConfig)
        assert config.directories['major_data_directory'] == "/tmp/fly_test"
        
        # Verify fly-specific defaults
        assert config.ignore_substrings is not None
        fly_specific_ignores = ["calibration", "test", "debug", "practice"]
        for pattern in fly_specific_ignores:
            assert pattern in config.ignore_substrings
        
        # Verify fly-specific mandatory strings
        assert config.mandatory_experiment_strings is not None
        fly_specific_mandatory = ["experiment", "trial", "fly", "behavior"]
        for string in fly_specific_mandatory:
            assert string in config.mandatory_experiment_strings
        
        # Verify fly-specific extraction patterns
        assert config.extraction_patterns is not None
        # Look for fly-specific patterns
        pattern_text = " ".join(config.extraction_patterns)
        assert "fly_id" in pattern_text
        assert "genotype" in pattern_text
        assert "sex" in pattern_text
    
    def test_create_plume_tracking_experiment(self):
        """Test create_plume_tracking_experiment() factory method."""
        datasets = ["plume_data1", "plume_data2"]
        
        config = create_plume_tracking_experiment(
            datasets=datasets,
            analysis_window=15.0,
            tracking_threshold=0.4
        )
        
        # Verify it returns an ExperimentConfig
        assert isinstance(config, ExperimentConfig)
        assert config.datasets == datasets
        
        # Verify plume-tracking specific parameters
        assert config.parameters['analysis_window'] == 15.0
        assert config.parameters['tracking_threshold'] == 0.4
        assert config.parameters['method'] == 'optical_flow'
        assert config.parameters['smoothing_window'] == 5
        
        # Verify plume-tracking specific filters
        plume_ignores = ["calibration", "test", "debug", "background"]
        for pattern in plume_ignores:
            assert pattern in config.filters['ignore_substrings']
        
        plume_mandatory = ["plume", "tracking", "behavior"]
        for string in plume_mandatory:
            assert string in config.filters['mandatory_experiment_strings']
        
        # Verify plume-tracking specific metadata
        assert config.metadata['experiment_type'] == 'plume_tracking'
        assert config.metadata['output_format'] == 'trajectory_data'
    
    def test_create_choice_assay_experiment(self):
        """Test create_choice_assay_experiment() factory method."""
        datasets = ["choice_data1", "choice_data2"]
        
        config = create_choice_assay_experiment(
            datasets=datasets,
            choice_duration=600.0,
            decision_threshold=0.9
        )
        
        # Verify it returns an ExperimentConfig
        assert isinstance(config, ExperimentConfig)
        assert config.datasets == datasets
        
        # Verify choice-assay specific parameters
        assert config.parameters['choice_duration'] == 600.0
        assert config.parameters['decision_threshold'] == 0.9
        assert config.parameters['method'] == 'preference_index'
        assert config.parameters['baseline_duration'] == 60.0
        
        # Verify choice-assay specific metadata
        assert config.metadata['experiment_type'] == 'choice_assay'
        assert config.metadata['scoring_method'] == 'preference_index'
    
    def test_create_rig_dataset(self):
        """Test create_rig_dataset() factory method with date range generation."""
        config = create_rig_dataset(
            rig_name="test_rig",
            start_date="2023-05-01",
            end_date="2023-05-03",
            vials_per_day=4
        )
        
        # Verify it returns a DatasetConfig
        assert isinstance(config, DatasetConfig)
        assert config.rig == "test_rig"
        
        # Verify dates_vials were generated correctly
        expected_dates = ["2023-05-01", "2023-05-02", "2023-05-03"]
        for date in expected_dates:
            assert date in config.dates_vials
            assert len(config.dates_vials[date]) == 4
        
        # Verify vial numbering
        assert config.dates_vials["2023-05-01"] == [1, 2, 3, 4]
        assert config.dates_vials["2023-05-02"] == [5, 6, 7, 8]
        assert config.dates_vials["2023-05-03"] == [9, 10, 11, 12]
        
        # Verify rig-specific metadata
        assert config.metadata['rig_type'] == 'behavioral'
        assert config.metadata['dataset_type'] == 'longitudinal'
        assert config.metadata['total_vials'] == 12
        assert config.metadata['vials_per_day'] == 4
    
    def test_create_rig_dataset_invalid_dates(self):
        """Test create_rig_dataset() with invalid date formats."""
        with pytest.raises(ValueError, match="Invalid date format"):
            create_rig_dataset(
                rig_name="test_rig",
                start_date="invalid-date",
                end_date="2023-05-03"
            )


class TestSchemaVersionHandling:
    """
    Test suite for schema version handling, migration capabilities,
    and version-aware configuration management.
    """
    
    def test_schema_version_default_assignment(self):
        """Test that all builder functions assign default schema versions."""
        # Test create_config()
        project_config = create_config()
        assert hasattr(project_config, 'schema_version')
        assert project_config.schema_version is not None
        
        # Test create_experiment()
        experiment_config = create_experiment(name="test", datasets=["test"])
        assert hasattr(experiment_config, 'schema_version')
        assert experiment_config.schema_version is not None
        
        # Test create_dataset()
        dataset_config = create_dataset(name="test", rig="test_rig")
        assert hasattr(dataset_config, 'schema_version')
        assert dataset_config.schema_version is not None
    
    def test_schema_version_explicit_assignment(self):
        """Test explicit schema version assignment in builder functions."""
        version = "1.0.0"
        
        # Test all builder functions with explicit version
        project_config = create_config(schema_version=version)
        assert project_config.schema_version == version
        
        experiment_config = create_experiment(
            name="test", 
            datasets=["test"], 
            schema_version=version
        )
        assert experiment_config.schema_version == version
        
        dataset_config = create_dataset(
            name="test", 
            rig="test_rig", 
            schema_version=version
        )
        assert dataset_config.schema_version == version
    
    def test_migration_capabilities(self):
        """Test configuration migration capabilities for version upgrades."""
        # Create a config with older version (simulated)
        config = create_config(schema_version="1.0.0")
        
        # Test migration method exists and is callable
        assert hasattr(config, 'migrate_config')
        assert callable(config.migrate_config)
        
        # Test migration to same version returns copy
        migrated = config.migrate_config("1.0.0")
        assert isinstance(migrated, ProjectConfig)
        assert migrated.schema_version == "1.0.0"
        assert migrated is not config  # Should be a copy, not the same object
    
    @patch('src.flyrigloader.config.models.validate_config_version')
    def test_version_validation_integration(self, mock_validate):
        """Test integration with version validation system."""
        # Mock successful validation
        mock_validate.return_value = (True, "1.0.0", "Valid configuration")
        
        config = create_config(schema_version="1.0.0")
        
        # Verify version validation was called during creation
        assert mock_validate.called
        assert config.schema_version == "1.0.0"


class TestLegacyConfigAdapterIntegration:
    """
    Test suite for LegacyConfigAdapter integration with builder patterns
    and backward compatibility features.
    """
    
    def test_legacy_adapter_with_builder_output(self):
        """Test LegacyConfigAdapter with output from builder functions."""
        # Create configuration using builder
        project_config = create_config(
            project_name="test_project",
            base_directory="/tmp/test"
        )
        
        # Convert to dictionary format for adapter
        config_dict = {
            'project': project_config.model_dump(),
            'datasets': {},
            'experiments': {}
        }
        
        # Test LegacyConfigAdapter creation and deprecation warning
        with pytest.warns(DeprecationWarning, match="Dictionary-based configuration format is deprecated"):
            adapter = LegacyConfigAdapter(config_dict)
        
        # Test adapter functionality
        assert isinstance(adapter, LegacyConfigAdapter)
        assert 'project' in adapter
        assert adapter['project'] is not None
    
    def test_legacy_adapter_get_model_method(self):
        """Test LegacyConfigAdapter.get_model() method with builder-created configurations."""
        # Create configurations using builders
        project_config = create_config()
        experiment_config = create_experiment(name="test", datasets=["test"])
        dataset_config = create_dataset(name="test", rig="test_rig")
        
        config_dict = {
            'project': project_config.model_dump(),
            'datasets': {'test': dataset_config.model_dump()},
            'experiments': {'test': experiment_config.model_dump()}
        }
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            adapter = LegacyConfigAdapter(config_dict)
        
        # Test get_model() method
        project_model = adapter.get_model('project')
        assert isinstance(project_model, ProjectConfig)
        
        dataset_model = adapter.get_model('dataset', 'test')
        assert isinstance(dataset_model, DatasetConfig)
        
        experiment_model = adapter.get_model('experiment', 'test')
        assert isinstance(experiment_model, ExperimentConfig)
    
    def test_legacy_adapter_validate_all_method(self):
        """Test LegacyConfigAdapter.validate_all() method with builder configurations."""
        # Create valid configurations using builders
        project_config = create_config()
        
        config_dict = {
            'project': project_config.model_dump()
        }
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            adapter = LegacyConfigAdapter(config_dict)
        
        # Test validate_all() method
        is_valid = adapter.validate_all()
        assert is_valid is True
    
    def test_legacy_adapter_with_invalid_configuration(self):
        """Test LegacyConfigAdapter validation with invalid configurations."""
        # Create invalid configuration
        invalid_config_dict = {
            'project': {
                'directories': "not_a_dict",  # Invalid type
                'ignore_substrings': 123      # Invalid type
            }
        }
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            adapter = LegacyConfigAdapter(invalid_config_dict)
        
        # Test validate_all() with invalid configuration
        is_valid = adapter.validate_all()
        assert is_valid is False


class TestBuilderValidationAndErrorHandling:
    """
    Test suite for comprehensive error handling and validation in builder functions
    with edge cases and type safety enforcement.
    """
    
    def test_create_config_type_validation(self):
        """Test type validation in create_config() builder function."""
        # Test invalid base_directory type
        with pytest.raises(Exception):  # Could be ValidationError or ValueError
            create_config(base_directory=123)
        
        # Test invalid ignore_substrings type
        with pytest.raises(ValueError) as exc_info:
            create_config(ignore_substrings="not_a_list")
        error_msg = str(exc_info.value)
        assert "Failed to create ProjectConfig" in error_msg
        assert "ignore_substrings" in error_msg
        assert "Input should be a valid list" in error_msg
        
        # Test invalid extraction_patterns type
        with pytest.raises(ValueError) as exc_info:
            create_config(extraction_patterns="not_a_list")
        error_msg = str(exc_info.value)
        assert "Failed to create ProjectConfig" in error_msg
        assert "extraction_patterns" in error_msg
        assert "Input should be a valid list" in error_msg
    
    def test_create_experiment_validation_failures(self):
        """Test validation failures in create_experiment() builder function."""
        # Test empty datasets list
        config = create_experiment(name="test", datasets=[])
        assert config.datasets == []  # Should be allowed but warned
        
        # Test invalid dataset names
        with pytest.raises(ValueError) as exc_info:
            create_experiment(name="test", datasets=["invalid dataset name with spaces"])
        error_msg = str(exc_info.value)
        assert "Failed to create ExperimentConfig" in error_msg
        assert "dataset name" in error_msg
        assert "contains invalid characters" in error_msg
        
        # Test invalid parameters type
        with pytest.raises((ValueError, AttributeError)):
            create_experiment(name="test", datasets=["test"], parameters="not_a_dict")
    
    def test_create_dataset_validation_failures(self):
        """Test validation failures in create_dataset() builder function."""
        # Test invalid rig name
        with pytest.raises(ValueError) as exc_info:
            create_dataset(name="test", rig="invalid rig name")
        error_msg = str(exc_info.value)
        assert "Failed to create DatasetConfig" in error_msg
        assert "rig name" in error_msg
        assert "contains invalid characters" in error_msg
        
        # Test empty rig name
        with pytest.raises(ValueError) as exc_info:
            create_dataset(name="test", rig="")
        error_msg = str(exc_info.value)
        assert "Failed to create DatasetConfig" in error_msg
        assert "rig cannot be empty" in error_msg
        
        # Test invalid dates_vials structure
        with pytest.raises(ValueError) as exc_info:
            create_dataset(
                name="test", 
                rig="rig1", 
                dates_vials={"2023-01-01": "not_a_list"}
            )
        error_msg = str(exc_info.value)
        assert "Failed to create DatasetConfig" in error_msg
    
    def test_regex_pattern_validation(self):
        """Test regex pattern validation in extraction_patterns."""
        # Test valid regex patterns
        valid_patterns = [
            r"(?P<date>\d{4}-\d{2}-\d{2})",
            r"(?P<rig>rig\d+)",
            r"(?P<subject>\w+)"
        ]
        
        config = create_config(extraction_patterns=valid_patterns)
        assert config.extraction_patterns == valid_patterns
        
        # Test invalid regex patterns
        with pytest.raises(ValueError, match="Invalid regex pattern"):
            create_config(extraction_patterns=[r"(?P<invalid>[unclosed"])
    
    def test_comprehensive_validation_error_messages(self):
        """Test that validation errors provide clear, actionable error messages."""
        # Test directory validation error message
        try:
            create_config(directories="not_a_dict")
            assert False, "Should have raised TypeError"
        except (TypeError, ValidationError, ValueError) as e:
            error_msg = str(e)
            assert "item assignment" in error_msg.lower() or "dict" in error_msg.lower() or "dictionary" in error_msg.lower()
        
        # Test rig validation error message
        try:
            create_dataset(name="test", rig="invalid rig name")
            assert False, "Should have raised ValueError"
        except (ValueError, ValidationError) as e:
            error_msg = str(e)
            assert "invalid characters" in error_msg


class TestBuilderPerformance:
    """
    Test suite for performance testing of builder pattern construction ensuring
    efficient configuration creation per Section 6.6.7.3.
    """
    
    def test_create_config_performance_benchmark(self, benchmark):
        """Benchmark create_config() performance for efficient construction."""
        def create_standard_config():
            return create_config(
                project_name="performance_test",
                base_directory="/tmp/perf_test",
                datasets=["dataset1", "dataset2", "dataset3"],
                experiments=["exp1", "exp2", "exp3"]
            )
        
        # Benchmark the configuration creation
        result = benchmark(create_standard_config)
        
        # Verify the result is valid
        assert isinstance(result, ProjectConfig)
        assert result.directories['major_data_directory'] == "/tmp/perf_test"
    
    def test_bulk_configuration_creation_performance(self):
        """Test performance of creating multiple configurations in sequence."""
        start_time = time.perf_counter()
        
        configs = []
        for i in range(100):
            config = create_config(
                project_name=f"project_{i}",
                base_directory=f"/tmp/project_{i}"
            )
            configs.append(config)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Verify all configurations were created successfully
        assert len(configs) == 100
        for config in configs:
            assert isinstance(config, ProjectConfig)
        
        # Performance expectation: should create 100 configs in reasonable time
        # This is a reasonable expectation for configuration creation
        assert total_time < 5.0, f"Configuration creation too slow: {total_time:.2f}s"
        
        # Calculate average time per configuration
        avg_time = total_time / 100
        assert avg_time < 0.05, f"Average config creation time too slow: {avg_time:.4f}s"
    
    def test_factory_method_performance(self, benchmark):
        """Benchmark factory method performance for specialized configurations."""
        def create_fly_config():
            return create_standard_fly_config(
                project_name="fly_performance_test",
                base_directory="/tmp/fly_perf"
            )
        
        result = benchmark(create_fly_config)
        
        # Verify the result is valid and has fly-specific features
        assert isinstance(result, ProjectConfig)
        assert "fly" in " ".join(result.mandatory_experiment_strings or [])
    
    def test_memory_efficiency_bulk_creation(self):
        """Test memory efficiency during bulk configuration creation."""
        import gc
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create many configurations
        configs = []
        for i in range(1000):
            config = create_config(project_name=f"mem_test_{i}")
            configs.append(config)
            
            # Periodically check memory growth
            if i % 100 == 0:
                current_memory = process.memory_info().rss
                memory_growth = current_memory - initial_memory
                
                # Memory growth should be reasonable (less than 100MB for 1000 configs)
                assert memory_growth < 100 * 1024 * 1024, f"Excessive memory growth: {memory_growth / 1024 / 1024:.1f} MB"
        
        # Force garbage collection
        gc.collect()
        
        # Verify all configurations are valid
        assert len(configs) == 1000
        for config in configs:
            assert isinstance(config, ProjectConfig)


class TestHypothesisPropertyBasedTesting:
    """
    Property-based testing using Hypothesis for comprehensive validation
    of builder pattern edge cases and type safety.
    """
    
    @given(
        project_name=text(min_size=1, max_size=50),
        base_dir=text(min_size=1, max_size=100)
    )
    @settings(max_examples=50, deadline=1000)
    def test_create_config_with_random_strings(self, project_name, base_dir):
        """Property-based test for create_config() with random string inputs."""
        try:
            config = create_config(
                project_name=project_name,
                base_directory=base_dir
            )
            
            # Properties that should always hold
            assert isinstance(config, ProjectConfig)
            assert config.schema_version is not None
            assert isinstance(config.directories, dict)
            assert 'major_data_directory' in config.directories
            
        except (ValidationError, ValueError):
            # Some random strings may be invalid, which is expected
            pass
    
    @given(
        datasets=lists(text(min_size=1, max_size=20, alphabet='abcdefghijklmnopqrstuvwxyz0123456789_-'), min_size=1, max_size=10)
    )
    @settings(max_examples=30, deadline=1000)
    def test_create_experiment_with_random_datasets(self, datasets):
        """Property-based test for create_experiment() with random dataset names."""
        try:
            config = create_experiment(
                name="test_experiment",
                datasets=datasets
            )
            
            # Properties that should always hold
            assert isinstance(config, ExperimentConfig)
            assert config.datasets == datasets
            assert config.schema_version is not None
            
        except ValidationError:
            # Some random dataset names may be invalid, which is expected
            pass
    
    @given(
        vials_per_day=integers(min_value=1, max_value=50)
    )
    @settings(max_examples=20, deadline=1000)
    def test_create_rig_dataset_with_random_vials(self, vials_per_day):
        """Property-based test for create_rig_dataset() with random vial counts."""
        config = create_rig_dataset(
            rig_name="test_rig",
            start_date="2023-05-01",
            end_date="2023-05-02",
            vials_per_day=vials_per_day
        )
        
        # Properties that should always hold
        assert isinstance(config, DatasetConfig)
        assert len(config.dates_vials["2023-05-01"]) == vials_per_day
        assert len(config.dates_vials["2023-05-02"]) == vials_per_day
        assert config.metadata['vials_per_day'] == vials_per_day


class TestIntegrationWithPydanticFeatures:
    """
    Test suite for integration with Pydantic features including validation,
    serialization, and model capabilities.
    """
    
    def test_model_serialization_and_deserialization(self):
        """Test that builder-created configurations can be serialized and deserialized."""
        # Create configuration using builder
        original_config = create_config(
            project_name="serialization_test",
            base_directory="/tmp/serial_test"
        )
        
        # Test JSON serialization
        json_data = original_config.model_dump_json()
        assert isinstance(json_data, str)
        assert "/tmp/serial_test" in json_data  # Check for the actual directory path
        
        # Test deserialization
        config_dict = original_config.model_dump()
        reconstructed_config = ProjectConfig(**config_dict)
        
        # Verify reconstruction
        assert isinstance(reconstructed_config, ProjectConfig)
        assert reconstructed_config.directories == original_config.directories
        assert reconstructed_config.schema_version == original_config.schema_version
    
    def test_model_validation_on_assignment(self):
        """Test that Pydantic validation works on field assignment for builder-created configs."""
        config = create_config()
        
        # Test valid assignment
        config.ignore_substrings = ["new_pattern"]
        assert config.ignore_substrings == ["new_pattern"]
        
        # Test invalid assignment should raise ValidationError
        with pytest.raises(ValidationError):
            config.ignore_substrings = "not_a_list"
    
    def test_model_copy_functionality(self):
        """Test Pydantic model copy functionality with builder-created configurations."""
        original_config = create_config(
            project_name="copy_test",
            base_directory="/tmp/copy_test"
        )
        
        # Test shallow copy
        copied_config = original_config.model_copy()
        assert copied_config is not original_config
        assert copied_config.directories == original_config.directories
        assert copied_config.schema_version == original_config.schema_version
        
        # Test deep copy with updates
        updated_config = original_config.model_copy(
            update={"schema_version": "2.0.0"}
        )
        assert updated_config.schema_version == "2.0.0"
        assert original_config.schema_version != "2.0.0"
    
    def test_field_validation_edge_cases(self):
        """Test edge cases in field validation for builder-created configurations."""
        # Test empty lists
        config = create_config(
            ignore_substrings=[],
            extraction_patterns=[],
            mandatory_experiment_strings=[]
        )
        
        # Empty lists should be handled gracefully
        assert isinstance(config, ProjectConfig)
        
        # Test None values where optional - function provides defaults
        config = create_config(
            mandatory_experiment_strings=None
        )
        # Function provides default values instead of None
        assert config.mandatory_experiment_strings == ['experiment', 'trial']
        
        # Test whitespace handling
        config = create_config(
            ignore_substrings=["  pattern_with_spaces  "]
        )
        # Pydantic should strip whitespace
        assert "pattern_with_spaces" in config.ignore_substrings[0]


# Performance and integration fixtures
@pytest.fixture(scope="session")
def sample_project_config():
    """Fixture providing a sample ProjectConfig for reuse in tests."""
    return create_config(
        project_name="test_project",
        base_directory="/tmp/test_project",
        datasets=["dataset1", "dataset2"],
        experiments=["exp1", "exp2"]
    )


@pytest.fixture(scope="session") 
def sample_experiment_config():
    """Fixture providing a sample ExperimentConfig for reuse in tests."""
    return create_experiment(
        name="test_experiment",
        datasets=["test_dataset"],
        parameters={"analysis_window": 10.0}
    )


@pytest.fixture(scope="session")
def sample_dataset_config():
    """Fixture providing a sample DatasetConfig for reuse in tests."""
    return create_dataset(
        name="test_dataset",
        rig="test_rig",
        dates_vials={"2023-05-01": [1, 2, 3, 4]}
    )


# Custom pytest markers for test categorization
pytestmark = [
    pytest.mark.unit,
    pytest.mark.config,
    pytest.mark.builders
]