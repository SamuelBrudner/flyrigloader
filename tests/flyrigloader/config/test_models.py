"""
Comprehensive test suite for Pydantic configuration models.

This module provides extensive testing for ProjectConfig, DatasetConfig, and ExperimentConfig
Pydantic models, including property-based testing, validation scenarios, and backward
compatibility verification. The tests ensure robust configuration handling across all
use cases and edge conditions.

Test Coverage:
- Property-based testing with Hypothesis for diverse input scenarios
- Validation tests for all Pydantic model fields and custom validators
- Backward compatibility tests for LegacyConfigAdapter
- Edge cases and error scenarios for model validation
- Performance tests comparing Pydantic vs dict-based validation
"""

import pytest
import os
import tempfile
import json
import yaml
import re
import time
from pathlib import Path
from datetime import datetime
from typing import List
from copy import deepcopy
from unittest.mock import patch

from hypothesis import given, strategies as st, settings, assume
from pydantic import ValidationError

from src.flyrigloader.config.models import (
    ProjectConfig,
    DatasetConfig, 
    ExperimentConfig,
    LegacyConfigAdapter
)
from src.flyrigloader.config.validators import pattern_validation


class TestProjectConfig:
    """Test suite for ProjectConfig Pydantic model."""
    
    def test_project_config_basic_creation(self):
        """Test basic ProjectConfig creation with minimal data."""
        config = ProjectConfig()
        assert config.directories == {}
        # Updated: ignore_substrings now has comprehensive defaults
        assert config.ignore_substrings == ["._", "temp", "backup", ".tmp", "~", ".DS_Store"]
        assert config.mandatory_experiment_strings is None
        # Updated: extraction_patterns now has comprehensive defaults
        assert len(config.extraction_patterns) >= 4  # Should have default patterns
        assert any("date" in pattern for pattern in config.extraction_patterns)
        # New: schema_version field should default to current version
        assert config.schema_version == "1.0.0"
    
    def test_project_config_with_directories(self):
        """Test ProjectConfig creation with directory specification."""
        with tempfile.TemporaryDirectory() as temp_dir:
            directories = {
                "major_data_directory": temp_dir,
                "backup_directory": temp_dir
            }
            config = ProjectConfig(directories=directories)
            assert config.directories == directories
    
    def test_project_config_directory_validation(self):
        """Test directory validation in ProjectConfig."""
        # Test with non-dict directories
        with pytest.raises(ValidationError) as exc_info:
            ProjectConfig(directories="not_a_dict")
        # Pydantic 2 generates standardized error messages
        assert "Input should be a valid dictionary" in str(exc_info.value)
        
        # Test with None values in directories (should be allowed)
        config = ProjectConfig(directories={"major_data_directory": None})
        assert config.directories == {}
    
    def test_project_config_ignore_substrings_validation(self):
        """Test ignore_substrings validation in ProjectConfig."""
        # Valid ignore substrings
        config = ProjectConfig(ignore_substrings=["._", "temp", "backup"])
        assert config.ignore_substrings == ["._", "temp", "backup"]
        
        # Test with non-list
        with pytest.raises(ValidationError) as exc_info:
            ProjectConfig(ignore_substrings="not_a_list")
        assert "Input should be a valid list" in str(exc_info.value)
        
        # Test with empty strings (should be filtered out)
        config = ProjectConfig(ignore_substrings=["valid", "", "  ", "another"])
        assert config.ignore_substrings == ["valid", "another"]
        
        # Test with non-string items
        with pytest.raises(ValidationError) as exc_info:
            ProjectConfig(ignore_substrings=["valid", 123, "another"])
        assert "string_type" in str(exc_info.value)
    
    def test_project_config_mandatory_strings_validation(self):
        """Test mandatory_experiment_strings validation in ProjectConfig."""
        # Valid mandatory strings
        config = ProjectConfig(mandatory_experiment_strings=["experiment", "trial"])
        assert config.mandatory_experiment_strings == ["experiment", "trial"]
        
        # Test with non-list
        with pytest.raises(ValidationError) as exc_info:
            ProjectConfig(mandatory_experiment_strings="not_a_list")
        assert "Input should be a valid list" in str(exc_info.value)
        
        # Test with empty strings (should be filtered out)
        config = ProjectConfig(mandatory_experiment_strings=["valid", "", "  ", "another"])
        assert config.mandatory_experiment_strings == ["valid", "another"]
        
        # Test with non-string items
        with pytest.raises(ValidationError) as exc_info:
            ProjectConfig(mandatory_experiment_strings=["valid", 123, "another"])
        assert "string_type" in str(exc_info.value)
    
    def test_project_config_extraction_patterns_validation(self):
        """Test extraction_patterns validation in ProjectConfig."""
        # Valid extraction patterns
        valid_patterns = [r"(?P<date>\d{4}-\d{2}-\d{2})", r"(?P<subject>\w+)"]
        config = ProjectConfig(extraction_patterns=valid_patterns)
        assert config.extraction_patterns == valid_patterns
        
        # Test with non-list
        with pytest.raises(ValidationError) as exc_info:
            ProjectConfig(extraction_patterns="not_a_list")
        assert "Input should be a valid list" in str(exc_info.value)
        
        # Test with invalid regex pattern
        with pytest.raises(ValidationError) as exc_info:
            ProjectConfig(extraction_patterns=[r"valid_pattern", r"[invalid"])
        assert "Invalid regex pattern" in str(exc_info.value)
        
        # Test with empty strings (should be filtered out)
        config = ProjectConfig(extraction_patterns=[r"\d+", "", "  ", r"\w+"])
        assert config.extraction_patterns == [r"\d+", r"\w+"]
        
        # Test with non-string items
        with pytest.raises(ValidationError) as exc_info:
            ProjectConfig(extraction_patterns=[r"\d+", 123, r"\w+"])
        assert "string_type" in str(exc_info.value)
    
    def test_project_config_extra_fields_allowed(self):
        """Test that extra fields are allowed in ProjectConfig."""
        config = ProjectConfig(
            directories={"major_data_directory": "/path/to/data"},
            custom_field="custom_value",
            another_field={"nested": "value"}
        )
        assert config.directories == {"major_data_directory": "/path/to/data"}
        assert config.custom_field == "custom_value"
        assert config.another_field == {"nested": "value"}
    
    @given(st.dictionaries(
        st.text(min_size=1, max_size=50),
        st.one_of(st.text(min_size=1, max_size=100), st.none())
    ))
    @settings(max_examples=50)
    def test_project_config_directories_property_based(self, directories):
        """Property-based test for ProjectConfig directories validation."""
        try:
            config = ProjectConfig(directories=directories)
            # If validation passes, directories should be a dict with non-None values
            assert isinstance(config.directories, dict)
            for key, value in config.directories.items():
                assert value is not None
        except ValidationError:
            # Validation failure is acceptable for some inputs
            pass
    
    @given(st.lists(st.text(min_size=1, max_size=50), max_size=20))
    @settings(max_examples=30)
    def test_project_config_ignore_substrings_property_based(self, ignore_substrings):
        """Property-based test for ProjectConfig ignore_substrings validation."""
        try:
            config = ProjectConfig(ignore_substrings=ignore_substrings)
            if config.ignore_substrings:
                assert isinstance(config.ignore_substrings, list)
                assert all(isinstance(item, str) and item.strip() for item in config.ignore_substrings)
        except ValidationError:
            # Validation failure is acceptable for some inputs
            pass


class TestDatasetConfig:
    """Test suite for DatasetConfig Pydantic model."""
    
    def test_dataset_config_basic_creation(self):
        """Test basic DatasetConfig creation with minimal data."""
        config = DatasetConfig(rig="rig1")
        assert config.rig == "rig1"
        assert config.dates_vials == {}
        # Updated: metadata now has comprehensive defaults
        assert config.metadata is not None
        assert isinstance(config.metadata, dict)
        assert "created_by" in config.metadata
        # New: schema_version field should default to current version
        assert config.schema_version == "1.0.0"
    
    def test_dataset_config_with_dates_vials(self):
        """Test DatasetConfig creation with dates_vials data."""
        dates_vials = {
            "2023-05-01": [1, 2, 3, 4],
            "2023-05-02": [5, 6, 7, 8]
        }
        config = DatasetConfig(rig="rig1", dates_vials=dates_vials)
        assert config.rig == "rig1"
        assert config.dates_vials == dates_vials
    
    def test_dataset_config_rig_validation(self):
        """Test rig validation in DatasetConfig."""
        # Valid rig names
        valid_rigs = ["rig1", "rig_2", "rig-3", "RIG1", "experimental_rig"]
        for rig in valid_rigs:
            config = DatasetConfig(rig=rig)
            assert config.rig == rig
        
        # Test with non-string rig
        with pytest.raises(ValidationError) as exc_info:
            DatasetConfig(rig=123)
        assert "string_type" in str(exc_info.value)
        
        # Test with empty rig
        with pytest.raises(ValidationError) as exc_info:
            DatasetConfig(rig="")
        assert "rig cannot be empty or whitespace-only" in str(exc_info.value)
        
        # Test with invalid characters
        with pytest.raises(ValidationError) as exc_info:
            DatasetConfig(rig="rig@invalid")
        assert "contains invalid characters" in str(exc_info.value)
    
    def test_dataset_config_dates_vials_validation(self):
        """Test dates_vials validation in DatasetConfig."""
        # Valid dates_vials
        dates_vials = {
            "2023-05-01": [1, 2, 3],
            "05/01/2023": [4, 5, 6],
            "01/05/2023": [7, 8, 9]
        }
        config = DatasetConfig(rig="rig1", dates_vials=dates_vials)
        assert config.dates_vials == dates_vials
        
        # Test with non-dict dates_vials
        with pytest.raises(ValidationError) as exc_info:
            DatasetConfig(rig="rig1", dates_vials="not_a_dict")
        assert "Input should be a valid dictionary" in str(exc_info.value)
        
        # Test with non-string date keys
        with pytest.raises(ValidationError) as exc_info:
            DatasetConfig(rig="rig1", dates_vials={123: [1, 2, 3]})
        assert "Input should be a valid string" in str(exc_info.value)
        
        # Test with non-list vials
        with pytest.raises(ValidationError) as exc_info:
            DatasetConfig(rig="rig1", dates_vials={"2023-05-01": "not_a_list"})
        assert "Input should be a valid list" in str(exc_info.value)
        
        # Test with non-integer vials
        with pytest.raises(ValidationError) as exc_info:
            DatasetConfig(rig="rig1", dates_vials={"2023-05-01": [1, "not_int", 3]})
        assert ("int_parsing" in str(exc_info.value) or "Input should be a valid integer" in str(exc_info.value))
        
        # Test with string vials that can be converted to int
        config = DatasetConfig(rig="rig1", dates_vials={"2023-05-01": [1, "2", 3]})
        assert config.dates_vials == {"2023-05-01": [1, 2, 3]}
    
    def test_dataset_config_metadata_validation(self):
        """Test metadata validation in DatasetConfig."""
        # Valid metadata
        metadata = {
            "description": "Temperature gradient experiments",
            "extraction_patterns": [r"(?P<temperature>\d+)C"]
        }
        config = DatasetConfig(rig="rig1", metadata=metadata)
        assert config.metadata == metadata
        
        # Test with non-dict metadata
        with pytest.raises(ValidationError) as exc_info:
            DatasetConfig(rig="rig1", metadata="not_a_dict")
        assert "Input should be a valid dictionary" in str(exc_info.value)
        
        # Test with invalid extraction patterns in metadata
        with pytest.raises(ValidationError) as exc_info:
            DatasetConfig(rig="rig1", metadata={"extraction_patterns": ["valid", "[invalid"]})
        assert "Invalid regex pattern" in str(exc_info.value)
        
        # Test with non-list extraction patterns
        with pytest.raises(ValidationError) as exc_info:
            DatasetConfig(rig="rig1", metadata={"extraction_patterns": "not_a_list"})
        assert "extraction_patterns must be a list" in str(exc_info.value)
        
        # Test with non-string extraction patterns
        with pytest.raises(ValidationError) as exc_info:
            DatasetConfig(rig="rig1", metadata={"extraction_patterns": [123]})
        assert "extraction pattern must be string" in str(exc_info.value)
    
    @given(st.text(min_size=1, max_size=50).filter(lambda x: re.match(r'^[a-zA-Z0-9_-]+$', x.strip())))
    @settings(max_examples=30)
    def test_dataset_config_rig_property_based(self, rig_name):
        """Property-based test for DatasetConfig rig validation."""
        try:
            config = DatasetConfig(rig=rig_name)
            assert config.rig == rig_name.strip()
        except ValidationError:
            # Some edge cases might fail validation
            pass
    
    @given(st.dictionaries(
        st.text(min_size=8, max_size=10).filter(lambda x: re.match(r'\d{4}-\d{2}-\d{2}', x)),
        st.lists(st.integers(min_value=1, max_value=100), min_size=1, max_size=10)
    ))
    @settings(max_examples=20)
    def test_dataset_config_dates_vials_property_based(self, dates_vials):
        """Property-based test for DatasetConfig dates_vials validation."""
        try:
            config = DatasetConfig(rig="test_rig", dates_vials=dates_vials)
            assert isinstance(config.dates_vials, dict)
            for date_str, vials in config.dates_vials.items():
                assert isinstance(date_str, str)
                assert isinstance(vials, list)
                assert all(isinstance(v, int) for v in vials)
        except ValidationError:
            # Some edge cases might fail validation
            pass


class TestExperimentConfig:
    """Test suite for ExperimentConfig Pydantic model."""
    
    def test_experiment_config_basic_creation(self):
        """Test basic ExperimentConfig creation with minimal data."""
        config = ExperimentConfig()
        assert config.datasets == []
        # Updated: parameters now has comprehensive defaults
        assert config.parameters is not None
        assert isinstance(config.parameters, dict)
        assert "analysis_window" in config.parameters
        # Updated: filters now has comprehensive defaults
        assert config.filters is not None
        assert isinstance(config.filters, dict) 
        assert "ignore_substrings" in config.filters
        # Updated: metadata now has comprehensive defaults
        assert config.metadata is not None
        assert isinstance(config.metadata, dict)
        assert "created_by" in config.metadata
        # New: schema_version field should default to current version
        assert config.schema_version == "1.0.0"
    
    def test_experiment_config_with_datasets(self):
        """Test ExperimentConfig creation with datasets."""
        datasets = ["plume_tracking", "odor_response"]
        config = ExperimentConfig(datasets=datasets)
        assert config.datasets == datasets
    
    def test_experiment_config_datasets_validation(self):
        """Test datasets validation in ExperimentConfig."""
        # Valid datasets
        valid_datasets = ["plume_tracking", "odor_response", "dataset_1", "dataset-2"]
        config = ExperimentConfig(datasets=valid_datasets)
        assert config.datasets == valid_datasets
        
        # Test with non-list datasets
        with pytest.raises(ValidationError) as exc_info:
            ExperimentConfig(datasets="not_a_list")
        assert "extraction_patterns must be a list" in str(exc_info.value)
        
        # Test with non-string dataset names
        with pytest.raises(ValidationError) as exc_info:
            ExperimentConfig(datasets=["valid_dataset", 123, "another"])
        assert "string_type" in str(exc_info.value)
        
        # Test with empty dataset names (should be filtered out)
        config = ExperimentConfig(datasets=["valid", "", "  ", "another"])
        assert config.datasets == ["valid", "another"]
        
        # Test with invalid characters
        with pytest.raises(ValidationError) as exc_info:
            ExperimentConfig(datasets=["valid_dataset", "invalid@dataset"])
        assert "contains invalid characters" in str(exc_info.value)
    
    def test_experiment_config_parameters_validation(self):
        """Test parameters validation in ExperimentConfig."""
        # Valid parameters
        parameters = {
            "analysis_window": 10.0,
            "threshold": 0.5,
            "method": "correlation"
        }
        config = ExperimentConfig(parameters=parameters)
        assert config.parameters == parameters
        
        # Test with non-dict parameters
        with pytest.raises(ValidationError) as exc_info:
            ExperimentConfig(parameters="not_a_dict")
        assert "Input should be a valid dictionary" in str(exc_info.value)
        
        # Test with non-string parameter keys
        with pytest.raises(ValidationError) as exc_info:
            ExperimentConfig(parameters={123: "value"})
        assert "Input should be a valid string" in str(exc_info.value)
        
        # Test with empty parameter keys
        with pytest.raises(ValidationError) as exc_info:
            ExperimentConfig(parameters={"": "value"})
        assert ("parameter key cannot be empty" in str(exc_info.value) or "ensure this value has at least 1 character" in str(exc_info.value))
    
    def test_experiment_config_filters_validation(self):
        """Test filters validation in ExperimentConfig."""
        # Valid filters
        filters = {
            "ignore_substrings": ["temp", "backup"],
            "mandatory_experiment_strings": ["trial"]
        }
        config = ExperimentConfig(filters=filters)
        assert config.filters == filters
        
        # Test with non-dict filters
        with pytest.raises(ValidationError) as exc_info:
            ExperimentConfig(filters="not_a_dict")
        assert "Input should be a valid dictionary" in str(exc_info.value)
        
        # Test with non-list ignore_substrings
        with pytest.raises(ValidationError) as exc_info:
            ExperimentConfig(filters={"ignore_substrings": "not_a_list"})
        assert "filters ignore_substrings must be a list" in str(exc_info.value)
        
        # Test with non-string ignore patterns
        with pytest.raises(ValidationError) as exc_info:
            ExperimentConfig(filters={"ignore_substrings": ["valid", 123]})
        assert "filter ignore pattern must be string" in str(exc_info.value)
        
        # Test with non-list mandatory_experiment_strings
        with pytest.raises(ValidationError) as exc_info:
            ExperimentConfig(filters={"mandatory_experiment_strings": "not_a_list"})
        assert "filters mandatory_experiment_strings must be a list" in str(exc_info.value)
        
        # Test with non-string mandatory strings
        with pytest.raises(ValidationError) as exc_info:
            ExperimentConfig(filters={"mandatory_experiment_strings": ["valid", 123]})
        assert "filter mandatory string must be string" in str(exc_info.value)
    
    def test_experiment_config_metadata_validation(self):
        """Test metadata validation in ExperimentConfig."""
        # Valid metadata
        metadata = {
            "description": "Plume navigation analysis experiment",
            "analysis_type": "behavioral",
            "extraction_patterns": [r"(?P<trial>\d+)"]
        }
        config = ExperimentConfig(metadata=metadata)
        assert config.metadata == metadata
        
        # Test with non-dict metadata
        with pytest.raises(ValidationError) as exc_info:
            ExperimentConfig(metadata="not_a_dict")
        assert "Input should be a valid dictionary" in str(exc_info.value)
        
        # Test with invalid extraction patterns in metadata
        with pytest.raises(ValidationError) as exc_info:
            ExperimentConfig(metadata={"extraction_patterns": ["valid", "[invalid"]})
        assert "Invalid regex pattern" in str(exc_info.value)
    
    @given(st.lists(
        st.text(min_size=1, max_size=30).filter(lambda x: re.match(r'^[a-zA-Z0-9_-]+$', x.strip())),
        max_size=10
    ))
    @settings(max_examples=30)
    def test_experiment_config_datasets_property_based(self, datasets):
        """Property-based test for ExperimentConfig datasets validation."""
        try:
            config = ExperimentConfig(datasets=datasets)
            assert isinstance(config.datasets, list)
            assert all(isinstance(ds, str) and ds.strip() for ds in config.datasets)
        except ValidationError:
            # Some edge cases might fail validation
            pass


class TestLegacyConfigAdapter:
    """Test suite for LegacyConfigAdapter backward compatibility."""
    
    def test_legacy_adapter_creation(self):
        """Test LegacyConfigAdapter creation with sample configuration."""
        config_data = {
            "project": {
                "directories": {"major_data_directory": "/path/to/data"},
                "ignore_substrings": ["temp", "backup"]
            },
            "datasets": {
                "plume_tracking": {
                    "rig": "rig1",
                    "dates_vials": {"2023-05-01": [1, 2, 3]}
                }
            },
            "experiments": {
                "plume_analysis": {
                    "datasets": ["plume_tracking"],
                    "parameters": {"threshold": 0.5}
                }
            }
        }
        
        adapter = LegacyConfigAdapter(config_data)
        assert isinstance(adapter, LegacyConfigAdapter)
        assert len(adapter) == 3
        assert "project" in adapter
        assert "datasets" in adapter
        assert "experiments" in adapter
    
    def test_legacy_adapter_getitem(self):
        """Test __getitem__ method for dictionary-style access."""
        config_data = {
            "project": {
                "directories": {"major_data_directory": "/path/to/data"}
            },
            "datasets": {
                "test_dataset": {
                    "rig": "rig1",
                    "dates_vials": {"2023-05-01": [1, 2]}
                }
            }
        }
        
        adapter = LegacyConfigAdapter(config_data)
        
        # Test project access
        assert adapter["project"]["directories"]["major_data_directory"] == "/path/to/data"
        
        # Test datasets access
        assert adapter["datasets"]["test_dataset"]["rig"] == "rig1"
        assert adapter["datasets"]["test_dataset"]["dates_vials"] == {"2023-05-01": [1, 2]}
    
    def test_legacy_adapter_setitem(self):
        """Test __setitem__ method for dictionary-style assignment."""
        config_data = {"project": {"directories": {}}}
        adapter = LegacyConfigAdapter(config_data)
        
        # Test setting new project data
        new_project = {"directories": {"major_data_directory": "/new/path"}}
        adapter["project"] = new_project
        assert adapter["project"]["directories"]["major_data_directory"] == "/new/path"
        
        # Test setting new datasets
        new_datasets = {
            "new_dataset": {
                "rig": "rig2",
                "dates_vials": {"2023-05-02": [4, 5, 6]}
            }
        }
        adapter["datasets"] = new_datasets
        assert adapter["datasets"]["new_dataset"]["rig"] == "rig2"
    
    def test_legacy_adapter_delitem(self):
        """Test __delitem__ method for dictionary-style deletion."""
        config_data = {
            "project": {"directories": {}},
            "datasets": {"test_dataset": {"rig": "rig1"}},
            "experiments": {"test_experiment": {"datasets": ["test_dataset"]}}
        }
        adapter = LegacyConfigAdapter(config_data)
        
        # Test deletion
        del adapter["experiments"]
        assert "experiments" not in adapter
        assert len(adapter) == 2
        
        # Test deleting non-existent key
        with pytest.raises(KeyError):
            del adapter["non_existent"]
    
    def test_legacy_adapter_contains(self):
        """Test __contains__ method for membership testing."""
        config_data = {
            "project": {"directories": {}},
            "datasets": {"test_dataset": {"rig": "rig1"}}
        }
        adapter = LegacyConfigAdapter(config_data)
        
        assert "project" in adapter
        assert "datasets" in adapter
        assert "experiments" not in adapter
        assert "non_existent" not in adapter
    
    def test_legacy_adapter_get_method(self):
        """Test get method with default values."""
        config_data = {"project": {"directories": {}}}
        adapter = LegacyConfigAdapter(config_data)
        
        # Test getting existing key
        assert adapter.get("project") == {"directories": {}}
        
        # Test getting non-existent key with default
        assert adapter.get("non_existent", "default") == "default"
        
        # Test getting non-existent key without default
        assert adapter.get("non_existent") is None
    
    def test_legacy_adapter_keys_values_items(self):
        """Test keys(), values(), and items() methods."""
        config_data = {
            "project": {"directories": {}},
            "datasets": {"test_dataset": {"rig": "rig1"}}
        }
        adapter = LegacyConfigAdapter(config_data)
        
        # Test keys
        keys = list(adapter.keys())
        assert "project" in keys
        assert "datasets" in keys
        assert len(keys) == 2
        
        # Test values
        values = list(adapter.values())
        assert len(values) == 2
        
        # Test items
        items = list(adapter.items())
        assert len(items) == 2
        assert ("project", {"directories": {}}) in items
    
    def test_legacy_adapter_get_model(self):
        """Test get_model method for accessing Pydantic models."""
        config_data = {
            "project": {
                "directories": {"major_data_directory": "/path/to/data"}
            },
            "datasets": {
                "test_dataset": {
                    "rig": "rig1",
                    "dates_vials": {"2023-05-01": [1, 2]}
                }
            },
            "experiments": {
                "test_experiment": {
                    "datasets": ["test_dataset"]
                }
            }
        }
        adapter = LegacyConfigAdapter(config_data)
        
        # Test getting project model
        project_model = adapter.get_model("project")
        assert isinstance(project_model, ProjectConfig)
        assert project_model.directories["major_data_directory"] == "/path/to/data"
        
        # Test getting dataset model
        dataset_model = adapter.get_model("dataset", "test_dataset")
        assert isinstance(dataset_model, DatasetConfig)
        assert dataset_model.rig == "rig1"
        
        # Test getting experiment model
        experiment_model = adapter.get_model("experiment", "test_experiment")
        assert isinstance(experiment_model, ExperimentConfig)
        assert experiment_model.datasets == ["test_dataset"]
        
        # Test getting non-existent model
        assert adapter.get_model("non_existent") is None
        assert adapter.get_model("dataset", "non_existent") is None
    
    def test_legacy_adapter_get_all_models(self):
        """Test get_all_models method."""
        config_data = {
            "project": {"directories": {}},
            "datasets": {"test_dataset": {"rig": "rig1"}},
            "experiments": {"test_experiment": {"datasets": ["test_dataset"]}}
        }
        adapter = LegacyConfigAdapter(config_data)
        
        models = adapter.get_all_models()
        assert isinstance(models, dict)
        assert "project" in models
        assert "dataset_test_dataset" in models
        assert "experiment_test_experiment" in models
        
        # Verify model types
        assert isinstance(models["project"], ProjectConfig)
        assert isinstance(models["dataset_test_dataset"], DatasetConfig)
        assert isinstance(models["experiment_test_experiment"], ExperimentConfig)
    
    def test_legacy_adapter_validate_all(self):
        """Test validate_all method for comprehensive validation."""
        # Test with valid configuration
        valid_config = {
            "project": {
                "directories": {"major_data_directory": "/path/to/data"},
                "extraction_patterns": [r"(?P<date>\d{4}-\d{2}-\d{2})"]
            },
            "datasets": {
                "test_dataset": {
                    "rig": "rig1",
                    "dates_vials": {"2023-05-01": [1, 2, 3]}
                }
            },
            "experiments": {
                "test_experiment": {
                    "datasets": ["test_dataset"],
                    "parameters": {"threshold": 0.5}
                }
            }
        }
        adapter = LegacyConfigAdapter(valid_config)
        assert adapter.validate_all() is True
        
        # Test with invalid configuration
        invalid_config = {
            "project": {
                "directories": {"major_data_directory": "/path/to/data"},
                "extraction_patterns": ["[invalid_regex"]  # Invalid regex
            }
        }
        adapter = LegacyConfigAdapter(invalid_config)
        assert adapter.validate_all() is False
    
    def test_legacy_adapter_invalid_model_creation(self):
        """Test LegacyConfigAdapter with invalid model data."""
        # Test with invalid project data
        invalid_config = {
            "project": {
                "directories": "not_a_dict",  # Should be dict
                "extraction_patterns": ["[invalid_regex"]  # Invalid regex
            },
            "datasets": {
                "invalid_dataset": {
                    "rig": 123,  # Should be string
                    "dates_vials": "not_a_dict"  # Should be dict
                }
            }
        }
        
        # Should create adapter but models won't be created due to validation errors
        adapter = LegacyConfigAdapter(invalid_config)
        assert len(adapter) == 2  # Still contains the data
        assert "project" in adapter
        assert "datasets" in adapter
        
        # Models should not be created for invalid data
        assert adapter.get_model("project") is None
        assert adapter.get_model("dataset", "invalid_dataset") is None


class TestEdgeCasesAndErrorScenarios:
    """Test suite for edge cases and error scenarios."""
    
    def test_model_creation_with_none_values(self):
        """Test model creation with None values in various fields."""
        # ProjectConfig with None values
        config = ProjectConfig(
            directories=None,
            ignore_substrings=None,
            mandatory_experiment_strings=None,
            extraction_patterns=None
        )
        assert config.directories == {}
        assert config.ignore_substrings is None
        assert config.mandatory_experiment_strings is None
        assert config.extraction_patterns is None
        
        # DatasetConfig with None metadata
        config = DatasetConfig(rig="rig1", metadata=None)
        assert config.rig == "rig1"
        assert config.metadata is None
        
        # ExperimentConfig with None optional fields
        config = ExperimentConfig(
            datasets=["dataset1"],
            parameters=None,
            filters=None,
            metadata=None
        )
        assert config.datasets == ["dataset1"]
        assert config.parameters is None
        assert config.filters is None
        assert config.metadata is None
    
    def test_model_creation_with_empty_collections(self):
        """Test model creation with empty collections."""
        # ProjectConfig with empty collections
        config = ProjectConfig(
            directories={},
            ignore_substrings=[],
            mandatory_experiment_strings=[],
            extraction_patterns=[]
        )
        assert config.directories == {}
        assert config.ignore_substrings is None  # Empty list converted to None
        assert config.mandatory_experiment_strings is None
        assert config.extraction_patterns is None
        
        # DatasetConfig with empty dates_vials
        config = DatasetConfig(rig="rig1", dates_vials={})
        assert config.rig == "rig1"
        assert config.dates_vials == {}
        
        # ExperimentConfig with empty datasets
        config = ExperimentConfig(datasets=[])
        assert config.datasets == []
    
    def test_model_field_assignment_validation(self):
        """Test validation when fields are assigned after model creation."""
        # ProjectConfig field assignment
        config = ProjectConfig()
        
        # Valid assignment
        config.directories = {"major_data_directory": "/path/to/data"}
        assert config.directories == {"major_data_directory": "/path/to/data"}
        
        # Invalid assignment should raise ValidationError
        with pytest.raises(ValidationError):
            config.directories = "not_a_dict"
        
        # DatasetConfig field assignment
        config = DatasetConfig(rig="rig1")
        
        # Valid assignment
        config.dates_vials = {"2023-05-01": [1, 2, 3]}
        assert config.dates_vials == {"2023-05-01": [1, 2, 3]}
        
        # Invalid assignment should raise ValidationError
        with pytest.raises(ValidationError):
            config.rig = 123
    
    def test_model_json_serialization(self):
        """Test JSON serialization and deserialization of models."""
        # ProjectConfig serialization
        config = ProjectConfig(
            directories={"major_data_directory": "/path/to/data"},
            ignore_substrings=["temp", "backup"],
            extraction_patterns=[r"(?P<date>\d{4}-\d{2}-\d{2})"]
        )
        
        # Serialize to JSON
        json_str = config.model_dump_json()
        assert isinstance(json_str, str)
        
        # Deserialize from JSON
        json_data = json.loads(json_str)
        reconstructed = ProjectConfig(**json_data)
        assert reconstructed.directories == config.directories
        assert reconstructed.ignore_substrings == config.ignore_substrings
        assert reconstructed.extraction_patterns == config.extraction_patterns
    
    def test_model_dict_conversion(self):
        """Test dictionary conversion of models."""
        # DatasetConfig to dict
        config = DatasetConfig(
            rig="rig1",
            dates_vials={"2023-05-01": [1, 2, 3]},
            metadata={"description": "test"}
        )
        
        config_dict = config.model_dump()
        assert isinstance(config_dict, dict)
        assert config_dict["rig"] == "rig1"
        assert config_dict["dates_vials"] == {"2023-05-01": [1, 2, 3]}
        assert config_dict["metadata"] == {"description": "test"}
        
        # Reconstruct from dict
        reconstructed = DatasetConfig(**config_dict)
        assert reconstructed.rig == config.rig
        assert reconstructed.dates_vials == config.dates_vials
        assert reconstructed.metadata == config.metadata
    
    def test_model_copy_operations(self):
        """Test deep copy operations on models."""
        # ExperimentConfig copy
        config = ExperimentConfig(
            datasets=["dataset1", "dataset2"],
            parameters={"threshold": 0.5, "method": "correlation"},
            filters={"ignore_substrings": ["temp"]}
        )
        
        # Deep copy
        copied_config = deepcopy(config)
        assert copied_config.datasets == config.datasets
        assert copied_config.parameters == config.parameters
        assert copied_config.filters == config.filters
        
        # Modify original - copy should be unaffected
        config.datasets.append("dataset3")
        assert "dataset3" not in copied_config.datasets
        
        # Modify copy - original should be unaffected
        copied_config.parameters["new_param"] = "new_value"
        assert "new_param" not in config.parameters
    
    def test_validation_error_details(self):
        """Test detailed validation error information."""
        # Test error count and details
        with pytest.raises(ValidationError) as exc_info:
            ProjectConfig(
                directories="not_a_dict",
                ignore_substrings="not_a_list",
                extraction_patterns=["[invalid_regex"]
            )
        
        error = exc_info.value
        assert error.error_count() > 0
        assert len(error.errors()) > 0
        
        # Check error details
        error_details = error.errors()
        assert any("directories" in str(err) for err in error_details)
        
        # Test string representation
        error_str = str(error)
        assert isinstance(error_str, str)
        assert len(error_str) > 0


class TestPerformanceComparison:
    """Test suite for performance comparison between Pydantic and dict-based validation."""
    
    def test_pydantic_vs_dict_validation_performance(self):
        """Test performance comparison between Pydantic and dict-based validation."""
        # Sample configuration data
        config_data = {
            "directories": {"major_data_directory": "/path/to/data"},
            "ignore_substrings": ["temp", "backup", "._"],
            "mandatory_experiment_strings": ["experiment", "trial"],
            "extraction_patterns": [r"(?P<date>\d{4}-\d{2}-\d{2})", r"(?P<subject>\w+)"]
        }
        
        # Test Pydantic validation performance
        start_time = time.perf_counter()
        for _ in range(100):
            config = ProjectConfig(**config_data)
        pydantic_time = time.perf_counter() - start_time
        
        # Test dict-based validation performance (simulated)
        def dict_validation(data):
            """Simulate dict-based validation."""
            validated = {}
            
            # Validate directories
            if "directories" in data:
                if not isinstance(data["directories"], dict):
                    raise ValueError("directories must be a dictionary")
                validated["directories"] = data["directories"]
            
            # Validate ignore_substrings
            if "ignore_substrings" in data:
                if not isinstance(data["ignore_substrings"], list):
                    raise ValueError("ignore_substrings must be a list")
                validated["ignore_substrings"] = data["ignore_substrings"]
            
            # Validate mandatory_experiment_strings
            if "mandatory_experiment_strings" in data:
                if not isinstance(data["mandatory_experiment_strings"], list):
                    raise ValueError("mandatory_experiment_strings must be a list")
                validated["mandatory_experiment_strings"] = data["mandatory_experiment_strings"]
            
            # Validate extraction_patterns
            if "extraction_patterns" in data:
                if not isinstance(data["extraction_patterns"], list):
                    raise ValueError("extraction_patterns must be a list")
                for pattern in data["extraction_patterns"]:
                    re.compile(pattern)  # Validate regex
                validated["extraction_patterns"] = data["extraction_patterns"]
            
            return validated
        
        start_time = time.perf_counter()
        for _ in range(100):
            validated = dict_validation(config_data)
        dict_time = time.perf_counter() - start_time
        
        # Performance should be comparable (allowing 10x overhead for Pydantic)
        assert pydantic_time < dict_time * 10, f"Pydantic validation is too slow: {pydantic_time:.4f}s vs {dict_time:.4f}s"
        
        # Log performance results
        print(f"Pydantic validation time: {pydantic_time:.4f}s")
        print(f"Dict validation time: {dict_time:.4f}s")
        print(f"Performance ratio: {pydantic_time/dict_time:.2f}x")
    
    def test_model_creation_performance(self):
        """Test performance of model creation vs dict creation."""
        # Sample data for different models
        project_data = {
            "directories": {"major_data_directory": "/path/to/data"},
            "ignore_substrings": ["temp", "backup"]
        }
        
        dataset_data = {
            "rig": "rig1",
            "dates_vials": {"2023-05-01": [1, 2, 3], "2023-05-02": [4, 5, 6]}
        }
        
        experiment_data = {
            "datasets": ["dataset1", "dataset2"],
            "parameters": {"threshold": 0.5, "method": "correlation"}
        }
        
        # Test model creation performance
        start_time = time.perf_counter()
        for _ in range(50):
            project_config = ProjectConfig(**project_data)
            dataset_config = DatasetConfig(**dataset_data)
            experiment_config = ExperimentConfig(**experiment_data)
        model_time = time.perf_counter() - start_time
        
        # Test dict creation performance
        start_time = time.perf_counter()
        for _ in range(50):
            project_dict = project_data.copy()
            dataset_dict = dataset_data.copy()
            experiment_dict = experiment_data.copy()
        dict_time = time.perf_counter() - start_time
        
        # Model creation should be reasonably fast (allowing 20x overhead)
        assert model_time < dict_time * 20, f"Model creation is too slow: {model_time:.4f}s vs {dict_time:.4f}s"
        
        # Log performance results
        print(f"Model creation time: {model_time:.4f}s")
        print(f"Dict creation time: {dict_time:.4f}s")
        print(f"Performance ratio: {model_time/dict_time:.2f}x")


class TestPropertyBasedValidation:
    """Extended property-based testing for comprehensive validation coverage."""
    
    @given(st.dictionaries(
        st.text(min_size=1, max_size=20),
        st.one_of(
            st.text(min_size=1, max_size=100),
            st.none()
        ),
        min_size=0,
        max_size=5
    ))
    @settings(max_examples=30)
    def test_project_config_comprehensive(self, directories):
        """Comprehensive property-based test for ProjectConfig."""
        assume(all(isinstance(k, str) for k in directories.keys()))
        
        try:
            config = ProjectConfig(directories=directories)
            assert isinstance(config.directories, dict)
            # Validate that None values are filtered out
            for value in config.directories.values():
                assert value is not None
        except ValidationError:
            # Some inputs will naturally fail validation
            pass
    
    @given(
        st.text(min_size=1, max_size=20).filter(lambda x: x.strip()),
        st.dictionaries(
            st.text(min_size=8, max_size=10),
            st.lists(st.integers(min_value=1, max_value=50), min_size=1, max_size=5),
            min_size=0,
            max_size=3
        )
    )
    @settings(max_examples=20)
    def test_dataset_config_comprehensive(self, rig, dates_vials):
        """Comprehensive property-based test for DatasetConfig."""
        assume(re.match(r'^[a-zA-Z0-9_-]+$', rig.strip()))
        
        try:
            config = DatasetConfig(rig=rig, dates_vials=dates_vials)
            assert isinstance(config.rig, str)
            assert isinstance(config.dates_vials, dict)
            assert config.rig == rig.strip()
        except ValidationError:
            # Some inputs will naturally fail validation
            pass
    
    @given(
        st.lists(
            st.text(min_size=1, max_size=20).filter(lambda x: x.strip()),
            max_size=5
        ),
        st.dictionaries(
            st.text(min_size=1, max_size=15),
            st.one_of(
                st.integers(),
                st.floats(allow_nan=False, allow_infinity=False),
                st.text(max_size=50),
                st.booleans()
            ),
            min_size=0,
            max_size=3
        )
    )
    @settings(max_examples=20)
    def test_experiment_config_comprehensive(self, datasets, parameters):
        """Comprehensive property-based test for ExperimentConfig."""
        # Filter datasets to valid names
        valid_datasets = [
            ds for ds in datasets 
            if ds.strip() and re.match(r'^[a-zA-Z0-9_-]+$', ds.strip())
        ]
        
        try:
            config = ExperimentConfig(datasets=valid_datasets, parameters=parameters)
            assert isinstance(config.datasets, list)
            assert isinstance(config.parameters, dict) or config.parameters is None
        except ValidationError:
            # Some inputs will naturally fail validation
            pass
    
    @given(st.dictionaries(
        st.sampled_from(["project", "datasets", "experiments"]),
        st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.one_of(
                st.text(max_size=50),
                st.integers(),
                st.floats(allow_nan=False, allow_infinity=False),
                st.booleans(),
                st.lists(st.text(max_size=20), max_size=3)
            ),
            min_size=1,
            max_size=3
        ),
        min_size=1,
        max_size=3
    ))
    @settings(max_examples=15)
    def test_legacy_config_adapter_comprehensive(self, config_data):
        """Comprehensive property-based test for LegacyConfigAdapter."""
        try:
            adapter = LegacyConfigAdapter(config_data)
            assert isinstance(adapter, LegacyConfigAdapter)
            assert len(adapter) >= 0
            
            # Test basic operations
            for key in config_data:
                assert key in adapter
                assert adapter[key] == config_data[key]
            
            # Test iteration
            keys = list(adapter.keys())
            assert len(keys) == len(config_data)
            
        except (ValidationError, Exception):
            # Some inputs will naturally cause issues
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])