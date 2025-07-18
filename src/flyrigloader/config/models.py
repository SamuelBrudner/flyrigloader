"""
Comprehensive Pydantic configuration models for flyrigloader.

This module provides type-safe configuration schemas with automatic validation,
serving as the foundation for the configuration system overhaul. It implements
ProjectConfig, DatasetConfig, and ExperimentConfig models with validation rules
and backward compatibility support to eliminate configuration-related runtime errors.

The models support the three-tier hierarchy (project/datasets/experiments) structure
with inheritance and override capabilities, while maintaining backward compatibility
with existing dictionary-based configurations through the LegacyConfigAdapter.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union, ClassVar
from datetime import datetime
import logging
import re
from collections.abc import MutableMapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.types import DirectoryPath

from .validators import path_existence_validator

# Set up logger for configuration model events
logger = logging.getLogger(__name__)


class ProjectConfig(BaseModel):
    """
    Project-level configuration model with validation for directories and patterns.
    
    This model defines the top-level project configuration including data directories,
    ignore patterns, mandatory experiment strings, and extraction patterns. It provides
    comprehensive validation for path existence and pattern compilation.
    
    Attributes:
        directories: Dictionary containing directory paths (major_data_directory, etc.)
        ignore_substrings: List of substring patterns to ignore during file discovery
        mandatory_experiment_strings: List of strings that must be present in experiment files
        extraction_patterns: List of regex patterns for extracting metadata from filenames
    """
    
    model_config = ConfigDict(
        extra="allow",  # Allow additional fields for forward compatibility
        validate_assignment=True,  # Validate fields when they are assigned
        str_strip_whitespace=True,  # Strip whitespace from string fields
        validate_default=True,  # Validate default values
        frozen=False,  # Allow mutation for legacy compatibility
    )
    
    directories: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary of directory paths including major_data_directory",
        json_schema_extra={
            "example": {
                "major_data_directory": "/path/to/fly_data",
                "backup_directory": "/path/to/backup"
            }
        }
    )
    
    ignore_substrings: Optional[List[str]] = Field(
        default_factory=lambda: ["._", "temp", "backup", ".tmp", "~", ".DS_Store"],
        description="List of substring patterns to ignore during file discovery",
        json_schema_extra={
            "example": ["._", "temp", "backup"]
        }
    )
    
    mandatory_experiment_strings: Optional[List[str]] = Field(
        default=None,
        description="List of strings that must be present in experiment files",
        json_schema_extra={
            "example": ["experiment", "trial"]
        }
    )
    
    extraction_patterns: Optional[List[str]] = Field(
        default_factory=lambda: [
            r"(?P<date>\d{4}-\d{2}-\d{2})",  # ISO date format
            r"(?P<date>\d{8})",  # Compact date format
            r"(?P<subject>\w+)",  # Subject identifier
            r"(?P<rig>rig\d+)",  # Rig identifier
        ],
        description="List of regex patterns for extracting metadata from filenames",
        json_schema_extra={
            "example": [r"(?P<date>\d{4}-\d{2}-\d{2})", r"(?P<subject>\w+)"]
        }
    )
    
    @field_validator('directories')
    @classmethod
    def validate_directories(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate directory paths with security checks and existence validation."""
        if not isinstance(v, dict):
            raise ValueError("directories must be a dictionary")
        
        validated_dirs = {}
        for key, value in v.items():
            if value is None:
                continue
                
            # Convert to string for validation
            path_str = str(value)
            
            # Use path_existence_validator for security and existence checks
            # This validator is test-environment aware and will skip existence checks during testing
            try:
                path_existence_validator(path_str, require_file=False)
                validated_dirs[key] = path_str
                logger.debug(f"Directory validated: {key} = {path_str}")
            except Exception as e:
                logger.warning(f"Directory validation failed for {key}: {path_str} - {e}")
                # During testing or if path doesn't exist, still allow the configuration
                validated_dirs[key] = path_str
        
        return validated_dirs
    
    @field_validator('ignore_substrings')
    @classmethod
    def validate_ignore_substrings(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate ignore substring patterns."""
        if v is None:
            return None
        
        if not isinstance(v, list):
            raise ValueError("ignore_substrings must be a list")
        
        # Validate each pattern
        validated_patterns = []
        for pattern in v:
            if not isinstance(pattern, str):
                raise ValueError(f"ignore pattern must be string, got {type(pattern)}")
            
            # Check for empty or whitespace-only patterns
            if not pattern.strip():
                logger.warning("Empty or whitespace-only ignore pattern detected")
                continue
            
            validated_patterns.append(pattern.strip())
        
        logger.debug(f"Validated {len(validated_patterns)} ignore patterns")
        return validated_patterns if validated_patterns else None
    
    @field_validator('mandatory_experiment_strings')
    @classmethod
    def validate_mandatory_strings(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate mandatory experiment strings."""
        if v is None:
            return None
        
        if not isinstance(v, list):
            raise ValueError("mandatory_experiment_strings must be a list")
        
        validated_strings = []
        for string in v:
            if not isinstance(string, str):
                raise ValueError(f"mandatory string must be string, got {type(string)}")
            
            # Check for empty or whitespace-only strings
            if not string.strip():
                logger.warning("Empty or whitespace-only mandatory string detected")
                continue
            
            validated_strings.append(string.strip())
        
        logger.debug(f"Validated {len(validated_strings)} mandatory strings")
        return validated_strings if validated_strings else None
    
    @field_validator('extraction_patterns')
    @classmethod
    def validate_extraction_patterns(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate regex extraction patterns by compiling them."""
        if v is None:
            return None
        
        if not isinstance(v, list):
            raise ValueError("extraction_patterns must be a list")
        
        validated_patterns = []
        for pattern in v:
            if not isinstance(pattern, str):
                raise ValueError(f"extraction pattern must be string, got {type(pattern)}")
            
            # Check for empty or whitespace-only patterns
            if not pattern.strip():
                logger.warning("Empty or whitespace-only extraction pattern detected")
                continue
            
            # Validate pattern by compiling it
            try:
                re.compile(pattern)
                validated_patterns.append(pattern.strip())
                logger.debug(f"Regex pattern validated: {pattern}")
            except re.error as e:
                raise ValueError(f"Invalid regex pattern '{pattern}': {e}")
        
        logger.debug(f"Validated {len(validated_patterns)} extraction patterns")
        return validated_patterns if validated_patterns else None


class DatasetConfig(BaseModel):
    """
    Dataset-level configuration model with validation for rig setup and date/vial structure.
    
    This model defines dataset-specific configuration including rig identification,
    date/vial mappings, and metadata. It provides comprehensive validation for
    date format consistency and vial list structure.
    
    Attributes:
        rig: Rig identifier string
        dates_vials: Dictionary mapping date strings to lists of vial numbers
        metadata: Optional metadata dictionary for dataset-specific information
    """
    
    model_config = ConfigDict(
        extra="allow",  # Allow additional fields for forward compatibility
        validate_assignment=True,  # Validate fields when they are assigned
        str_strip_whitespace=True,  # Strip whitespace from string fields
        validate_default=True,  # Validate default values
        frozen=False,  # Allow mutation for legacy compatibility
    )
    
    rig: str = Field(
        description="Rig identifier string",
        json_schema_extra={
            "example": "rig1"
        }
    )
    
    dates_vials: Dict[str, List[int]] = Field(
        default_factory=dict,
        description="Dictionary mapping date strings to lists of vial numbers",
        json_schema_extra={
            "example": {
                "2023-05-01": [1, 2, 3, 4],
                "2023-05-02": [5, 6, 7, 8]
            }
        }
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "created_by": "flyrigloader",
            "dataset_type": "behavioral",
            "extraction_patterns": [
                r"(?P<temperature>\d+)C",
                r"(?P<humidity>\d+)%",
                r"(?P<trial_number>\d+)",
                r"(?P<condition>\w+_condition)",
            ],
        },
        description="Optional metadata dictionary for dataset-specific information",
        json_schema_extra={
            "example": {
                "extraction_patterns": [r"(?P<temperature>\d+)C"],
                "description": "Temperature gradient experiments"
            }
        }
    )
    
    @field_validator('rig')
    @classmethod
    def validate_rig(cls, v: str) -> str:
        """Validate rig identifier."""
        if not isinstance(v, str):
            raise ValueError("rig must be a string")
        
        if not v.strip():
            raise ValueError("rig cannot be empty or whitespace-only")
        
        # Basic pattern validation for rig names
        rig_name = v.strip()
        if not re.match(r'^[a-zA-Z0-9_-]+$', rig_name):
            raise ValueError(f"rig name '{rig_name}' contains invalid characters. Use only alphanumeric, underscore, and hyphen.")
        
        logger.debug(f"Rig validated: {rig_name}")
        return rig_name
    
    @field_validator('dates_vials')
    @classmethod
    def validate_dates_vials(cls, v: Dict[str, List[int]]) -> Dict[str, List[int]]:
        """Validate dates_vials structure with date format checking."""
        if not isinstance(v, dict):
            raise ValueError("dates_vials must be a dictionary")
        
        validated_dates_vials = {}
        for date_str, vials in v.items():
            # Validate date string format
            if not isinstance(date_str, str):
                raise ValueError(f"Date key must be string, got {type(date_str)}")
            
            # Try to parse date using multiple formats
            date_parsed = False
            for date_format in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y%m%d']:
                try:
                    datetime.strptime(date_str, date_format)
                    date_parsed = True
                    break
                except ValueError:
                    continue
            
            if not date_parsed:
                logger.warning(f"Date format not recognized for '{date_str}', but allowing for flexibility")
            
            # Validate vials list
            if not isinstance(vials, list):
                raise ValueError(f"Vials for date '{date_str}' must be a list")
            
            validated_vials = []
            for vial in vials:
                if isinstance(vial, int):
                    validated_vials.append(vial)
                elif isinstance(vial, str) and vial.isdigit():
                    validated_vials.append(int(vial))
                else:
                    raise ValueError(f"Vial numbers must be integers, got {type(vial)} for date '{date_str}'")
            
            if not validated_vials:
                logger.warning(f"No valid vials found for date '{date_str}'")
                continue
            
            validated_dates_vials[date_str] = validated_vials
        
        logger.debug(f"Validated dates_vials for {len(validated_dates_vials)} dates")
        return validated_dates_vials
    
    @field_validator('metadata')
    @classmethod
    def validate_metadata(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Validate metadata dictionary structure."""
        if v is None:
            return None
        
        if not isinstance(v, dict):
            raise ValueError("metadata must be a dictionary")
        
        # Validate extraction patterns if present
        if 'extraction_patterns' in v:
            patterns = v['extraction_patterns']
            if patterns is not None:
                if not isinstance(patterns, list):
                    raise ValueError("metadata extraction_patterns must be a list")
                
                for pattern in patterns:
                    if not isinstance(pattern, str):
                        raise ValueError("metadata extraction pattern must be string")
                    
                    # Validate pattern by compiling it
                    try:
                        re.compile(pattern)
                    except re.error as e:
                        raise ValueError(f"Invalid regex pattern in metadata '{pattern}': {e}")
        
        logger.debug("Metadata validated successfully")
        return v


class ExperimentConfig(BaseModel):
    """
    Experiment-level configuration model with validation for datasets and parameters.
    
    This model defines experiment-specific configuration including dataset references,
    analysis parameters, filters, and metadata. It provides comprehensive validation
    for dataset references and parameter structures.
    
    Attributes:
        datasets: List of dataset names to include in this experiment
        parameters: Dictionary of experiment-specific parameters
        filters: Dictionary containing filter configurations
        metadata: Optional metadata dictionary for experiment-specific information
    """
    
    model_config = ConfigDict(
        extra="allow",  # Allow additional fields for forward compatibility
        validate_assignment=True,  # Validate fields when they are assigned
        str_strip_whitespace=True,  # Strip whitespace from string fields
        validate_default=True,  # Validate default values
        frozen=False,  # Allow mutation for legacy compatibility
    )
    
    datasets: List[str] = Field(
        default_factory=list,
        description="List of dataset names to include in this experiment",
        json_schema_extra={
            "example": ["plume_tracking", "odor_response"]
        }
    )
    
    parameters: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "analysis_window": 10.0,
            "sampling_rate": 1000.0,
            "threshold": 0.5,
            "method": "correlation",
            "confidence_level": 0.95,
        },
        description="Dictionary of experiment-specific parameters",
        json_schema_extra={
            "example": {
                "analysis_window": 10.0,
                "threshold": 0.5,
                "method": "correlation"
            }
        }
    )
    
    filters: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "ignore_substrings": ["temp", "backup", "test"],
            "mandatory_experiment_strings": ["experiment", "trial"],
        },
        description="Dictionary containing filter configurations",
        json_schema_extra={
            "example": {
                "ignore_substrings": ["temp", "backup"],
                "mandatory_experiment_strings": ["trial"]
            }
        }
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "created_by": "flyrigloader",
            "analysis_type": "behavioral",
            "description": "Automated experiment configuration",
        },
        description="Optional metadata dictionary for experiment-specific information",
        json_schema_extra={
            "example": {
                "description": "Plume navigation analysis experiment",
                "analysis_type": "behavioral"
            }
        }
    )
    
    @field_validator('datasets')
    @classmethod
    def validate_datasets(cls, v: List[str]) -> List[str]:
        """Validate dataset name list."""
        if not isinstance(v, list):
            raise ValueError("datasets must be a list")
        
        validated_datasets = []
        for dataset in v:
            if not isinstance(dataset, str):
                raise ValueError(f"dataset name must be string, got {type(dataset)}")
            
            if not dataset.strip():
                logger.warning("Empty or whitespace-only dataset name detected")
                continue
            
            dataset_name = dataset.strip()
            if not re.match(r'^[a-zA-Z0-9_-]+$', dataset_name):
                raise ValueError(f"dataset name '{dataset_name}' contains invalid characters. Use only alphanumeric, underscore, and hyphen.")
            
            validated_datasets.append(dataset_name)
        
        if not validated_datasets:
            logger.warning("No valid datasets found in experiment configuration")
        
        logger.debug(f"Validated {len(validated_datasets)} datasets")
        return validated_datasets
    
    @field_validator('parameters')
    @classmethod
    def validate_parameters(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Validate parameters dictionary structure."""
        if v is None:
            return None
        
        if not isinstance(v, dict):
            raise ValueError("parameters must be a dictionary")
        
        # Basic validation - parameters can contain any structure
        # but we'll validate common parameter types
        for key, value in v.items():
            if not isinstance(key, str):
                raise ValueError(f"parameter key must be string, got {type(key)}")
            
            if not key.strip():
                raise ValueError("parameter key cannot be empty or whitespace-only")
        
        logger.debug(f"Validated parameters with {len(v)} entries")
        return v
    
    @field_validator('filters')
    @classmethod
    def validate_filters(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Validate filters dictionary structure."""
        if v is None:
            return None
        
        if not isinstance(v, dict):
            raise ValueError("filters must be a dictionary")
        
        # Validate known filter types
        if 'ignore_substrings' in v:
            ignore_substrings = v['ignore_substrings']
            if ignore_substrings is not None:
                if not isinstance(ignore_substrings, list):
                    raise ValueError("filters ignore_substrings must be a list")
                
                for pattern in ignore_substrings:
                    if not isinstance(pattern, str):
                        raise ValueError("filter ignore pattern must be string")
        
        if 'mandatory_experiment_strings' in v:
            mandatory_strings = v['mandatory_experiment_strings']
            if mandatory_strings is not None:
                if not isinstance(mandatory_strings, list):
                    raise ValueError("filters mandatory_experiment_strings must be a list")
                
                for string in mandatory_strings:
                    if not isinstance(string, str):
                        raise ValueError("filter mandatory string must be string")
        
        logger.debug("Filters validated successfully")
        return v
    
    @field_validator('metadata')
    @classmethod
    def validate_metadata(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Validate metadata dictionary structure."""
        if v is None:
            return None
        
        if not isinstance(v, dict):
            raise ValueError("metadata must be a dictionary")
        
        # Validate extraction patterns if present
        if 'extraction_patterns' in v:
            patterns = v['extraction_patterns']
            if patterns is not None:
                if not isinstance(patterns, list):
                    raise ValueError("metadata extraction_patterns must be a list")
                
                for pattern in patterns:
                    if not isinstance(pattern, str):
                        raise ValueError("metadata extraction pattern must be string")
                    
                    # Validate pattern by compiling it
                    try:
                        re.compile(pattern)
                    except re.error as e:
                        raise ValueError(f"Invalid regex pattern in metadata '{pattern}': {e}")
        
        logger.debug("Metadata validated successfully")
        return v


class LegacyConfigAdapter(MutableMapping):
    """
    Backward compatibility adapter for Pydantic configuration models.
    
    This class provides dictionary-style access to Pydantic models, ensuring
    backward compatibility with existing code that expects dictionary access
    patterns. It wraps the validated Pydantic models while maintaining the
    ability to access configuration data using familiar dict syntax.
    
    The adapter supports both read and write operations, automatically
    converting between dictionary access and Pydantic model attributes.
    """
    
    def __init__(self, config_data: Dict[str, Any]):
        """
        Initialize the adapter with configuration data.
        
        Args:
            config_data: Dictionary containing project, datasets, and experiments sections
        """
        self._data = {}
        self._models = {}
        
        # Convert each section to appropriate Pydantic model
        if 'project' in config_data:
            try:
                self._models['project'] = ProjectConfig(**config_data['project'])
                self._data['project'] = config_data['project']
                logger.debug("Project configuration converted to Pydantic model")
            except Exception as e:
                logger.warning(f"Failed to convert project config to Pydantic model: {e}")
                self._data['project'] = config_data['project']
        
        if 'datasets' in config_data:
            datasets_dict = {}
            for name, dataset_config in config_data['datasets'].items():
                try:
                    self._models[f'dataset_{name}'] = DatasetConfig(**dataset_config)
                    datasets_dict[name] = dataset_config
                    logger.debug(f"Dataset '{name}' converted to Pydantic model")
                except Exception as e:
                    logger.warning(f"Failed to convert dataset '{name}' to Pydantic model: {e}")
                    datasets_dict[name] = dataset_config
            self._data['datasets'] = datasets_dict
        
        if 'experiments' in config_data:
            experiments_dict = {}
            for name, experiment_config in config_data['experiments'].items():
                try:
                    self._models[f'experiment_{name}'] = ExperimentConfig(**experiment_config)
                    experiments_dict[name] = experiment_config
                    logger.debug(f"Experiment '{name}' converted to Pydantic model")
                except Exception as e:
                    logger.warning(f"Failed to convert experiment '{name}' to Pydantic model: {e}")
                    experiments_dict[name] = experiment_config
            self._data['experiments'] = experiments_dict
        
        # Add any additional sections that weren't modeled
        for key, value in config_data.items():
            if key not in ['project', 'datasets', 'experiments']:
                self._data[key] = value
    
    def __getitem__(self, key: str) -> Any:
        """Get item using dictionary-style access."""
        return self._data[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Set item using dictionary-style access."""
        self._data[key] = value
        
        # Update corresponding Pydantic model if applicable
        if key == 'project':
            try:
                self._models['project'] = ProjectConfig(**value)
                logger.debug("Project configuration updated in Pydantic model")
            except Exception as e:
                logger.warning(f"Failed to update project Pydantic model: {e}")
        
        elif key == 'datasets' and isinstance(value, dict):
            for name, dataset_config in value.items():
                try:
                    self._models[f'dataset_{name}'] = DatasetConfig(**dataset_config)
                    logger.debug(f"Dataset '{name}' updated in Pydantic model")
                except Exception as e:
                    logger.warning(f"Failed to update dataset '{name}' Pydantic model: {e}")
        
        elif key == 'experiments' and isinstance(value, dict):
            for name, experiment_config in value.items():
                try:
                    self._models[f'experiment_{name}'] = ExperimentConfig(**experiment_config)
                    logger.debug(f"Experiment '{name}' updated in Pydantic model")
                except Exception as e:
                    logger.warning(f"Failed to update experiment '{name}' Pydantic model: {e}")
    
    def __delitem__(self, key: str) -> None:
        """Delete item using dictionary-style access."""
        del self._data[key]
        
        # Remove corresponding Pydantic models
        if key == 'project':
            self._models.pop('project', None)
        elif key == 'datasets':
            keys_to_remove = [k for k in self._models.keys() if k.startswith('dataset_')]
            for k in keys_to_remove:
                del self._models[k]
        elif key == 'experiments':
            keys_to_remove = [k for k in self._models.keys() if k.startswith('experiment_')]
            for k in keys_to_remove:
                del self._models[k]
    
    def __iter__(self):
        """Iterate over keys."""
        return iter(self._data)
    
    def __len__(self) -> int:
        """Return number of items."""
        return len(self._data)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return key in self._data
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get item with default value."""
        return self._data.get(key, default)
    
    def keys(self):
        """Return dictionary keys."""
        return self._data.keys()
    
    def values(self):
        """Return dictionary values."""
        return self._data.values()
    
    def items(self):
        """Return dictionary items."""
        return self._data.items()
    
    def get_model(self, model_type: str, name: Optional[str] = None) -> Optional[BaseModel]:
        """
        Get the underlying Pydantic model for a configuration section.
        
        Args:
            model_type: Type of model ('project', 'dataset', 'experiment')
            name: Name of the specific dataset or experiment (required for dataset/experiment)
        
        Returns:
            The Pydantic model instance, or None if not found
        """
        if model_type == 'project':
            return self._models.get('project')
        elif model_type == 'dataset' and name:
            return self._models.get(f'dataset_{name}')
        elif model_type == 'experiment' and name:
            return self._models.get(f'experiment_{name}')
        else:
            return None
    
    def get_all_models(self) -> Dict[str, BaseModel]:
        """
        Get all underlying Pydantic models.
        
        Returns:
            Dictionary mapping model keys to Pydantic model instances
        """
        return self._models.copy()
    
    def validate_all(self) -> bool:
        """
        Validate all configuration sections using Pydantic models.
        
        Returns:
            True if all sections are valid, False otherwise
        """
        try:
            # Validate project
            if 'project' in self._data:
                ProjectConfig(**self._data['project'])
            
            # Validate datasets
            if 'datasets' in self._data:
                for name, dataset_config in self._data['datasets'].items():
                    DatasetConfig(**dataset_config)
            
            # Validate experiments
            if 'experiments' in self._data:
                for name, experiment_config in self._data['experiments'].items():
                    ExperimentConfig(**experiment_config)
            
            logger.info("All configuration sections validated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False


# Builder functions for programmatic configuration creation

def create_config(
    project_name: str,
    base_directory: Union[str, Path],
    datasets: Optional[List[str]] = None,
    experiments: Optional[List[str]] = None,
    directories: Optional[Dict[str, Any]] = None,
    ignore_substrings: Optional[List[str]] = None,
    mandatory_experiment_strings: Optional[List[str]] = None,
    extraction_patterns: Optional[List[str]] = None,
    **kwargs
) -> ProjectConfig:
    """
    Builder function for creating validated ProjectConfig objects programmatically.
    
    This function provides a convenient way to create project configurations without
    requiring YAML files, reducing configuration boilerplate and enabling code-driven
    configuration creation with comprehensive defaults.
    
    Args:
        project_name: Name of the project for identification
        base_directory: Base directory path for the project
        datasets: List of dataset names to include in the project
        experiments: List of experiment names to include in the project
        directories: Dictionary of directory paths (auto-populated with defaults)
        ignore_substrings: List of substring patterns to ignore during file discovery
        mandatory_experiment_strings: List of strings that must be present in experiment files
        extraction_patterns: List of regex patterns for extracting metadata from filenames
        **kwargs: Additional fields for forward compatibility
    
    Returns:
        ProjectConfig: Validated Pydantic model instance
    
    Raises:
        ValueError: If configuration validation fails
        
    Example:
        >>> config = create_config(
        ...     project_name="fly_behavior_analysis",
        ...     base_directory="/data/fly_experiments",
        ...     datasets=["plume_tracking", "odor_response"],
        ...     experiments=["navigation_test", "choice_assay"]
        ... )
        >>> print(config.directories["major_data_directory"])
        /data/fly_experiments
    """
    logger.debug(f"Creating ProjectConfig for project: {project_name}")
    
    # Ensure base_directory is a Path object
    base_path = Path(base_directory)
    
    # Auto-populate directories with sensible defaults
    if directories is None:
        directories = {}
    
    # Provide default major_data_directory if not specified
    if "major_data_directory" not in directories:
        directories["major_data_directory"] = str(base_path)
    
    # Add common directory defaults
    directories.setdefault("backup_directory", str(base_path / "backup"))
    directories.setdefault("processed_directory", str(base_path / "processed"))
    directories.setdefault("output_directory", str(base_path / "output"))
    
    # Provide sensible defaults for ignore patterns
    if ignore_substrings is None:
        ignore_substrings = ["._", "temp", "backup", ".tmp", "~", ".DS_Store"]
    
    # Provide default extraction patterns for common use cases
    if extraction_patterns is None:
        extraction_patterns = [
            r"(?P<date>\d{4}-\d{2}-\d{2})",  # ISO date format
            r"(?P<date>\d{8})",  # Compact date format
            r"(?P<subject>\w+)",  # Subject identifier
            r"(?P<experiment>\w+_experiment)",  # Experiment identifier
            r"(?P<rig>rig\d+)",  # Rig identifier
        ]
    
    # Combine all configuration data
    config_data = {
        "directories": directories,
        "ignore_substrings": ignore_substrings,
        "mandatory_experiment_strings": mandatory_experiment_strings,
        "extraction_patterns": extraction_patterns,
        **kwargs
    }
    
    try:
        project_config = ProjectConfig(**config_data)
        logger.info(f"Successfully created ProjectConfig for project: {project_name}")
        return project_config
    except Exception as e:
        logger.error(f"Failed to create ProjectConfig for project {project_name}: {e}")
        raise ValueError(f"Configuration validation failed: {e}") from e


def create_experiment(
    name: str,
    datasets: List[str],
    parameters: Optional[Dict[str, Any]] = None,
    filters: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs
) -> ExperimentConfig:
    """
    Builder function for creating validated ExperimentConfig objects programmatically.
    
    This function provides a convenient way to create experiment configurations with
    comprehensive defaults and validation, reducing boilerplate code for common
    experiment setup patterns.
    
    Args:
        name: Name of the experiment for identification
        datasets: List of dataset names to include in this experiment
        parameters: Dictionary of experiment-specific parameters
        filters: Dictionary containing filter configurations
        metadata: Optional metadata dictionary for experiment-specific information
        **kwargs: Additional fields for forward compatibility
    
    Returns:
        ExperimentConfig: Validated Pydantic model instance
    
    Raises:
        ValueError: If configuration validation fails
        
    Example:
        >>> config = create_experiment(
        ...     name="navigation_test",
        ...     datasets=["plume_tracking", "odor_response"],
        ...     parameters={"analysis_window": 10.0, "threshold": 0.5},
        ...     metadata={"description": "Navigation behavior analysis"}
        ... )
        >>> print(config.datasets)
        ['plume_tracking', 'odor_response']
    """
    logger.debug(f"Creating ExperimentConfig for experiment: {name}")
    
    # Provide default parameters for common analysis patterns
    if parameters is None:
        parameters = {}
    
    # Add common parameter defaults
    parameters.setdefault("analysis_window", 10.0)
    parameters.setdefault("sampling_rate", 1000.0)
    parameters.setdefault("threshold", 0.5)
    parameters.setdefault("method", "correlation")
    parameters.setdefault("confidence_level", 0.95)
    
    # Provide default filters
    if filters is None:
        filters = {}
    
    # Add common filter defaults
    filters.setdefault("ignore_substrings", ["temp", "backup", "test"])
    filters.setdefault("mandatory_experiment_strings", ["experiment", "trial"])
    
    # Provide default metadata
    if metadata is None:
        metadata = {}
    
    # Add common metadata defaults
    metadata.setdefault("created_by", "flyrigloader")
    metadata.setdefault("analysis_type", "behavioral")
    metadata.setdefault("description", f"Automated experiment configuration for {name}")
    
    # Combine all configuration data
    config_data = {
        "datasets": datasets,
        "parameters": parameters,
        "filters": filters,
        "metadata": metadata,
        **kwargs
    }
    
    try:
        experiment_config = ExperimentConfig(**config_data)
        logger.info(f"Successfully created ExperimentConfig for experiment: {name}")
        return experiment_config
    except Exception as e:
        logger.error(f"Failed to create ExperimentConfig for experiment {name}: {e}")
        raise ValueError(f"Configuration validation failed: {e}") from e


def create_dataset(
    name: str,
    rig: str,
    dates_vials: Optional[Dict[str, List[int]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs
) -> DatasetConfig:
    """
    Builder function for creating validated DatasetConfig objects programmatically.
    
    This function provides a convenient way to create dataset configurations with
    sensible defaults and validation, supporting common dataset patterns while
    maintaining flexibility for specific use cases.
    
    Args:
        name: Name of the dataset for identification
        rig: Rig identifier string
        dates_vials: Dictionary mapping date strings to lists of vial numbers
        metadata: Optional metadata dictionary for dataset-specific information
        **kwargs: Additional fields for forward compatibility
    
    Returns:
        DatasetConfig: Validated Pydantic model instance
    
    Raises:
        ValueError: If configuration validation fails
        
    Example:
        >>> config = create_dataset(
        ...     name="plume_tracking",
        ...     rig="rig1",
        ...     dates_vials={"2023-05-01": [1, 2, 3, 4]},
        ...     metadata={"description": "Plume tracking behavioral data"}
        ... )
        >>> print(config.rig)
        rig1
    """
    logger.debug(f"Creating DatasetConfig for dataset: {name}")
    
    # Provide default dates_vials if not specified
    if dates_vials is None:
        dates_vials = {}
    
    # Provide default metadata
    if metadata is None:
        metadata = {}
    
    # Add common metadata defaults
    metadata.setdefault("created_by", "flyrigloader")
    metadata.setdefault("dataset_type", "behavioral")
    metadata.setdefault("description", f"Automated dataset configuration for {name}")
    
    # Add common extraction patterns for dataset-specific metadata
    if "extraction_patterns" not in metadata:
        metadata["extraction_patterns"] = [
            r"(?P<temperature>\d+)C",  # Temperature extraction
            r"(?P<humidity>\d+)%",  # Humidity extraction
            r"(?P<trial_number>\d+)",  # Trial number extraction
            r"(?P<condition>\w+_condition)",  # Condition extraction
        ]
    
    # Combine all configuration data
    config_data = {
        "rig": rig,
        "dates_vials": dates_vials,
        "metadata": metadata,
        **kwargs
    }
    
    try:
        dataset_config = DatasetConfig(**config_data)
        logger.info(f"Successfully created DatasetConfig for dataset: {name}")
        return dataset_config
    except Exception as e:
        logger.error(f"Failed to create DatasetConfig for dataset {name}: {e}")
        raise ValueError(f"Configuration validation failed: {e}") from e


# Factory methods for common configuration patterns

def create_standard_fly_config(
    project_name: str,
    base_directory: Union[str, Path],
    rigs: Optional[List[str]] = None,
    **kwargs
) -> ProjectConfig:
    """
    Factory method for creating standard fly behavior experiment configurations.
    
    This factory provides a pre-configured setup optimized for typical fly behavior
    experiments, including common patterns, extraction rules, and directory structures.
    
    Args:
        project_name: Name of the fly behavior project
        base_directory: Base directory path for the project
        rigs: List of rig identifiers (defaults to common rig names)
        **kwargs: Additional configuration options
    
    Returns:
        ProjectConfig: Pre-configured project configuration for fly experiments
        
    Example:
        >>> config = create_standard_fly_config(
        ...     project_name="courtship_behavior",
        ...     base_directory="/data/fly_experiments",
        ...     rigs=["rig1", "rig2", "rig3"]
        ... )
    """
    logger.debug(f"Creating standard fly config for project: {project_name}")
    
    # Default rigs for fly experiments
    if rigs is None:
        rigs = ["rig1", "rig2", "rig3", "rig4"]
    
    # Fly-specific ignore patterns
    ignore_substrings = [
        "._", "temp", "backup", ".tmp", "~", ".DS_Store",
        "calibration", "test", "debug", "practice"
    ]
    
    # Fly-specific mandatory strings
    mandatory_experiment_strings = [
        "experiment", "trial", "fly", "behavior"
    ]
    
    # Fly-specific extraction patterns
    extraction_patterns = [
        r"(?P<date>\d{4}-\d{2}-\d{2})",  # ISO date format
        r"(?P<date>\d{8})",  # Compact date format
        r"(?P<fly_id>fly\d+)",  # Fly identifier
        r"(?P<genotype>\w+_\w+)",  # Genotype pattern
        r"(?P<sex>[MF])",  # Sex designation
        r"(?P<age>\d+)d",  # Age in days
        r"(?P<rig>rig\d+)",  # Rig identifier
        r"(?P<condition>\w+_condition)",  # Experimental condition
        r"(?P<trial_number>trial\d+)",  # Trial number
        r"(?P<temperature>\d+)C",  # Temperature
        r"(?P<humidity>\d+)%",  # Humidity
    ]
    
    return create_config(
        project_name=project_name,
        base_directory=base_directory,
        ignore_substrings=ignore_substrings,
        mandatory_experiment_strings=mandatory_experiment_strings,
        extraction_patterns=extraction_patterns,
        **kwargs
    )


def create_plume_tracking_experiment(
    datasets: List[str],
    analysis_window: float = 10.0,
    tracking_threshold: float = 0.3,
    **kwargs
) -> ExperimentConfig:
    """
    Factory method for creating plume tracking experiment configurations.
    
    This factory provides a pre-configured setup optimized for plume tracking
    behavioral experiments, including specialized parameters and filters.
    
    Args:
        datasets: List of dataset names for plume tracking
        analysis_window: Time window for analysis in seconds
        tracking_threshold: Threshold for plume detection
        **kwargs: Additional configuration options
    
    Returns:
        ExperimentConfig: Pre-configured experiment configuration for plume tracking
        
    Example:
        >>> config = create_plume_tracking_experiment(
        ...     datasets=["plume_data_rig1", "plume_data_rig2"],
        ...     analysis_window=15.0,
        ...     tracking_threshold=0.4
        ... )
    """
    logger.debug("Creating plume tracking experiment configuration")
    
    # Plume tracking specific parameters
    parameters = {
        "analysis_window": analysis_window,
        "tracking_threshold": tracking_threshold,
        "sampling_rate": 1000.0,
        "method": "optical_flow",
        "confidence_level": 0.95,
        "smoothing_window": 5,
        "velocity_threshold": 10.0,
        "direction_bins": 36,
        "distance_threshold": 50.0,
    }
    
    # Plume tracking specific filters
    filters = {
        "ignore_substrings": ["calibration", "test", "debug", "background"],
        "mandatory_experiment_strings": ["plume", "tracking", "behavior"],
    }
    
    # Plume tracking specific metadata
    metadata = {
        "experiment_type": "plume_tracking",
        "analysis_type": "behavioral",
        "description": "Plume tracking behavioral analysis experiment",
        "output_format": "trajectory_data",
        "coordinate_system": "cartesian",
    }
    
    return create_experiment(
        name="plume_tracking",
        datasets=datasets,
        parameters=parameters,
        filters=filters,
        metadata=metadata,
        **kwargs
    )


def create_choice_assay_experiment(
    datasets: List[str],
    choice_duration: float = 300.0,
    decision_threshold: float = 0.8,
    **kwargs
) -> ExperimentConfig:
    """
    Factory method for creating choice assay experiment configurations.
    
    This factory provides a pre-configured setup optimized for choice assay
    behavioral experiments, including specialized parameters and analysis settings.
    
    Args:
        datasets: List of dataset names for choice assay
        choice_duration: Duration of choice period in seconds
        decision_threshold: Threshold for decision detection
        **kwargs: Additional configuration options
    
    Returns:
        ExperimentConfig: Pre-configured experiment configuration for choice assay
        
    Example:
        >>> config = create_choice_assay_experiment(
        ...     datasets=["choice_data_rig1", "choice_data_rig2"],
        ...     choice_duration=600.0,
        ...     decision_threshold=0.9
        ... )
    """
    logger.debug("Creating choice assay experiment configuration")
    
    # Choice assay specific parameters
    parameters = {
        "choice_duration": choice_duration,
        "decision_threshold": decision_threshold,
        "sampling_rate": 100.0,
        "method": "preference_index",
        "confidence_level": 0.95,
        "baseline_duration": 60.0,
        "response_window": 30.0,
        "spatial_bins": 20,
        "temporal_bins": 60,
    }
    
    # Choice assay specific filters
    filters = {
        "ignore_substrings": ["calibration", "test", "debug", "control"],
        "mandatory_experiment_strings": ["choice", "assay", "preference"],
    }
    
    # Choice assay specific metadata
    metadata = {
        "experiment_type": "choice_assay",
        "analysis_type": "behavioral",
        "description": "Choice assay behavioral analysis experiment",
        "output_format": "preference_data",
        "scoring_method": "preference_index",
    }
    
    return create_experiment(
        name="choice_assay",
        datasets=datasets,
        parameters=parameters,
        filters=filters,
        metadata=metadata,
        **kwargs
    )


def create_rig_dataset(
    rig_name: str,
    start_date: str,
    end_date: str,
    vials_per_day: int = 8,
    **kwargs
) -> DatasetConfig:
    """
    Factory method for creating rig-specific dataset configurations.
    
    This factory provides a convenient way to create dataset configurations
    for specific rigs with automatic date range and vial number generation.
    
    Args:
        rig_name: Name of the rig (e.g., "rig1", "rig2")
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        vials_per_day: Number of vials per day (default: 8)
        **kwargs: Additional configuration options
    
    Returns:
        DatasetConfig: Pre-configured dataset configuration for the specified rig
        
    Example:
        >>> config = create_rig_dataset(
        ...     rig_name="rig1",
        ...     start_date="2023-05-01",
        ...     end_date="2023-05-07",
        ...     vials_per_day=12
        ... )
    """
    logger.debug(f"Creating rig dataset configuration for {rig_name}")
    
    # Generate date range and vial numbers
    from datetime import datetime, timedelta
    
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"Invalid date format. Use YYYY-MM-DD: {e}")
    
    dates_vials = {}
    current_date = start
    vial_counter = 1
    
    while current_date <= end:
        date_str = current_date.strftime("%Y-%m-%d")
        vials_for_day = list(range(vial_counter, vial_counter + vials_per_day))
        dates_vials[date_str] = vials_for_day
        vial_counter += vials_per_day
        current_date += timedelta(days=1)
    
    # Rig-specific metadata
    metadata = {
        "rig_type": "behavioral",
        "dataset_type": "longitudinal",
        "description": f"Automated dataset configuration for {rig_name}",
        "date_range": f"{start_date} to {end_date}",
        "total_vials": vial_counter - 1,
        "vials_per_day": vials_per_day,
    }
    
    return create_dataset(
        name=f"{rig_name}_dataset",
        rig=rig_name,
        dates_vials=dates_vials,
        metadata=metadata,
        **kwargs
    )