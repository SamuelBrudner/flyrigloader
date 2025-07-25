"""
Kedro catalog configuration helpers for FlyRigLoader integration.

This module provides comprehensive catalog configuration utilities that simplify the creation
and validation of FlyRigLoader datasets in Kedro catalogs. It enables seamless catalog.yml
integration with parameter validation, programmatic catalog construction for dynamic pipeline
generation, and template generation for common experimental workflow patterns.

Key Features:
- Programmatic catalog entry creation with comprehensive parameter validation
- Template generation for common experimental workflow patterns  
- Multi-experiment catalog creation with parameter injection support
- Pre-flight validation of catalog configurations against FlyRigLoader schema requirements
- Catalog parameter extraction and validation utilities
- Comprehensive error handling with structured logging

Functions:
    create_flyrigloader_catalog_entry: Create catalog entries for FlyRigLoader datasets
    validate_catalog_config: Validate catalog configurations against schema requirements
    get_dataset_parameters: Extract and validate dataset-specific parameters
    generate_catalog_template: Generate catalog templates for experimental workflows
    create_multi_experiment_catalog: Create catalog entries for multiple experiments
    inject_catalog_parameters: Inject parameters into existing catalog configurations
    create_workflow_catalog_entries: Create catalog entries for complete workflows
    validate_catalog_against_schema: Validate catalog structure against FlyRigLoader schema

Usage Examples:
    # Basic catalog entry creation
    >>> entry = create_flyrigloader_catalog_entry(
    ...     dataset_name="experiment_data",
    ...     config_path="/data/config/experiment.yaml",
    ...     experiment_name="baseline_study"
    ... )
    
    # Multi-experiment catalog generation  
    >>> catalog = create_multi_experiment_catalog(
    ...     base_config_path="/data/config/experiments.yaml",
    ...     experiments=["baseline", "treatment_1", "treatment_2"],
    ...     output_format="yaml"
    ... )
    
    # Catalog validation
    >>> is_valid, errors = validate_catalog_config(catalog_config)
    >>> if not is_valid:
    ...     for error in errors:
    ...         print(f"Validation error: {error}")
"""

from typing import Union, Any, Dict, List, Optional, Tuple
from pathlib import Path
import logging
import yaml
import warnings

# Internal imports
from flyrigloader.config.models import ExperimentConfig
from flyrigloader.config.yaml_config import load_config
from flyrigloader.kedro.datasets import FlyRigLoaderDataSet
from flyrigloader.migration.versions import detect_config_version
from flyrigloader.exceptions import KedroIntegrationError
from flyrigloader.api import validate_manifest

# Set up module-level logging  
logger = logging.getLogger(__name__)


def create_flyrigloader_catalog_entry(
    dataset_name: str,
    config_path: Union[str, Path],
    experiment_name: str,
    *,
    recursive: bool = True,
    extract_metadata: bool = True,
    parse_dates: bool = True,
    transform_options: Optional[Dict[str, Any]] = None,
    dataset_type: str = "flyrigloader.FlyRigLoaderDataSet",
    validate_entry: bool = True
) -> Dict[str, Any]:
    """
    Create a Kedro catalog entry for FlyRigLoader dataset with comprehensive parameter validation.
    
    This function enables programmatic catalog construction by generating properly formatted
    catalog entries that can be directly integrated into Kedro catalog.yml files or used
    for dynamic catalog generation in pipeline workflows.
    
    Args:
        dataset_name: Name for the dataset in the Kedro catalog
        config_path: Path to the FlyRigLoader configuration file
        experiment_name: Name of the experiment to load data for
        recursive: Whether to search recursively in directories (default: True)
        extract_metadata: Whether to extract metadata from filenames (default: True)  
        parse_dates: Whether to parse dates from filenames (default: True)
        transform_options: Additional options for DataFrame transformation
        dataset_type: Full import path for the dataset class (default: flyrigloader.FlyRigLoaderDataSet)
        validate_entry: Whether to validate the created entry (default: True)
        
    Returns:
        Dict[str, Any]: Kedro catalog entry dictionary ready for catalog integration
        
    Raises:
        KedroIntegrationError: If catalog entry creation or validation fails
        ValueError: If required parameters are missing or invalid
        
    Example:
        >>> entry = create_flyrigloader_catalog_entry(
        ...     dataset_name="plume_tracking_data",
        ...     config_path="config/experiments.yaml", 
        ...     experiment_name="plume_navigation",
        ...     recursive=True,
        ...     extract_metadata=True,
        ...     transform_options={"include_kedro_metadata": True}
        ... )
        >>> print(yaml.safe_dump({entry['name']: entry['config']}))
        plume_tracking_data:
          type: flyrigloader.FlyRigLoaderDataSet
          filepath: config/experiments.yaml
          experiment_name: plume_navigation
          recursive: true
          extract_metadata: true
    """
    logger.info(f"Creating Kedro catalog entry for dataset '{dataset_name}'")
    logger.debug(f"Entry parameters: config_path={config_path}, experiment={experiment_name}")
    
    # Validate required parameters
    if not dataset_name or not isinstance(dataset_name, str):
        raise KedroIntegrationError(
            f"dataset_name must be a non-empty string, got: {dataset_name}",
            error_code="KEDRO_003",
            context={
                "parameter": "dataset_name", 
                "value": dataset_name,
                "function": "create_flyrigloader_catalog_entry"
            }
        )
    
    if not config_path:
        raise KedroIntegrationError(
            "config_path parameter is required",
            error_code="KEDRO_003",
            context={
                "parameter": "config_path",
                "value": config_path,
                "function": "create_flyrigloader_catalog_entry"
            }
        )
    
    if not experiment_name or not isinstance(experiment_name, str):
        raise KedroIntegrationError(
            f"experiment_name must be a non-empty string, got: {experiment_name}",
            error_code="KEDRO_003",
            context={
                "parameter": "experiment_name",
                "value": experiment_name, 
                "function": "create_flyrigloader_catalog_entry"
            }
        )
    
    # Validate configuration file exists
    config_path_obj = Path(config_path)
    if not config_path_obj.exists():
        raise KedroIntegrationError(
            f"Configuration file not found: {config_path}",
            error_code="KEDRO_003",
            context={
                "config_path": str(config_path),
                "exists": False,
                "function": "create_flyrigloader_catalog_entry"
            }
        )
    
    try:
        # Build catalog entry configuration
        catalog_config = {
            "type": dataset_type,
            "filepath": str(config_path),
            "experiment_name": experiment_name,
            "recursive": recursive,
            "extract_metadata": extract_metadata,
            "parse_dates": parse_dates
        }
        
        # Add transform options if provided
        if transform_options:
            catalog_config["transform_options"] = transform_options
        
        # Create complete catalog entry
        catalog_entry = {
            "name": dataset_name,
            "config": catalog_config,
            "metadata": {
                "created_by": "flyrigloader.kedro.catalog",
                "dataset_type": "FlyRigLoaderDataSet",
                "experiment_name": experiment_name,
                "config_file": str(config_path)
            }
        }
        
        # Validate entry if requested
        if validate_entry:
            logger.debug(f"Validating catalog entry for dataset '{dataset_name}'")
            validation_result = validate_catalog_config(catalog_config)
            
            if not validation_result["valid"]:
                error_details = "; ".join(validation_result["errors"])
                raise KedroIntegrationError(
                    f"Catalog entry validation failed: {error_details}",
                    error_code="KEDRO_003",
                    context={
                        "dataset_name": dataset_name,
                        "validation_errors": validation_result["errors"],
                        "function": "create_flyrigloader_catalog_entry"
                    }
                )
        
        logger.info(f"✓ Successfully created catalog entry for dataset '{dataset_name}'")
        return catalog_entry
        
    except Exception as e:
        if isinstance(e, KedroIntegrationError):
            raise
        
        error_msg = f"Failed to create catalog entry for dataset '{dataset_name}': {e}"
        logger.error(error_msg)
        raise KedroIntegrationError(
            error_msg,
            error_code="KEDRO_003",
            context={
                "dataset_name": dataset_name,
                "original_error": str(e),
                "function": "create_flyrigloader_catalog_entry"
            }
        ) from e


def validate_catalog_config(
    catalog_config: Dict[str, Any],
    config_path: Optional[Union[str, Path]] = None,
    strict_validation: bool = True
) -> Dict[str, Any]:
    """
    Validate catalog configuration against FlyRigLoader schema requirements with pre-flight checks.
    
    This function performs comprehensive validation of catalog configurations to ensure they
    meet FlyRigLoader requirements before deployment. It checks parameter validity, configuration
    file accessibility, experiment existence, and schema compatibility.
    
    Args:
        catalog_config: Catalog configuration dictionary to validate
        config_path: Optional path to configuration file for enhanced validation
        strict_validation: Whether to perform comprehensive validation checks
        
    Returns:
        Dict[str, Any]: Validation report containing:
        - 'valid': Boolean indicating validation success
        - 'errors': List of validation errors
        - 'warnings': List of validation warnings  
        - 'metadata': Additional validation metadata
        
    Raises:
        KedroIntegrationError: For critical validation failures
        ValueError: If catalog_config format is invalid
        
    Example:
        >>> config = {
        ...     "type": "flyrigloader.FlyRigLoaderDataSet",
        ...     "filepath": "config/experiments.yaml",
        ...     "experiment_name": "baseline_study"
        ... }
        >>> result = validate_catalog_config(config)
        >>> if result['valid']:
        ...     print("✓ Catalog configuration is valid")
        >>> else:
        ...     print(f"Validation errors: {result['errors']}")
    """
    logger.info("Validating catalog configuration against FlyRigLoader schema")
    logger.debug(f"Validation mode: {'strict' if strict_validation else 'basic'}")
    
    # Validate input parameters
    if not isinstance(catalog_config, dict):
        raise ValueError(f"catalog_config must be a dictionary, got {type(catalog_config)}")
    
    # Initialize validation report
    validation_report = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "metadata": {
            "validation_type": "strict" if strict_validation else "basic",
            "parameters_checked": [],
            "config_version": None,
            "experiment_validated": False
        }
    }
    
    try:
        # Validate required catalog parameters
        required_params = ["type", "filepath", "experiment_name"]
        for param in required_params:
            validation_report["metadata"]["parameters_checked"].append(param)
            
            if param not in catalog_config:
                validation_report["errors"].append(f"Missing required parameter: {param}")
                continue
            
            # Validate parameter values
            param_value = catalog_config[param]
            
            if param == "type":
                if not isinstance(param_value, str) or "FlyRigLoaderDataSet" not in param_value:
                    validation_report["errors"].append(
                        f"Invalid dataset type: {param_value}. Must contain 'FlyRigLoaderDataSet'"
                    )
            
            elif param == "filepath":
                if not param_value or not isinstance(param_value, str):
                    validation_report["errors"].append(f"Invalid filepath: {param_value}")
                elif strict_validation:
                    filepath_obj = Path(param_value)
                    if not filepath_obj.exists():
                        validation_report["errors"].append(f"Configuration file not found: {param_value}")
                    elif not filepath_obj.is_file():
                        validation_report["errors"].append(f"Configuration path is not a file: {param_value}")
            
            elif param == "experiment_name":
                if not param_value or not isinstance(param_value, str):
                    validation_report["errors"].append(f"Invalid experiment_name: {param_value}")
        
        # Validate optional parameters
        optional_params = ["recursive", "extract_metadata", "parse_dates", "transform_options"]
        for param in optional_params:
            if param in catalog_config:
                validation_report["metadata"]["parameters_checked"].append(param)
                param_value = catalog_config[param]
                
                if param in ["recursive", "extract_metadata", "parse_dates"]:
                    if not isinstance(param_value, bool):
                        validation_report["warnings"].append(
                            f"Parameter {param} should be boolean, got {type(param_value)}"
                        )
                
                elif param == "transform_options":
                    if not isinstance(param_value, dict):
                        validation_report["errors"].append(
                            f"transform_options must be a dictionary, got {type(param_value)}"
                        )
        
        # Enhanced validation if configuration file is accessible
        if (strict_validation and 
            "filepath" in catalog_config and 
            catalog_config["filepath"] and
            Path(catalog_config["filepath"]).exists()):
            
            try:
                logger.debug("Performing enhanced validation with configuration file")
                config_file_path = catalog_config["filepath"]
                experiment_name = catalog_config.get("experiment_name")
                
                # Load and validate configuration
                config_data = load_config(config_file_path)
                
                # Detect configuration version
                detected_version = detect_config_version(config_data)
                validation_report["metadata"]["config_version"] = str(detected_version)
                logger.debug(f"Detected configuration version: {detected_version}")
                
                # Validate experiment exists in configuration
                if experiment_name:
                    if hasattr(config_data, 'get'):
                        experiments = config_data.get("experiments", {})
                    else:
                        experiments = getattr(config_data, 'experiments', {}) if hasattr(config_data, 'experiments') else {}
                    
                    if experiment_name not in experiments:
                        validation_report["errors"].append(
                            f"Experiment '{experiment_name}' not found in configuration"
                        )
                    else:
                        validation_report["metadata"]["experiment_validated"] = True
                        logger.debug(f"✓ Experiment '{experiment_name}' found in configuration")
                
            except Exception as e:
                validation_report["warnings"].append(f"Enhanced validation failed: {e}")
                logger.warning(f"Enhanced validation error: {e}")
        
        # Determine overall validation status
        validation_report["valid"] = len(validation_report["errors"]) == 0
        
        # Log validation results
        if validation_report["valid"]:
            logger.info("✓ Catalog configuration validation successful")
            if validation_report["warnings"]:
                logger.info(f"  Warnings: {len(validation_report['warnings'])}")
        else:
            logger.error(f"✗ Catalog configuration validation failed: {len(validation_report['errors'])} errors")
        
        return validation_report
        
    except Exception as e:
        error_msg = f"Catalog configuration validation failed: {e}"
        logger.error(error_msg)
        raise KedroIntegrationError(
            error_msg,
            error_code="KEDRO_003",
            context={
                "function": "validate_catalog_config",
                "original_error": str(e),
                "catalog_config": catalog_config
            }
        ) from e


def get_dataset_parameters(
    dataset_config: Dict[str, Any],
    parameter_filters: Optional[List[str]] = None,
    include_metadata: bool = True
) -> Dict[str, Any]:
    """
    Extract and validate dataset-specific parameters from catalog configuration.
    
    This function provides a utility for extracting and validating dataset parameters
    from catalog configurations, enabling parameter inspection and validation workflows
    in dynamic pipeline generation scenarios.
    
    Args:
        dataset_config: Dataset configuration dictionary from catalog
        parameter_filters: Optional list of parameter names to include (default: all)
        include_metadata: Whether to include metadata about parameter extraction
        
    Returns:
        Dict[str, Any]: Extracted parameters with validation metadata
        
    Raises:
        KedroIntegrationError: If parameter extraction fails
        ValueError: If dataset_config format is invalid
        
    Example:
        >>> config = {
        ...     "type": "flyrigloader.FlyRigLoaderDataSet",
        ...     "filepath": "config/experiments.yaml",
        ...     "experiment_name": "baseline_study",
        ...     "recursive": True,
        ...     "extract_metadata": True
        ... }
        >>> params = get_dataset_parameters(config)
        >>> print(f"Dataset parameters: {list(params['parameters'].keys())}")
        Dataset parameters: ['filepath', 'experiment_name', 'recursive', 'extract_metadata']
    """
    logger.info("Extracting dataset parameters from catalog configuration")
    logger.debug(f"Parameter filters: {parameter_filters}")
    
    # Validate input
    if not isinstance(dataset_config, dict):
        raise ValueError(f"dataset_config must be a dictionary, got {type(dataset_config)}")
    
    try:
        # Define FlyRigLoader-specific parameters
        flyrigloader_params = {
            # Required parameters
            "filepath": {"type": str, "required": True, "description": "Path to configuration file"},
            "experiment_name": {"type": str, "required": True, "description": "Name of experiment to load"},
            
            # Optional parameters
            "recursive": {"type": bool, "required": False, "default": True, "description": "Enable recursive directory scanning"},
            "extract_metadata": {"type": bool, "required": False, "default": True, "description": "Extract metadata from filenames"},
            "parse_dates": {"type": bool, "required": False, "default": True, "description": "Parse dates from filenames"},
            "transform_options": {"type": dict, "required": False, "default": {}, "description": "DataFrame transformation options"},
            
            # Advanced parameters  
            "date_range": {"type": list, "required": False, "description": "Date range filter for experiments"},
            "rig_names": {"type": list, "required": False, "description": "List of rig names to include"},
            "file_patterns": {"type": list, "required": False, "description": "Custom file patterns for discovery"}
        }
        
        # Extract parameters from configuration
        extracted_params = {}
        validation_info = {
            "extracted_count": 0,
            "missing_required": [],
            "invalid_types": [],
            "unknown_params": []
        }
        
        # Process each parameter in the configuration
        for param_name, param_value in dataset_config.items():
            # Skip type parameter as it's not a FlyRigLoader parameter
            if param_name == "type":
                continue
            
            if param_name in flyrigloader_params:
                param_spec = flyrigloader_params[param_name]
                
                # Type validation
                expected_type = param_spec["type"]
                if not isinstance(param_value, expected_type):
                    validation_info["invalid_types"].append({
                        "parameter": param_name,
                        "expected_type": expected_type.__name__,
                        "actual_type": type(param_value).__name__,
                        "value": param_value
                    })
                
                # Apply parameter filters if specified
                if parameter_filters is None or param_name in parameter_filters:
                    extracted_params[param_name] = param_value
                    validation_info["extracted_count"] += 1
            else:
                validation_info["unknown_params"].append(param_name)
        
        # Check for missing required parameters
        for param_name, param_spec in flyrigloader_params.items():
            if param_spec["required"] and param_name not in dataset_config:
                validation_info["missing_required"].append(param_name)
        
        # Build result
        result = {
            "parameters": extracted_params,
            "validation": validation_info
        }
        
        # Add metadata if requested
        if include_metadata:
            result["metadata"] = {
                "total_flyrigloader_params": len(flyrigloader_params),
                "available_params": list(flyrigloader_params.keys()),
                "parameter_specs": flyrigloader_params,
                "extraction_successful": len(validation_info["missing_required"]) == 0 and len(validation_info["invalid_types"]) == 0
            }
        
        logger.info(f"✓ Extracted {validation_info['extracted_count']} parameters")
        if validation_info["missing_required"]:
            logger.warning(f"Missing required parameters: {validation_info['missing_required']}")
        if validation_info["invalid_types"]:
            logger.warning(f"Type validation issues: {len(validation_info['invalid_types'])}")
        
        return result
        
    except Exception as e:
        error_msg = f"Parameter extraction failed: {e}"
        logger.error(error_msg)
        raise KedroIntegrationError(
            error_msg,
            error_code="KEDRO_003",
            context={
                "function": "get_dataset_parameters",
                "original_error": str(e),
                "dataset_config": dataset_config
            }
        ) from e


def generate_catalog_template(
    template_type: str = "multi_experiment",
    base_config_path: Optional[Union[str, Path]] = None,
    experiments: Optional[List[str]] = None,
    dataset_prefix: str = "experiment",
    output_format: str = "yaml"
) -> Union[Dict[str, Any], str]:
    """
    Generate catalog templates for common experimental workflow patterns.
    
    This function provides template generation utilities for common experimental
    workflow patterns, enabling rapid catalog creation for standard research
    configurations and reducing boilerplate code in catalog setup.
    
    Args:
        template_type: Type of template to generate ("single_experiment", "multi_experiment", "workflow")
        base_config_path: Base path to configuration file (optional for template generation)
        experiments: List of experiment names for multi-experiment templates
        dataset_prefix: Prefix for dataset names in the catalog
        output_format: Output format ("yaml", "dict")
        
    Returns:
        Union[Dict[str, Any], str]: Generated catalog template in requested format
        
    Raises:
        KedroIntegrationError: If template generation fails
        ValueError: If template parameters are invalid
        
    Example:
        >>> template = generate_catalog_template(
        ...     template_type="multi_experiment",
        ...     experiments=["baseline", "treatment_1", "treatment_2"],
        ...     dataset_prefix="fly_behavior",
        ...     output_format="yaml"
        ... )
        >>> print(template)
        fly_behavior_baseline:
          type: flyrigloader.FlyRigLoaderDataSet
          filepath: "${base_dir}/config/experiments.yaml"
          experiment_name: baseline
          recursive: true
          extract_metadata: true
    """
    logger.info(f"Generating catalog template: {template_type}")
    logger.debug(f"Template parameters: experiments={experiments}, prefix={dataset_prefix}")
    
    # Validate template type
    valid_template_types = ["single_experiment", "multi_experiment", "workflow"]
    if template_type not in valid_template_types:
        raise ValueError(f"Invalid template_type: {template_type}. Must be one of {valid_template_types}")
    
    # Validate output format
    valid_formats = ["yaml", "dict"]
    if output_format not in valid_formats:
        raise ValueError(f"Invalid output_format: {output_format}. Must be one of {valid_formats}")
    
    try:
        catalog_template = {}
        
        if template_type == "single_experiment":
            # Single experiment template
            experiment_name = experiments[0] if experiments else "experiment_1"
            dataset_name = f"{dataset_prefix}_{experiment_name}"
            
            catalog_template[dataset_name] = {
                "type": "flyrigloader.FlyRigLoaderDataSet",
                "filepath": base_config_path or "${base_dir}/config/experiment_config.yaml",
                "experiment_name": experiment_name,
                "recursive": True,
                "extract_metadata": True,
                "parse_dates": True
            }
        
        elif template_type == "multi_experiment":
            # Multi-experiment template
            if not experiments:
                experiments = ["baseline", "treatment_1", "treatment_2", "control"]
            
            for experiment in experiments:
                dataset_name = f"{dataset_prefix}_{experiment}"
                catalog_template[dataset_name] = {
                    "type": "flyrigloader.FlyRigLoaderDataSet",
                    "filepath": base_config_path or "${base_dir}/config/experiments.yaml",
                    "experiment_name": experiment,
                    "recursive": True,
                    "extract_metadata": True,
                    "parse_dates": True,
                    "transform_options": {
                        "include_kedro_metadata": True,
                        "experiment_name": experiment
                    }
                }
        
        elif template_type == "workflow":
            # Complete workflow template with manifests and processed data
            if not experiments:
                experiments = ["baseline", "treatment"]
            
            for experiment in experiments:
                # Raw data dataset
                raw_dataset_name = f"{dataset_prefix}_{experiment}_raw"
                catalog_template[raw_dataset_name] = {
                    "type": "flyrigloader.FlyRigLoaderDataSet",
                    "filepath": base_config_path or "${base_dir}/config/experiments.yaml",
                    "experiment_name": experiment,
                    "recursive": True,
                    "extract_metadata": True,
                    "parse_dates": True
                }
                
                # Manifest dataset  
                manifest_dataset_name = f"{dataset_prefix}_{experiment}_manifest"
                catalog_template[manifest_dataset_name] = {
                    "type": "flyrigloader.FlyRigManifestDataSet",
                    "filepath": base_config_path or "${base_dir}/config/experiments.yaml",
                    "experiment_name": experiment,
                    "recursive": True,
                    "include_stats": True
                }
                
                # Processed data output
                processed_dataset_name = f"{dataset_prefix}_{experiment}_processed"
                catalog_template[processed_dataset_name] = {
                    "type": "pandas.CSVDataset",
                    "filepath": f"${{base_dir}}/processed/{experiment}_processed.csv",
                    "save_args": {"index": False}
                }
        
        # Add template metadata
        template_metadata = {
            "_template_info": {
                "generated_by": "flyrigloader.kedro.catalog",
                "template_type": template_type,
                "creation_time": "auto_generated",
                "experiments": experiments or ["template_experiment"],
                "dataset_count": len(catalog_template)
            }
        }
        catalog_template.update(template_metadata)
        
        # Return in requested format
        if output_format == "yaml":
            yaml_output = yaml.safe_dump(catalog_template, default_flow_style=False, sort_keys=False)
            logger.info(f"✓ Generated YAML catalog template with {len(catalog_template)-1} datasets")
            return yaml_output
        else:
            logger.info(f"✓ Generated dictionary catalog template with {len(catalog_template)-1} datasets")
            return catalog_template
    
    except Exception as e:
        error_msg = f"Template generation failed for type '{template_type}': {e}"
        logger.error(error_msg)
        raise KedroIntegrationError(
            error_msg,
            error_code="KEDRO_003",
            context={
                "template_type": template_type,
                "function": "generate_catalog_template",
                "original_error": str(e)
            }
        ) from e


def create_multi_experiment_catalog(
    base_config_path: Union[str, Path],
    experiments: List[str],
    dataset_prefix: str = "experiment",
    include_manifests: bool = True,
    validate_experiments: bool = True,
    output_format: str = "dict"
) -> Union[Dict[str, Any], str]:
    """
    Create catalog entries for multiple experiments with parameter validation and injection.
    
    This function enables efficient multi-experiment catalog generation with automatic
    parameter injection and validation. It supports both data and manifest datasets
    for comprehensive experimental pipeline workflows.
    
    Args:
        base_config_path: Path to the base FlyRigLoader configuration file
        experiments: List of experiment names to create catalog entries for
        dataset_prefix: Prefix for dataset names in the catalog
        include_manifests: Whether to create manifest datasets alongside data datasets
        validate_experiments: Whether to validate experiments exist in configuration
        output_format: Output format ("dict", "yaml")
        
    Returns:
        Union[Dict[str, Any], str]: Complete catalog configuration for all experiments
        
    Raises:
        KedroIntegrationError: If catalog creation fails
        FileNotFoundError: If configuration file is not found
        ValueError: If experiment validation fails
        
    Example:
        >>> catalog = create_multi_experiment_catalog(
        ...     base_config_path="config/experiments.yaml",
        ...     experiments=["baseline", "treatment_1", "treatment_2"],
        ...     dataset_prefix="fly_behavior",
        ...     include_manifests=True
        ... )
        >>> print(f"Created catalog with {len(catalog)} entries")
    """
    logger.info(f"Creating multi-experiment catalog for {len(experiments)} experiments")
    logger.debug(f"Config path: {base_config_path}, prefix: {dataset_prefix}")
    
    # Validate parameters
    if not experiments or not isinstance(experiments, list):
        raise ValueError("experiments must be a non-empty list")
    
    config_path_obj = Path(base_config_path)
    if not config_path_obj.exists():
        raise FileNotFoundError(f"Configuration file not found: {base_config_path}")
    
    try:
        catalog = {}
        
        # Validate experiments if requested
        if validate_experiments:
            logger.debug("Validating experiments exist in configuration")
            config_data = load_config(base_config_path)
            
            # Get experiments from configuration
            if hasattr(config_data, 'get'):
                config_experiments = config_data.get("experiments", {})
            else:
                config_experiments = getattr(config_data, 'experiments', {}) if hasattr(config_data, 'experiments') else {}
            
            # Validate each experiment
            missing_experiments = []
            for experiment in experiments:
                if experiment not in config_experiments:
                    missing_experiments.append(experiment)
            
            if missing_experiments:
                error_msg = f"Experiments not found in configuration: {missing_experiments}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            logger.info(f"✓ All {len(experiments)} experiments validated in configuration")
        
        # Create catalog entries for each experiment
        for experiment in experiments:
            logger.debug(f"Creating catalog entries for experiment: {experiment}")
            
            # Main data dataset
            data_dataset_name = f"{dataset_prefix}_{experiment}_data"
            catalog[data_dataset_name] = {
                "type": "flyrigloader.FlyRigLoaderDataSet",
                "filepath": str(base_config_path),
                "experiment_name": experiment,
                "recursive": True,
                "extract_metadata": True,
                "parse_dates": True,
                "transform_options": {
                    "include_kedro_metadata": True,
                    "experiment_name": experiment
                }
            }
            
            # Manifest dataset if requested
            if include_manifests:
                manifest_dataset_name = f"{dataset_prefix}_{experiment}_manifest"
                catalog[manifest_dataset_name] = {
                    "type": "flyrigloader.FlyRigManifestDataSet",
                    "filepath": str(base_config_path),
                    "experiment_name": experiment,
                    "recursive": True,
                    "include_stats": True,
                    "extract_metadata": True,
                    "parse_dates": True
                }
        
        # Inject additional parameters using inject_catalog_parameters
        parameter_overrides = {
            "recursive": True,
            "extract_metadata": True,
            "parse_dates": True
        }
        catalog = inject_catalog_parameters(catalog, parameter_overrides)
        
        # Add catalog metadata
        catalog["_catalog_metadata"] = {
            "created_by": "flyrigloader.kedro.catalog.create_multi_experiment_catalog",
            "base_config_path": str(base_config_path),
            "experiments": experiments,
            "dataset_prefix": dataset_prefix,
            "includes_manifests": include_manifests,
            "total_datasets": len(catalog) - 1  # Exclude metadata entry
        }
        
        logger.info(f"✓ Successfully created multi-experiment catalog with {len(catalog)-1} datasets")
        
        # Return in requested format
        if output_format == "yaml":
            return yaml.safe_dump(catalog, default_flow_style=False, sort_keys=False)
        else:
            return catalog
            
    except Exception as e:
        if isinstance(e, (ValueError, FileNotFoundError)):
            raise
        
        error_msg = f"Multi-experiment catalog creation failed: {e}"
        logger.error(error_msg)
        raise KedroIntegrationError(
            error_msg,
            error_code="KEDRO_003",
            context={
                "function": "create_multi_experiment_catalog",
                "experiments": experiments,
                "config_path": str(base_config_path),
                "original_error": str(e)
            }
        ) from e


def inject_catalog_parameters(
    catalog_config: Dict[str, Any],
    parameter_overrides: Dict[str, Any],
    target_datasets: Optional[List[str]] = None,
    parameter_filters: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Inject parameters into existing catalog configurations with selective targeting.
    
    This function provides parameter injection capabilities for dynamic catalog
    modification, enabling parameter updates across multiple dataset configurations
    with selective targeting and validation.
    
    Args:
        catalog_config: Existing catalog configuration dictionary
        parameter_overrides: Parameters to inject or override
        target_datasets: List of dataset names to target (default: all FlyRigLoader datasets)
        parameter_filters: List of parameter names to inject (default: all)
        
    Returns:
        Dict[str, Any]: Updated catalog configuration with injected parameters
        
    Raises:
        KedroIntegrationError: If parameter injection fails
        ValueError: If catalog_config format is invalid
        
    Example:
        >>> catalog = {
        ...     "data1": {"type": "flyrigloader.FlyRigLoaderDataSet", "recursive": False},
        ...     "data2": {"type": "flyrigloader.FlyRigLoaderDataSet", "extract_metadata": False}
        ... }
        >>> overrides = {"recursive": True, "extract_metadata": True, "parse_dates": True}
        >>> updated_catalog = inject_catalog_parameters(catalog, overrides)
        >>> # All FlyRigLoader datasets now have the updated parameters
    """
    logger.info(f"Injecting parameters into catalog configuration")
    logger.debug(f"Parameter overrides: {list(parameter_overrides.keys())}")
    
    # Validate inputs
    if not isinstance(catalog_config, dict):
        raise ValueError(f"catalog_config must be a dictionary, got {type(catalog_config)}")
    
    if not isinstance(parameter_overrides, dict):
        raise ValueError(f"parameter_overrides must be a dictionary, got {type(parameter_overrides)}")
    
    try:
        updated_catalog = catalog_config.copy()
        injection_report = {
            "datasets_processed": 0,
            "parameters_injected": 0,
            "skipped_datasets": [],
            "injection_details": {}
        }
        
        # Process each dataset in the catalog
        for dataset_name, dataset_config in catalog_config.items():
            # Skip metadata entries
            if dataset_name.startswith("_"):
                continue
            
            # Check if this dataset should be targeted
            if target_datasets is not None and dataset_name not in target_datasets:
                injection_report["skipped_datasets"].append(dataset_name)
                continue
            
            # Check if this is a FlyRigLoader dataset
            if not isinstance(dataset_config, dict) or "type" not in dataset_config:
                injection_report["skipped_datasets"].append(dataset_name)
                continue
            
            dataset_type = dataset_config.get("type", "")
            if "FlyRigLoaderDataSet" not in dataset_type and "FlyRigManifestDataSet" not in dataset_type:
                injection_report["skipped_datasets"].append(dataset_name)
                continue
            
            # Inject parameters
            injection_report["datasets_processed"] += 1
            injection_report["injection_details"][dataset_name] = []
            
            for param_name, param_value in parameter_overrides.items():
                # Apply parameter filters if specified
                if parameter_filters is not None and param_name not in parameter_filters:
                    continue
                
                # Record original value if it exists
                original_value = dataset_config.get(param_name, "<not_set>")
                
                # Inject the parameter
                updated_catalog[dataset_name][param_name] = param_value
                injection_report["parameters_injected"] += 1
                
                injection_report["injection_details"][dataset_name].append({
                    "parameter": param_name,
                    "original_value": original_value,
                    "new_value": param_value
                })
                
                logger.debug(f"Injected {param_name}={param_value} into {dataset_name}")
        
        # Add injection metadata
        updated_catalog["_injection_metadata"] = {
            "injected_by": "flyrigloader.kedro.catalog.inject_catalog_parameters",
            "injection_report": injection_report,
            "parameter_overrides": parameter_overrides,
            "target_datasets": target_datasets,
            "parameter_filters": parameter_filters
        }
        
        logger.info(f"✓ Parameter injection completed: {injection_report['datasets_processed']} datasets, "
                   f"{injection_report['parameters_injected']} parameters injected")
        
        return updated_catalog
        
    except Exception as e:
        error_msg = f"Parameter injection failed: {e}"
        logger.error(error_msg)
        raise KedroIntegrationError(
            error_msg,
            error_code="KEDRO_003",
            context={
                "function": "inject_catalog_parameters",
                "parameter_overrides": parameter_overrides,
                "original_error": str(e)
            }
        ) from e


def create_workflow_catalog_entries(
    workflow_name: str,
    base_config_path: Union[str, Path],
    experiments: List[str],
    pipeline_stages: Optional[List[str]] = None,
    include_intermediates: bool = True,
    validate_config: bool = True
) -> Dict[str, Any]:
    """
    Create comprehensive catalog entries for complete experimental workflows.
    
    This function generates complete catalog configurations for end-to-end experimental
    workflows, including raw data, intermediate processing stages, and final outputs
    with proper dependency management and validation.
    
    Args:
        workflow_name: Name of the workflow for dataset naming
        base_config_path: Path to FlyRigLoader configuration file
        experiments: List of experiment names in the workflow
        pipeline_stages: List of pipeline stage names (default: ["raw", "processed", "analyzed"])
        include_intermediates: Whether to include intermediate processing datasets
        validate_config: Whether to validate configuration and experiments
        
    Returns:
        Dict[str, Any]: Complete workflow catalog configuration
        
    Raises:
        KedroIntegrationError: If workflow catalog creation fails
        FileNotFoundError: If configuration file is not found
        ValueError: If workflow parameters are invalid
        
    Example:
        >>> workflow_catalog = create_workflow_catalog_entries(
        ...     workflow_name="plume_analysis",
        ...     base_config_path="config/experiments.yaml",
        ...     experiments=["baseline", "high_concentration"],
        ...     pipeline_stages=["raw", "filtered", "analyzed", "summary"]
        ... )
        >>> print(f"Created workflow with {len(workflow_catalog)} catalog entries")
    """
    logger.info(f"Creating workflow catalog entries for '{workflow_name}'")
    logger.debug(f"Experiments: {experiments}, stages: {pipeline_stages}")
    
    # Validate parameters
    if not workflow_name or not isinstance(workflow_name, str):
        raise ValueError("workflow_name must be a non-empty string")
    
    if not experiments or not isinstance(experiments, list):
        raise ValueError("experiments must be a non-empty list")
    
    config_path_obj = Path(base_config_path)
    if not config_path_obj.exists():
        raise FileNotFoundError(f"Configuration file not found: {base_config_path}")
    
    # Set default pipeline stages
    if pipeline_stages is None:
        pipeline_stages = ["raw", "processed", "analyzed"]
    
    try:
        workflow_catalog = {}
        
        # Validate configuration if requested
        if validate_config:
            logger.debug("Validating configuration for workflow")
            config_data = load_config(base_config_path)
            
            # Detect configuration version
            config_version = detect_config_version(config_data)
            logger.debug(f"Configuration version: {config_version}")
            
            # Validate experiments exist
            if hasattr(config_data, 'get'):
                config_experiments = config_data.get("experiments", {})
            else:
                config_experiments = getattr(config_data, 'experiments', {}) if hasattr(config_data, 'experiments') else {}
            
            missing_experiments = [exp for exp in experiments if exp not in config_experiments]
            if missing_experiments:
                raise ValueError(f"Experiments not found in configuration: {missing_experiments}")
            
            logger.info(f"✓ Configuration validation successful for {len(experiments)} experiments")
        
        # Create catalog entries for each experiment and stage
        for experiment in experiments:
            logger.debug(f"Creating workflow entries for experiment: {experiment}")
            
            for stage_idx, stage in enumerate(pipeline_stages):
                dataset_name = f"{workflow_name}_{experiment}_{stage}"
                
                if stage == "raw":
                    # Raw data from FlyRigLoader
                    workflow_catalog[dataset_name] = {
                        "type": "flyrigloader.FlyRigLoaderDataSet",
                        "filepath": str(base_config_path),
                        "experiment_name": experiment,
                        "recursive": True,
                        "extract_metadata": True,
                        "parse_dates": True,
                        "transform_options": {
                            "include_kedro_metadata": True,
                            "experiment_name": experiment,
                            "workflow_stage": stage
                        }
                    }
                    
                    # Also create manifest dataset for raw data
                    manifest_dataset_name = f"{workflow_name}_{experiment}_manifest"
                    workflow_catalog[manifest_dataset_name] = {
                        "type": "flyrigloader.FlyRigManifestDataSet",
                        "filepath": str(base_config_path),
                        "experiment_name": experiment,
                        "recursive": True,
                        "include_stats": True
                    }
                
                elif stage in ["processed", "filtered", "cleaned"]:
                    # Intermediate processing datasets
                    if include_intermediates:
                        workflow_catalog[dataset_name] = {
                            "type": "pandas.ParquetDataset",
                            "filepath": f"${{base_dir}}/intermediate/{workflow_name}/{experiment}_{stage}.parquet",
                            "save_args": {"compression": "snappy"},
                            "load_args": {"engine": "pyarrow"}
                        }
                
                elif stage in ["analyzed", "results"]:
                    # Analysis results datasets
                    workflow_catalog[dataset_name] = {
                        "type": "pandas.CSVDataset",
                        "filepath": f"${{base_dir}}/results/{workflow_name}/{experiment}_{stage}.csv",
                        "save_args": {"index": False},
                        "load_args": {"parse_dates": True}
                    }
                
                elif stage in ["summary", "report"]:
                    # Summary/report datasets
                    workflow_catalog[dataset_name] = {
                        "type": "pandas.ExcelWriter",
                        "filepath": f"${{base_dir}}/reports/{workflow_name}/{experiment}_{stage}.xlsx",
                        "save_args": {"index": False, "engine": "openpyxl"}
                    }
                
                else:
                    # Generic datasets for custom stages
                    workflow_catalog[dataset_name] = {
                        "type": "pickle.PickleDataset",
                        "filepath": f"${{base_dir}}/custom/{workflow_name}/{experiment}_{stage}.pkl"
                    }
        
        # Create workflow summary dataset
        summary_dataset_name = f"{workflow_name}_summary"
        workflow_catalog[summary_dataset_name] = {
            "type": "pandas.CSVDataset",
            "filepath": f"${{base_dir}}/reports/{workflow_name}/workflow_summary.csv",
            "save_args": {"index": False}
        }
        
        # Add workflow metadata
        workflow_catalog["_workflow_metadata"] = {
            "created_by": "flyrigloader.kedro.catalog.create_workflow_catalog_entries",
            "workflow_name": workflow_name,
            "base_config_path": str(base_config_path),
            "experiments": experiments,
            "pipeline_stages": pipeline_stages,
            "include_intermediates": include_intermediates,
            "total_datasets": len(workflow_catalog) - 1,  # Exclude metadata
            "raw_datasets": len(experiments),
            "intermediate_datasets": len(experiments) * len(pipeline_stages) if include_intermediates else 0
        }
        
        logger.info(f"✓ Created workflow catalog with {len(workflow_catalog)-1} entries")
        logger.debug(f"Workflow stages: {pipeline_stages}")
        
        return workflow_catalog
        
    except Exception as e:
        if isinstance(e, (ValueError, FileNotFoundError)):
            raise
        
        error_msg = f"Workflow catalog creation failed: {e}"
        logger.error(error_msg)
        raise KedroIntegrationError(
            error_msg,
            error_code="KEDRO_003",
            context={
                "function": "create_workflow_catalog_entries",
                "workflow_name": workflow_name,
                "experiments": experiments,
                "original_error": str(e)
            }
        ) from e


def validate_catalog_against_schema(
    catalog_dict: Dict[str, Any],
    schema_requirements: Optional[Dict[str, Any]] = None,
    strict_mode: bool = True
) -> Dict[str, Any]:
    """
    Validate entire catalog structure against FlyRigLoader schema requirements.
    
    This function performs comprehensive validation of catalog configurations against
    FlyRigLoader schema requirements, ensuring all datasets are properly configured
    and compatible with the FlyRigLoader system before deployment.
    
    Args:
        catalog_dict: Complete catalog dictionary to validate
        schema_requirements: Optional custom schema requirements (uses defaults if None)
        strict_mode: Whether to perform strict validation with file existence checks
        
    Returns:
        Dict[str, Any]: Comprehensive validation report with detailed analysis
        
    Raises:
        KedroIntegrationError: For critical schema validation failures
        ValueError: If catalog format is invalid
        
    Example:
        >>> catalog = {
        ...     "experiment_1": {"type": "flyrigloader.FlyRigLoaderDataSet", ...},
        ...     "experiment_2": {"type": "flyrigloader.FlyRigLoaderDataSet", ...}
        ... }
        >>> validation_result = validate_catalog_against_schema(catalog)
        >>> if validation_result['valid']:
        ...     print(f"✓ Catalog validation successful: {validation_result['dataset_count']} datasets")
        >>> else:
        ...     print(f"✗ Validation failed: {len(validation_result['errors'])} errors")
    """
    logger.info(f"Validating catalog structure against FlyRigLoader schema")
    logger.debug(f"Catalog entries: {len(catalog_dict)}, strict_mode: {strict_mode}")
    
    # Validate input
    if not isinstance(catalog_dict, dict):
        raise ValueError(f"catalog_dict must be a dictionary, got {type(catalog_dict)}")
    
    # Set default schema requirements
    if schema_requirements is None:
        schema_requirements = {
            "required_dataset_fields": ["type", "filepath", "experiment_name"],
            "valid_dataset_types": [
                "flyrigloader.FlyRigLoaderDataSet",
                "flyrigloader.FlyRigManifestDataSet"
            ],
            "parameter_types": {
                "filepath": str,
                "experiment_name": str,
                "recursive": bool,
                "extract_metadata": bool,
                "parse_dates": bool,
                "transform_options": dict
            },
            "deprecated_parameters": [],
            "recommended_parameters": ["recursive", "extract_metadata", "parse_dates"]
        }
    
    try:
        # Initialize validation report
        validation_report = {
            "valid": True,
            "dataset_count": 0,
            "flyrigloader_datasets": 0,
            "other_datasets": 0,
            "errors": [],
            "warnings": [],
            "recommendations": [],
            "dataset_details": {},
            "schema_compliance": {
                "total_checks": 0,
                "passed_checks": 0,
                "failed_checks": 0
            }
        }
        
        # Process each catalog entry
        for dataset_name, dataset_config in catalog_dict.items():
            # Skip metadata entries
            if dataset_name.startswith("_"):
                continue
            
            validation_report["dataset_count"] += 1
            dataset_validation = {
                "dataset_name": dataset_name,
                "is_flyrigloader": False,
                "errors": [],
                "warnings": [],
                "recommendations": [],
                "parameter_validation": {}
            }
            
            logger.debug(f"Validating dataset: {dataset_name}")
            
            # Basic structure validation
            if not isinstance(dataset_config, dict):
                error_msg = f"Dataset '{dataset_name}' configuration must be a dictionary"
                dataset_validation["errors"].append(error_msg)
                validation_report["errors"].append(error_msg)
                validation_report["dataset_details"][dataset_name] = dataset_validation
                continue
            
            # Check for dataset type
            if "type" not in dataset_config:
                error_msg = f"Dataset '{dataset_name}' missing required 'type' field"
                dataset_validation["errors"].append(error_msg)
                validation_report["errors"].append(error_msg)
                validation_report["dataset_details"][dataset_name] = dataset_validation
                continue
            
            dataset_type = dataset_config["type"]
            
            # Check if this is a FlyRigLoader dataset
            if any(flyrig_type in dataset_type for flyrig_type in schema_requirements["valid_dataset_types"]):
                dataset_validation["is_flyrigloader"] = True
                validation_report["flyrigloader_datasets"] += 1
                
                # Validate FlyRigLoader-specific requirements
                for required_field in schema_requirements["required_dataset_fields"]:
                    validation_report["schema_compliance"]["total_checks"] += 1
                    
                    if required_field not in dataset_config:
                        error_msg = f"FlyRigLoader dataset '{dataset_name}' missing required field: {required_field}"
                        dataset_validation["errors"].append(error_msg)
                        validation_report["errors"].append(error_msg)
                        validation_report["schema_compliance"]["failed_checks"] += 1
                    else:
                        validation_report["schema_compliance"]["passed_checks"] += 1
                        
                        # Validate parameter types
                        field_value = dataset_config[required_field]
                        expected_type = schema_requirements["parameter_types"].get(required_field)
                        
                        if expected_type and not isinstance(field_value, expected_type):
                            warning_msg = (f"Dataset '{dataset_name}' field '{required_field}' "
                                         f"expected {expected_type.__name__}, got {type(field_value).__name__}")
                            dataset_validation["warnings"].append(warning_msg)
                            validation_report["warnings"].append(warning_msg)
                
                # Check for recommended parameters
                for recommended_param in schema_requirements["recommended_parameters"]:
                    if recommended_param not in dataset_config:
                        rec_msg = f"Dataset '{dataset_name}' missing recommended parameter: {recommended_param}"
                        dataset_validation["recommendations"].append(rec_msg)
                        validation_report["recommendations"].append(rec_msg)
                
                # Validate configuration file exists if strict mode
                if strict_mode and "filepath" in dataset_config:
                    filepath = dataset_config["filepath"]
                    # Skip template variables
                    if "${" not in str(filepath):
                        filepath_obj = Path(filepath)
                        if not filepath_obj.exists():
                            error_msg = f"Configuration file not found for dataset '{dataset_name}': {filepath}"
                            dataset_validation["errors"].append(error_msg)
                            validation_report["errors"].append(error_msg)
                
                # Validate experiment in configuration if strict mode  
                if (strict_mode and 
                    "filepath" in dataset_config and 
                    "experiment_name" in dataset_config):
                    
                    filepath = dataset_config["filepath"]
                    experiment_name = dataset_config["experiment_name"]
                    
                    # Skip validation for template variables
                    if "${" not in str(filepath):
                        try:
                            config_data = load_config(filepath)
                            if hasattr(config_data, 'get'):
                                experiments = config_data.get("experiments", {})
                            else:
                                experiments = getattr(config_data, 'experiments', {}) if hasattr(config_data, 'experiments') else {}
                            
                            if experiment_name not in experiments:
                                error_msg = f"Experiment '{experiment_name}' not found in config for dataset '{dataset_name}'"
                                dataset_validation["errors"].append(error_msg)
                                validation_report["errors"].append(error_msg)
                        except Exception as e:
                            warning_msg = f"Could not validate experiment for dataset '{dataset_name}': {e}"
                            dataset_validation["warnings"].append(warning_msg)
                            validation_report["warnings"].append(warning_msg)
            
            else:
                validation_report["other_datasets"] += 1
            
            validation_report["dataset_details"][dataset_name] = dataset_validation
        
        # Determine overall validation status
        validation_report["valid"] = len(validation_report["errors"]) == 0
        
        # Calculate compliance percentage
        total_checks = validation_report["schema_compliance"]["total_checks"]
        passed_checks = validation_report["schema_compliance"]["passed_checks"]
        compliance_percentage = (passed_checks / total_checks * 100) if total_checks > 0 else 100
        validation_report["schema_compliance"]["compliance_percentage"] = compliance_percentage
        
        # Log validation summary
        if validation_report["valid"]:
            logger.info(f"✓ Catalog schema validation successful")
            logger.info(f"  Datasets: {validation_report['dataset_count']} total, "
                       f"{validation_report['flyrigloader_datasets']} FlyRigLoader")
            logger.info(f"  Schema compliance: {compliance_percentage:.1f}%")
            
            if validation_report["warnings"]:
                logger.info(f"  Warnings: {len(validation_report['warnings'])}")
            if validation_report["recommendations"]:
                logger.info(f"  Recommendations: {len(validation_report['recommendations'])}")
        else:
            logger.error(f"✗ Catalog schema validation failed")
            logger.error(f"  Errors: {len(validation_report['errors'])}")
            logger.error(f"  Warnings: {len(validation_report['warnings'])}")
        
        return validation_report
        
    except Exception as e:
        error_msg = f"Catalog schema validation failed: {e}"
        logger.error(error_msg)
        raise KedroIntegrationError(
            error_msg,
            error_code="KEDRO_003",
            context={
                "function": "validate_catalog_against_schema",
                "catalog_size": len(catalog_dict),
                "original_error": str(e)
            }
        ) from e


# Export all public functions
__all__ = [
    "create_flyrigloader_catalog_entry",
    "validate_catalog_config", 
    "get_dataset_parameters",
    "generate_catalog_template",
    "create_multi_experiment_catalog",
    "inject_catalog_parameters",
    "create_workflow_catalog_entries",
    "validate_catalog_against_schema"
]