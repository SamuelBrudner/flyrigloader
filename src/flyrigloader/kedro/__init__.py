"""
Kedro integration package for FlyRigLoader.

This package provides comprehensive Kedro integration through AbstractDataset implementations,
catalog configuration helpers, and factory functions for seamless pipeline integration.
It enables first-class support for Kedro's data catalog and pipeline workflows as part
of the FlyRigLoader 1.0 refactoring initiative.

Key Features:
- AbstractDataset implementations (FlyRigLoaderDataSet, FlyRigManifestDataSet) for complete
  Kedro compatibility with proper lifecycle management and error handling  
- Catalog configuration utilities for programmatic catalog construction and validation
- Factory functions for streamlined dataset creation with parameter validation
- Template generation for common experimental workflow patterns
- Multi-experiment catalog support with parameter injection capabilities
- Schema validation against FlyRigLoader requirements for pre-deployment verification

Primary Components:
    FlyRigLoaderDataSet: Full data loading AbstractDataset with DataFrame transformation
    FlyRigManifestDataSet: Lightweight manifest-only dataset for file discovery workflows
    create_kedro_dataset: Factory function for dataset creation with validation
    Catalog helpers: Comprehensive utilities for catalog configuration and management

Usage Examples:
    # Direct dataset usage
    >>> from flyrigloader.kedro import FlyRigLoaderDataSet
    >>> dataset = FlyRigLoaderDataSet(
    ...     filepath="config/experiment.yaml",
    ...     experiment_name="baseline_study"
    ... )
    >>> data = dataset.load()
    
    # Factory function usage  
    >>> from flyrigloader.kedro import create_kedro_dataset
    >>> dataset = create_kedro_dataset(
    ...     config_path="config/experiment.yaml",
    ...     experiment_name="baseline_study",
    ...     recursive=True
    ... )
    
    # Catalog configuration
    >>> from flyrigloader.kedro import create_flyrigloader_catalog_entry
    >>> entry = create_flyrigloader_catalog_entry(
    ...     dataset_name="experiment_data",
    ...     config_path="/data/config/experiment.yaml", 
    ...     experiment_name="baseline_study"
    ... )

Integration with catalog.yml:
    experiment_data:
      type: flyrigloader.FlyRigLoaderDataSet
      filepath: "${base_dir}/config/experiment_config.yaml"
      experiment_name: "baseline_study"
      recursive: true
      extract_metadata: true
      parse_dates: true

This module implements the Kedro integration requirements specified in Section 0.2.2
of the FlyRigLoader refactoring specification, providing plugin-style extensibility
and unified configuration interfaces for seamless Kedro catalog and pipeline workflows.
"""

import logging

# Import dataset classes from kedro.datasets module
from flyrigloader.kedro.datasets import (
    FlyRigLoaderDataSet,
    FlyRigManifestDataSet
)

# Import factory function from api module  
from flyrigloader.api import create_kedro_dataset

# Import catalog configuration helpers from kedro.catalog module
from flyrigloader.kedro.catalog import (
    create_flyrigloader_catalog_entry,
    validate_catalog_config,
    get_dataset_parameters,
    generate_catalog_template,
    create_multi_experiment_catalog,
    inject_catalog_parameters,
    create_workflow_catalog_entries,
    validate_catalog_against_schema
)

# Configure module-level logging
logger = logging.getLogger(__name__)

# Log package initialization for debugging and audit trail
logger.debug("Initializing flyrigloader.kedro package with Kedro integration components")
logger.info("FlyRigLoader Kedro integration package loaded successfully")

# Export all public interfaces for unified access
__all__ = [
    # Dataset classes - Core AbstractDataset implementations for Kedro pipelines
    "FlyRigLoaderDataSet",      # Full data loading with DataFrame transformation
    "FlyRigManifestDataSet",    # Lightweight manifest-only operations
    
    # Factory functions - Simplified dataset creation with validation
    "create_kedro_dataset",     # Primary factory for dataset instantiation
    
    # Catalog configuration helpers - Programmatic catalog construction utilities
    "create_flyrigloader_catalog_entry",    # Single dataset catalog entry creation
    "validate_catalog_config",              # Catalog configuration validation
    "generate_catalog_template",            # Template generation for workflows
    "create_multi_experiment_catalog",      # Multi-experiment catalog creation
    "get_dataset_parameters",               # Parameter extraction utilities  
    "inject_catalog_parameters",            # Parameter injection for dynamic catalogs
    "create_workflow_catalog_entries",      # Complete workflow catalog generation
    "validate_catalog_against_schema"       # Schema validation against FlyRigLoader requirements
]

# Package metadata for debugging and version tracking
__version__ = "1.0.0"
__author__ = "FlyRigLoader Development Team"
__description__ = "Kedro integration package for FlyRigLoader experimental data workflows"

# Log successful initialization with component count for audit trail
logger.debug(f"Kedro integration package initialized with {len(__all__)} public components")
logger.debug(f"Available dataset classes: FlyRigLoaderDataSet, FlyRigManifestDataSet")
logger.debug(f"Available catalog helpers: {len(__all__) - 3} configuration utilities")