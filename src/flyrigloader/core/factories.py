"""
Factory functions for creating component instances.

This module contains factory functions that create and return instances of
various components used throughout the flyrigloader package. These factories
help prevent circular dependencies and provide a clean interface for component
creation.
"""

from typing import Dict, Any, Optional, List, Union
from pathlib import Path

from .utils.path_utils import PathLike


def create_lineage_tracker(**kwargs):
    """
    Create and return a lineage tracker instance.
    
    Args:
        **kwargs: Keyword arguments to pass to the LineageTracker constructor
        
    Returns:
        LineageTracker instance
    """
    from .dataframe_tracking import create_tracker
    return create_tracker(**kwargs)


def create_minimal_lineage_tracker(source, description="Data loaded"):
    """
    Create and return a minimal lineage tracker instance.
    
    Args:
        source: Path to the data source or a descriptive string
        description: Short description of the data source
        
    Returns:
        MinimalLineageTracker instance
    """
    from .dataframe_tracking import create_tracker
    tracker = create_tracker(name=f"track_{str(source)[:10]}")
    tracker.add_source(source, description)
    return tracker


def create_null_lineage_tracker(**kwargs):
    """
    Create and return a null lineage tracker that implements the same interface
    but doesn't perform any actual tracking.
    
    Args:
        **kwargs: Keyword arguments (ignored)
        
    Returns:
        NullLineageTracker instance
    """
    from .dataframe_tracking import create_null_tracker
    return create_null_tracker(**kwargs)


def create_schema_validator(schema_dict: Optional[Dict[str, Any]] = None, **kwargs):
    """
    Create and return a schema validator instance.
    
    Args:
        schema_dict: Dictionary representation of the schema
        **kwargs: Additional keyword arguments to pass to the validator constructor
        
    Returns:
        SchemaValidator instance
    """
    from ..schema.validator import SchemaValidator
    return SchemaValidator(schema_dict, **kwargs)


def create_config_manager(
    base_dir: Optional[PathLike] = None,
    config_dir: Optional[PathLike] = None,
    run_specific_config_path: Optional[PathLike] = None,
    **kwargs: Any
) -> 'ConfigManager':
    """
    Create and return a configuration manager instance.
    
    Args:
        base_dir: Base directory for configuration files
        config_dir: Directory containing configuration files
        run_specific_config_path: Optional path to run-specific config
        **kwargs: Additional keyword arguments for configuration loading
        
    Returns:
        ConfigManager instance
    """
    from ..config_utils.manager import ConfigManager, ConfigSettings
    from pathlib import Path
    from ..core.utils import ensure_path
    
    # If config_dir is provided but base_dir is not, use config_dir's parent as base_dir
    if config_dir is not None and base_dir is None:
        config_dir_path = ensure_path(config_dir)
        base_dir = config_dir_path.parent
    
    # Create settings if config_dir is specified
    settings = None
    if config_dir is not None:
        settings = ConfigSettings(config_dir=ensure_path(config_dir))
    
    return ConfigManager(
        base_dir=base_dir,
        config_settings=settings,
        run_specific_config_path=run_specific_config_path,
        **kwargs
    )


def create_data_pipeline(
    config: Optional[Dict[str, Any]] = None,
    lineage_tracker: Optional[Any] = None,
    **kwargs
):
    """
    Create and return a data pipeline instance.
    
    Args:
        config: Configuration dictionary
        lineage_tracker: Optional LineageTracker instance
        **kwargs: Additional keyword arguments for the data pipeline
        
    Returns:
        DataPipeline instance
    """
    from ..pipeline.data_pipeline import DataPipeline
    return DataPipeline(
        config=config,
        lineage_tracker=lineage_tracker,
        **kwargs
    )


def create_metadata_extractor(**kwargs):
    """
    Create and return a metadata extractor instance.
    
    Args:
        **kwargs: Keyword arguments for the metadata extractor
        
    Returns:
        MetadataExtractor instance
    """
    from ..discovery.metadata import MetadataExtractor
    return MetadataExtractor(**kwargs)


def create_file_reader(file_type: Optional[str] = None, **kwargs):
    """
    Create and return an appropriate file reader based on file type.
    
    Args:
        file_type: Type of file reader to create (e.g., 'csv', 'parquet', 'pickle')
        **kwargs: Additional keyword arguments for the file reader
        
    Returns:
        File reader function or object
    """
    if file_type == 'pickle':
        from ..readers.pickle import load_pickle
        return load_pickle
    else:
        from ..readers.formats import get_reader_for_format
        return get_reader_for_format(file_type, **kwargs)
