"""
standard.py - Standard user-facing API for lineage tracking.

This module provides the LineageTracker class, which is the main interface for
lineage tracking in the flyrigloader package. It builds on the core infrastructure
to provide a complete API for tracking lineage information in DataFrames.
"""

import inspect
import os
import sys
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union, TypeVar, cast, overload

import pandas as pd
from loguru import logger

from ...core.utils import PathLike, ensure_path
from ..core import (
    LineageRecord,
    create_lineage_record,
    LineageStorage,
    create_storage,
    validate_lineage_storage_params,
    ensure_dataframe,
    verify_callable
)

# Type variables for better type hints
DF = TypeVar('DF', bound=pd.DataFrame)
F = TypeVar('F', bound=Callable)


class LineageTracker:
    """
    Standard lineage tracker for data processing pipelines.
    
    This class provides a complete API for tracking lineage information
    throughout a data processing pipeline. It maintains information about:
    - Input sources (file paths, timestamps, versions)
    - Processing steps applied to the data
    - Configuration parameters used
    - Any metadata needed for complete reproducibility
    
    The lineage information can be stored either in DataFrame attributes or
    in a central registry, depending on configuration.
    """
    
    def __init__(
        self, 
        name: Optional[str] = None,
        storage: Optional[LineageStorage] = None,
        attributes: Optional[Dict[str, Any]] = None,
        source_path: Optional[PathLike] = None
    ):
        """
        Initialize a new lineage tracker.
        
        Args:
            name: Optional name for this lineage tracker
            storage: Optional storage backend (creates new one if None)
            attributes: Additional attributes to store
            source_path: Optional source file path to add
            
        Raises:
            ValueError: If initialization fails
        """
        try:
            # Create record and storage
            self.record = create_lineage_record(attributes=attributes)
            self.storage = storage or create_storage()
            
            # Set tracker name
            self._name = name or f"lineage-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            # Register lineage ID in the storage
            self.storage.register(self.record)
            
            # Add source if provided
            if source_path is not None:
                self.add_source(source_path)
                
            logger.debug(f"Created LineageTracker: {self.name}")
        except Exception as e:
            logger.error(f"Failed to initialize LineageTracker: {str(e)}")
            raise ValueError(f"Failed to initialize lineage tracker: {str(e)}") from e
    
    @property
    def name(self) -> str:
        """Get the name of this tracker."""
        return self._name
        
    @name.setter
    def name(self, value: str) -> None:
        """Set the name of this tracker."""
        self._name = value
        
    @property
    def lineage_id(self) -> str:
        """Get the unique ID for this lineage."""
        return self.record.id
        
    def add_source(
        self, 
        source: PathLike, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'LineageTracker':
        """
        Add a data source to this lineage.
        
        Args:
            source: Path to the data source
            metadata: Additional metadata about the source
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If source cannot be added
        """
        try:
            path = ensure_path(source)
            
            # Create metadata if none provided
            if metadata is None:
                metadata = {}
                
                # Build metadata based on path properties
                if path.exists() and path.is_file():
                    # File exists, extract detailed stats
                    stats = path.stat()
                    metadata.update({
                        'path': str(path),
                        'name': path.name,
                        'size': stats.st_size,
                        'modified': datetime.fromtimestamp(stats.st_mtime).isoformat(),
                        'type': path.suffix.lstrip('.').lower()
                    })
                elif hasattr(path, 'name'):
                    # Path exists but isn't a file or has no name attribute
                    metadata.update({
                        'path': str(path),
                        'name': path.name
                    })
                else:
                    # Fallback for paths with no name attribute
                    metadata.update({
                        'path': str(path),
                        'name': 'unknown'
                    })
            else:
                # Ensure path is in metadata
                if 'path' not in metadata:
                    metadata['path'] = str(path)
                    
            # Add to lineage record
            self.record.add_source(str(path), metadata)
            return self
        except Exception as e:
            logger.error(f"Failed to add source {source}: {str(e)}")
            raise ValueError(f"Failed to add source: {str(e)}") from e
            
    def add_step(
        self, 
        name: str, 
        description: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'LineageTracker':
        """
        Add a processing step to this lineage.
        
        Args:
            name: Short name for the step
            description: Detailed description of what the step does
            metadata: Optional metadata about the step (e.g. parameters)
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If step cannot be added
        """
        try:
            self.record.add_step(name, description, metadata or {})
            return self
        except Exception as e:
            logger.error(f"Failed to add step {name}: {str(e)}")
            raise ValueError(f"Failed to add step: {str(e)}") from e
            
    def attach_to_dataframe(self, df: DF) -> DF:
        """
        Attach lineage information to a DataFrame.
        
        Args:
            df: DataFrame to attach lineage to
            
        Returns:
            DataFrame with lineage attached
            
        Raises:
            ValueError: If the DataFrame is invalid or attachment fails
        """
        try:
            # Validate DataFrame
            df = ensure_dataframe(df)
            
            # Store lineage in the configured storage
            return self.storage.store(df, self.record)
        except Exception as e:
            logger.error(f"Failed to attach lineage to DataFrame: {str(e)}")
            raise ValueError(f"Failed to attach lineage: {str(e)}") from e
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> Optional['LineageTracker']:
        """
        Create a LineageTracker from a DataFrame's lineage information.
        
        Args:
            df: DataFrame to extract lineage from
            
        Returns:
            New LineageTracker instance, or None if extraction fails
        """
        try:
            # Try all storage backends to find lineage
            for storage_type in ["attribute", "registry"]:
                storage = create_storage(storage_type)
                lineage_record = storage.retrieve(df)
                
                if lineage_record is not None:
                    # Create a new tracker with the same storage type
                    tracker = cls()
                    # Replace the record with the retrieved one
                    tracker.record = lineage_record
                    return tracker
            
            return None
        except Exception as e:
            logger.error(f"Failed to create LineageTracker from DataFrame: {str(e)}")
            raise ValueError(f"Failed to create LineageTracker: {str(e)}") from e
    
    def track_transformation(
        self, 
        input_df: pd.DataFrame, 
        output_df: DF, 
        step_name: str, 
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        skip_validation: bool = False
    ) -> DF:
        """
        Track a transformation from input DataFrame to output DataFrame.
        
        Args:
            input_df: Input DataFrame
            output_df: Output DataFrame
            step_name: Name of the transformation step
            description: Description of the transformation
            metadata: Additional metadata about the transformation
            skip_validation: If True, skip DataFrame compatibility validation
            
        Returns:
            Output DataFrame with updated lineage
            
        Raises:
            ValueError: If tracking fails or DataFrames are incompatible
        """
        try:
            # Validate DataFrames
            input_df = ensure_dataframe(input_df)
            output_df = ensure_dataframe(output_df)
            
            # Validate DataFrame compatibility unless explicitly skipped
            if not skip_validation:
                # Validate that the transformation is logically sound
                if input_df.shape[0] < output_df.shape[0] and input_df.shape[0] > 0 and output_df.shape[0] > input_df.shape[0] * 10:
                    # Output has significantly more rows than input (>10x) - might be suspicious
                    logger.warning(
                        f"Suspicious transformation: Output has {output_df.shape[0]} rows, "
                        f"which is more than 10x the input's {input_df.shape[0]} rows"
                    )
                    
                # Check for extreme column changes
                if len(input_df.columns) > 0 and len(output_df.columns) == 0:
                    raise ValueError("Invalid transformation: Output DataFrame has no columns")
                
                # If output has columns not in input, log them for information
                if len(set(output_df.columns) - set(input_df.columns)) > 0:
                    new_cols = set(output_df.columns) - set(input_df.columns)
                    if metadata is None:
                        metadata = {}
                    metadata["added_columns"] = list(new_cols)
                    
                # If output is missing columns from input, log them for information
                if len(set(input_df.columns) - set(output_df.columns)) > 0:
                    removed_cols = set(input_df.columns) - set(output_df.columns)
                    if metadata is None:
                        metadata = {}
                    metadata["removed_columns"] = list(removed_cols)
            
            # Try to get lineage from input DataFrame
            input_tracker = self.from_dataframe(input_df)
            
            if input_tracker is not None:
                # Merge the input lineage with our own
                self.record.merge(input_tracker.record)
            
            # Add transformation step
            self.add_step(step_name, description, metadata)
            
            # Attach to output DataFrame
            return self.attach_to_dataframe(output_df)
        except Exception as e:
            logger.error(f"Failed to track transformation: {str(e)}")
            raise ValueError(f"Failed to track transformation: {str(e)}") from e
    
    def track_function(self, function: F) -> F:
        """
        Decorator to track lineage for a function that processes DataFrames.
        
        The decorated function must take a DataFrame as its first argument
        and return a DataFrame.
        
        Args:
            function: Function to track
            
        Returns:
            Wrapped function with lineage tracking
            
        Raises:
            ValueError: If the function is not compatible with tracking
        """
        try:
            verify_callable(function)
            
            @wraps(function)
            def wrapper(df: pd.DataFrame, *args, **kwargs):
                # Verify df is a DataFrame
                df = ensure_dataframe(df)
                
                # Get function name and information for the step
                func_name = function.__name__
                func_module = function.__module__
                
                # Build metadata
                metadata = {
                    "function": func_name,
                    "module": func_module,
                    "args": str(args),
                    "kwargs": str(kwargs)
                }
                
                # Call the original function
                result = function(df, *args, **kwargs)
                
                # Check that the result is a DataFrame
                if not isinstance(result, pd.DataFrame):
                    logger.warning(f"Function {func_name} did not return a DataFrame, skipping lineage tracking")
                    return result
                
                # Track the transformation
                step_name = f"{func_module}.{func_name}"
                description = function.__doc__ or f"Applied function {func_name}"
                result = self.track_transformation(df, result, step_name, description, metadata)
                
                return result
                
            return cast(F, wrapper)
        except Exception as e:
            logger.error(f"Failed to track function: {str(e)}")
            raise ValueError(f"Failed to track function: {str(e)}") from e

    @staticmethod
    def get_lineage_from_dataframe(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Extract lineage information from a DataFrame.
        
        Args:
            df: DataFrame to extract lineage from
            
        Returns:
            Dictionary of lineage information, or None if not found
        """
        try:
            # Try all storage backends to find lineage
            for storage_type in ["attribute", "registry"]:
                storage = create_storage(storage_type)
                lineage_record = storage.retrieve(df)
                
                if lineage_record is not None:
                    return lineage_record.to_dict()
            
            return None
        except Exception as e:
            logger.error(f"Failed to extract lineage from DataFrame: {str(e)}")
            raise ValueError(f"Failed to extract lineage: {str(e)}") from e
    
    def merge(self, other: 'LineageTracker') -> 'LineageTracker':
        """
        Merge another LineageTracker into this one.
        
        Args:
            other: Another LineageTracker to merge
            
        Returns:
            Self with merged information
            
        Raises:
            ValueError: If merger fails
        """
        try:
            self.record.merge(other.record)
            return self
        except Exception as e:
            logger.error(f"Failed to merge LineageTracker: {str(e)}")
            raise ValueError(f"Failed to merge LineageTracker: {str(e)}") from e
    
    @classmethod
    def merge_from_dataframes(
        cls, 
        dfs: List[pd.DataFrame], 
        name: Optional[str] = None
    ) -> Optional['LineageTracker']:
        """
        Merge lineage information from multiple DataFrames.
        
        Args:
            dfs: List of DataFrames to merge lineage from
            name: Optional name for the merged lineage tracker
            
        Returns:
            New LineageTracker with merged information, or None if no lineage found
            
        Raises:
            ValueError: If merger fails
        """
        try:
            trackers = []
            
            # Try to get lineage from each DataFrame
            for df in dfs:
                tracker = cls.from_dataframe(df)
                if tracker is not None:
                    trackers.append(tracker)
            
            if not trackers:
                return None
            
            # Create a new tracker for the merged lineage
            merged = cls(name=name or f"merged-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
            
            # Merge all trackers into the new one
            for tracker in trackers:
                merged.merge(tracker)
            
            return merged
        except Exception as e:
            logger.error(f"Failed to merge lineage from DataFrames: {str(e)}")
            raise ValueError(f"Failed to merge lineage: {str(e)}") from e
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert lineage information to a dictionary.
        
        Returns:
            Dictionary representation of the lineage
        """
        try:
            return self.record.to_dict()
        except Exception as e:
            logger.error(f"Failed to convert lineage to dictionary: {str(e)}")
            raise ValueError(f"Failed to convert lineage: {str(e)}") from e


def get_save_function(output_path: PathLike) -> Optional[Callable]:
    """
    Get an appropriate save function based on file extension.
    
    This utility function examines the file extension of the given path
    and returns the appropriate pandas save function.
    
    Args:
        output_path: Path to save to, used to determine file type
        
    Returns:
        A callable function that can save a DataFrame to the given path,
        or None if no appropriate function could be determined
    """
    try:
        path_str = str(ensure_path(output_path))
        
        # Determine save function based on file extension
        if path_str.endswith('.csv'):
            return lambda df, path, **kwargs: df.to_csv(path, **kwargs)
        elif path_str.endswith('.parquet'):
            return lambda df, path, **kwargs: df.to_parquet(path, **kwargs)
        elif path_str.endswith('.feather'):
            return lambda df, path, **kwargs: df.to_feather(path, **kwargs)
        elif path_str.endswith('.json'):
            return lambda df, path, **kwargs: df.to_json(path, **kwargs)
        elif path_str.endswith('.xlsx') or path_str.endswith('.xls'):
            return lambda df, path, **kwargs: df.to_excel(path, **kwargs)
        elif path_str.endswith('.h5') or path_str.endswith('.hdf5'):
            # Note: HDF requires a key, so this is just a placeholder
            # Callers should pass key in kwargs
            return lambda df, path, **kwargs: df.to_hdf(path, key=kwargs.pop('key', 'data'), **kwargs)
        else:
            logger.warning(f"Unknown file extension for {path_str}, unable to determine save function")
            return None
    except Exception as e:
        logger.error(f"Error determining save function for {output_path}: {str(e)}")
        return None


def create_tracker(
    name: Optional[str] = None,
    storage: Optional[LineageStorage] = None,
    attributes: Optional[Dict[str, Any]] = None,
    source_path: Optional[PathLike] = None
) -> LineageTracker:
    """
    Create a new LineageTracker.
    
    This is a convenience function to create a new LineageTracker
    with the specified parameters.
    
    Args:
        name: Optional name for the lineage tracker
        storage: Optional storage backend (creates new one if None)
        attributes: Additional attributes to store
        source_path: Optional source file path to add
        
    Returns:
        New LineageTracker instance
    """
    return LineageTracker(
        name=name,
        storage=storage,
        attributes=attributes,
        source_path=source_path
    )
