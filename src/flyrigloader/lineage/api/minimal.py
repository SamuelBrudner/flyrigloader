"""
minimal.py - Simplified API for lineage tracking.

This module provides the MinimalLineageTracker class, which is a simplified interface
for lineage tracking in the flyrigloader package. It focuses on ease of use for
basic lineage tracking needs, while still leveraging the full power of the core
LineageTracker internally.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, TypeVar, cast

import pandas as pd
from loguru import logger

from ...core.utils import PathLike, ensure_path, create_metadata, create_error_metadata
from ..core import (
    ATTR_MINIMAL_FLAG,
    LineageRecord,
    create_storage,
    ensure_dataframe
)

# Import LineageTracker type for type hints only, but avoid importing the actual class
# This prevents circular imports while still allowing proper type checking
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .standard import LineageTracker

# Type variables for better type hints
DF = TypeVar('DF', bound=pd.DataFrame)
F = TypeVar('F', bound=Callable)


class MinimalLineageTracker:
    """
    Simplified lineage tracker for basic data processing pipelines.
    
    This class provides a minimal API for tracking lineage information,
    focusing on ease of use for common scenarios. It internally uses the
    full LineageTracker with a simplified interface.
    
    Key features:
    - Automatic source tracking from file paths
    - Simplified step tracking with sensible defaults
    - Chainable API for fluent usage
    """
    
    def __init__(
            self, 
            source: Optional[PathLike] = None, 
            description: str = "Data loaded",
            storage_type: str = "attribute",
            name: Optional[str] = None
        ):
        """
        Initialize a new MinimalLineageTracker.
        
        Args:
            source: Optional path to the data source
            description: Description for the initial source
            storage_type: Type of storage backend to use ("attribute", "registry")
            name: Optional name for this lineage tracker
            
        Raises:
            ValueError: If initialization fails with invalid parameters
            TypeError: If parameters are of incorrect types
            RuntimeError: If tracker initialization fails
        """
        # Create the internal tracker
        try:
            # Lazy import to avoid circular dependencies
            from .standard import create_tracker
            
            self._tracker = create_tracker(
                name=name or f"minimal-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                storage_type=storage_type
            )
        except TypeError as e:
            logger.error(f"Type error initializing MinimalLineageTracker: {str(e)}")
            raise TypeError(f"Invalid parameter type: {str(e)}") from e
        except ValueError as e:
            logger.error(f"Value error initializing MinimalLineageTracker: {str(e)}")
            raise ValueError(f"Invalid parameter value: {str(e)}") from e
        except KeyError as e:
            logger.error(f"Missing key when initializing MinimalLineageTracker: {str(e)}")
            raise ValueError(f"Missing required configuration key: {str(e)}") from e
        except Exception as e:
            # For unexpected errors, log clearly and re-raise with context
            logger.error(f"Unexpected error initializing MinimalLineageTracker: {str(e)}")
            raise RuntimeError(f"Failed to initialize lineage tracker: {str(e)}") from e
            
        # Set minimal flag in metadata
        try:
            self._tracker.record.metadata["minimal"] = True
        except AttributeError as e:
            logger.error(f"Failed to set minimal flag in metadata: {str(e)}")
            raise RuntimeError(f"Tracker initialization incomplete: {str(e)}") from e
        
        # Add source if provided
        if source is not None:
            try:
                result, metadata = self.add_source(source, description)
                if not metadata["success"]:
                    logger.warning(f"Created tracker but failed to add initial source: {metadata.get('error', 'Unknown error')}")
            except (ValueError, TypeError, FileNotFoundError) as e:
                logger.error(f"Failed to add initial source {source}: {str(e)}")
                # We still have a valid tracker, so just warn about the source failure
                logger.warning(f"Created tracker without initial source due to error: {str(e)}")
    
    @property
    def lineage_id(self) -> str:
        """
        Get the unique identifier for this lineage tracker.
        
        Returns:
            The lineage ID string
        """
        return self._tracker.lineage_id
    
    @property
    def name(self) -> str:
        """
        Get the name of this lineage tracker.
        
        Returns:
            The name string
        """
        return self._tracker.name
    
    @name.setter
    def name(self, value: str) -> None:
        """
        Set the name of this lineage tracker.
        
        Args:
            value: New name to set
        """
        self._tracker.name = value
    
    def add_source(
        self, 
        source: Union[PathLike, pd.DataFrame], 
        description: str = "Data source"
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Add a source to the lineage.
        
        This records where the data originally came from (file, API, etc).
        
        Args:
            source: Path to the data source or a DataFrame
            description: Description of this source
            
        Returns:
            Tuple of (success, metadata) where success is a boolean indicating if 
            the operation was successful and metadata contains status information 
            and any error details
            
        Raises:
            ValueError: If source is invalid
            TypeError: If parameters are of incorrect types
        """
        try:
            metadata = create_metadata()
            
            if isinstance(source, (str, Path)):
                # Handle file path source
                path = ensure_path(source)
                
                if not os.path.exists(path):
                    metadata["error"] = f"Source file does not exist: {path}"
                    return False, metadata
                    
                self._tracker.add_source(
                    source=str(path), 
                    source_type="file",
                    description=description
                )
            elif isinstance(source, pd.DataFrame):
                # Handle DataFrame source
                self._tracker.add_source(
                    source="dataframe", 
                    source_type="dataframe",
                    description=description
                )
            else:
                metadata["error"] = f"Unsupported source type: {type(source).__name__}"
                return False, metadata
                
            metadata["success"] = True
            return True, metadata
            
        except Exception as e:
            logger.error(f"Failed to add source: {str(e)}")
            return False, create_error_metadata(e)
    
    def add_step(
        self, 
        name: str, 
        description: Optional[str] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Add a processing step to the lineage.
        
        Args:
            name: Name of the processing step
            description: Description of what the step does
            
        Returns:
            Tuple of (success, metadata) where success is a boolean indicating if 
            the operation was successful and metadata contains status information 
            and any error details
            
        Raises:
            ValueError: If the step information is invalid
        """
        try:
            metadata = create_metadata()
            self._tracker.add_step(name, description or f"Applied {name}")
            metadata["success"] = True
            return True, metadata
        except Exception as e:
            logger.error(f"Failed to add step {name}: {str(e)}")
            return False, create_error_metadata(e)
    
    def apply(
        self, 
        df: pd.DataFrame, 
        function: Callable[[pd.DataFrame], pd.DataFrame],
        step_name: str,
        description: str = "Applied transformation",
        **kwargs
    ) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """
        Apply a function to the DataFrame and track the step.
        
        Args:
            df: DataFrame to transform
            function: Function to apply to the DataFrame
            step_name: Name of this processing step
            description: Description of what this step does
            **kwargs: Additional keyword arguments to pass to the function
            
        Returns:
            Tuple of (result_df, metadata) where result_df is the transformed
            DataFrame or None if the operation failed, and metadata contains 
            status information and any error details
        """
        try:
            metadata = create_metadata()
            
            if not callable(function):
                metadata["error"] = "Function must be callable"
                return None, metadata
                
            # Ensure we have a valid DataFrame
            try:
                df = ensure_dataframe(df)
            except (TypeError, ValueError) as e:
                metadata["error"] = f"Invalid DataFrame: {str(e)}"
                return None, metadata
                
            # Record the step and apply function
            step_id = self._tracker.add_step(
                step_name=step_name, 
                description=description
            )
            
            # Try to apply the function
            try:
                result_df = function(df, **kwargs)
                
                # Ensure the result is a DataFrame
                if not isinstance(result_df, pd.DataFrame):
                    metadata["error"] = f"Function returned {type(result_df).__name__}, expected DataFrame"
                    return None, metadata
                    
                # Attach the lineage to the result
                self._tracker.attach_to_dataframe(result_df)
                
                metadata["success"] = True
                metadata["step_id"] = step_id
                return result_df, metadata
                
            except Exception as e:
                # Failed to apply function
                logger.error(f"Error applying function {function.__name__}: {str(e)}")
                
                # Mark step as failed
                self._tracker.update_step_metadata(
                    step_id=step_id,
                    metadata={"status": "failed", "error": str(e)}
                )
                
                metadata["error"] = f"Function application failed: {str(e)}"
                metadata["step_id"] = step_id
                return None, metadata
                
        except Exception as e:
            logger.error(f"Failed to apply function and track step: {str(e)}")
            return None, create_error_metadata(e)
    
    def attach(self, df: DF) -> DF:
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
            df = ensure_dataframe(df)
            result = self._tracker.attach_to_dataframe(df)
            
            # Set the minimal flag
            if hasattr(result, 'attrs'):
                result.attrs[ATTR_MINIMAL_FLAG] = True
            return result
        except Exception as e:
            logger.error(f"Failed to attach lineage to DataFrame: {str(e)}")
            raise ValueError(f"Failed to attach lineage: {str(e)}") from e
    
    def process(
        self, 
        df: pd.DataFrame, 
        step_name: str, 
        output_df: Optional[DF] = None,
        description: Optional[str] = None
    ) -> Tuple[Optional[DF], Dict[str, Any]]:
        """
        Track a processing step on a DataFrame.
        
        Args:
            df: Input DataFrame
            step_name: Name of the processing step
            output_df: Optional output DataFrame (if None, input DataFrame is used)
            description: Description of the processing step
            
        Returns:
            Tuple of (result_df, metadata) where result_df is the DataFrame with 
            updated lineage or None if the operation failed, and metadata contains 
            status information and any error details
            
        Raises:
            ValueError: If tracking fails
        """
        try:
            df = ensure_dataframe(df)
            output = output_df if output_df is not None else df
            output = ensure_dataframe(output)
            
            # Track the transformation
            result, metadata = self.apply(df, lambda x: x, step_name, description)
            
            # Set the minimal flag
            if hasattr(result, 'attrs'):
                result.attrs[ATTR_MINIMAL_FLAG] = True
            return result, metadata
        except Exception as e:
            logger.error(f"Failed to track processing step {step_name}: {str(e)}")
            return None, create_error_metadata(e)
    
    def track_loaded_df(
        self, 
        df: pd.DataFrame, 
        source: PathLike, 
        description: str = "Loaded from file"
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Track a DataFrame that was loaded from a file.
        
        Args:
            df: DataFrame that was loaded
            source: Path to the source file
            description: Description of the source
            
        Returns:
            Tuple of (df, metadata) where df is the input DataFrame with lineage 
            attached and metadata contains status information and any error details
        """
        try:
            metadata = create_metadata()
            
            # Ensure path is valid
            try:
                source_path = ensure_path(source)
                if not os.path.exists(source_path):
                    metadata["error"] = f"Source file does not exist: {source_path}"
                    return df, metadata
            except (TypeError, ValueError) as e:
                metadata["error"] = f"Invalid source path: {str(e)}"
                return df, metadata
                
            # Add source to tracker
            source_add_result, source_metadata = self.add_source(source, description)
            if not source_metadata["success"]:
                metadata["error"] = source_metadata.get("error", "Failed to add source")
                return df, metadata
                
            # Attach lineage to the DataFrame
            try:
                self._tracker.attach_to_dataframe(df)
                metadata["success"] = True
                return df, metadata
            except Exception as e:
                metadata["error"] = f"Failed to attach lineage to DataFrame: {str(e)}"
                return df, metadata
                
        except Exception as e:
            logger.error(f"Failed to track loaded DataFrame: {str(e)}")
            return df, create_error_metadata(e)
    
    def save_df(
        self, 
        df: pd.DataFrame, 
        output_path: PathLike, 
        save_function: Optional[Callable] = None,
        description: str = "Saved to file",
        **kwargs
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Save a DataFrame to a file and track the output.
        
        Args:
            df: DataFrame to save
            output_path: Path where to save the DataFrame
            save_function: Function to use for saving (default: auto-detect from extension)
            description: Description of this output
            **kwargs: Additional arguments to pass to the save function
            
        Returns:
            Tuple of (success, metadata) where success is a boolean indicating if 
            the operation was successful and metadata contains status information
            and any error details
            
        Raises:
            ValueError: If saving fails
        """
        try:
            metadata = create_metadata()
            
            # Ensure we have a valid DataFrame
            try:
                df = ensure_dataframe(df)
            except (TypeError, ValueError) as e:
                metadata["error"] = f"Invalid DataFrame: {str(e)}"
                return False, metadata
                
            # Ensure output path is valid
            try:
                output_path = ensure_path(output_path)
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            except (TypeError, ValueError, OSError) as e:
                metadata["error"] = f"Invalid output path: {str(e)}"
                return False, metadata
                
            # Determine save function if not provided
            if save_function is None:
                # Lazy import to avoid circular dependencies
                from .standard import get_save_function
                save_function = get_save_function(output_path)
                
            # Check if we have a valid save function
            if save_function is None:
                metadata["error"] = f"Could not determine save function for {output_path}"
                return False, metadata
                    
            # Save the file
            try:
                save_function(df, output_path, **kwargs)
            except Exception as e:
                metadata["error"] = f"Failed to save DataFrame: {str(e)}"
                return False, metadata
                
            # Record the output
            self._tracker.add_output(
                output=str(output_path),
                output_type="file",
                description=description
            )
            
            metadata["success"] = True
            metadata["output_path"] = str(output_path)
            return True, metadata
            
        except Exception as e:
            logger.error(f"Failed to save DataFrame: {str(e)}")
            return False, create_error_metadata(e)
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> Optional['MinimalLineageTracker']:
        """
        Create a MinimalLineageTracker from a DataFrame's lineage information.
        
        Args:
            df: DataFrame to extract lineage from
            
        Returns:
            New MinimalLineageTracker instance, or None if extraction fails
        """
        try:
            # Try to get the underlying tracker
            # Lazy import to avoid circular dependencies
            from .standard import LineageTracker
            tracker = LineageTracker.from_dataframe(df)
            
            if tracker is not None:
                # Create a new minimal tracker
                minimal = cls()
                # Replace the internal tracker
                minimal._tracker = tracker
                return minimal
        except Exception as e:
            logger.error(f"Failed to create MinimalLineageTracker from DataFrame: {str(e)}")
            raise ValueError(f"Failed to create MinimalLineageTracker: {str(e)}") from e
        
        return None
    
    def save(self, path: PathLike) -> Tuple[bool, Dict[str, Any]]:
        """
        Save lineage information to a file.
        
        Args:
            path: Path to save the lineage to
            
        Returns:
            Tuple of (success, metadata) where success is a boolean indicating if 
            the operation was successful and metadata contains status information
            and any error details
            
        Raises:
            ValueError: If saving fails
        """
        try:
            metadata = create_metadata()
            self._tracker.save(path)
            metadata["success"] = True
            return True, metadata
        except Exception as e:
            logger.error(f"Failed to save lineage to {path}: {str(e)}")
            return False, create_error_metadata(e)
    
    @classmethod
    def load(cls, path: PathLike) -> Tuple[Optional['MinimalLineageTracker'], Dict[str, Any]]:
        """
        Load lineage information from a file.
        
        Args:
            path: Path to load the lineage from
            
        Returns:
            Tuple of (tracker, metadata) where tracker is the loaded 
            MinimalLineageTracker instance or None if loading fails, and metadata 
            contains status information and any error details
        """
        try:
            # Lazy import to avoid circular dependencies
            from .standard import LineageTracker
            tracker = LineageTracker.load(path)
            if tracker is None:
                raise ValueError("Failed to load lineage tracker")
            
            minimal = cls()
            minimal._tracker = tracker
            return minimal, create_metadata({"success": True})
        except Exception as e:
            logger.error(f"Failed to load lineage from {path}: {str(e)}")
            return None, create_error_metadata(e)
    
    @staticmethod
    def get_source(df: pd.DataFrame) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Get the primary source path from a DataFrame.
        
        Args:
            df: DataFrame to extract source from
            
        Returns:
            Tuple of (source, metadata) where source is the source path if found, 
            and metadata contains status information and any error details
        """
        try:
            # Lazy import to avoid circular dependencies
            from .standard import LineageTracker
            lineage_dict = LineageTracker.get_lineage_from_dataframe(df)
            
            if lineage_dict and 'sources' in lineage_dict and lineage_dict['sources']:
                return lineage_dict['sources'][0].get('path'), create_metadata({"success": True})
        except Exception as e:
            logger.error(f"Failed to get source from DataFrame: {str(e)}")
            return None, create_error_metadata(e)
        
        return None, create_metadata({"error": "No source found"})
    
    @staticmethod
    def get_steps(df: pd.DataFrame) -> Tuple[Optional[List[Dict[str, Any]]], Dict[str, Any]]:
        """
        Get the processing steps from a DataFrame.
        
        Args:
            df: DataFrame to extract steps from
            
        Returns:
            Tuple of (steps, metadata) where steps is the list of processing steps 
            if found, and metadata contains status information and any error details
        """
        try:
            # Lazy import to avoid circular dependencies
            from .standard import LineageTracker
            lineage_dict = LineageTracker.get_lineage_from_dataframe(df)
            
            if lineage_dict and 'steps' in lineage_dict:
                return lineage_dict['steps'], create_metadata({"success": True})
        except Exception as e:
            logger.error(f"Failed to get steps from DataFrame: {str(e)}")
            return None, create_error_metadata(e)
        
        return None, create_metadata({"error": "No steps found"})
    
    def track_function(self, function: F) -> F:
        """
        Decorator to track lineage for a function that processes DataFrames.
        
        The decorated function must take a DataFrame as its first argument
        and return a DataFrame.
        
        Args:
            function: Function to track
            
        Returns:
            Wrapped function with lineage tracking
        """
        try:
            # Lazy import to avoid circular dependencies
            from .standard import LineageTracker
            return LineageTracker.track_function(function)
        except Exception as e:
            logger.error(f"Failed to track function {function.__name__}: {str(e)}")
            raise ValueError(f"Failed to track function: {str(e)}") from e


def create_minimal_tracker(
    source: Optional[PathLike] = None,
    description: str = "Data loaded",
    storage_type: str = "attribute",
    name: Optional[str] = None
) -> Tuple[Optional[MinimalLineageTracker], Dict[str, Any]]:
    """
    Create a new MinimalLineageTracker.
    
    This is a convenience function to create a new MinimalLineageTracker
    with the specified parameters.
    
    Args:
        source: Optional path to the data source
        description: Description for the initial source
        storage_type: Type of storage backend to use
        name: Optional name for the lineage tracker
        
    Returns:
        Tuple of (tracker, metadata) where tracker is the new MinimalLineageTracker 
        instance or None if creation fails, and metadata contains status information 
        and any error details
    """
    try:
        return MinimalLineageTracker(
            source=source,
            description=description,
            storage_type=storage_type,
            name=name
        ), create_metadata({"success": True})
    except Exception as e:
        logger.error(f"Failed to create MinimalLineageTracker: {str(e)}")
        return None, create_error_metadata(e)
