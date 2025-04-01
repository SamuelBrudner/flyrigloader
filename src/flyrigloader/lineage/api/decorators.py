"""
decorators.py - Function decorators for lineage tracking.

This module provides decorators that can be used to automatically track
lineage for functions that process DataFrames, making it easy to add
lineage tracking to existing code with minimal changes.
"""

import functools
import inspect
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast

import pandas as pd
from loguru import logger

from ..core import ensure_dataframe
from .minimal import MinimalLineageTracker
from .standard import LineageTracker

# Type variables for better type hints
F = TypeVar('F', bound=Callable)
DF = TypeVar('DF', bound=pd.DataFrame)


def track_lineage(
    step_name: Optional[str] = None,
    description: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    minimal: bool = False
) -> Callable[[F], F]:
    """
    Decorator to track lineage for a function that processes DataFrames.
    
    This decorator automatically creates a LineageTracker (or MinimalLineageTracker)
    for each function call, adds a processing step for the function, and attaches
    the updated lineage to the output DataFrame.
    
    The decorated function must:
    - Take a DataFrame as its first positional argument
    - Return a DataFrame
    
    Args:
        step_name: Name of the processing step (defaults to function name)
        description: Description of the processing step (defaults to function docstring)
        metadata: Additional metadata for the processing step
        minimal: Whether to use MinimalLineageTracker (True) or LineageTracker (False)
        
    Returns:
        Decorated function with lineage tracking
        
    Example:
        @track_lineage(description="Filter rows by value")
        def filter_rows(df, column, value):
            return df[df[column] == value]
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(df: DF, *args, **kwargs) -> DF:
            # Ensure df is a DataFrame
            df = ensure_dataframe(df)
            
            # Determine step name and description
            nonlocal step_name, description
            function_name = func.__name__
            actual_step_name = step_name or f"{func.__module__}.{function_name}"
            actual_description = description or func.__doc__ or f"Applied function {function_name}"
            
            # Create metadata for the step
            actual_metadata = metadata or {}
            actual_metadata.update({
                "function": function_name,
                "module": func.__module__,
                "args": str(args) if args else "",
                "kwargs": str(kwargs) if kwargs else ""
            })
            
            # Call the original function
            result = func(df, *args, **kwargs)
            
            # Ensure result is a DataFrame
            if not isinstance(result, pd.DataFrame):
                logger.warning(f"Function {function_name} did not return a DataFrame, skipping lineage tracking")
                return result
            
            # Track lineage
            try:
                # Check if input DataFrame already has lineage
                existing_tracker = MinimalLineageTracker.from_dataframe(df) if minimal else LineageTracker.from_dataframe(df)
                
                if existing_tracker is not None:
                    # Use the existing tracker to track this transformation
                    if minimal:
                        return existing_tracker.process(df, actual_step_name, result, actual_description)
                    return existing_tracker.track_transformation(
                        df, result, actual_step_name, actual_description, actual_metadata
                    )
                
                # Create a new tracker
                if minimal:
                    tracker = MinimalLineageTracker()
                    tracker.add_step(actual_step_name, actual_description)
                    return tracker.attach(result)
                
                tracker = LineageTracker(name=f"lineage-{function_name}")
                tracker.add_step(actual_step_name, actual_description, actual_metadata)
                return tracker.attach_to_dataframe(result)
            except Exception as e:
                logger.warning(f"Failed to track lineage for {function_name}: {str(e)}")
                # Return the result anyway, even if lineage tracking failed
                return result
        
        return cast(F, wrapper)
    
    return decorator


def track_minimal_lineage(
    step_name: Optional[str] = None,
    description: Optional[str] = None
) -> Callable[[F], F]:
    """
    Decorator to track lineage using MinimalLineageTracker.
    
    This is a convenience wrapper around track_lineage with minimal=True.
    It provides a simplified lineage tracking experience focused on ease of use.
    
    Args:
        step_name: Name of the processing step (defaults to function name)
        description: Description of the processing step (defaults to function docstring)
        
    Returns:
        Decorated function with minimal lineage tracking
        
    Example:
        @track_minimal_lineage(description="Clean data")
        def clean_data(df):
            # cleaning operations
            return cleaned_df
    """
    return track_lineage(step_name=step_name, description=description, minimal=True)


def propagate_lineage(
    input_arg_index: int = 0,
    output_arg_index: Optional[int] = None
) -> Callable[[F], F]:
    """
    Decorator to propagate lineage from input to output object.
    
    This decorator is useful for functions that transform one object to another
    where lineage should be preserved, but the actual lineage content doesn't need
    to be modified.
    
    Args:
        input_arg_index: Index of the input argument to get lineage from
        output_arg_index: Index of the output object to add lineage to (None if returning)
        
    Returns:
        Decorated function with lineage propagation
        
    Example:
        # For a function that returns a new DataFrame
        @propagate_lineage()
        def pivot_data(df):
            return df.pivot(...)
            
        # For a function that modifies an output argument
        @propagate_lineage(input_arg_index=0, output_arg_index=1)
        def process_data(input_df, output_df):
            # process input_df and update output_df
            pass
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Call the original function
            result = func(*args, **kwargs)
            
            try:
                # Ensure we have enough arguments
                if len(args) <= input_arg_index:
                    return result
                
                input_obj = args[input_arg_index]
                
                # Check if input is a DataFrame with lineage
                if not isinstance(input_obj, pd.DataFrame):
                    return result
                
                # Try to find existing lineage
                lineage_tracker = LineageTracker.from_dataframe(input_obj)
                if lineage_tracker is None:
                    lineage_tracker = MinimalLineageTracker.from_dataframe(input_obj)
                
                if lineage_tracker is None:
                    # No lineage to propagate
                    return result
                
                # Determine the output object
                if output_arg_index is not None:
                    # Update an argument in-place
                    if len(args) <= output_arg_index:
                        return result
                    
                    output_obj = args[output_arg_index]
                    if not isinstance(output_obj, pd.DataFrame):
                        return result
                    
                    # Attach lineage to the output argument
                    if isinstance(lineage_tracker, MinimalLineageTracker):
                        lineage_tracker.attach(output_obj)
                    else:
                        lineage_tracker.attach_to_dataframe(output_obj)
                else:
                    # Update the returned object
                    if not isinstance(result, pd.DataFrame):
                        return result
                    
                    # Attach lineage to the returned object
                    if isinstance(lineage_tracker, MinimalLineageTracker):
                        result = lineage_tracker.attach(result)
                    else:
                        result = lineage_tracker.attach_to_dataframe(result)
            except Exception as e:
                logger.warning(f"Failed to propagate lineage: {str(e)}")
                
            return result
        
        return cast(F, wrapper)
    
    return decorator
