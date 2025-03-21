"""
Data assembly for fly rig experiments.

This module provides low-level functions for transforming raw file data into structured
DataFrames with proper typing, metadata, and lineage tracking. It handles the actual
data loading and transformation logic, while higher-level pipeline orchestration is 
handled by the data_pipeline module.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
from pathlib import Path
from datetime import datetime
from loguru import logger

from ..schema.validator import apply_schema
from ..readers.pickle import read_pickle_any_format
from ..lineage.tracker import LineageTracker


def load_file_into_dataframe(
    item: Union[str, Path, Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None,
    schema: Optional[Dict[str, Any]] = None,
    lineage: Optional[LineageTracker] = None,
    track_lineage: bool = True
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Read a file and transform it into a standardized DataFrame with metadata.
    
    This function supports multiple calling patterns:
    1. Dict item with metadata: {'path': Path, 'rig': 'rig1', ...}
    2. Direct file path: 'data/experiment/file.pkl'
    
    Args:
        item: File path or metadata dictionary with 'path' key
        config: Optional configuration dictionary
        schema: Optional schema for validation and transformation
        lineage: Optional LineageTracker for data provenance
        track_lineage: Whether to track data lineage
        
    Returns:
        Tuple of (DataFrame, metadata_dict)
        If processing fails, DataFrame will be None and metadata_dict will contain error info.
    """
    # Handle different input types
    if isinstance(item, (str, Path)):
        path = Path(item)
        item_dict = {
            'path': path,
            'file_path': str(path),
            'file_name': path.name,
            'date': datetime.now().strftime("%Y-%m-%d")
        }
    else:
        item_dict = item
        # Check if 'path' key exists in the dictionary
        if 'path' not in item_dict:
            processing_meta = {
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "error": "'path' key missing in item dictionary"
            }
            logger.error("Required 'path' key missing in item dictionary")
            return None, processing_meta
            
        path = Path(item_dict['path'])
    
    # Initialize processing metadata
    processing_meta = {
        "file_processed": str(path),
        "timestamp": datetime.now().isoformat(),
        "success": False,
        "rows_processed": 0
    }

    # Create or use lineage tracker
    if track_lineage:
        if lineage is None:
            from ..lineage.minimal import MinimalLineageTracker
            tracker = MinimalLineageTracker(source=path, description=f"Loading file {path.name}")
            lineage = tracker._tracker

        # Record source in lineage
        lineage.add_source(path, metadata=item_dict)
        lineage.add_step("load_file", f"Loading data from {path.name}", {
            "item_metadata": item_dict
        })

    # Load data from file
    try:
        # Get file reader based on extension
        obj = read_pickle_any_format(path)
        
        if obj is None:
            processing_meta["error"] = "Failed to read file"
            return None, processing_meta
            
        # Convert to DataFrame if it's a dictionary
        if isinstance(obj, dict):
            df = _dict_to_dataframe(obj, item_dict, lineage)
        elif isinstance(obj, pd.DataFrame):
            df = obj
            if lineage:
                lineage.add_step(
                    "use_dataframe",
                    "Using existing DataFrame",
                    {"original_columns": list(df.columns)}
                )
        else:
            # Unsupported type
            processing_meta["error"] = f"Unsupported data type: {type(obj)}"
            return None, processing_meta
            
        # Attach metadata
        df, overwritten_cols = _attach_metadata(df, item_dict)
        
        # Update processing metadata
        processing_meta["rows_processed"] = len(df)
        if overwritten_cols:
            processing_meta["overwritten_columns"] = overwritten_cols
            
        # Add metadata attachment to lineage
        if lineage:
            metadata_info = {"metadata_added": {k: v for k, v in item_dict.items() if k != 'path'}}
            if overwritten_cols:
                metadata_info["overwritten_columns"] = overwritten_cols
                
            lineage.add_step(
                "assign_metadata",
                "Adding metadata columns",
                metadata_info
            )
        
        # Apply schema if provided
        if schema:
            # Transform DataFrame according to schema
            df = apply_schema(df, schema)
            
            if lineage:
                lineage.add_step(
                    "apply_schema",
                    "Applying schema transformations",
                    {
                        "schema_version": schema.get("schema_version", "unknown"),
                        "schema_name": schema.get("schema_name", "unnamed"),
                        "columns_after_schema": list(df.columns)
                    }
                )
        elif lineage:
            lineage.add_step(
                "skip_schema",
                "No schema provided, skipping schema application",
                {"columns": list(df.columns)}
            )
            
        # Mark processing as successful
        processing_meta["success"] = True
        return df, processing_meta
        
    except Exception as e:
        # Log any errors during processing
        error_msg = str(e)
        logger.error(f"Error processing {path}: {error_msg}")
        processing_meta["error"] = error_msg
        return None, processing_meta


def _dict_to_dataframe(
    data_dict: Dict[str, Any],
    metadata: Dict[str, Any],
    lineage: Optional[LineageTracker] = None
) -> pd.DataFrame:
    """
    Convert a raw data dictionary into a DataFrame.
    
    Args:
        data_dict: Raw dictionary containing data arrays
        metadata: Metadata dictionary with 'path', 'rig', etc.
        lineage: Optional lineage tracker
        
    Returns:
        DataFrame with basic structure
    """
    from ..schema.operations import ensure_1d

    # Handle time array, assuming 't' is the standard time key
    if 't' not in data_dict:
        raise ValueError("Data dictionary must contain a 't' array")

    t = ensure_1d(data_dict['t'], 't')

    # Start with basic metadata
    result_dict = {
        't': t,
        'rig': metadata.get('rig', 'unknown'),
        'date': metadata.get('date', 'unknown'),
        'file_path': str(metadata['path'])
    }

    # Add standard data columns
    standard_columns = [
        'trjn', 'x', 'y', 'x_smooth', 'y_smooth', 'theta', 'theta_smooth',
        'dtheta_smooth', 'vx_smooth', 'vy_smooth', 'spd_smooth',
        'signal', 'jumps', 'headx_smooth', 'heady_smooth'
    ]

    # Add each column if it exists
    for col in standard_columns:
        if col in data_dict:
            result_dict[col] = ensure_1d(data_dict[col], col)
        else:
            # Create empty column filled with NaN
            import numpy as np
            result_dict[col] = np.full_like(t, np.nan, dtype=float)

    # Handle special case for signal_disp (which is 2D)
    if 'signal_disp' in data_dict:
        # Convert 2D array to list of arrays
        sd = data_dict['signal_disp']
        if sd.ndim == 2:
            # Match to time dimension
            T = len(t)
            n, m = sd.shape

            if n == T:
                # Already in correct orientation
                pass
            elif m == T:
                # Transpose to match time
                sd = sd.T
            else:
                if lineage:
                    lineage.add_step(
                        "warning",
                        f"signal_disp shape {sd.shape} does not match time length {T}"
                    )
                import numpy as np
                result_dict['signal_disp'] = pd.Series([[] for _ in range(len(t))], index=range(len(t)))

            # Store as Series of arrays
            result_dict['signal_disp'] = pd.Series(list(sd), index=range(T), name='signal_disp')

        elif lineage:
            lineage.add_step(
                "warning", 
                f"signal_disp has unexpected shape {sd.shape}, expected 2D array"
            )
    # Create the DataFrame
    df = pd.DataFrame(result_dict)

    # Record in lineage
    if lineage:
        lineage.add_step(
            "make_dataframe",
            "Converting dictionary to DataFrame",
            {"original_keys": list(data_dict.keys())},
        )

    return df


def _attach_metadata(
    df: pd.DataFrame,
    metadata: Dict[str, Any]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Attach metadata to a DataFrame as columns.
    
    Args:
        df: DataFrame to attach metadata to
        metadata: Dictionary of metadata to attach
        
    Returns:
        Tuple of (modified_dataframe, list_of_overwritten_columns)
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    overwritten_columns = []
    
    # Add each metadata field as a column
    for key, value in metadata.items():
        if key != 'path':
            # Check if column already exists
            if key in result_df.columns:
                overwritten_columns.append(key)
                logger.warning(f"Metadata column '{key}' is overwriting an existing column")
                
            # Add as column
            result_df[key] = value
    
    return result_df, overwritten_columns


def combine_dataframes(
    dataframes: List[pd.DataFrame],
    lineage: Optional[LineageTracker] = None
) -> pd.DataFrame:
    """
    Combine multiple DataFrames into a single result.
    
    Args:
        dataframes: List of DataFrames to combine
        lineage: Optional lineage tracker
        
    Returns:
        Combined DataFrame with merged lineage
    """
    if not dataframes:
        return pd.DataFrame()
        
    # Concatenate all dataframes
    result_df = pd.concat(dataframes, ignore_index=True)
    
    # Record in lineage
    if lineage:
        lineage.add_step(
            "concatenate",
            "Concatenated all DataFrames",
            {
                "total_dataframes": len(dataframes),
                "total_rows": len(result_df),
                "resulting_columns": list(result_df.columns)
            }
        )
        
        # Attach lineage to DataFrame
        from ..lineage.tracker import attach_lineage_to_dataframe
        result_df = attach_lineage_to_dataframe(result_df, lineage)
        
    return result_df
