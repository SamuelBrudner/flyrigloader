"""
Simple DataFrame tracking utilities.

This module provides basic functionality for tracking DataFrame origins and transformations
without depending on the lineage module. It uses DataFrame attributes for storage.
"""

import pandas as pd
import uuid
import json
from typing import Dict, Any, Optional, Union
from datetime import datetime
from pathlib import Path

from .utils import ensure_path, PathLike

# Constants for DataFrame attribute keys
ATTR_TRACKING = "__flyrig_tracking"
ATTR_TRACKING_ID = "__flyrig_tracking_id"


class SimpleTracker:
    """
    Simple implementation of a DataFrame tracker.
    
    This class provides basic tracking functionality without depending on the 
    lineage module. It stores tracking information in DataFrame attributes.
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize a new tracker.
        
        Args:
            name: Optional name for the tracker
        """
        self.tracking_id = str(uuid.uuid4())
        self.name = name or f"track_{self.tracking_id[:8]}"
        self.history = []
        self.sources = []
        self.outputs = []
        self.created_at = datetime.now().isoformat()
    
    def add_source(self, source: Union[str, PathLike], description: str = "Data source") -> None:
        """
        Add a source to the tracking history.
        
        Args:
            source: Path or identifier for the data source
            description: Description of the source
        """
        try:
            source_path = ensure_path(source)
            source_str = str(source_path)
        except:
            source_str = str(source)
            
        source_entry = {
            "source": source_str,
            "description": description,
            "timestamp": datetime.now().isoformat()
        }
        self.sources.append(source_entry)
        self.history.append({
            "type": "source",
            "timestamp": source_entry["timestamp"],
            "details": source_entry
        })
    
    def add_step(self, step_name: str, description: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a processing step to the tracking history.
        
        Args:
            step_name: Name of the processing step
            description: Description of what the step does
            metadata: Additional metadata about the step
        """
        step_entry = {
            "step_name": step_name,
            "description": description,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }
        self.history.append({
            "type": "step",
            "timestamp": step_entry["timestamp"],
            "details": step_entry
        })
    
    def add_output(self, output: str, description: str = "Data output") -> None:
        """
        Add an output to the tracking history.
        
        Args:
            output: Path or identifier for the data output
            description: Description of the output
        """
        try:
            output_path = ensure_path(output)
            output_str = str(output_path)
        except:
            output_str = str(output)
            
        output_entry = {
            "output": output_str,
            "description": description,
            "timestamp": datetime.now().isoformat()
        }
        self.outputs.append(output_entry)
        self.history.append({
            "type": "output",
            "timestamp": output_entry["timestamp"],
            "details": output_entry
        })
    
    def attach_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Attach tracking information to a DataFrame.
        
        Args:
            df: DataFrame to attach tracking to
            
        Returns:
            DataFrame with tracking information attached
        """
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Ensure attrs dict exists
        if not hasattr(df_copy, 'attrs'):
            df_copy.attrs = {}
            
        # Create tracking dict
        tracking_dict = {
            "id": self.tracking_id,
            "name": self.name,
            "created_at": self.created_at,
            "history": self.history,
            "sources": self.sources,
            "outputs": self.outputs
        }
        
        # Store in DataFrame attributes
        df_copy.attrs[ATTR_TRACKING] = tracking_dict
        df_copy.attrs[ATTR_TRACKING_ID] = self.tracking_id
        
        return df_copy
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> Optional['SimpleTracker']:
        """
        Create a tracker from a DataFrame that has tracking information.
        
        Args:
            df: DataFrame with tracking information
            
        Returns:
            SimpleTracker instance or None if no tracking information found
        """
        if not hasattr(df, 'attrs') or ATTR_TRACKING not in df.attrs:
            return None
            
        # Create new tracker
        tracker = cls()
        
        # Load data from DataFrame
        tracking_dict = df.attrs[ATTR_TRACKING]
        tracker.tracking_id = tracking_dict.get("id", tracker.tracking_id)
        tracker.name = tracking_dict.get("name", tracker.name)
        tracker.created_at = tracking_dict.get("created_at", tracker.created_at)
        tracker.history = tracking_dict.get("history", [])
        tracker.sources = tracking_dict.get("sources", [])
        tracker.outputs = tracking_dict.get("outputs", [])
        
        return tracker
    
    def save(self, path: PathLike) -> None:
        """
        Save tracking information to a JSON file.
        
        Args:
            path: Path to save the tracking information to
        """
        tracking_dict = {
            "id": self.tracking_id,
            "name": self.name,
            "created_at": self.created_at,
            "history": self.history,
            "sources": self.sources,
            "outputs": self.outputs
        }
        
        output_path = ensure_path(path)
        with open(output_path, 'w') as f:
            json.dump(tracking_dict, f, indent=2)


class NullTracker:
    """
    A no-op tracker that implements the same interface as SimpleTracker.
    
    This tracker doesn't perform any tracking but provides the same methods
    as SimpleTracker for compatibility.
    """
    
    def __init__(self, **kwargs):
        """Initialize a null tracker (does nothing)."""
        pass
    
    def add_source(self, source, description="Data source"):
        """No-op method."""
        pass
    
    def add_step(self, step_name, description, metadata=None):
        """No-op method."""
        pass
    
    def add_output(self, output, description="Data output"):
        """No-op method."""
        pass
    
    def attach_to_dataframe(self, df):
        """Returns the DataFrame unchanged."""
        return df
    
    @classmethod
    def from_dataframe(cls, df):
        """Returns a new NullTracker instance."""
        return cls()
    
    def save(self, path):
        """No-op method."""
        pass


def create_tracker(name: Optional[str] = None) -> SimpleTracker:
    """
    Create and return a simple tracker instance.
    
    Args:
        name: Optional name for the tracker
    
    Returns:
        SimpleTracker instance
    """
    return SimpleTracker(name=name)


def create_null_tracker(**kwargs) -> NullTracker:
    """
    Create and return a null tracker instance.
    
    Args:
        **kwargs: Keyword arguments (ignored)
    
    Returns:
        NullTracker instance
    """
    return NullTracker(**kwargs)


def has_tracking(df: pd.DataFrame) -> bool:
    """
    Check if a DataFrame has tracking information.
    
    Args:
        df: DataFrame to check
    
    Returns:
        True if the DataFrame has tracking information, False otherwise
    """
    return hasattr(df, 'attrs') and ATTR_TRACKING in df.attrs


def get_tracking_from_dataframe(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Extract tracking information from a DataFrame.
    
    Args:
        df: DataFrame to extract tracking from
    
    Returns:
        Dictionary with tracking information or None if no tracking found
    """
    if not has_tracking(df):
        return None
    
    return df.attrs.get(ATTR_TRACKING)


def attach_tracking_to_dataframe(df: pd.DataFrame, tracker: Union[SimpleTracker, NullTracker]) -> pd.DataFrame:
    """
    Attach tracking from a tracker to a DataFrame.
    
    Args:
        df: DataFrame to attach tracking to
        tracker: Tracker to attach
    
    Returns:
        DataFrame with tracking attached
    """
    return tracker.attach_to_dataframe(df)


def save_df(
    df: pd.DataFrame, 
    output_path: PathLike, 
    tracker: Optional[Union[SimpleTracker, NullTracker]] = None
) -> None:
    """
    Save a DataFrame to a file with tracking information.
    
    Args:
        df: DataFrame to save
        output_path: Path to save the DataFrame to
        tracker: Optional tracker to attach to the DataFrame
    """
    path = ensure_path(output_path)
    
    # Attach tracking if provided
    if tracker:
        df = tracker.attach_to_dataframe(df)
        tracker.add_output(path, f"Saved DataFrame to {path}")
    
    # Determine save function based on file extension
    ext = path.suffix.lower()
    
    if ext == '.csv':
        df.to_csv(path, index=False)
    elif ext == '.parquet':
        df.to_parquet(path, index=False)
    elif ext == '.pkl' or ext == '.pickle':
        df.to_pickle(path)
    elif ext == '.xlsx':
        df.to_excel(path, index=False)
    elif ext == '.json':
        df.to_json(path, orient='records', lines=True)
    else:
        # Default to parquet
        df.to_parquet(f"{path}.parquet", index=False)
