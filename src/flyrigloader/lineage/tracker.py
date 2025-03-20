"""
tracker.py - Core module for tracking data lineage throughout the pipeline.

This module provides functionality for:
1. Creating and tracking data lineage information (data sources, processing steps)
2. Attaching lineage metadata to DataFrames
3. Exporting and importing lineage information to/from various formats

Data lineage allows tracing how a dataset was generated, from raw inputs to final outputs.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Set
import uuid
import pandas as pd
import yaml
from loguru import logger


class LineageTracker:
    """
    Tracks data lineage information throughout the data processing pipeline.
    
    This class maintains information about:
    - Input sources (file paths, timestamps, versions)
    - Processing steps applied to the data
    - Configuration parameters used
    - Any metadata needed for complete reproducibility
    """
    
    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize a new LineageTracker.
        
        Args:
            name: Optional name for this lineage tracker
            config: Optional configuration dictionary used for this pipeline run
        """
        self.lineage_id = str(uuid.uuid4())
        self.name = name or f"lineage-{self.lineage_id[:8]}"
        self.creation_time = datetime.now().isoformat()
        self.sources = []
        self.steps = []
        self.config = config or {}
        self.metadata = {
            "user": os.environ.get("USER", "unknown"),
            "system": sys.platform,
            "python_version": sys.version,
            "timestamp": time.time(),
            "date": datetime.now().strftime("%Y-%m-%d")
        }
        # Keep track of parent lineages when multiple are merged
        self.parent_lineages = []
    
    def add_source(self, path: Union[str, Path], metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a data source to the lineage.
        
        Args:
            path: Path to the data source file
            metadata: Additional metadata about this source
        """
        source_path = Path(path)
        source_entry = {
            "path": str(source_path.absolute()),
            "filename": source_path.name,
            "size_bytes": source_path.stat().st_size if source_path.exists() else None,
            "last_modified": datetime.fromtimestamp(source_path.stat().st_mtime).isoformat() if source_path.exists() else None,
            "added_at": datetime.now().isoformat(),
            **(metadata or {})
        }
        self.sources.append(source_entry)
        logger.debug(f"Added source to lineage: {source_path.name}")
    
    def add_step(self, name: str, description: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> 'LineageTracker':
        """
        Add a processing step to the lineage.
        
        Args:
            name: Name of the processing step
            description: Description of what the step does
            metadata: Additional metadata about this step
            
        Returns:
            Self for method chaining
        """
        self.steps.append({
            "name": name,
            "description": description or name,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        })
        logger.debug(f"Added processing step to lineage: {name}")
        return self
    
    def get_id(self) -> str:
        """
        Get the unique identifier for this lineage tracker.
        
        Returns:
            The lineage ID string
        """
        return self.lineage_id
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert lineage information to a dictionary.
        
        Returns:
            Dictionary representation of lineage information
        """
        return {
            "lineage_id": self.lineage_id,
            "name": self.name,
            "creation_time": self.creation_time,
            "sources": self.sources,
            "steps": self.steps,
            "config": self.config,
            "metadata": self.metadata,
            "parent_lineages": self.parent_lineages
        }
    
    def to_json(self, indent: int = 2) -> str:
        """
        Convert lineage information to JSON.
        
        Args:
            indent: Indentation level for JSON formatting
            
        Returns:
            JSON string representation of lineage information
        """
        return json.dumps(self.to_dict(), indent=indent)
    
    def save_to_file(self, path: Union[str, Path]) -> None:
        """
        Save lineage information to a file.
        
        Args:
            path: Path to save the lineage information to
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use set for extension checking
        json_extensions = {'.json'}
        yaml_extensions = {'.yaml', '.yml'}
        
        if output_path.suffix.lower() in json_extensions:
            with open(output_path, 'w') as f:
                f.write(self.to_json())
        elif output_path.suffix.lower() in yaml_extensions:
            with open(output_path, 'w') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
        else:
            # Default to JSON if extension is not recognized
            with open(output_path, 'w') as f:
                f.write(self.to_json())
                
        logger.info(f"Saved lineage information to {output_path}")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LineageTracker':
        """
        Create a LineageTracker from a dictionary.
        
        Args:
            data: Dictionary containing lineage information
            
        Returns:
            LineageTracker instance with loaded data
        """
        tracker = cls()
        tracker.lineage_id = data.get("lineage_id", str(uuid.uuid4()))
        tracker.name = data.get("name", f"lineage-{tracker.lineage_id[:8]}")
        tracker.creation_time = data.get("creation_time", datetime.now().isoformat())
        tracker.sources = data.get("sources", [])
        tracker.steps = data.get("steps", [])
        tracker.config = data.get("config", {})
        tracker.metadata = data.get("metadata", {})
        tracker.parent_lineages = data.get("parent_lineages", [])
        return tracker
    
    @classmethod
    def load_from_file(cls, path: Union[str, Path]) -> 'LineageTracker':
        """
        Load lineage information from a file.
        
        Args:
            path: Path to load the lineage information from
            
        Returns:
            LineageTracker instance with loaded data
        """
        input_path = Path(path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Lineage file {input_path} not found")
        
        # Use set for extension checking
        json_extensions = {'.json'}
        yaml_extensions = {'.yaml', '.yml'}
        
        # Use named expression to simplify conditional
        if input_path.suffix.lower() in json_extensions:
            with open(input_path, 'r') as f:
                data = json.load(f)
        elif input_path.suffix.lower() in yaml_extensions:
            with open(input_path, 'r') as f:
                data = yaml.safe_load(f)
        else:
            # Default to JSON if extension is not recognized, but warn the user
            logger.warning(
                f"Unrecognized file extension '{input_path.suffix}' for lineage file '{input_path}'. "
                f"Attempting to load as JSON. Supported extensions: {json_extensions | yaml_extensions}"
            )
            try:
                with open(input_path, 'r') as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to load '{input_path}' as JSON. Consider using a standard extension (.json, .yaml, .yml): {e}")
                
        return cls.from_dict(data)
    
    @classmethod
    def merge_lineages(cls, lineages: List["LineageTracker"], 
                      name: Optional[str] = None, 
                      description: str = "Merged lineages") -> "LineageTracker":
        """
        Merge multiple lineage trackers into a new composite tracker.
        
        This is useful when multiple DataFrames with their own lineage are
        concatenated or merged into a single DataFrame.
        
        Args:
            lineages: List of LineageTracker instances to merge
            name: Optional name for the new merged lineage
            description: Description of the merge operation
            
        Returns:
            A new LineageTracker that contains the combined history
        """
        if not lineages:
            return cls(name=name or "empty-merged-lineage")
            
        # Create a new lineage with a merged name if not provided
        merged_name = name or f"merged-{len(lineages)}-lineages"
        merged = cls(name=merged_name)
        
        # Keep track of parent lineages
        merged.parent_lineages = [lineage.lineage_id for lineage in lineages]
        
        # Add a merge step
        merged.add_step(
            "merge_lineages", 
            description, 
            {"parent_count": len(lineages), 
             "parent_names": [lineage.name for lineage in lineages]}
        )
        
        # Combine all sources from all parent lineages
        for lineage in lineages:
            for source in lineage.sources:
                if source not in merged.sources:
                    merged.sources.append(source)
        
        # Log the merge operation
        logger.info(f"Merged {len(lineages)} lineages into {merged.name}")
        
        return merged


def attach_lineage_to_dataframe(df: pd.DataFrame, lineage: LineageTracker) -> pd.DataFrame:
    """
    Attach lineage information to a DataFrame as attributes.
    
    Args:
        df: DataFrame to attach lineage information to
        lineage: LineageTracker containing lineage information
        
    Returns:
        DataFrame with lineage information attached
    """
    # Make a shallow copy to avoid modifying the original
    df_copy = df.copy(deep=False)
    
    # If the DataFrame already has a lineage tracker, check how to handle it
    existing_lineage = get_lineage_from_dataframe(df_copy)
    
    if existing_lineage is None:
        # First time attaching lineage
        df_copy.attrs['lineage'] = lineage
        df_copy.attrs['_lineage_ids'] = [lineage.lineage_id]
        df_copy.attrs['_lineages'] = {lineage.lineage_id: lineage.to_dict()}
    elif isinstance(df_copy.attrs.get('_lineage_ids'), list):
        # We're already tracking multiple lineages, add this one
        lineage_ids = df_copy.attrs['_lineage_ids']
        if lineage.lineage_id not in lineage_ids:
            lineage_ids.append(lineage.lineage_id)
            df_copy.attrs['_lineages'][lineage.lineage_id] = lineage.to_dict()
    else:
        # We have a single lineage, convert to the multi-lineage format
        existing_id = existing_lineage.lineage_id
        df_copy.attrs['_lineage_ids'] = [existing_id, lineage.lineage_id]
        df_copy.attrs['_lineages'] = {
            existing_id: existing_lineage.to_dict(),
            lineage.lineage_id: lineage.to_dict()
        }
        # Keep the main lineage reference for backward compatibility
        df_copy.attrs['lineage'] = lineage
    
    logger.debug(f"Attached lineage {lineage.name} to DataFrame")
    return df_copy


def get_lineage_from_dataframe(df: pd.DataFrame) -> Optional[LineageTracker]:
    """
    Get lineage information from a DataFrame.
    
    Args:
        df: DataFrame to get lineage information from
        
    Returns:
        LineageTracker containing lineage information, or None if not found
    """
    if 'lineage' not in df.attrs:
        return None
    
    if isinstance(df.attrs['lineage'], LineageTracker):
        return df.attrs['lineage']
    
    return LineageTracker.from_dict(df.attrs['lineage'])


def get_all_lineages_from_dataframe(df: pd.DataFrame) -> List[LineageTracker]:
    """
    Get all lineage trackers associated with a DataFrame.
    
    This is useful for DataFrames that have been created by concatenating
    multiple DataFrames with different lineage trackers.
    
    Args:
        df: DataFrame to get lineage information from
        
    Returns:
        List of LineageTracker objects, empty list if none found
    """
    # Check if we have multi-lineage tracking
    if '_lineages' in df.attrs and '_lineage_ids' in df.attrs:
        lineages = []
        for lineage_id in df.attrs['_lineage_ids']:
            if lineage_id in df.attrs['_lineages']:
                lineage_dict = df.attrs['_lineages'][lineage_id]
                lineages.append(LineageTracker.from_dict(lineage_dict))
        return lineages
    
    # Fall back to single lineage for backward compatibility
    lineage = get_lineage_from_dataframe(df)
    return [lineage] if lineage else []


def merge_dataframe_lineages(df: pd.DataFrame, name: Optional[str] = None) -> pd.DataFrame:
    """
    Merge all lineages in a DataFrame into a single unified lineage.
    
    This is useful when a DataFrame has multiple lineages (e.g., from 
    concatenating multiple DataFrames) and you want to create a unified
    lineage that represents the combined history.
    
    Args:
        df: DataFrame with multiple lineages
        name: Optional name for the merged lineage
        
    Returns:
        DataFrame with a single merged lineage
    """
    lineages = get_all_lineages_from_dataframe(df)
    
    if not lineages:
        logger.warning("No lineages found in DataFrame to merge")
        return df
    
    if len(lineages) == 1:
        logger.info("Only one lineage found, no merging needed")
        return df
    
    # Create merged lineage
    merged_lineage = LineageTracker.merge_lineages(
        lineages, 
        name=name or f"merged-{len(lineages)}-dataframe-lineages",
        description=f"Merged {len(lineages)} lineages from dataframe concatenation"
    )
    
    # Make a shallow copy to avoid modifying the original
    df_copy = df.copy(deep=False)
    
    # Replace the existing lineage information
    df_copy.attrs['lineage'] = merged_lineage
    df_copy.attrs['_lineage_ids'] = [merged_lineage.lineage_id]
    df_copy.attrs['_lineages'] = {merged_lineage.lineage_id: merged_lineage.to_dict()}
    
    logger.info(f"Merged {len(lineages)} lineages into a single unified lineage")
    return df_copy


def export_dataframe_with_lineage(
    df: pd.DataFrame, 
    data_path: Union[str, Path], 
    lineage_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Export a DataFrame along with its lineage information.
    
    Args:
        df: DataFrame to export
        data_path: Path to save the DataFrame to
        lineage_path: Optional path to save the lineage information separately
                     If None, lineage is saved in the same directory with a _lineage suffix
    """
    # Save the DataFrame
    output_path = Path(data_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine the appropriate storage format
    if output_path.suffix.lower() == '.csv':
        df.to_csv(output_path, index=False)
    elif output_path.suffix.lower() == '.parquet':
        df.to_parquet(output_path, index=False)
    elif output_path.suffix.lower() == '.pkl':
        df.to_pickle(output_path)
    else:
        # Default to pickle
        df.to_pickle(output_path)

    logger.info(f"Saved DataFrame to {output_path}")

    if lineage := get_lineage_from_dataframe(df):
        if lineage_path is None:
            # Create a default lineage path next to the data file
            stem = output_path.stem
            lineage_path = output_path.with_name(f"{stem}_lineage.json")

        lineage.save_to_file(lineage_path)