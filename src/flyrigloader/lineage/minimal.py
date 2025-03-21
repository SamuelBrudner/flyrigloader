"""
minimal.py - A simple, OOP-style lineage wrapper for everyday usage.

This class provides:
  1) A single source file reference
  2) Optional step annotations (e.g., "cleaned data", "merged frames")
  3) Minimal methods to attach lineage to a DataFrame
  4) A method to save both DataFrame and lineage in one call

Internally, it uses the advanced 'LineageTracker' so you still have access to
power-user features if needed, but typically you can stay with this simpler interface.

Usage Example:
    from flyrigloader.lineage.minimal import MinimalLineageTracker

    df = pd.read_csv("my_data.csv")
    # Initialize the minimal lineage, referencing the original file
    tracker = MinimalLineageTracker(source="my_data.csv", description="Initial data load")
    
    # Attach lineage attributes to the DataFrame
    df = tracker.attach_to_dataframe(df)
    
    # If you do more transformations, record them as 'steps'
    tracker.note_step("clean_data", "Removed negative rows in column x")
    
    # Save data + lineage in CSV or pickle
    tracker.save_dataframe(df, "processed.csv")
    # This also writes "processed_lineage.json" next to the CSV.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Union, Optional, Dict, Any, List
from loguru import logger

# Import the advanced lineage system
from .tracker import LineageTracker, attach_lineage_to_dataframe, get_lineage_from_dataframe


class NullLineageTracker:
    """
    A "do-nothing" implementation of the LineageTracker interface.
    
    This class implements the same interface as LineageTracker but doesn't
    perform any actual tracking. It's used when lineage tracking is disabled
    to maintain a consistent interface without having to use conditionals
    throughout the codebase.
    
    All methods are no-ops that return appropriate empty values or the input
    parameters unchanged.
    """
    
    def __init__(self, name: str = None, config: Dict[str, Any] = None):
        """
        Initialize a NullLineageTracker (does nothing).
        
        Args:
            name: Ignored
            config: Ignored
        """
        self.lineage_id = "null"
        self.name = "null_lineage_tracker"
        self.creation_time = datetime.now().isoformat()
        self.sources = []
        self.steps = []
        self.config = {}
        self.metadata = {}
    
    def add_source(self, path: Union[str, Path], metadata: Optional[Dict[str, Any]] = None) -> None:
        """No-op implementation of add_source."""
        pass
    
    def add_step(self, name: str, description: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> 'NullLineageTracker':
        """No-op implementation of add_step."""
        return self
    
    def get_id(self) -> str:
        """No-op implementation of get_id."""
        return self.lineage_id
    
    def to_dict(self) -> Dict[str, Any]:
        """No-op implementation of to_dict."""
        return {
            "lineage_id": self.lineage_id,
            "name": self.name,
            "creation_time": self.creation_time,
            "sources": [],
            "steps": [],
            "config": {},
            "metadata": {}
        }
    
    def to_json(self, indent: int = 2) -> str:
        """No-op implementation of to_json."""
        return "{}"
    
    def save_to_file(self, path: Union[str, Path]) -> None:
        """No-op implementation of save_to_file."""
        logger.debug(f"NullLineageTracker: Skipping save_to_file ({path})")


class MinimalLineageTracker:
    """
    An OOP-style minimal lineage tool that uses LineageTracker internally but
    exposes only a small subset of features:
      - A single 'source_path' to note where the DataFrame originated
      - A short textual description
      - Optional step annotations
      - Easy saving of both DataFrame and lineage

    For advanced multi-source or multi-step usage, you can still call
    'LineageTracker' directly. This class aims to be simpler for everyday usage.
    """

    def __init__(self, source: Union[str, Path], description: str = "Data loaded"):
        """
        Initialize a MinimalLineageTracker with one data source.

        Parameters
        ----------
        source : Union[str, Path]
            Path to the raw data file or other data source you loaded.
            Can also be a descriptive string if no actual file path exists.
        description : str
            Short description, e.g. "Initial load" or "Main experiment data".
        """
        # Handle the source path more robustly to avoid errors on non-existent paths
        if isinstance(source, (str, Path)):
            try:
                # First check if this is a valid path
                path_obj = Path(source)
                # Store resolved path if exists, otherwise store as is
                self.source_path = path_obj.resolve() if path_obj.exists() else path_obj
            except (ValueError, TypeError):
                # Handle cases where source is a string but not a valid path
                self.source_path = Path(str(source))
        else:
            # Fallback for any other type
            self.source_path = Path(str(source))
            
        self.description = description
        self.creation_time = datetime.now().isoformat()

        # Create an internal advanced tracker
        self._tracker = LineageTracker(name="MinimalLineage")

        # Register the source
        self._tracker.add_source(
            self.source_path,
            metadata={
                "short_description": self.description,
                "created_at": self.creation_time
            }
        )
        logger.debug(f"MinimalLineageTracker initialized with source {self.source_path}")

    def attach_to_dataframe(self, df: pd.DataFrame, overwrite: bool = True) -> pd.DataFrame:
        """
        Attach this tracker's lineage to a DataFrame as attributes.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to attach lineage to.
        overwrite : bool, default=True
            If True and the DataFrame already has lineage, the old lineage is overwritten.
            If False, the new lineage will be appended to any existing lineage tracking,
            enabling multi-lineage tracking for advanced usage.

        Returns
        -------
        pd.DataFrame
            A new copy of 'df' with lineage info embedded.
        """
        if overwrite:
            return attach_lineage_to_dataframe(df, self._tracker)
            
        # For non-overwrite mode (append mode):
        # Check if we need to create a new copy of the DataFrame
        # (The attach_lineage_to_dataframe function already makes a copy)
        existing_lineage = get_lineage_from_dataframe(df)
        if existing_lineage is None or existing_lineage.lineage_id == self._tracker.lineage_id:
            # No existing lineage or same lineage ID, simple case
            return attach_lineage_to_dataframe(df, self._tracker)
            
        # We have an existing different lineage, and we want to append
        df_copy = df.copy(deep=False)
        return attach_lineage_to_dataframe(df_copy, self._tracker)

    def note_step(self, step_name: str, info: str = "") -> None:
        """
        Record an additional step to note some transformation or analysis.

        Parameters
        ----------
        step_name : str
            A short label for the step (e.g. 'clean_data').
        info : str
            A longer description, e.g. 'Removed outliers below x=0'.
        """
        self._tracker.add_step(step_name, description=info)
        logger.debug(f"Noted step: {step_name} - {info}")

    def save_lineage(self, path: Union[str, Path]) -> None:
        """
        Save just the lineage info to a JSON or YAML file.

        Parameters
        ----------
        path : Union[str, Path]
            Where to write the lineage file (auto-detected extension: .json or .yaml)
        """
        self._tracker.save_to_file(path)
        logger.debug(f"Lineage saved to {path}")

    def get_id(self) -> str:
        """
        Get a unique identifier for this lineage tracker.
        
        Returns
        -------
        str
            The unique ID of the underlying LineageTracker instance.
            This can be used to track and merge lineage information
            when multiple DataFrames are combined.
        """
        return self._tracker.lineage_id

    def save_dataframe(self, df: pd.DataFrame, output_path: Union[str, Path]) -> None:
        """
        Save a DataFrame (which hopefully has this tracker's lineage) to a file,
        plus automatically save the lineage in a .json next to it.

        - If output_path ends with .csv, we do df.to_csv()
        - Otherwise we default to a .pkl file

        The lineage is written to output_path with a '_lineage.json' suffix.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to save (with or without lineage).
        output_path : Union[str, Path]
            Target file path, e.g. 'results/processed.csv' or 'results/processed.pkl'.
        """
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        # Attach lineage if missing
        current_lineage = get_lineage_from_dataframe(df)
        if current_lineage is None:
            # If the user never attached this tracker's lineage, do so now
            df = attach_lineage_to_dataframe(df, self._tracker)

        # Decide on format
        if p.suffix.lower() == ".csv":
            df.to_csv(p, index=False)
            logger.debug(f"Saved DataFrame to CSV at {p}")
        else:
            # If not CSV, default to .pkl
            if p.suffix.lower() != ".pkl":
                logger.warning(f"Unrecognized extension '{p.suffix}'. Using .pkl instead.")
                p = p.with_suffix(".pkl")
            df.to_pickle(p)
            logger.debug(f"Saved DataFrame to pickle at {p}")

        # Save the lineage
        lineage_file = p.with_name(f"{p.stem}_lineage.json")
        self._tracker.save_to_file(lineage_file)
        logger.debug(f"Saved lineage to {lineage_file}")