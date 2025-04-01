"""
tracker.py - Core implementation of lineage tracking functionality.

This module provides the fundamental LineageRecord class which stores
lineage information including sources, processing steps, and metadata.
It is used by higher-level interfaces but can also be used directly
for advanced use cases.
"""

import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Union, TypeVar

import pandas as pd
import yaml
from loguru import logger

from ...core.utils import PathLike, ensure_path


class LineageRecord:
    """
    Core lineage record that tracks data sources, processing steps, and metadata.
    
    This class maintains information about:
    - Input sources (file paths, timestamps, versions)
    - Processing steps applied to the data
    - Configuration parameters used
    - Any metadata needed for complete reproducibility
    
    It serves as the fundamental data structure for lineage tracking,
    used by higher-level interfaces like LineageTracker.
    """
    
    def __init__(
        self, 
        name: Optional[str] = None, 
        config: Optional[Dict[str, Any]] = None,
        lineage_id: Optional[str] = None
    ):
        """
        Initialize a new LineageRecord.
        
        Args:
            name: Optional name for this lineage record
            config: Optional configuration dictionary used for this pipeline run
            lineage_id: Optional unique identifier for this lineage record
        """
        try:
            self.lineage_id = lineage_id or str(uuid.uuid4())
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
            
            logger.debug(f"Created new LineageRecord: {self.name} ({self.lineage_id})")
        except Exception as e:
            raise ValueError(f"Failed to initialize LineageRecord: {str(e)}")
    
    def add_source(
        self, 
        path: PathLike, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'LineageRecord':
        """
        Add a data source to the lineage.
        
        Args:
            path: Path to the data source file
            metadata: Additional metadata about this source
            
        Returns:
            Self for method chaining
        """
        try:
            # Convert path to string representation
            path_str = str(ensure_path(path))
        
            # Create the source entry
            source_entry = {
                "path": path_str,
                "added_at": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
        
            # Add to sources list
            self.sources.append(source_entry)
            logger.debug(f"Added source to lineage {self.lineage_id}: {path_str}")
        
            return self
        except Exception as e:
            raise ValueError(f"Failed to add source to lineage: {str(e)}")
    
    def add_step(
        self, 
        name: str, 
        description: Optional[str] = None, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'LineageRecord':
        """
        Add a processing step to the lineage.
        
        Args:
            name: Name of the processing step
            description: Description of what the step does
            metadata: Additional metadata about this step
            
        Returns:
            Self for method chaining
        """
        try:
            if not name:
                raise ValueError("Step name cannot be empty")
        
            # Create the step entry
            step_entry = {
                "name": name,
                "description": description or "",
                "added_at": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
        
            # Add to steps list
            self.steps.append(step_entry)
            logger.debug(f"Added step to lineage {self.lineage_id}: {name}")
        
            return self
        except Exception as e:
            raise ValueError(f"Failed to add step to lineage: {str(e)}")
    
    def get_id(self) -> str:
        """
        Get the unique identifier for this lineage record.
        
        Returns:
            The lineage ID string
        """
        return self.lineage_id
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert lineage information to a dictionary.
        
        Returns:
            Dictionary representation of the lineage
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
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Optional['LineageRecord']:
        """
        Create a LineageRecord from a dictionary.
        
        Args:
            data: Dictionary containing lineage information
            
        Returns:
            New LineageRecord instance, or None if creation fails
        """
        try:
            if not isinstance(data, dict):
                logger.error(f"Expected dict, got {type(data)}")
                return None
        
            # Create a new instance with the basic attributes
            record = cls(
                name=data.get("name"),
                config=data.get("config", {}),
                lineage_id=data.get("lineage_id")
            )
        
            # Set the remaining attributes
            record.creation_time = data.get("creation_time", record.creation_time)
            record.sources = data.get("sources", [])
            record.steps = data.get("steps", [])
            record.metadata = data.get("metadata", {})
            record.parent_lineages = data.get("parent_lineages", [])
        
            return record
        except Exception as e:
            logger.error(f"Failed to create lineage from dict: {str(e)}")
            return None
    
    def to_json(self) -> str:
        """
        Convert lineage information to a JSON string.
        
        Returns:
            JSON string representation of the lineage
            
        Raises:
            TypeError: If lineage data contains types that cannot be serialized to JSON
            ValueError: If lineage data conversion fails
        """
        try:
            # Get the dictionary representation first
            data_dict = self.to_dict()
            
            # Convert to JSON string
            try:
                json_str = json.dumps(data_dict, indent=2)
                return json_str
            except TypeError as e:
                logger.error(f"Cannot serialize lineage data to JSON due to invalid types: {e}")
                raise TypeError(f"Cannot serialize lineage to JSON: {str(e)}") from e
        except Exception as e:
            logger.error(f"Failed to convert lineage to JSON: {type(e).__name__}: {e}")
            raise ValueError(f"Failed to convert lineage to JSON: {str(e)}") from e
    
    @classmethod
    def from_json(cls, json_str: str) -> Optional['LineageRecord']:
        """
        Create a LineageRecord from a JSON string.
        
        Args:
            json_str: JSON string containing lineage information
            
        Returns:
            New LineageRecord instance, or None if creation fails
            
        Raises:
            TypeError: If json_str is not a string
            json.JSONDecodeError: If JSON parsing fails
            ValueError: If the data cannot be converted to a LineageRecord
        """
        if not isinstance(json_str, str):
            logger.error(f"Expected string, got {type(json_str).__name__}")
            raise TypeError(f"Expected string, got {type(json_str).__name__}")
            
        try:
            # Parse JSON string to dictionary
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON format: {e}")
                raise json.JSONDecodeError(f"Invalid JSON format: {str(e)}", e.doc, e.pos) from e
                
            # Validate data structure
            if not isinstance(data, dict):
                error_msg = f"Invalid lineage data format: expected dict, got {type(data).__name__}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Create LineageRecord from dictionary
            record = cls.from_dict(data)
            if record is None:
                logger.error("Failed to create LineageRecord from JSON data")
                return None
                
            return record
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            # Re-raise specific exceptions for better error handling
            raise
        except Exception as e:
            logger.error(f"Unexpected error creating lineage from JSON: {type(e).__name__}: {e}")
            return None
    
    def save_to_file(self, path: PathLike, format: str = "json") -> None:
        """
        Save lineage information to a file.
        
        Args:
            path: Path to save the lineage to
            format: Format to save in ("json" or "yaml")
            
        Raises:
            TypeError: If path is not a valid path type
            ValueError: If the format is not supported or data cannot be serialized
            FileNotFoundError: If the parent directory does not exist
            PermissionError: If there are permission issues with the path
            OSError: If there are file system errors
        """
        try:
            # Validate format
            format = format.lower()
            if format not in ["json", "yaml"]:
                raise ValueError(f"Unsupported format: {format}")
                
            # Ensure path is valid
            try:
                file_path = ensure_path(path)
            except (TypeError, ValueError) as e:
                logger.error(f"Invalid path for saving lineage: {e}")
                raise TypeError(f"Invalid path for saving lineage: {str(e)}") from e
                
            # Create parent directories if they don't exist
            try:
                if not file_path.parent.exists():
                    logger.debug(f"Creating parent directories for lineage file: {file_path.parent}")
                    file_path.parent.mkdir(parents=True, exist_ok=True)
            except PermissionError as e:
                logger.error(f"Permission denied when creating directories: {e}")
                raise PermissionError(f"Permission denied when creating directories: {str(e)}") from e
            except OSError as e:
                logger.error(f"OS error when creating directories: {e}")
                raise OSError(f"Failed to create directories: {str(e)}") from e
                
            # Convert lineage to dictionary
            data = self.to_dict()
            
            # Save the file in the specified format
            try:
                with open(file_path, 'w') as f:
                    if format == "json":
                        json.dump(data, f, indent=2)
                    elif format == "yaml":
                        yaml.dump(data, f, default_flow_style=False)
                
                logger.debug(f"Successfully saved lineage {self.lineage_id} to {file_path}")
            except PermissionError as e:
                logger.error(f"Permission denied when writing to {file_path}: {e}")
                raise PermissionError(f"Cannot write to file: {str(e)}") from e
            except (IOError, OSError) as e:
                logger.error(f"I/O error when writing to {file_path}: {e}")
                raise OSError(f"Failed to write to file: {str(e)}") from e
            except (TypeError, ValueError, json.JSONDecodeError) as e:
                logger.error(f"Serialization error when saving lineage to {format} format: {e}")
                raise ValueError(f"Failed to serialize lineage data: {str(e)}") from e
                
        except Exception as e:
            # Catch truly unexpected errors
            logger.error(f"Unexpected error saving lineage to file: {type(e).__name__}: {e}")
            raise ValueError(f"Failed to save lineage to file: {str(e)}") from e
    
    @classmethod
    def load_from_file(cls, path: PathLike) -> Optional['LineageRecord']:
        """
        Load lineage information from a file.
        
        Args:
            path: Path to load the lineage from
            
        Returns:
            New LineageRecord instance, or None if loading fails
            
        Raises:
            TypeError: If path is not a valid path type
            FileNotFoundError: If the file does not exist
            PermissionError: If there are permission issues with the path
            ValueError: If the file format is invalid or data cannot be parsed
            OSError: If there are file system errors
        """
        try:
            # Ensure path is valid
            try:
                file_path = ensure_path(path)
            except (TypeError, ValueError) as e:
                logger.error(f"Invalid path for loading lineage: {e}")
                raise TypeError(f"Invalid path for loading lineage: {str(e)}") from e
            
            # Check file existence
            if not file_path.exists():
                logger.error(f"Lineage file not found: {file_path}")
                raise FileNotFoundError(f"Lineage file not found: {file_path}")
                
            if not file_path.is_file():
                logger.error(f"Path is not a file: {file_path}")
                raise ValueError(f"Path is not a file: {file_path}")
                
            logger.debug(f"Loading lineage from {file_path}")
            
            # Determine format from file extension
            is_yaml = str(file_path).lower().endswith(('.yaml', '.yml'))
            
            # Read and parse the file
            try:
                with open(file_path, 'r') as f:
                    if is_yaml:
                        try:
                            data = yaml.safe_load(f)
                        except yaml.YAMLError as e:
                            logger.error(f"Failed to parse YAML file {file_path}: {e}")
                            raise ValueError(f"Invalid YAML format: {str(e)}") from e
                    else:
                        try:
                            data = json.load(f)
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse JSON file {file_path}: {e}")
                            raise ValueError(f"Invalid JSON format: {str(e)}") from e
            except PermissionError as e:
                logger.error(f"Permission denied when reading {file_path}: {e}")
                raise PermissionError(f"Cannot read file: {str(e)}") from e
            except (IOError, OSError) as e:
                logger.error(f"I/O error when reading {file_path}: {e}")
                raise OSError(f"Failed to read file: {str(e)}") from e
            
            # Validate data structure
            if not isinstance(data, dict):
                error_msg = f"Invalid lineage data format: expected dict, got {type(data).__name__}"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            # Create lineage record from data
            record = cls.from_dict(data)
            if record is None:
                raise ValueError("Failed to create LineageRecord from file data")
                
            logger.debug(f"Successfully loaded lineage {record.lineage_id} from {file_path}")
            return record
            
        except (TypeError, FileNotFoundError, PermissionError, ValueError, OSError) as e:
            # Re-raise these exceptions for specific error handling by caller
            raise
        except Exception as e:
            # For truly unexpected errors, log and return None
            logger.error(f"Unexpected error loading lineage from file: {type(e).__name__}: {e}")
            return None
    
    def merge(
        self, 
        other: 'LineageRecord', 
        preserve_history: bool = True
    ) -> 'LineageRecord':
        """
        Merge another LineageRecord into this one.
        
        Args:
            other: Another LineageRecord to merge
            preserve_history: Whether to keep parent lineage references
            
        Returns:
            Self with merged information
        """
        try:
            if not isinstance(other, LineageRecord):
                raise ValueError(f"Expected LineageRecord, got {type(other)}")
        
            # Merge sources (avoiding duplicates)
            existing_paths = {source["path"] for source in self.sources}
            for source in other.sources:
                if source["path"] not in existing_paths:
                    self.sources.append(source)
                    existing_paths.add(source["path"])
        
            # Merge steps (keeping all, including duplicates)
            self.steps.extend(other.steps)
        
            # Update metadata with new information, preserving existing
            for key, value in other.metadata.items():
                if key not in self.metadata:
                    self.metadata[key] = value
        
            # Track the merged lineage's ID in parent_lineages
            if preserve_history and other.lineage_id not in self.parent_lineages:
                self.parent_lineages.append(other.lineage_id)
                # Also include the other lineage's parents
                for parent_id in other.parent_lineages:
                    if parent_id not in self.parent_lineages:
                        self.parent_lineages.append(parent_id)
        
            logger.debug(f"Merged lineage {other.lineage_id} into {self.lineage_id}")
            return self
        except Exception as e:
            raise ValueError(f"Failed to merge lineages: {str(e)}")


# Factory function to create LineageRecord instances
def create_lineage_record(
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    lineage_id: Optional[str] = None
) -> LineageRecord:
    """
    Create a new LineageRecord.
    
    Args:
        name: Optional name for the lineage record
        config: Optional configuration dictionary
        lineage_id: Optional unique identifier
        
    Returns:
        New LineageRecord instance
    """
    return LineageRecord(name=name, config=config, lineage_id=lineage_id)
