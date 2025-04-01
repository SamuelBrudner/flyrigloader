"""
storage.py - Abstract storage interfaces and implementations for lineage tracking.

This module provides a clean abstraction over different storage mechanisms for
lineage data, allowing the tracking system to use different backends without
changing the core logic.

Available storage backends:
- AttributeStorage: Stores lineage in DataFrame attributes (traditional approach)
- RegistryStorage: Stores lineage in a central registry (more robust)
- NullStorage: No-op storage for testing and when lineage is disabled
"""

import pickle
import weakref
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Optional, Union, Any, List, Set, TypeVar, Generic

import pandas as pd
from loguru import logger

from ...core.utils import PathLike, ensure_path

# Type variables for better type hinting
DF = TypeVar('DF', bound=pd.DataFrame)
Lineage = TypeVar('Lineage')

# Constants for attribute storage
ATTR_MAIN_LINEAGE = '__flyrig_lineage'
ATTR_LINEAGE_IDS = '__flyrig_lineage_ids'
ATTR_LINEAGES_DICT = '__flyrig_lineages_dict'
ATTR_MINIMAL_FLAG = '__flyrig_minimal_lineage'

# Registry constants
DF_LINEAGE_ID_ATTR = '_flyrig_lineage_id'


class LineageStorage(Generic[DF, Lineage], ABC):
    """
    Abstract base class for lineage storage backends.
    
    This defines the interface for storing and retrieving lineage information
    associated with DataFrames, regardless of the actual storage mechanism.
    """
    
    @abstractmethod
    def store(self, df: DF, lineage: Lineage) -> DF:
        """
        Store lineage information for a DataFrame.
        
        Args:
            df: DataFrame to store lineage for
            lineage: Lineage information to store
            
        Returns:
            DataFrame with storage mechanism applied (may be a copy)
            
        Raises:
            ValueError: If storage fails
        """
        pass
    
    @abstractmethod
    def retrieve(self, df: DF) -> Optional[Lineage]:
        """
        Retrieve lineage information for a DataFrame.
        
        Args:
            df: DataFrame to retrieve lineage for
            
        Returns:
            Lineage information if found, None otherwise
            
        Raises:
            ValueError: If retrieval fails (not including "not found" cases)
        """
        pass
    
    @abstractmethod
    def retrieve_all(self, df: DF) -> List[Lineage]:
        """
        Retrieve all lineage information for a DataFrame.
        
        This is useful for DataFrames that have been created by combining
        multiple DataFrames, each with its own lineage.
        
        Args:
            df: DataFrame to retrieve lineage for
            
        Returns:
            List of lineage objects, empty list if none found
            
        Raises:
            ValueError: If retrieval fails
        """
        pass
    
    @abstractmethod
    def remove(self, df: DF) -> bool:
        """
        Remove lineage information for a DataFrame.
        
        Args:
            df: DataFrame to remove lineage for
            
        Returns:
            True if removed successfully, False if not found
            
        Raises:
            ValueError: If removal fails
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """
        Clear all stored lineage information.
        
        Raises:
            ValueError: If clearing fails
        """
        pass


class AttributeStorage(LineageStorage[pd.DataFrame, Any]):
    """
    Store lineage information in DataFrame attributes.
    
    This is the traditional approach used in earlier versions of the library.
    It stores lineage information directly in DataFrame.attrs with namespaced keys.
    
    Advantages:
    - Lineage travels with the DataFrame
    - Simple implementation
    - No external dependencies
    
    Disadvantages:
    - Lineage can be lost during some DataFrame operations
    - Less memory-efficient for large lineage objects
    """
    
    def store(self, df: pd.DataFrame, lineage: Any) -> pd.DataFrame:
        """Store lineage in DataFrame attributes."""
        try:
            # Create a deep copy to avoid modifying the original
            df_copy = df.copy(deep=True)
            
            # Make a deep copy of the attrs dictionary to avoid modifying shared references
            if hasattr(df_copy, 'attrs'):
                df_copy.attrs = df_copy.attrs.copy()
            else:
                df_copy.attrs = {}
            
            # Initialize the lineage attributes dictionary if not present
            if ATTR_LINEAGES_DICT not in df_copy.attrs:
                df_copy.attrs[ATTR_LINEAGES_DICT] = {}
            else:
                # Make a copy of the existing lineages dictionary
                df_copy.attrs[ATTR_LINEAGES_DICT] = df_copy.attrs[ATTR_LINEAGES_DICT].copy()
                
            # Get the lineage ID
            lineage_id = getattr(lineage, "lineage_id", id(lineage))
            
            # Store the lineage in the dictionary
            df_copy.attrs[ATTR_LINEAGES_DICT][lineage_id] = lineage
            
            # Update or initialize the list of lineage IDs
            if ATTR_LINEAGE_IDS not in df_copy.attrs:
                df_copy.attrs[ATTR_LINEAGE_IDS] = []
            else:
                # Make a copy of the existing lineage IDs list
                df_copy.attrs[ATTR_LINEAGE_IDS] = df_copy.attrs[ATTR_LINEAGE_IDS].copy()
                
            if lineage_id not in df_copy.attrs[ATTR_LINEAGE_IDS]:
                df_copy.attrs[ATTR_LINEAGE_IDS].append(lineage_id)
                
            # Set the main lineage
            df_copy.attrs[ATTR_MAIN_LINEAGE] = lineage
            
            return df_copy
            
        except Exception as e:
            raise ValueError(f"Failed to store lineage in DataFrame attributes: {str(e)}")
    
    def retrieve(self, df: pd.DataFrame) -> Optional[Any]:
        """Retrieve the main lineage from DataFrame attributes."""
        try:
            return df.attrs.get(ATTR_MAIN_LINEAGE)
        except Exception as e:
            raise ValueError(f"Failed to retrieve lineage from DataFrame attributes: {str(e)}")
    
    def retrieve_all(self, df: pd.DataFrame) -> List[Any]:
        """Retrieve all lineages from DataFrame attributes."""
        try:
            if ATTR_LINEAGES_DICT not in df.attrs or ATTR_LINEAGE_IDS not in df.attrs:
                return []
            
            return [df.attrs[ATTR_LINEAGES_DICT][lineage_id] 
                    for lineage_id in df.attrs[ATTR_LINEAGE_IDS] 
                    if lineage_id in df.attrs[ATTR_LINEAGES_DICT]]
        except Exception as e:
            raise ValueError(f"Failed to retrieve all lineages from DataFrame attributes: {str(e)}")
    
    def remove(self, df: pd.DataFrame) -> bool:
        """Remove lineage from DataFrame attributes."""
        try:
            # Check if the DataFrame has lineage information
            if ATTR_MAIN_LINEAGE not in df.attrs:
                return False
                
            # Remove all lineage attributes
            for attr in [ATTR_MAIN_LINEAGE, ATTR_LINEAGE_IDS, ATTR_LINEAGES_DICT]:
                if attr in df.attrs:
                    del df.attrs[attr]
                    
            return True
        except Exception as e:
            raise ValueError(f"Failed to remove lineage from DataFrame attributes: {str(e)}")
    
    def clear(self) -> None:
        """No-op for attribute storage as there's no central store."""
        # Nothing to do, as attribute storage has no central state
        pass


class RegistryStorage(LineageStorage[pd.DataFrame, Any]):
    """
    Store lineage information in a central registry.
    
    This approach maintains a mapping between DataFrame identifiers and
    their associated lineage information in a central store.
    
    Advantages:
    - Robustness: Lineage won't be lost during DataFrame transformations
    - Memory efficiency: Lineage is stored once, not duplicated
    - Centralized management: Easy to query all lineage information
    - Persistence options: Both in-memory and on-disk storage
    
    Disadvantages:
    - Requires more setup and management
    - DataFrame must be tracked by ID
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get the singleton instance of the registry."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """Initialize a new registry."""
        # Dictionary mapping DataFrame IDs to lineage objects
        self._registry = {}
        
        # Dictionary mapping lineage IDs to sets of DataFrame IDs
        self._lineage_to_df_map = {}
        
        # WeakRef dictionary to track DataFrames
        self._df_refs = {}
        
        # Registry metadata
        self._metadata = {
            "created_at": datetime.now().isoformat(),
            "lineage_count": 0,
            "dataframe_count": 0
        }
    
    def _get_df_id(self, df: pd.DataFrame) -> str:
        """
        Get a unique ID for a DataFrame.
        
        Args:
            df: DataFrame to get ID for
            
        Returns:
            DataFrame ID
        """
        # Check if the DataFrame already has an ID
        if DF_LINEAGE_ID_ATTR in df.attrs:
            return df.attrs[DF_LINEAGE_ID_ATTR]
        
        # Generate a new ID
        return f"df_{id(df)}"
    
    def _set_df_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Set a unique ID for a DataFrame.
        
        Args:
            df: DataFrame to set ID for
            
        Returns:
            DataFrame with ID attribute set
        """
        df_copy = df.copy(deep=False)
        df_id = self._get_df_id(df)
        df_copy.attrs[DF_LINEAGE_ID_ATTR] = df_id
        
        # Store a weak reference to the DataFrame
        self._df_refs[df_id] = weakref.ref(df_copy)
        
        return df_copy
    
    def _get_lineage_id(self, lineage: Any) -> Any:
        """Get a consistent ID for a lineage object."""
        return getattr(lineage, "lineage_id", id(lineage))

    def store(self, df: pd.DataFrame, lineage: Any) -> pd.DataFrame:
        """Store lineage in the registry."""
        try:
            # Set ID on DataFrame if not already present
            df_with_id = self._set_df_id(df)
            df_id = df_with_id.attrs[DF_LINEAGE_ID_ATTR]
            
            # Get lineage ID
            lineage_id = self._get_lineage_id(lineage)
            
            # Store lineage in registry
            self._registry[df_id] = lineage
            
            # Update lineage to DataFrame mapping
            if lineage_id not in self._lineage_to_df_map:
                self._lineage_to_df_map[lineage_id] = set()
            self._lineage_to_df_map[lineage_id].add(df_id)
            
            # Update metadata
            self._metadata["lineage_count"] = len(self._lineage_to_df_map)
            self._metadata["dataframe_count"] = len(self._registry)
            
            return df_with_id
        except Exception as e:
            raise ValueError(f"Failed to store lineage in registry: {str(e)}")
    
    def retrieve(self, df: pd.DataFrame) -> Optional[Any]:
        """Retrieve lineage from the registry."""
        try:
            df_id = df.attrs.get(DF_LINEAGE_ID_ATTR)
            return self._registry.get(df_id)
        except Exception as e:
            raise ValueError(f"Failed to retrieve lineage from registry: {str(e)}")
    
    def retrieve_all(self, df: pd.DataFrame) -> List[Any]:
        """Retrieve all lineages from the registry for this DataFrame."""
        try:
            df_id = df.attrs.get(DF_LINEAGE_ID_ATTR)
            if df_id is None or df_id not in self._registry:
                return []
            
            return [self._registry[df_id]]
        except Exception as e:
            raise ValueError(f"Failed to retrieve all lineages from registry: {str(e)}")
    
    def remove(self, df: pd.DataFrame) -> bool:
        """Remove lineage from the registry."""
        try:
            df_id = df.attrs.get(DF_LINEAGE_ID_ATTR)
            if df_id is None or df_id not in self._registry:
                return False
                
            # Get the lineage before removal
            lineage = self._registry[df_id]
            lineage_id = self._get_lineage_id(lineage)
            
            # Remove from registry
            del self._registry[df_id]
            
            # Update lineage to DataFrame mapping
            if lineage_id in self._lineage_to_df_map:
                self._lineage_to_df_map[lineage_id].discard(df_id)
                if not self._lineage_to_df_map[lineage_id]:
                    del self._lineage_to_df_map[lineage_id]
            
            # Remove weak reference
            if df_id in self._df_refs:
                del self._df_refs[df_id]
                
            # Update metadata
            self._metadata["lineage_count"] = len(self._lineage_to_df_map)
            self._metadata["dataframe_count"] = len(self._registry)
            
            return True
        except Exception as e:
            raise ValueError(f"Failed to remove lineage from registry: {str(e)}")
    
    def clear(self) -> None:
        """Clear the registry."""
        try:
            self._registry.clear()
            self._lineage_to_df_map.clear()
            self._df_refs.clear()
            
            # Update metadata
            self._metadata["lineage_count"] = 0
            self._metadata["dataframe_count"] = 0
        except Exception as e:
            raise ValueError(f"Failed to clear registry: {str(e)}")
    
    def save(self, path: PathLike) -> None:
        """
        Save the registry to disk.
        
        Args:
            path: Path to save the registry to
            
        Raises:
            TypeError: If path is not a valid path type
            FileNotFoundError: If the parent directory does not exist
            PermissionError: If there are permission issues with the path
            pickle.PickleError: If serialization fails
            OSError: If there are file system errors
            ValueError: If saving fails for other reasons
        """
        try:
            file_path = ensure_path(path)
            
            # Ensure parent directory exists
            if not file_path.parent.exists():
                logger.debug(f"Creating parent directories for registry save: {file_path.parent}")
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
            logger.debug(f"Saving registry with {len(self._registry)} dataframes to {file_path}")
            
            # Prepare data for serialization
            save_data = {
                'registry': self._registry,
                'lineage_to_df_map': self._lineage_to_df_map,
                'metadata': {
                    **self._metadata,
                    'last_saved': datetime.now().isoformat()
                }
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(save_data, f)
                
            logger.debug(f"Successfully saved registry to {file_path}")
            
        except TypeError as e:
            logger.error(f"Invalid path type for registry save: {e}")
            raise TypeError(f"Invalid path type for registry save: {str(e)}") from e
        except FileNotFoundError as e:
            logger.error(f"Directory not found for registry save: {e}")
            raise FileNotFoundError(f"Directory not found for registry save: {str(e)}") from e
        except PermissionError as e:
            logger.error(f"Permission denied when saving registry: {e}")
            raise PermissionError(f"Permission denied when saving registry: {str(e)}") from e
        except pickle.PickleError as e:
            logger.error(f"Failed to serialize registry data: {e}")
            raise pickle.PickleError(f"Failed to serialize registry data: {str(e)}") from e
        except OSError as e:
            logger.error(f"OS error when saving registry to {path}: {e}")
            raise OSError(f"OS error when saving registry: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error saving registry to {path}: {type(e).__name__}: {e}")
            raise ValueError(f"Failed to save registry: {str(e)}") from e
    
    def load(self, path: PathLike) -> None:
        """
        Load a registry from disk.
        
        Args:
            path: Path to load the registry from
            
        Raises:
            TypeError: If path is not a valid path type
            FileNotFoundError: If the file does not exist
            PermissionError: If there are permission issues with the path
            pickle.UnpicklingError: If deserialization fails
            OSError: If there are file system errors
            ValueError: If loading fails for other reasons or data is invalid
        """
        try:
            file_path = ensure_path(path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"Registry file not found: {file_path}")
                
            if not file_path.is_file():
                raise ValueError(f"Registry path is not a file: {file_path}")
                
            logger.debug(f"Loading registry from {file_path}")
            
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                
            # Validate the data structure
            if not isinstance(data, dict):
                raise ValueError(f"Invalid registry data format: expected dict, got {type(data).__name__}")
                
            # Extract and validate components
            if 'registry' not in data:
                raise ValueError("Invalid registry data: missing 'registry' key")
                
            if 'lineage_to_df_map' not in data:
                raise ValueError("Invalid registry data: missing 'lineage_to_df_map' key")
                
            # Update the registry
            self._registry = data.get('registry', {})
            self._lineage_to_df_map = data.get('lineage_to_df_map', {})
            
            # Update metadata, ensuring required fields are present
            base_metadata = {
                "created_at": datetime.now().isoformat(),
                "lineage_count": len(self._lineage_to_df_map),
                "dataframe_count": len(self._registry),
                "loaded_at": datetime.now().isoformat()
            }
            self._metadata = {**base_metadata, **data.get('metadata', {})}
            
            # Clear and recreate weak references (not saved in file)
            self._df_refs.clear()
            
            logger.debug(f"Successfully loaded registry with {len(self._registry)} dataframes")
            
        except TypeError as e:
            logger.error(f"Invalid path type for registry load: {e}")
            raise TypeError(f"Invalid path type for registry load: {str(e)}") from e
        except FileNotFoundError as e:
            logger.error(f"Registry file not found: {e}")
            raise FileNotFoundError(f"Registry file not found: {str(e)}") from e
        except PermissionError as e:
            logger.error(f"Permission denied when loading registry: {e}")
            raise PermissionError(f"Permission denied when loading registry: {str(e)}") from e
        except pickle.UnpicklingError as e:
            logger.error(f"Failed to deserialize registry data: {e}")
            raise pickle.UnpicklingError(f"Failed to deserialize registry data: {str(e)}") from e
        except OSError as e:
            logger.error(f"OS error when loading registry from {path}: {e}")
            raise OSError(f"OS error when loading registry: {str(e)}") from e
        except (ValueError, KeyError) as e:
            logger.error(f"Invalid registry data structure: {e}")
            raise ValueError(f"Invalid registry data structure: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error loading registry from {path}: {type(e).__name__}: {e}")
            raise ValueError(f"Failed to load registry: {str(e)}") from e


class NullStorage(LineageStorage[pd.DataFrame, Any]):
    """
    A no-op storage implementation.
    
    This is useful for testing or when lineage tracking is disabled.
    """
    
    def store(self, df: pd.DataFrame, lineage: Any) -> pd.DataFrame:
        """No-op store."""
        return df.copy(deep=False)
    
    def retrieve(self, df: pd.DataFrame) -> Optional[Any]:
        """No-op retrieve."""
        return None
    
    def retrieve_all(self, df: pd.DataFrame) -> List[Any]:
        """No-op retrieve all."""
        return []
    
    def remove(self, df: pd.DataFrame) -> bool:
        """No-op remove."""
        return False
    
    def clear(self) -> None:
        """No-op clear."""
        pass


def create_storage(storage_type: str = "attribute") -> LineageStorage:
    """
    Create a lineage storage backend.
    
    Args:
        storage_type: Type of storage to create ("attribute", "registry", "null")
        
    Returns:
        LineageStorage implementation
        
    Raises:
        ValueError: If storage_type is invalid
    """
    storage_map = {
        "attribute": AttributeStorage,
        "registry": lambda: RegistryStorage.get_instance(),
        "null": NullStorage
    }
    
    if storage_type not in storage_map:
        raise ValueError(f"Invalid storage type: {storage_type}. Valid options: {', '.join(storage_map.keys())}")
    
    storage_factory = storage_map[storage_type]
    return storage_factory()
