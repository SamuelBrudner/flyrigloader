"""
Kedro AbstractDataset implementations for FlyRigLoader integration.

This module provides first-class Kedro integration through FlyRigLoaderDataSet and 
FlyRigManifestDataSet classes that implement Kedro's AbstractDataset interface. These 
datasets enable seamless integration with Kedro pipelines by wrapping FlyRigLoader's 
discovery and loading capabilities with proper lifecycle management, error handling, 
and metadata support.

Key Features:
- AbstractDataset compliance with _load, _save, _exists, _describe methods
- Read-only operation model with comprehensive error handling
- Thread-safe operations for Kedro's parallel execution patterns
- Comprehensive parameter injection for experiment-specific configuration
- Kedro-compatible exception propagation and structured logging
- Lazy evaluation with manifest-based discovery optimization
- Configuration file validation and existence checking

Classes:
    FlyRigLoaderDataSet: Full data loading with transformation pipeline
    FlyRigManifestDataSet: Manifest-only operations without full data loading

Usage Example:
    # Direct instantiation
    >>> dataset = FlyRigLoaderDataSet(
    ...     filepath="config/experiment_config.yaml",
    ...     experiment_name="baseline_study"
    ... )
    >>> data = dataset.load()
    
    # Kedro catalog integration
    experiment_data:
      type: flyrigloader.FlyRigLoaderDataSet
      filepath: "${base_dir}/config/experiment_config.yaml"
      experiment_name: "baseline_study"
      recursive: true
"""

import logging
from pathlib import Path
from typing import Union, Any, Dict, Optional
from threading import RLock
from copy import deepcopy

# External imports
from kedro.io import AbstractDataSet
import pandas as pd

# Internal imports  
from flyrigloader.config.models import ExperimentConfig
from flyrigloader.exceptions import ConfigError
from flyrigloader.discovery.files import discover_experiment_manifest
from flyrigloader.io.loaders import load_data_file
from flyrigloader.io.transformers import transform_to_dataframe
from flyrigloader.config.yaml_config import load_config

# Configure module-level logging
logger = logging.getLogger(__name__)

# Thread-safe lock for concurrent operations
_dataset_lock = RLock()


class FlyRigLoaderDataSet(AbstractDataSet[None, pd.DataFrame]):
    """
    Kedro AbstractDataset implementation for FlyRigLoader with full data loading capabilities.
    
    This dataset provides first-class Kedro integration by wrapping FlyRigLoader's complete
    discovery and loading pipeline. It implements Kedro's AbstractDataset interface with
    proper lifecycle management, comprehensive error handling, and thread-safe operations
    for parallel execution patterns.
    
    The dataset operates in read-only mode, returning pandas DataFrames with Kedro-compatible
    metadata columns. It supports experiment-specific parameter injection and lazy evaluation
    through manifest-based file discovery.
    
    Attributes:
        filepath: Path to the FlyRigLoader configuration file
        experiment_name: Name of the experiment to load data for
        _kwargs: Additional configuration parameters for discovery and loading
        _config: Cached configuration object after initial load
        _manifest: Cached file manifest for optimized repeated access
    
    Examples:
        >>> # Basic usage
        >>> dataset = FlyRigLoaderDataSet(
        ...     filepath="configs/experiment_config.yaml",
        ...     experiment_name="baseline_study"
        ... )
        >>> dataframe = dataset.load()
        >>> print(f"Loaded {len(dataframe)} rows")
        
        >>> # With additional parameters
        >>> dataset = FlyRigLoaderDataSet(
        ...     filepath="configs/experiment_config.yaml", 
        ...     experiment_name="behavioral_analysis",
        ...     recursive=True,
        ...     extract_metadata=True,
        ...     date_range=["2024-01-01", "2024-01-31"]
        ... )
    """
    
    def __init__(
        self,
        filepath: Union[str, Path],
        experiment_name: str,
        **kwargs: Any
    ) -> None:
        """
        Initialize FlyRigLoaderDataSet with configuration parameters.
        
        Args:
            filepath: Path to the FlyRigLoader configuration file
            experiment_name: Name of the experiment to load data for
            **kwargs: Additional parameters for discovery and loading including:
                - recursive: Enable recursive directory scanning
                - extract_metadata: Extract metadata from filenames
                - date_range: Date range filter for experiments
                - rig_names: List of rig names to include
                - file_patterns: Custom file patterns for discovery
                - transform_options: Options for DataFrame transformation
        
        Raises:
            ConfigError: If filepath or experiment_name are invalid
            TypeError: If parameters have incorrect types
        """
        logger.debug(f"Initializing FlyRigLoaderDataSet for experiment '{experiment_name}'")
        
        # Validate required parameters
        if not filepath:
            raise ConfigError(
                "filepath parameter is required",
                error_code="CONFIG_007",
                context={"parameter": "filepath", "value": filepath}
            )
        
        if not experiment_name or not isinstance(experiment_name, str):
            raise ConfigError(
                "experiment_name must be a non-empty string",
                error_code="CONFIG_007", 
                context={"parameter": "experiment_name", "value": experiment_name}
            )
        
        # Store configuration parameters with deep copy for thread safety
        self.filepath = Path(filepath)
        self.experiment_name = experiment_name.strip()
        self._kwargs = deepcopy(kwargs)
        
        # Initialize cached objects for performance optimization
        self._config: Optional[Any] = None
        self._manifest: Optional[Any] = None
        
        # Validate filepath exists and is accessible
        try:
            resolved_path = self.filepath.resolve()
            if not resolved_path.exists():
                logger.warning(f"Configuration file does not exist: {resolved_path}")
                # Don't raise error here - let _exists() handle this for Kedro compatibility
        except (OSError, RuntimeError) as e:
            logger.warning(f"Error accessing configuration file {filepath}: {e}")
        
        logger.info(f"FlyRigLoaderDataSet initialized for experiment '{self.experiment_name}'")
    
    def _load(self) -> pd.DataFrame:
        """
        Load experiment data using FlyRigLoader's discovery and transformation pipeline.
        
        This method implements the core data loading logic by:
        1. Loading and validating the configuration file
        2. Discovering experiment files using manifest-based discovery
        3. Loading raw data from discovered files using registry-based loaders
        4. Transforming data to Kedro-compatible pandas DataFrame
        
        Returns:
            pd.DataFrame: Transformed experimental data with Kedro-compatible metadata columns
            
        Raises:
            ConfigError: If configuration loading or validation fails
            FileNotFoundError: If configuration file or experiment data files are not found
            ValueError: If experiment data cannot be loaded or transformed
            
        Examples:
            >>> dataset = FlyRigLoaderDataSet("config.yaml", "experiment1")
            >>> df = dataset._load()
            >>> assert isinstance(df, pd.DataFrame)
            >>> assert 'experiment_name' in df.columns  # Kedro metadata column
        """
        with _dataset_lock:
            logger.info(f"Loading data for experiment '{self.experiment_name}'")
            
            try:
                # Step 1: Load and validate configuration
                if self._config is None:
                    logger.debug(f"Loading configuration from {self.filepath}")
                    self._config = load_config(self.filepath)
                    logger.debug("Configuration loaded and validated successfully")
                
                # Step 2: Discover experiment files using manifest-based approach
                if self._manifest is None:
                    logger.debug(f"Discovering files for experiment '{self.experiment_name}'")
                    
                    # Create experiment-specific configuration parameters
                    discovery_params = {
                        'enable_kedro_metadata': True,
                        'kedro_namespace': self.experiment_name,
                        **self._kwargs  # Include user-provided parameters
                    }
                    
                    self._manifest = discover_experiment_manifest(
                        config=self._config,
                        experiment_name=self.experiment_name,
                        **discovery_params
                    )
                    
                    logger.info(f"Discovered {len(self._manifest.files)} files for experiment")
                
                # Step 3: Load raw data from discovered files
                if not self._manifest.files:
                    logger.warning(f"No files found for experiment '{self.experiment_name}'")
                    # Return empty DataFrame with proper structure for consistency
                    return pd.DataFrame({
                        'experiment_name': [],
                        'file_path': [],
                        'timestamp': []
                    })
                
                logger.debug(f"Loading data from {len(self._manifest.files)} files")
                loaded_data = []
                
                for file_info in self._manifest.files:
                    try:
                        # Load individual file using registry-based loader
                        raw_data = load_data_file(file_info.path)
                        
                        # Add file-level metadata for traceability
                        if isinstance(raw_data, dict):
                            raw_data.update({
                                '_file_path': str(file_info.path),
                                '_file_size': file_info.size_bytes,
                                '_modified_time': file_info.modified_time,
                                '_experiment_name': self.experiment_name
                            })
                        
                        loaded_data.append(raw_data)
                        logger.debug(f"Successfully loaded data from {file_info.path}")
                        
                    except Exception as e:
                        logger.error(f"Failed to load data from {file_info.path}: {e}")
                        # Continue with other files rather than failing completely
                        continue
                
                if not loaded_data:
                    raise ValueError(f"Failed to load any data files for experiment '{self.experiment_name}'")
                
                # Step 4: Transform to Kedro-compatible DataFrame
                logger.debug("Transforming data to pandas DataFrame")
                
                # Merge transformation options from constructor and defaults
                transform_options = {
                    'include_kedro_metadata': True,
                    'experiment_name': self.experiment_name,
                    **self._kwargs.get('transform_options', {})
                }
                
                # Transform loaded data to standardized DataFrame
                dataframe = transform_to_dataframe(
                    data=loaded_data,
                    **transform_options
                )
                
                # Ensure Kedro-compatible metadata columns are present
                if 'experiment_name' not in dataframe.columns:
                    dataframe['experiment_name'] = self.experiment_name
                
                if 'dataset_source' not in dataframe.columns:
                    dataframe['dataset_source'] = 'flyrigloader'
                
                if 'load_timestamp' not in dataframe.columns:
                    from datetime import datetime
                    dataframe['load_timestamp'] = datetime.now().isoformat()
                
                logger.info(f"Successfully loaded DataFrame with shape {dataframe.shape}")
                logger.debug(f"DataFrame columns: {list(dataframe.columns)}")
                
                return dataframe
                
            except ConfigError:
                # Re-raise configuration errors as-is for proper error handling
                raise
            except FileNotFoundError as e:
                error_msg = f"Configuration or data files not found for experiment '{self.experiment_name}': {e}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg) from e
            except Exception as e:
                error_msg = f"Failed to load data for experiment '{self.experiment_name}': {e}"
                logger.error(error_msg)
                raise ValueError(error_msg) from e
    
    def _save(self, data: pd.DataFrame) -> None:
        """
        Save operation is not supported for FlyRigLoaderDataSet (read-only dataset).
        
        FlyRigLoaderDataSet is designed as a read-only dataset for loading experimental
        data. Save operations are not supported to maintain data integrity and prevent
        accidental modification of source experimental data files.
        
        Args:
            data: DataFrame that would be saved (unused)
            
        Raises:
            NotImplementedError: Always raised as save operations are not supported
        """
        error_msg = (
            f"FlyRigLoaderDataSet for experiment '{self.experiment_name}' is read-only. "
            "Save operations are not supported to maintain experimental data integrity. "
            "Use appropriate data output datasets in your Kedro pipeline for saving results."
        )
        logger.error(error_msg)
        raise NotImplementedError(error_msg)
    
    def _exists(self) -> bool:
        """
        Check if the dataset exists by validating configuration file presence.
        
        This method performs existence validation by checking if the configuration
        file exists and is readable. It does not validate that experiment data
        files exist, as this would require expensive file system traversal.
        
        Returns:
            bool: True if configuration file exists and is readable, False otherwise
            
        Examples:
            >>> dataset = FlyRigLoaderDataSet("existing_config.yaml", "experiment1")
            >>> assert dataset._exists() == True
            >>> 
            >>> dataset = FlyRigLoaderDataSet("missing_config.yaml", "experiment1") 
            >>> assert dataset._exists() == False
        """
        try:
            # Check if configuration file exists and is accessible
            resolved_path = self.filepath.resolve()
            exists = resolved_path.exists() and resolved_path.is_file()
            
            logger.debug(f"Dataset existence check for '{self.experiment_name}': {exists}")
            return exists
            
        except (OSError, RuntimeError) as e:
            # Handle path resolution errors (symlink loops, permission issues, etc.)
            logger.warning(f"Error checking dataset existence for '{self.experiment_name}': {e}")
            return False
    
    def _describe(self) -> Dict[str, Any]:
        """
        Return comprehensive dataset configuration metadata for Kedro lineage tracking.
        
        This method provides detailed information about the dataset configuration,
        experiment parameters, and operational settings for Kedro's data lineage
        and catalog management systems.
        
        Returns:
            Dict[str, Any]: Comprehensive dataset metadata including:
                - filepath: Configuration file path
                - experiment_name: Experiment identifier
                - dataset_type: Dataset class name
                - parameters: Configuration parameters
                - kedro_metadata: Kedro-specific metadata
                
        Examples:
            >>> dataset = FlyRigLoaderDataSet("config.yaml", "exp1", recursive=True)
            >>> metadata = dataset._describe()
            >>> assert metadata['experiment_name'] == 'exp1'
            >>> assert metadata['parameters']['recursive'] == True
        """
        # Create comprehensive metadata dictionary
        metadata = {
            # Core dataset information
            'dataset_type': self.__class__.__name__,
            'filepath': str(self.filepath),
            'experiment_name': self.experiment_name,
            
            # Configuration parameters
            'parameters': dict(self._kwargs),
            
            # Kedro-specific metadata
            'kedro_metadata': {
                'data_type': 'pandas.DataFrame',
                'operation_mode': 'read_only',
                'supports_versioning': False,
                'supports_parallel_execution': True,
                'thread_safe': True
            },
            
            # Runtime information
            'runtime_info': {
                'config_loaded': self._config is not None,
                'manifest_cached': self._manifest is not None,
                'file_exists': self._exists()
            }
        }
        
        # Add manifest information if available
        if self._manifest is not None:
            metadata['manifest_info'] = {
                'file_count': len(self._manifest.files),
                'total_size_bytes': sum(f.size_bytes for f in self._manifest.files if f.size_bytes),
                'file_extensions': list(set(f.path.suffix for f in self._manifest.files))
            }
        
        logger.debug(f"Generated dataset description for experiment '{self.experiment_name}'")
        return metadata


class FlyRigManifestDataSet(AbstractDataSet[None, Any]):
    """
    Kedro AbstractDataset implementation for manifest-only operations without full data loading.
    
    This dataset provides lightweight file discovery operations by returning file manifests
    without loading actual experimental data. It's optimized for pipeline workflows that
    need file inventory, metadata extraction, or selective processing based on discovered
    files rather than full data loading.
    
    The dataset returns FileManifest objects containing file metadata, statistics, and
    discovery information with minimal memory overhead and fast execution times.
    
    Attributes:
        filepath: Path to the FlyRigLoader configuration file
        experiment_name: Name of the experiment to discover files for
        _kwargs: Additional configuration parameters for discovery
        _config: Cached configuration object after initial load
    
    Examples:
        >>> # Basic manifest discovery
        >>> dataset = FlyRigManifestDataSet(
        ...     filepath="configs/experiment_config.yaml",
        ...     experiment_name="baseline_study"
        ... )
        >>> manifest = dataset.load()
        >>> print(f"Found {len(manifest.files)} files")
        
        >>> # Use manifest for selective processing
        >>> large_files = [f for f in manifest.files if f.size_bytes > 1000000]
        >>> print(f"Found {len(large_files)} large files")
    """
    
    def __init__(
        self,
        filepath: Union[str, Path],
        experiment_name: str,
        **kwargs: Any
    ) -> None:
        """
        Initialize FlyRigManifestDataSet with configuration parameters.
        
        Args:
            filepath: Path to the FlyRigLoader configuration file
            experiment_name: Name of the experiment to discover files for
            **kwargs: Additional parameters for discovery including:
                - recursive: Enable recursive directory scanning
                - extract_metadata: Extract metadata from filenames
                - include_stats: Include file statistics in manifest
                - patterns: Custom patterns for file discovery
                - filters: Additional filters for file selection
        
        Raises:
            ConfigError: If filepath or experiment_name are invalid
            TypeError: If parameters have incorrect types
        """
        logger.debug(f"Initializing FlyRigManifestDataSet for experiment '{experiment_name}'")
        
        # Validate required parameters
        if not filepath:
            raise ConfigError(
                "filepath parameter is required",
                error_code="CONFIG_007",
                context={"parameter": "filepath", "value": filepath}
            )
        
        if not experiment_name or not isinstance(experiment_name, str):
            raise ConfigError(
                "experiment_name must be a non-empty string",
                error_code="CONFIG_007",
                context={"parameter": "experiment_name", "value": experiment_name}
            )
        
        # Store configuration parameters with deep copy for thread safety
        self.filepath = Path(filepath)
        self.experiment_name = experiment_name.strip()
        self._kwargs = deepcopy(kwargs)
        
        # Initialize cached configuration
        self._config: Optional[Any] = None
        
        # Validate filepath accessibility
        try:
            resolved_path = self.filepath.resolve()
            if not resolved_path.exists():
                logger.warning(f"Configuration file does not exist: {resolved_path}")
        except (OSError, RuntimeError) as e:
            logger.warning(f"Error accessing configuration file {filepath}: {e}")
        
        logger.info(f"FlyRigManifestDataSet initialized for experiment '{self.experiment_name}'")
    
    def _load(self) -> Any:
        """
        Discover experiment files and return manifest without loading data.
        
        This method performs lightweight file discovery operations to create a
        FileManifest containing metadata about discovered files without actually
        loading the file contents. This enables efficient pipeline workflows
        that need file inventory or selective processing capabilities.
        
        Returns:
            FileManifest: Object containing discovered file metadata, statistics,
                         and discovery information
            
        Raises:
            ConfigError: If configuration loading or validation fails
            FileNotFoundError: If configuration file is not found
            ValueError: If file discovery fails
            
        Examples:
            >>> dataset = FlyRigManifestDataSet("config.yaml", "experiment1")
            >>> manifest = dataset._load()
            >>> print(f"Discovered {len(manifest.files)} files")
            >>> for file_info in manifest.files:
            ...     print(f"File: {file_info.path}, Size: {file_info.size_bytes}")
        """
        with _dataset_lock:
            logger.info(f"Discovering manifest for experiment '{self.experiment_name}'")
            
            try:
                # Load and validate configuration
                if self._config is None:
                    logger.debug(f"Loading configuration from {self.filepath}")
                    self._config = load_config(self.filepath)
                    logger.debug("Configuration loaded and validated successfully")
                
                # Create discovery parameters optimized for manifest-only operations
                discovery_params = {
                    'include_stats': self._kwargs.get('include_stats', True),
                    'parse_dates': self._kwargs.get('parse_dates', True),
                    'enable_kedro_metadata': True,
                    'kedro_namespace': self.experiment_name,
                    **{k: v for k, v in self._kwargs.items() 
                       if k not in ['include_stats', 'parse_dates']}
                }
                
                # Discover experiment files using manifest-based approach
                logger.debug(f"Discovering files for experiment '{self.experiment_name}'")
                manifest = discover_experiment_manifest(
                    config=self._config,
                    experiment_name=self.experiment_name,
                    **discovery_params
                )
                
                logger.info(f"Successfully discovered manifest with {len(manifest.files)} files")
                logger.debug(f"Total size: {sum(f.size_bytes for f in manifest.files if f.size_bytes)} bytes")
                
                return manifest
                
            except ConfigError:
                # Re-raise configuration errors as-is
                raise
            except FileNotFoundError as e:
                error_msg = f"Configuration file not found for experiment '{self.experiment_name}': {e}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg) from e
            except Exception as e:
                error_msg = f"Failed to discover manifest for experiment '{self.experiment_name}': {e}"
                logger.error(error_msg)
                raise ValueError(error_msg) from e
    
    def _save(self, data: Any) -> None:
        """
        Save operation is not supported for FlyRigManifestDataSet (read-only dataset).
        
        FlyRigManifestDataSet is designed as a read-only dataset for file discovery
        and manifest generation. Save operations are not supported as manifests
        are generated dynamically from file system state.
        
        Args:
            data: Manifest data that would be saved (unused)
            
        Raises:
            NotImplementedError: Always raised as save operations are not supported
        """
        error_msg = (
            f"FlyRigManifestDataSet for experiment '{self.experiment_name}' is read-only. "
            "Save operations are not supported as manifests are generated dynamically. "
            "Use appropriate data output datasets in your Kedro pipeline for saving results."
        )
        logger.error(error_msg)
        raise NotImplementedError(error_msg)
    
    def _exists(self) -> bool:
        """
        Check if the dataset exists by validating configuration file presence.
        
        This method performs existence validation by checking if the configuration
        file exists and is readable. It does not validate experiment file existence
        as manifest discovery handles missing files gracefully.
        
        Returns:
            bool: True if configuration file exists and is readable, False otherwise
            
        Examples:
            >>> dataset = FlyRigManifestDataSet("existing_config.yaml", "experiment1")
            >>> assert dataset._exists() == True
            >>> 
            >>> dataset = FlyRigManifestDataSet("missing_config.yaml", "experiment1")
            >>> assert dataset._exists() == False
        """
        try:
            # Check if configuration file exists and is accessible
            resolved_path = self.filepath.resolve()
            exists = resolved_path.exists() and resolved_path.is_file()
            
            logger.debug(f"Manifest dataset existence check for '{self.experiment_name}': {exists}")
            return exists
            
        except (OSError, RuntimeError) as e:
            # Handle path resolution errors
            logger.warning(f"Error checking manifest dataset existence for '{self.experiment_name}': {e}")
            return False
    
    def _describe(self) -> Dict[str, Any]:
        """
        Return comprehensive dataset configuration metadata for Kedro lineage tracking.
        
        This method provides detailed information about the manifest dataset
        configuration, discovery parameters, and operational settings for
        Kedro's data lineage and catalog management systems.
        
        Returns:
            Dict[str, Any]: Comprehensive dataset metadata including:
                - filepath: Configuration file path
                - experiment_name: Experiment identifier
                - dataset_type: Dataset class name
                - parameters: Discovery parameters
                - kedro_metadata: Kedro-specific metadata
                
        Examples:
            >>> dataset = FlyRigManifestDataSet("config.yaml", "exp1", recursive=True)
            >>> metadata = dataset._describe()
            >>> assert metadata['experiment_name'] == 'exp1'
            >>> assert metadata['kedro_metadata']['data_type'] == 'FileManifest'
        """
        # Create comprehensive metadata dictionary
        metadata = {
            # Core dataset information
            'dataset_type': self.__class__.__name__,
            'filepath': str(self.filepath),
            'experiment_name': self.experiment_name,
            
            # Discovery parameters
            'parameters': dict(self._kwargs),
            
            # Kedro-specific metadata
            'kedro_metadata': {
                'data_type': 'FileManifest',
                'operation_mode': 'read_only',
                'supports_versioning': False,
                'supports_parallel_execution': True,
                'thread_safe': True,
                'lightweight_operation': True
            },
            
            # Runtime information
            'runtime_info': {
                'config_loaded': self._config is not None,
                'file_exists': self._exists(),
                'discovery_only': True
            }
        }
        
        logger.debug(f"Generated manifest dataset description for experiment '{self.experiment_name}'")
        return metadata


# Export public interface
__all__ = [
    'FlyRigLoaderDataSet',
    'FlyRigManifestDataSet'
]