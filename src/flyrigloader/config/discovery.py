"""
Config-aware file discovery utilities with Pydantic model support.

Combines YAML configuration with file discovery functionality, now supporting
both legacy dictionary-based configurations and new Pydantic model-based
configurations for improved type safety and validation.

Implements dependency injection patterns for comprehensive testing support
and provides enhanced configuration validation through Pydantic model integration.
"""
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable, Protocol
from abc import ABC, abstractmethod

from flyrigloader.discovery.files import discover_files
from flyrigloader.config.yaml_config import (
    load_config,
    get_ignore_patterns,
    get_mandatory_substrings,
    get_dataset_info,
    get_experiment_info,
    get_extraction_patterns
)
from flyrigloader.config.models import ExperimentConfig, LegacyConfigAdapter
from flyrigloader.config.validators import date_format_validator
from flyrigloader import logger

class PathProvider(Protocol):
    """Protocol for configurable path operations supporting test mocking."""
    
    def resolve_path(self, path: Union[str, Path]) -> Path:
        """Resolve a path to an absolute Path object."""
        ...
    
    def exists(self, path: Union[str, Path]) -> bool:
        """Check if a path exists."""
        ...
    
    def list_directories(self, base_path: Union[str, Path]) -> List[Path]:
        """List directories in the given base path."""
        ...


class FileDiscoveryProvider(Protocol):
    """Protocol for configurable file discovery operations supporting test injection."""
    
    def discover_files(
        self,
        directory: Union[str, List[str]],
        pattern: str,
        recursive: bool = False,
        extensions: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None,
        mandatory_substrings: Optional[List[str]] = None,
        extract_patterns: Optional[List[str]] = None,
        parse_dates: bool = False,
        include_stats: bool = False
    ) -> Union[List[str], Dict[str, Dict[str, Any]]]:
        """Discover files with the given criteria."""
        ...


class ConfigurationProvider(Protocol):
    """Protocol for configuration access supporting test mocking and Pydantic models."""
    
    def get_ignore_patterns(
        self, config: Union[Dict[str, Any], 'ExperimentConfig'], experiment: Optional[str] = None
    ) -> List[str]:
        """Get ignore patterns from configuration (dict or Pydantic model)."""
        ...
    
    def get_mandatory_substrings(
        self, config: Union[Dict[str, Any], 'ExperimentConfig'], experiment: Optional[str] = None
    ) -> List[str]:
        """Get mandatory substrings from configuration (dict or Pydantic model)."""
        ...
    
    def get_dataset_info(self, config: Union[Dict[str, Any], 'ExperimentConfig'], dataset_name: str) -> Dict[str, Any]:
        """Get dataset information from configuration (dict or Pydantic model)."""
        ...
    
    def get_experiment_info(self, config: Union[Dict[str, Any], 'ExperimentConfig'], experiment_name: str) -> Dict[str, Any]:
        """Get experiment information from configuration (dict or Pydantic model)."""
        ...
    
    def get_extraction_patterns(
        self,
        config: Union[Dict[str, Any], 'ExperimentConfig'],
        experiment: Optional[str] = None,
        dataset_name: Optional[str] = None
    ) -> Optional[List[str]]:
        """Get extraction patterns from configuration (dict or Pydantic model)."""
        ...


class DefaultPathProvider:
    """Default implementation of PathProvider using standard pathlib operations."""
    
    def resolve_path(self, path: Union[str, Path]) -> Path:
        """Resolve a path to an absolute Path object."""
        return Path(path).resolve()
    
    def exists(self, path: Union[str, Path]) -> bool:
        """Check if a path exists."""
        return Path(path).exists()
    
    def list_directories(self, base_path: Union[str, Path]) -> List[Path]:
        """List directories in the given base path."""
        base_path = Path(base_path)
        if not base_path.exists():
            return []
        return [p for p in base_path.iterdir() if p.is_dir()]


class DefaultFileDiscoveryProvider:
    """Default implementation of FileDiscoveryProvider using standard discovery functions."""
    
    def discover_files(
        self,
        directory: Union[str, List[str]],
        pattern: str,
        recursive: bool = False,
        extensions: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None,
        mandatory_substrings: Optional[List[str]] = None,
        extract_patterns: Optional[List[str]] = None,
        parse_dates: bool = False,
        include_stats: bool = False
    ) -> Union[List[str], Dict[str, Dict[str, Any]]]:
        """Discover files using the standard discovery function."""
        return discover_files(
            directory=directory,
            pattern=pattern,
            recursive=recursive,
            extensions=extensions,
            ignore_patterns=ignore_patterns,
            mandatory_substrings=mandatory_substrings,
            extract_patterns=extract_patterns,
            parse_dates=parse_dates,
            include_stats=include_stats
        )


class DefaultConfigurationProvider:
    """
    Default implementation of ConfigurationProvider with Pydantic model support.
    
    This provider supports both legacy dictionary-based configurations and new
    Pydantic model-based configurations, providing type safety and validation
    while maintaining backward compatibility.
    """
    
    def _ensure_dict_config(self, config: Union[Dict[str, Any], ExperimentConfig]) -> Dict[str, Any]:
        """
        Convert Pydantic model to dictionary format for legacy function compatibility.
        
        Args:
            config: Configuration as either dict or Pydantic model
            
        Returns:
            Dictionary representation of the configuration
        """
        if isinstance(config, ExperimentConfig):
            # Convert ExperimentConfig to dict format expected by legacy functions
            return {
                'experiments': {
                    # Create a synthetic experiment entry for the provided config
                    'current': {
                        'datasets': config.datasets,
                        'parameters': config.parameters or {},
                        'filters': config.filters or {},
                        'metadata': config.metadata or {}
                    }
                }
            }
        elif hasattr(config, 'model_dump'):
            # Handle other Pydantic models
            return config.model_dump()
        else:
            # Already a dictionary
            return config
    
    def get_ignore_patterns(
        self, config: Union[Dict[str, Any], ExperimentConfig], experiment: Optional[str] = None
    ) -> List[str]:
        """
        Get ignore patterns from configuration with Pydantic model support.
        
        Args:
            config: Configuration as either dict or Pydantic model
            experiment: Optional experiment name
            
        Returns:
            List of ignore patterns
        """
        # Handle ExperimentConfig directly for better type safety
        if isinstance(config, ExperimentConfig):
            if config.filters and 'ignore_substrings' in config.filters:
                patterns = config.filters['ignore_substrings']
                if isinstance(patterns, list):
                    return patterns
            return []
        
        # Fallback to legacy dictionary-based approach
        dict_config = self._ensure_dict_config(config)
        return get_ignore_patterns(dict_config, experiment)
    
    def get_mandatory_substrings(
        self, config: Union[Dict[str, Any], ExperimentConfig], experiment: Optional[str] = None
    ) -> List[str]:
        """
        Get mandatory substrings from configuration with Pydantic model support.
        
        Args:
            config: Configuration as either dict or Pydantic model
            experiment: Optional experiment name
            
        Returns:
            List of mandatory substrings
        """
        # Handle ExperimentConfig directly for better type safety
        if isinstance(config, ExperimentConfig):
            if config.filters and 'mandatory_experiment_strings' in config.filters:
                strings = config.filters['mandatory_experiment_strings']
                if isinstance(strings, list):
                    return strings
            return []
        
        # Fallback to legacy dictionary-based approach
        dict_config = self._ensure_dict_config(config)
        return get_mandatory_substrings(dict_config, experiment)
    
    def get_dataset_info(self, config: Union[Dict[str, Any], ExperimentConfig], dataset_name: str) -> Dict[str, Any]:
        """
        Get dataset information from configuration with Pydantic model support.
        
        Args:
            config: Configuration as either dict or Pydantic model
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary containing dataset information
        """
        # For ExperimentConfig, we can validate that the dataset is in the datasets list
        if isinstance(config, ExperimentConfig):
            if dataset_name not in config.datasets:
                raise KeyError(f"Dataset '{dataset_name}' not found in experiment configuration")
            # We still need to fall back to the full config for dataset details
            # as ExperimentConfig only contains dataset references, not full definitions
            logger.warning(f"ExperimentConfig only contains dataset references. Full dataset info requires complete configuration.")
            return {'rig': 'unknown', 'dates_vials': {}}
        
        # Fallback to legacy dictionary-based approach
        dict_config = self._ensure_dict_config(config)
        return get_dataset_info(dict_config, dataset_name)
    
    def get_experiment_info(self, config: Union[Dict[str, Any], ExperimentConfig], experiment_name: str) -> Dict[str, Any]:
        """
        Get experiment information from configuration with Pydantic model support.
        
        Args:
            config: Configuration as either dict or Pydantic model
            experiment_name: Name of the experiment
            
        Returns:
            Dictionary containing experiment information
        """
        # Handle ExperimentConfig directly for better type safety
        if isinstance(config, ExperimentConfig):
            # Direct access to validated Pydantic model properties
            return {
                'datasets': config.datasets,
                'parameters': config.parameters or {},
                'filters': config.filters or {},
                'metadata': config.metadata or {}
            }
        
        # Fallback to legacy dictionary-based approach
        dict_config = self._ensure_dict_config(config)
        return get_experiment_info(dict_config, experiment_name)
    
    def get_extraction_patterns(
        self,
        config: Union[Dict[str, Any], ExperimentConfig],
        experiment: Optional[str] = None,
        dataset_name: Optional[str] = None
    ) -> Optional[List[str]]:
        """
        Get extraction patterns from configuration with Pydantic model support.
        
        Args:
            config: Configuration as either dict or Pydantic model
            experiment: Optional experiment name
            dataset_name: Optional dataset name
            
        Returns:
            List of extraction patterns or None if not found
        """
        # Handle ExperimentConfig directly for better type safety
        if isinstance(config, ExperimentConfig):
            if config.metadata and 'extraction_patterns' in config.metadata:
                patterns = config.metadata['extraction_patterns']
                if isinstance(patterns, list):
                    return patterns
            return None
        
        # Fallback to legacy dictionary-based approach
        dict_config = self._ensure_dict_config(config)
        return get_extraction_patterns(dict_config, experiment, dataset_name)


class ConfigDiscoveryEngine:
    """
    Enhanced configuration-driven file discovery engine with dependency injection support.
    
    This class implements dependency injection patterns for improved testability,
    supporting pytest.monkeypatch scenarios and comprehensive mocking strategies.
    """
    
    def __init__(
        self,
        path_provider: Optional[PathProvider] = None,
        file_discovery_provider: Optional[FileDiscoveryProvider] = None,
        config_provider: Optional[ConfigurationProvider] = None
    ):
        """
        Initialize the discovery engine with configurable dependencies.
        
        Args:
            path_provider: Provider for path operations (defaults to DefaultPathProvider)
            file_discovery_provider: Provider for file discovery (defaults to DefaultFileDiscoveryProvider)
            config_provider: Provider for configuration access (defaults to DefaultConfigurationProvider)
        """
        self.path_provider = path_provider or DefaultPathProvider()
        self.file_discovery_provider = file_discovery_provider or DefaultFileDiscoveryProvider()
        self.config_provider = config_provider or DefaultConfigurationProvider()
        
        logger.debug(
            "ConfigDiscoveryEngine initialized with providers: "
            f"path={type(self.path_provider).__name__}, "
            f"discovery={type(self.file_discovery_provider).__name__}, "
            f"config={type(self.config_provider).__name__}"
        )

    def discover_files_with_config(
        self,
        config: Union[Dict[str, Any], ExperimentConfig],
        directory: Union[str, List[str]],
        pattern: str,
        recursive: bool = False,
        extensions: Optional[List[str]] = None,
        experiment: Optional[str] = None,
        extract_metadata: bool = False,
        parse_dates: bool = False
    ) -> Union[List[str], Dict[str, Dict[str, Any]]]:
        """
        Discover files using configuration-aware filtering with enhanced error handling and Pydantic model support.
        
        This uses the configuration to determine ignore patterns and mandatory substrings
        based on project-wide settings and experiment-specific overrides. Now supports
        both legacy dictionary configurations and new Pydantic model configurations.
        
        Args:
            config: The loaded configuration (dictionary, Pydantic model, or Kedro-style parameters)
            directory: The directory or list of directories to search in
            pattern: File pattern to match (glob format)
            recursive: If True, search recursively through subdirectories
            extensions: Optional list of file extensions to filter by (without the dot)
            experiment: Optional experiment name to use experiment-specific filters
            extract_metadata: If True, extract metadata using patterns from config
            parse_dates: If True, attempt to parse dates from filenames
            
        Returns:
            If extract_metadata or parse_dates is True: Dictionary mapping file paths to metadata
            Otherwise: List of file paths matching the criteria
            
        Raises:
            ValueError: If config is invalid or directory parameters are malformed
            KeyError: If experiment-specific configuration is referenced but not found
        """
        # Enhanced parameter validation with Pydantic model support
        if isinstance(config, ExperimentConfig):
            # Validate using Pydantic model validation
            try:
                # Use model_validate to ensure the configuration is properly validated
                validated_config = ExperimentConfig.model_validate(config.model_dump())
                logger.debug("ExperimentConfig validated successfully")
            except Exception as e:
                raise ValueError(f"Invalid ExperimentConfig: {e}")
        elif not isinstance(config, (dict, LegacyConfigAdapter)):
            raise ValueError(f"Configuration must be a dictionary, LegacyConfigAdapter, or ExperimentConfig, got {type(config).__name__}")
        
        if not directory:
            raise ValueError("Directory parameter cannot be empty")
        
        if not pattern:
            raise ValueError("Pattern parameter cannot be empty")
        
        # Enhanced experiment validation for dictionary configs
        if isinstance(config, (dict, LegacyConfigAdapter)) and experiment and experiment not in config.get("experiments", {}):
            logger.warning(f"Experiment '{experiment}' not found in configuration, using project-level settings only")
        
        # For ExperimentConfig, validate that the experiment parameter is consistent
        if isinstance(config, ExperimentConfig) and experiment:
            logger.debug(f"Using ExperimentConfig directly, experiment parameter '{experiment}' will be used for logging only")
        
        logger.debug(
            f"Discovering files with config type={type(config).__name__} for experiment={experiment}, "
            f"pattern={pattern}, recursive={recursive}, extract_metadata={extract_metadata}"
        )
        
        try:
            # Get ignore patterns from config (project + experiment level)
            ignore_patterns = self.config_provider.get_ignore_patterns(config, experiment)
            logger.debug(f"Retrieved {len(ignore_patterns)} ignore patterns")
            
            # Get mandatory substrings from config (project + experiment level)
            mandatory_substrings = self.config_provider.get_mandatory_substrings(config, experiment)
            logger.debug(f"Retrieved {len(mandatory_substrings)} mandatory substrings")
            
            # Get extraction patterns from config (if requested)
            extract_patterns = None
            if extract_metadata:
                extract_patterns = self.config_provider.get_extraction_patterns(config, experiment)
                logger.debug(f"Retrieved {len(extract_patterns) if extract_patterns else 0} extraction patterns")
            
            # Use the configurable discovery provider with config-derived filters
            result = self.file_discovery_provider.discover_files(
                directory=directory,
                pattern=pattern,
                recursive=recursive,
                extensions=extensions,
                ignore_patterns=ignore_patterns,
                mandatory_substrings=mandatory_substrings,
                extract_patterns=extract_patterns,
                parse_dates=parse_dates
            )
            
            logger.info(
                f"Discovery completed: found {len(result) if isinstance(result, (list, dict)) else 'unknown'} files/entries"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error during configuration-driven file discovery: {e}")
            raise

    def discover_experiment_files(
        self,
        config: Union[Dict[str, Any], ExperimentConfig],
        experiment_name: str,
        base_directory: Union[str, Path],
        pattern: str = "*.*",
        recursive: bool = True,
        extensions: Optional[List[str]] = None,
        extract_metadata: bool = False,
        parse_dates: bool = False
    ) -> Union[List[str], Dict[str, Dict[str, Any]]]:
        """
        Discover files related to a specific experiment with enhanced validation, logging, and Pydantic model support.
        
        This uses the experiment's dataset definitions and filter settings
        to find relevant files with improved error handling, test observability,
        and type safety through Pydantic model validation.
        
        Args:
            config: The loaded configuration (dictionary, ExperimentConfig, or Kedro-style parameters)
            experiment_name: Name of the experiment to use for discovery
            base_directory: Base directory to search in (often the major_data_directory)
            pattern: File pattern to match (glob format), defaults to all files
            recursive: If True, search recursively through subdirectories
            extensions: Optional list of file extensions to filter by
            extract_metadata: If True, extract metadata using patterns from config
            parse_dates: If True, attempt to parse dates from filenames
            
        Returns:
            If extract_metadata or parse_dates is True: Dictionary mapping file paths to metadata
            Otherwise: List of file paths relevant to the experiment
            
        Raises:
            KeyError: If the experiment does not exist in the configuration
            ValueError: If parameters are invalid or base_directory is not accessible
        """
        # Enhanced parameter validation with Pydantic model support
        if isinstance(config, ExperimentConfig):
            # Use Pydantic model validation for type safety
            try:
                validated_config = ExperimentConfig.model_validate(config.model_dump())
                logger.debug(f"ExperimentConfig validated successfully for experiment '{experiment_name}'")
            except Exception as e:
                raise ValueError(f"Invalid ExperimentConfig: {e}")
        elif not isinstance(config, (dict, LegacyConfigAdapter)):
            raise ValueError(f"Configuration must be a dictionary, LegacyConfigAdapter, or ExperimentConfig, got {type(config).__name__}")
        
        if not experiment_name or not isinstance(experiment_name, str):
            raise ValueError(f"Experiment name must be a non-empty string, got {experiment_name}")
        
        if not base_directory:
            raise ValueError("Base directory cannot be empty")
        
        logger.info(f"Starting experiment file discovery for experiment '{experiment_name}' with config type {type(config).__name__}")
        
        try:
            # Get experiment information with enhanced error handling and Pydantic model support
            if isinstance(config, ExperimentConfig):
                # Direct access to validated Pydantic model properties for better type safety
                experiment_info = {
                    'datasets': config.datasets,
                    'parameters': config.parameters or {},
                    'filters': config.filters or {},
                    'metadata': config.metadata or {}
                }
                logger.debug(f"Retrieved experiment info from ExperimentConfig for '{experiment_name}': {len(experiment_info)} keys")
            else:
                # Legacy dictionary-based approach
                experiment_info = self.config_provider.get_experiment_info(config, experiment_name)
                logger.debug(f"Retrieved experiment info for '{experiment_name}': {len(experiment_info)} keys")
            
            # Get the list of datasets for this experiment
            dataset_names = experiment_info.get("datasets", [])
            logger.debug(f"Found {len(dataset_names)} datasets for experiment '{experiment_name}': {dataset_names}")
            
            # Collect all date-specific directories to search in
            search_dirs = []
            base_path = self.path_provider.resolve_path(base_directory)
            
            for dataset_name in dataset_names:
                try:
                    # Note: For ExperimentConfig, we don't have full dataset info,
                    # so we'll need to handle this case differently
                    if isinstance(config, ExperimentConfig):
                        # When using ExperimentConfig, we can only validate that the dataset
                        # is in the datasets list, but we can't get full dataset info
                        # without the complete configuration. This is a limitation of
                        # using only ExperimentConfig for discovery.
                        logger.warning(f"ExperimentConfig doesn't contain full dataset info for '{dataset_name}'. Using base directory.")
                        if self.path_provider.exists(base_directory):
                            search_dirs = [str(base_directory)]
                        break
                    else:
                        # Legacy approach with full configuration
                        dataset_info = self.config_provider.get_dataset_info(config, dataset_name)
                        # Get all date directories for this dataset
                        dates = dataset_info.get("dates_vials", {}).keys()
                        logger.debug(f"Dataset '{dataset_name}' has {len(dates)} date entries")
                        
                        for date in dates:
                            date_dir = base_path / str(date)
                            if self.path_provider.exists(date_dir):
                                search_dirs.append(str(date_dir))
                                logger.debug(f"Added search directory: {date_dir}")
                            else:
                                logger.warning(f"Date directory does not exist: {date_dir}")
                            
                except KeyError as e:
                    logger.warning(f"Dataset '{dataset_name}' not found in configuration: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing dataset '{dataset_name}': {e}")
                    continue
            
            # If no search directories were found, use the base directory
            if not search_dirs:
                if self.path_provider.exists(base_directory):
                    search_dirs = [str(base_directory)]
                    logger.info(f"No date-specific directories found, using base directory: {base_directory}")
                else:
                    raise ValueError(f"Base directory does not exist: {base_directory}")
            
            logger.info(f"Searching in {len(search_dirs)} directories for experiment '{experiment_name}'")
            
            # Discover files using config-aware filtering with enhanced Pydantic model support
            return self.discover_files_with_config(
                config=config,
                directory=search_dirs,
                pattern=pattern,
                recursive=recursive,
                extensions=extensions,
                experiment=experiment_name,
                extract_metadata=extract_metadata,
                parse_dates=parse_dates
            )
            
        except Exception as e:
            logger.error(f"Error during experiment file discovery for '{experiment_name}': {e}")
            raise

    def discover_dataset_files(
        self,
        config: Union[Dict[str, Any], ExperimentConfig],
        dataset_name: str,
        base_directory: Union[str, Path],
        pattern: str = "*.*",
        recursive: bool = True,
        extensions: Optional[List[str]] = None,
        extract_metadata: bool = False,
        parse_dates: bool = False
    ) -> Union[List[str], Dict[str, Dict[str, Any]]]:
        """
        Discover files related to a specific dataset with enhanced validation, logging, and Pydantic model support.
        
        This uses the dataset's date-vial definitions to find relevant files
        with improved error handling, test observability, and type safety
        through Pydantic model validation.
        
        Args:
            config: The loaded configuration (dictionary, ExperimentConfig, or Kedro-style parameters)
            dataset_name: Name of the dataset to use for discovery
            base_directory: Base directory to search in (often the major_data_directory)
            pattern: File pattern to match (glob format), defaults to all files
            recursive: If True, search recursively through subdirectories
            extensions: Optional list of file extensions to filter by
            extract_metadata: If True, extract metadata using patterns from config
            parse_dates: If True, attempt to parse dates from filenames
            
        Returns:
            If extract_metadata or parse_dates is True: Dictionary mapping file paths to metadata
            Otherwise: List of file paths relevant to the dataset
            
        Raises:
            KeyError: If the dataset does not exist in the configuration
            ValueError: If parameters are invalid or base_directory is not accessible
        """
        # Enhanced parameter validation with Pydantic model support
        if isinstance(config, ExperimentConfig):
            # Use Pydantic model validation for type safety
            try:
                validated_config = ExperimentConfig.model_validate(config.model_dump())
                
                # Validate that the dataset is in the experiment's datasets list
                if dataset_name not in config.datasets:
                    raise KeyError(f"Dataset '{dataset_name}' not found in ExperimentConfig datasets: {config.datasets}")
                
                logger.debug(f"ExperimentConfig validated successfully for dataset '{dataset_name}'")
            except Exception as e:
                raise ValueError(f"Invalid ExperimentConfig: {e}")
        elif not isinstance(config, (dict, LegacyConfigAdapter)):
            raise ValueError(f"Configuration must be a dictionary, LegacyConfigAdapter, or ExperimentConfig, got {type(config).__name__}")
        
        if not dataset_name or not isinstance(dataset_name, str):
            raise ValueError(f"Dataset name must be a non-empty string, got {dataset_name}")
        
        if not base_directory:
            raise ValueError("Base directory cannot be empty")
        
        logger.info(f"Starting dataset file discovery for dataset '{dataset_name}' with config type {type(config).__name__}")
        
        try:
            # Handle ExperimentConfig vs dictionary-based configurations
            if isinstance(config, ExperimentConfig):
                # For ExperimentConfig, we don't have full dataset info (dates_vials),
                # so we'll use the base directory and rely on filters from the experiment
                logger.warning(f"ExperimentConfig doesn't contain dataset date/vial info. Using base directory for dataset '{dataset_name}'.")
                search_dirs = []
                base_path = self.path_provider.resolve_path(base_directory)
                
                if self.path_provider.exists(base_directory):
                    search_dirs = [str(base_directory)]
                    logger.info(f"Using base directory for ExperimentConfig dataset discovery: {base_directory}")
                else:
                    raise ValueError(f"Base directory does not exist: {base_directory}")
                
                # Get patterns from the ExperimentConfig filters
                ignore_patterns = self.config_provider.get_ignore_patterns(config)
                mandatory_substrings = self.config_provider.get_mandatory_substrings(config)
                extract_patterns = None
                if extract_metadata:
                    extract_patterns = self.config_provider.get_extraction_patterns(config, dataset_name=dataset_name)
                
            else:
                # Legacy dictionary-based approach with full dataset information
                dataset_info = self.config_provider.get_dataset_info(config, dataset_name)
                logger.debug(f"Retrieved dataset info for '{dataset_name}': {len(dataset_info)} keys")
                
                # Get the dates for this dataset
                dates = dataset_info.get("dates_vials", {}).keys()
                logger.debug(f"Dataset '{dataset_name}' has {len(dates)} date entries")
                
                # Collect all date-specific directories to search in
                search_dirs = []
                base_path = self.path_provider.resolve_path(base_directory)
                dates_were_provided = len(dates) > 0
                
                for date in dates:
                    # Validate date format before using it for directory search
                    try:
                        if not date_format_validator(str(date)):
                            logger.warning(f"Invalid date format '{date}', skipping directory search")
                            continue
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Invalid date format '{date}': {e}, skipping directory search")
                        continue
                    
                    date_dir = base_path / str(date)
                    if self.path_provider.exists(date_dir):
                        search_dirs.append(str(date_dir))
                        logger.debug(f"Added search directory: {date_dir}")
                    else:
                        logger.warning(f"Date directory does not exist: {date_dir}")
                
                # If no search directories were found, decide what to do based on whether dates were provided
                if not search_dirs:
                    if dates_were_provided:
                        # Dates were provided but all were invalid or directories don't exist
                        logger.info(f"All provided dates were invalid or directories don't exist, returning empty results")
                        return []
                    else:
                        # No dates were provided, fall back to base directory
                        if self.path_provider.exists(base_directory):
                            search_dirs = [str(base_directory)]
                            logger.info(f"No date-specific configuration found, using base directory: {base_directory}")
                        else:
                            raise ValueError(f"Base directory does not exist: {base_directory}")
                
                # Get project-level ignore patterns (no experiment-specific ones)
                ignore_patterns = self.config_provider.get_ignore_patterns(config)
                mandatory_substrings = self.config_provider.get_mandatory_substrings(config)
                extract_patterns = None
                if extract_metadata:
                    extract_patterns = self.config_provider.get_extraction_patterns(config, dataset_name=dataset_name)
            
            logger.info(f"Searching in {len(search_dirs)} directories for dataset '{dataset_name}'")
            logger.debug(f"Retrieved {len(ignore_patterns)} ignore patterns")
            logger.debug(f"Retrieved {len(mandatory_substrings)} mandatory substrings")
            logger.debug(f"Retrieved {len(extract_patterns) if extract_patterns else 0} extraction patterns")
            
            # Use the configurable discovery provider with dataset-specific directories
            result = self.file_discovery_provider.discover_files(
                directory=search_dirs,
                pattern=pattern,
                recursive=recursive,
                extensions=extensions,
                ignore_patterns=ignore_patterns,
                mandatory_substrings=mandatory_substrings,
                extract_patterns=extract_patterns,
                parse_dates=parse_dates
            )
            
            logger.info(
                f"Dataset discovery completed: found {len(result) if isinstance(result, (list, dict)) else 'unknown'} files/entries"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error during dataset file discovery for '{dataset_name}': {e}")
            raise


# Global discovery engine instance for backward compatibility and standard usage
_default_discovery_engine = ConfigDiscoveryEngine()


def set_discovery_providers(
    path_provider: Optional[PathProvider] = None,
    file_discovery_provider: Optional[FileDiscoveryProvider] = None,
    config_provider: Optional[ConfigurationProvider] = None
) -> None:
    """
    Test-specific entry point for configuring discovery providers.
    
    This function enables comprehensive mocking scenarios for pytest.monkeypatch
    and dependency injection testing patterns per TST-REF-003 requirements.
    
    Args:
        path_provider: Custom path provider for filesystem operations
        file_discovery_provider: Custom file discovery provider
        config_provider: Custom configuration provider
        
    Note:
        This function is intended for testing purposes and should not be used
        in production code. Providers will persist until explicitly reset.
    """
    global _default_discovery_engine
    
    logger.debug("Setting custom discovery providers for testing")
    _default_discovery_engine = ConfigDiscoveryEngine(
        path_provider=path_provider,
        file_discovery_provider=file_discovery_provider,
        config_provider=config_provider
    )


def reset_discovery_providers() -> None:
    """
    Reset discovery providers to default implementations.
    
    This function restores the default behavior after testing scenarios
    to ensure clean state between test runs.
    """
    global _default_discovery_engine
    
    logger.debug("Resetting discovery providers to defaults")
    _default_discovery_engine = ConfigDiscoveryEngine()


def get_discovery_engine() -> ConfigDiscoveryEngine:
    """
    Get the current discovery engine instance.
    
    This provides access to the discovery engine for advanced usage scenarios
    and testing purposes.
    
    Returns:
        The current ConfigDiscoveryEngine instance
    """
    return _default_discovery_engine


# Backward-compatible module-level functions with enhanced implementation and Pydantic model support
def discover_files_with_config(
    config: Union[Dict[str, Any], ExperimentConfig],
    directory: Union[str, List[str]],
    pattern: str,
    recursive: bool = False,
    extensions: Optional[List[str]] = None,
    experiment: Optional[str] = None,
    extract_metadata: bool = False,
    parse_dates: bool = False
) -> Union[List[str], Dict[str, Dict[str, Any]]]:
    """
    Discover files using configuration-aware filtering with Pydantic model support.
    
    This is a backward-compatible wrapper around the enhanced ConfigDiscoveryEngine
    implementation, providing the same interface while supporting dependency injection
    and new Pydantic model-based configurations for improved type safety.
    
    Args:
        config: The loaded configuration (dictionary, ExperimentConfig, or Kedro-style parameters)
        directory: The directory or list of directories to search in
        pattern: File pattern to match (glob format)
        recursive: If True, search recursively through subdirectories
        extensions: Optional list of file extensions to filter by (without the dot)
        experiment: Optional experiment name to use experiment-specific filters
        extract_metadata: If True, extract metadata using patterns from config
        parse_dates: If True, attempt to parse dates from filenames
        
    Returns:
        If extract_metadata or parse_dates is True: Dictionary mapping file paths to metadata
        Otherwise: List of file paths matching the criteria
    """
    return _default_discovery_engine.discover_files_with_config(
        config=config,
        directory=directory,
        pattern=pattern,
        recursive=recursive,
        extensions=extensions,
        experiment=experiment,
        extract_metadata=extract_metadata,
        parse_dates=parse_dates
    )


def discover_experiment_files(
    config: Union[Dict[str, Any], ExperimentConfig],
    experiment_name: str,
    base_directory: Union[str, Path],
    pattern: str = "*.*",
    recursive: bool = True,
    extensions: Optional[List[str]] = None,
    extract_metadata: bool = False,
    parse_dates: bool = False
) -> Union[List[str], Dict[str, Dict[str, Any]]]:
    """
    Discover files related to a specific experiment with Pydantic model support.
    
    This is a backward-compatible wrapper around the enhanced ConfigDiscoveryEngine
    implementation, providing the same interface while supporting dependency injection
    and new Pydantic model-based configurations for improved type safety.
    
    Args:
        config: The loaded configuration (dictionary, ExperimentConfig, or Kedro-style parameters)
        experiment_name: Name of the experiment to use for discovery
        base_directory: Base directory to search in (often the major_data_directory)
        pattern: File pattern to match (glob format), defaults to all files
        recursive: If True, search recursively through subdirectories
        extensions: Optional list of file extensions to filter by
        extract_metadata: If True, extract metadata using patterns from config
        parse_dates: If True, attempt to parse dates from filenames
        
    Returns:
        If extract_metadata or parse_dates is True: Dictionary mapping file paths to metadata
        Otherwise: List of file paths relevant to the experiment
        
    Raises:
        KeyError: If the experiment does not exist in the configuration
    """
    return _default_discovery_engine.discover_experiment_files(
        config=config,
        experiment_name=experiment_name,
        base_directory=base_directory,
        pattern=pattern,
        recursive=recursive,
        extensions=extensions,
        extract_metadata=extract_metadata,
        parse_dates=parse_dates
    )


def discover_dataset_files(
    config: Union[Dict[str, Any], ExperimentConfig],
    dataset_name: str,
    base_directory: Union[str, Path],
    pattern: str = "*.*",
    recursive: bool = True,
    extensions: Optional[List[str]] = None,
    extract_metadata: bool = False,
    parse_dates: bool = False
) -> Union[List[str], Dict[str, Dict[str, Any]]]:
    """
    Discover files related to a specific dataset with Pydantic model support.
    
    This is a backward-compatible wrapper around the enhanced ConfigDiscoveryEngine
    implementation, providing the same interface while supporting dependency injection
    and new Pydantic model-based configurations for improved type safety.
    
    Args:
        config: The loaded configuration (dictionary, ExperimentConfig, or Kedro-style parameters)
        dataset_name: Name of the dataset to use for discovery
        base_directory: Base directory to search in (often the major_data_directory)
        pattern: File pattern to match (glob format), defaults to all files
        recursive: If True, search recursively through subdirectories
        extensions: Optional list of file extensions to filter by
        extract_metadata: If True, extract metadata using patterns from config
        parse_dates: If True, attempt to parse dates from filenames
        
    Returns:
        If extract_metadata or parse_dates is True: Dictionary mapping file paths to metadata
        Otherwise: List of file paths relevant to the dataset
        
    Raises:
        KeyError: If the dataset does not exist in the configuration
    """
    return _default_discovery_engine.discover_dataset_files(
        config=config,
        dataset_name=dataset_name,
        base_directory=base_directory,
        pattern=pattern,
        recursive=recursive,
        extensions=extensions,
        extract_metadata=extract_metadata,
        parse_dates=parse_dates
    )