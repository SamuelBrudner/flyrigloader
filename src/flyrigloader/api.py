"""
High-level API for flyrigloader.

This module provides simple entry points for external projects to use flyrigloader
functionality without having to directly import from multiple submodules.

Enhanced with comprehensive dependency injection patterns for improved testability
and configurable dependency providers supporting pytest.monkeypatch scenarios.
"""
from pathlib import Path
import copy
from typing import Dict, List, Any, Optional, Union, Protocol, Callable
from abc import ABC, abstractmethod
import pandas as pd
from flyrigloader.io.pickle import (
    read_pickle_any_format as _read_pickle_any_format,
)
from flyrigloader.io.transformers import (
    make_dataframe_from_config as _make_dataframe_from_config,
)
from flyrigloader.io.column_models import get_config_from_source as _get_config_from_source

# Re-export helpers for convenience
read_pickle_any_format = _read_pickle_any_format
make_dataframe_from_config = _make_dataframe_from_config
get_config_from_source = _get_config_from_source

__all__ = [
    "process_experiment_data",
    "read_pickle_any_format",
    "make_dataframe_from_config",
    "get_config_from_source",
    "FlyRigLoaderError",
]
from flyrigloader import logger


MISSING_DATA_DIR_ERROR = (
    "No data directory specified. Either provide base_directory parameter "
    "or ensure 'major_data_directory' is set in config."
)


class FlyRigLoaderError(Exception):
    """Custom exception for flyrigloader user-visible failures."""


class ConfigProvider(Protocol):
    """Protocol for configuration providers supporting dependency injection."""
    
    def load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from path."""
        ...
    
    def get_ignore_patterns(self, config: Dict[str, Any], experiment: Optional[str] = None) -> List[str]:
        """Get ignore patterns from configuration."""
        ...
    
    def get_mandatory_substrings(self, config: Dict[str, Any], experiment: Optional[str] = None) -> List[str]:
        """Get mandatory substrings from configuration."""
        ...
    
    def get_dataset_info(self, config: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
        """Get dataset information."""
        ...
    
    def get_experiment_info(self, config: Dict[str, Any], experiment_name: str) -> Dict[str, Any]:
        """Get experiment information."""
        ...


class DiscoveryProvider(Protocol):
    """Protocol for file discovery providers supporting dependency injection."""
    
    def discover_files_with_config(
        self,
        config: Dict[str, Any],
        directory: Union[str, List[str]],
        pattern: str,
        recursive: bool = False,
        extensions: Optional[List[str]] = None,
        experiment: Optional[str] = None,
        extract_metadata: bool = False,
        parse_dates: bool = False
    ) -> Union[List[str], Dict[str, Dict[str, Any]]]:
        """Discover files using configuration-aware filtering."""
        ...
    
    def discover_experiment_files(
        self,
        config: Dict[str, Any],
        experiment_name: str,
        base_directory: Union[str, Path],
        pattern: str = "*.*",
        recursive: bool = True,
        extensions: Optional[List[str]] = None,
        extract_metadata: bool = False,
        parse_dates: bool = False
    ) -> Union[List[str], Dict[str, Dict[str, Any]]]:
        """Discover files related to a specific experiment."""
        ...
    
    def discover_dataset_files(
        self,
        config: Dict[str, Any],
        dataset_name: str,
        base_directory: Union[str, Path],
        pattern: str = "*.*",
        recursive: bool = True,
        extensions: Optional[List[str]] = None,
        extract_metadata: bool = False,
        parse_dates: bool = False
    ) -> Union[List[str], Dict[str, Dict[str, Any]]]:
        """Discover files related to a specific dataset."""
        ...


class IOProvider(Protocol):
    """Protocol for I/O providers supporting dependency injection."""
    
    def read_pickle_any_format(self, path: Union[str, Path]) -> Any:
        """Read pickle files in any format."""
        ...
    
    def make_dataframe_from_config(
        self,
        exp_matrix: Dict[str, Any],
        config_source: Optional[Union[str, Path, Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Create DataFrame from experimental matrix using configuration."""
        ...
    
    def get_config_from_source(self, config_source: Optional[Union[str, Path, Dict[str, Any]]] = None) -> Any:
        """Get configuration from various sources."""
        ...


class UtilsProvider(Protocol):
    """Protocol for utility providers supporting dependency injection."""
    
    def get_file_stats(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Get file statistics."""
        ...
    
    def get_relative_path(self, path: Union[str, Path], base_dir: Union[str, Path]) -> Path:
        """Get relative path."""
        ...
    
    def get_absolute_path(self, path: Union[str, Path], base_dir: Union[str, Path]) -> Path:
        """Get absolute path."""
        ...
    
    def check_file_exists(self, path: Union[str, Path]) -> bool:
        """Check if file exists."""
        ...
    
    def ensure_directory_exists(self, path: Union[str, Path]) -> Path:
        """Ensure directory exists."""
        ...
    
    def find_common_base_directory(self, paths: List[Union[str, Path]]) -> Optional[Path]:
        """Find common base directory."""
        ...


class DefaultDependencyProvider:
    """Default implementation of dependency providers using actual modules."""
    
    def __init__(self):
        """Initialize with lazy imports to avoid circular dependencies."""
        self._config_module = None
        self._discovery_module = None
        self._io_module = None
        self._utils_module = None
        logger.debug("Initialized DefaultDependencyProvider with lazy loading")
    
    @property
    def config(self) -> ConfigProvider:
        """Get configuration provider with lazy loading."""
        if self._config_module is None:
            logger.debug("Loading configuration module dependencies")
            from flyrigloader.config import yaml_config as _yaml_config

            _load_config = _yaml_config.load_config
            _get_ignore_patterns = _yaml_config.get_ignore_patterns
            _get_mandatory_substrings = _yaml_config.get_mandatory_substrings
            _get_dataset_info = _yaml_config.get_dataset_info
            _get_experiment_info = _yaml_config.get_experiment_info
            
            class ConfigModule:
                load_config = staticmethod(_load_config)
                get_ignore_patterns = staticmethod(_get_ignore_patterns)
                get_mandatory_substrings = staticmethod(_get_mandatory_substrings)
                get_dataset_info = staticmethod(_get_dataset_info)
                get_experiment_info = staticmethod(_get_experiment_info)
            
            self._config_module = ConfigModule()
        return self._config_module
    
    @property
    def discovery(self) -> DiscoveryProvider:
        """Get discovery provider with lazy loading."""
        if self._discovery_module is None:
            logger.debug("Loading discovery module dependencies")
            import importlib
            discovery_mod = importlib.import_module("flyrigloader.config.discovery")

            class DiscoveryModule:
                discover_files_with_config = staticmethod(discovery_mod.discover_files_with_config)
                discover_experiment_files = staticmethod(discovery_mod.discover_experiment_files)
                discover_dataset_files = staticmethod(discovery_mod.discover_dataset_files)
            
            self._discovery_module = DiscoveryModule()
        return self._discovery_module
    
    @property
    def io(self) -> IOProvider:
        """Get I/O provider with lazy loading."""
        if self._io_module is None:
            logger.debug("Loading I/O module dependencies")
            from flyrigloader.io.pickle import (
                read_pickle_any_format,
                make_dataframe_from_config
            )
            # get_config_from_source already imported at module level
            
            class IOModule:
                read_pickle_any_format = staticmethod(read_pickle_any_format)
                make_dataframe_from_config = staticmethod(make_dataframe_from_config)
                get_config_from_source = staticmethod(get_config_from_source)
            
            self._io_module = IOModule()
        return self._io_module
    
    @property
    def utils(self) -> UtilsProvider:
        """Get utilities provider with lazy loading."""
        if self._utils_module is None:
            logger.debug("Loading utilities module dependencies")
            from flyrigloader.discovery.stats import get_file_stats
            from flyrigloader.utils.paths import (
                get_relative_path,
                get_absolute_path,
                check_file_exists,
                ensure_directory_exists,
                find_common_base_directory
            )
            
            class UtilsModule:
                get_file_stats = staticmethod(get_file_stats)
                get_relative_path = staticmethod(get_relative_path)
                get_absolute_path = staticmethod(get_absolute_path)
                check_file_exists = staticmethod(check_file_exists)
                ensure_directory_exists = staticmethod(ensure_directory_exists)
                find_common_base_directory = staticmethod(find_common_base_directory)
            
            self._utils_module = UtilsModule()
        return self._utils_module


# Global dependency provider instance with test override capability
_dependency_provider: DefaultDependencyProvider = DefaultDependencyProvider()


def set_dependency_provider(provider: DefaultDependencyProvider) -> None:
    """
    Set the global dependency provider for testing purposes.
    
    This function enables pytest.monkeypatch scenarios for comprehensive unit testing
    by allowing test code to inject mock dependencies.
    
    Args:
        provider: Dependency provider instance to use globally
        
    Examples:
        >>> # In tests, mock the entire provider
        >>> mock_provider = Mock(spec=DefaultDependencyProvider)
        >>> set_dependency_provider(mock_provider)
        >>> 
        >>> # Or patch specific methods using monkeypatch
        >>> def test_with_mocked_config(monkeypatch):
        ...     mock_config = Mock(spec=ConfigProvider)
        ...     mock_provider = DefaultDependencyProvider()
        ...     monkeypatch.setattr(mock_provider, 'config', mock_config)
        ...     set_dependency_provider(mock_provider)
    """
    global _dependency_provider
    logger.debug(f"Setting dependency provider to {type(provider).__name__}")
    _dependency_provider = provider


def get_dependency_provider() -> DefaultDependencyProvider:
    """
    Get the current global dependency provider.
    
    Returns:
        Current dependency provider instance
    """
    return _dependency_provider


def reset_dependency_provider() -> None:
    """Reset dependency provider to default implementation for test cleanup."""
    global _dependency_provider
    logger.debug("Resetting dependency provider to default")
    _dependency_provider = DefaultDependencyProvider()


def _validate_config_parameters(
    config_path: Optional[Union[str, Path]],
    config: Optional[Dict[str, Any]],
    operation_name: str
) -> None:
    """
    Validate that exactly one of config_path or config is provided.
    
    Args:
        config_path: Path to configuration file
        config: Pre-loaded configuration dictionary
        operation_name: Name of the operation for error messaging
        
    Raises:
        ValueError: If validation fails with detailed error message
    """
    logger.debug(f"Validating config parameters for {operation_name}")

    if config_path is None and config is None:
        _extracted_from__validate_config_parameters_20(
            operation_name,
            ": Either 'config_path' or 'config' must be provided, but both are None. Please provide either a path to a YAML configuration file or a pre-loaded configuration dictionary.",
        )
    if config_path is not None and config is not None:
        _extracted_from__validate_config_parameters_20(
            operation_name,
            ": Both 'config_path' and 'config' are provided, but only one is allowed. Please provide either a path to a configuration file OR a configuration dictionary, not both.",
        )
    logger.debug(f"Config parameter validation successful for {operation_name}")


# TODO Rename this here and in `_validate_config_parameters`
def _extracted_from__validate_config_parameters_20(operation_name, arg1):
    error_msg = f"Invalid configuration parameters for {operation_name}{arg1}"
    logger.error(error_msg)
    raise FlyRigLoaderError(error_msg)


def _load_and_validate_config(
    config_path: Optional[Union[str, Path]],
    config: Optional[Union[Dict[str, Any], Any]],
    operation_name: str,
    deps: Optional[DefaultDependencyProvider] = None
) -> Dict[str, Any]:
    """
    Load and validate configuration with enhanced error handling.
    
    Supports both dictionary configurations and LegacyConfigAdapter objects
    for backward compatibility with enhanced Pydantic-based configurations.
    
    Args:
        config_path: Path to configuration file
        config: Pre-loaded configuration dictionary or LegacyConfigAdapter
        operation_name: Name of the operation for error context
        deps: Dependency provider for testing injection
        
    Returns:
        Validated configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
        FileNotFoundError: If config file doesn't exist
    """
    if deps is None:
        deps = get_dependency_provider()
    
    logger.debug(f"Loading and validating config for {operation_name}")
    
    if config_path is not None:
        try:
            logger.info(f"Loading configuration from file: {config_path}")
            config_dict = deps.config.load_config(config_path)
            logger.debug(f"Successfully loaded config from {config_path}")
        except FileNotFoundError as e:
            error_msg = (
                f"Configuration file not found for {operation_name}: {config_path}. "
                "Please ensure the file exists and the path is correct."
            )
            logger.error(error_msg)
            raise FlyRigLoaderError(error_msg) from e
        except Exception as e:
            error_msg = (
                f"Failed to load configuration for {operation_name} from {config_path}: {e}. "
                "Please check the file format and syntax."
            )
            logger.error(error_msg)
            raise FlyRigLoaderError(error_msg) from e
    else:
        logger.debug(f"Using pre-loaded configuration for {operation_name}")
        config_dict = copy.deepcopy(config)
    
    # Check if config is a LegacyConfigAdapter (duck typing approach)
    if hasattr(config_dict, 'keys') and hasattr(config_dict, '__getitem__') and hasattr(config_dict, 'get'):
        # This supports both dict and LegacyConfigAdapter (which implements MutableMapping)
        logger.debug(f"Configuration is dict-like for {operation_name}")
        
        # For LegacyConfigAdapter, we need to convert to dict for backward compatibility
        if not isinstance(config_dict, dict):
            try:
                # Convert LegacyConfigAdapter to dictionary for internal use
                dict_config = {}
                for key in config_dict.keys():
                    dict_config[key] = config_dict[key]
                config_dict = dict_config
                logger.debug(f"Converted LegacyConfigAdapter to dictionary for {operation_name}")
            except Exception as e:
                error_msg = (
                    f"Failed to convert configuration to dictionary for {operation_name}: {e}. "
                    "Configuration must be convertible to dictionary structure."
                )
                logger.error(error_msg)
                raise FlyRigLoaderError(error_msg) from e
    elif not isinstance(config_dict, dict):
        error_msg = (
            f"Invalid configuration format for {operation_name}: "
            f"Expected dictionary or dict-like object, got {type(config_dict).__name__}. "
            "Configuration must be a valid dictionary structure or LegacyConfigAdapter."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)
    
    logger.debug(f"Configuration validation successful for {operation_name}")
    return config_dict


def _resolve_base_directory(
    config: Dict[str, Any], 
    base_directory: Optional[Union[str, Path]],
    operation_name: str
) -> Union[str, Path]:
    """
    Resolve base directory with enhanced validation and error messages.
    
    Args:
        config: Configuration dictionary
        base_directory: Optional override for base directory
        operation_name: Name of operation for error context
        
    Returns:
        Resolved base directory path
        
    Raises:
        ValueError: If no valid base directory can be resolved
    """
    logger.debug(f"Resolving base directory for {operation_name}")
    
    if base_directory is None:
        logger.debug("No base_directory provided, checking config for major_data_directory")
        base_directory = (
            config.get("project", {})
            .get("directories", {})
            .get("major_data_directory")
        )
        
        if base_directory:
            logger.info(f"Resolved base directory from config: {base_directory}")
        else:
            logger.debug("No major_data_directory found in config")

    if not base_directory:
        error_msg = (
            f"No data directory specified for {operation_name}. "
            "Either provide 'base_directory' parameter or ensure "
            "'project.directories.major_data_directory' is set in your configuration. "
            "The configuration should have the structure: "
            "project:\n  directories:\n    major_data_directory: /path/to/data"
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)
    
    # Convert to Path for validation
    base_path = Path(base_directory)
    if not base_path.exists():
        logger.warning(f"Base directory does not exist: {base_path}")
    
    logger.debug(f"Successfully resolved base directory: {base_directory}")
    return base_directory


# Import types for backward compatibility and direct usage when needed
from flyrigloader.io.column_models import (
    ColumnConfig,
    ColumnConfigDict,
    ColumnDimension,
    get_default_config_path,
    load_column_config
)

# Direct imports maintained for specific utility functions that don't need injection
from flyrigloader.discovery.files import discover_files


def load_experiment_files(
    config_path: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    experiment_name: str = "",
    base_directory: Optional[Union[str, Path]] = None,
    pattern: str = "*.*",
    recursive: bool = True,
    extensions: Optional[List[str]] = None,
    extract_metadata: bool = False,
    parse_dates: bool = False,
    _deps: Optional[DefaultDependencyProvider] = None
) -> Union[List[str], Dict[str, Dict[str, Any]]]:
    """
    High-level function to load files for a specific experiment with enhanced testability.
    
    This function supports comprehensive dependency injection for testing scenarios
    through the _deps parameter, enabling pytest.monkeypatch patterns.
    
    Args:
        config_path: Path to the YAML configuration file
        config: Pre-loaded configuration dictionary (can be a Kedro-style parameters dictionary)
        experiment_name: Name of the experiment to load files for
        base_directory: Optional override for the data directory (if not specified, uses config)
        pattern: File pattern to search for in glob format (e.g., "*.csv", "data_*.pkl")
        recursive: Whether to search recursively (defaults to True)
        extensions: Optional list of file extensions to filter by
        extract_metadata: If True, extract metadata from filenames using config patterns
        parse_dates: If True, attempt to parse dates from filenames
        _deps: Optional dependency provider for testing injection (internal parameter)
        
    Returns:
        If extract_metadata or parse_dates is True: Dictionary mapping file paths to metadata
        Otherwise: List of file paths for the experiment
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        KeyError: If the experiment doesn't exist in the config
        ValueError: If neither config_path nor config is provided, or if both are provided
    """
    operation_name = "load_experiment_files"
    
    # Initialize dependency provider for testability
    if _deps is None:
        _deps = get_dependency_provider()
    
    logger.info(f"Loading experiment files for experiment '{experiment_name}'")
    logger.debug(f"Parameters: pattern={pattern}, recursive={recursive}, "
                f"extensions={extensions}, extract_metadata={extract_metadata}, "
                f"parse_dates={parse_dates}")
    
    # Enhanced parameter validation with detailed error messages
    _validate_config_parameters(config_path, config, operation_name)
    
    # Validate experiment_name parameter
    if not experiment_name or not isinstance(experiment_name, str):
        error_msg = (
            f"Invalid experiment_name for {operation_name}: '{experiment_name}'. "
            "experiment_name must be a non-empty string representing the experiment identifier."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)
    
    # Load and validate configuration with enhanced error handling
    config_dict = _load_and_validate_config(config_path, config, operation_name, _deps)
    
    # Determine the data directory with enhanced validation
    base_directory = _resolve_base_directory(config_dict, base_directory, operation_name)
    
    logger.debug(f"Using base directory: {base_directory}")
    
    # Validate experiment exists in configuration
    try:
        experiment_info = _deps.config.get_experiment_info(config_dict, experiment_name)
        logger.debug(f"Found experiment configuration for '{experiment_name}'")
    except KeyError as e:
        available_experiments = list(config_dict.get("experiments", {}).keys())
        error_msg = (
            f"Experiment '{experiment_name}' not found in configuration. "
            f"Available experiments: {available_experiments}. "
            "Please check the experiment name and ensure it's defined in your configuration."
        )
        logger.error(error_msg)
        raise KeyError(error_msg) from e
    
    # Discover experiment files with dependency injection
    try:
        logger.debug(f"Discovering files for experiment '{experiment_name}'")
        result = _deps.discovery.discover_experiment_files(
            config=config_dict,
            experiment_name=experiment_name,
            base_directory=base_directory,
            pattern=pattern,
            recursive=recursive,
            extensions=extensions,
            extract_metadata=extract_metadata,
            parse_dates=parse_dates
        )
        
        file_count = len(result) if isinstance(result, (list, dict)) else 0
        logger.info(f"Successfully discovered {file_count} files for experiment '{experiment_name}'")
        return result
        
    except Exception as e:
        error_msg = (
            f"Failed to discover files for experiment '{experiment_name}': {e}. "
            "Please check the experiment configuration and data directory structure."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def load_dataset_files(
    config_path: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    dataset_name: str = "",
    base_directory: Optional[Union[str, Path]] = None,
    pattern: str = "*.*",
    recursive: bool = True,
    extensions: Optional[List[str]] = None,
    extract_metadata: bool = False,
    parse_dates: bool = False,
    _deps: Optional[DefaultDependencyProvider] = None
) -> Union[List[str], Dict[str, Dict[str, Any]]]:
    """
    High-level function to load files for a specific dataset with enhanced testability.
    
    This function supports comprehensive dependency injection for testing scenarios
    through the _deps parameter, enabling pytest.monkeypatch patterns.
    
    Args:
        config_path: Path to the YAML configuration file
        config: Pre-loaded configuration dictionary (can be a Kedro-style parameters dictionary)
        dataset_name: Name of the dataset to load files for
        base_directory: Optional override for the data directory (if not specified, uses config)
        pattern: File pattern to search for in glob format (e.g., "*.csv", "data_*.pkl")
        recursive: Whether to search recursively (defaults to True)
        extensions: Optional list of file extensions to filter by
        extract_metadata: If True, extract metadata from filenames using config patterns
        parse_dates: If True, attempt to parse dates from filenames
        _deps: Optional dependency provider for testing injection (internal parameter)
        
    Returns:
        If extract_metadata or parse_dates is True: Dictionary mapping file paths to metadata
        Otherwise: List of file paths for the dataset
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        KeyError: If the dataset doesn't exist in the config
        ValueError: If neither config_path nor config is provided, or if both are provided
    """
    operation_name = "load_dataset_files"
    
    # Initialize dependency provider for testability
    if _deps is None:
        _deps = get_dependency_provider()
    
    logger.info(f"Loading dataset files for dataset '{dataset_name}'")
    logger.debug(f"Parameters: pattern={pattern}, recursive={recursive}, "
                f"extensions={extensions}, extract_metadata={extract_metadata}, "
                f"parse_dates={parse_dates}")
    
    # Enhanced parameter validation with detailed error messages
    _validate_config_parameters(config_path, config, operation_name)
    
    # Validate dataset_name parameter
    if not dataset_name or not isinstance(dataset_name, str):
        error_msg = (
            f"Invalid dataset_name for {operation_name}: '{dataset_name}'. "
            "dataset_name must be a non-empty string representing the dataset identifier."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)
    
    # Load and validate configuration with enhanced error handling
    config_dict = _load_and_validate_config(config_path, config, operation_name, _deps)
    
    # Determine the data directory with enhanced validation
    base_directory = _resolve_base_directory(config_dict, base_directory, operation_name)
    
    logger.debug(f"Using base directory: {base_directory}")
    
    # Validate dataset exists in configuration
    try:
        dataset_info = _deps.config.get_dataset_info(config_dict, dataset_name)
        logger.debug(f"Found dataset configuration for '{dataset_name}'")
    except KeyError as e:
        available_datasets = list(config_dict.get("datasets", {}).keys())
        error_msg = (
            f"Dataset '{dataset_name}' not found in configuration. "
            f"Available datasets: {available_datasets}. "
            "Please check the dataset name and ensure it's defined in your configuration."
        )
        logger.error(error_msg)
        raise KeyError(error_msg) from e
    
    # Discover dataset files with dependency injection
    try:
        logger.debug(f"Discovering files for dataset '{dataset_name}'")
        result = _deps.discovery.discover_dataset_files(
            config=config_dict,
            dataset_name=dataset_name,
            base_directory=base_directory,
            pattern=pattern,
            recursive=recursive,
            extensions=extensions,
            extract_metadata=extract_metadata,
            parse_dates=parse_dates
        )
        
        file_count = len(result) if isinstance(result, (list, dict)) else 0
        logger.info(f"Successfully discovered {file_count} files for dataset '{dataset_name}'")
        return result
        
    except Exception as e:
        error_msg = (
            f"Failed to discover files for dataset '{dataset_name}': {e}. "
            "Please check the dataset configuration and data directory structure."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def get_experiment_parameters(
    config_path: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    experiment_name: str = "",
    _deps: Optional[DefaultDependencyProvider] = None
) -> Dict[str, Any]:
    """
    Get parameters for a specific experiment with enhanced testability.
    
    This function supports comprehensive dependency injection for testing scenarios
    through the _deps parameter, enabling pytest.monkeypatch patterns.
    
    Args:
        config_path: Path to the YAML configuration file
        config: Pre-loaded configuration dictionary (can be a Kedro-style parameters dictionary)
        experiment_name: Name of the experiment to get parameters for
        _deps: Optional dependency provider for testing injection (internal parameter)
        
    Returns:
        Dictionary containing experiment parameters
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        KeyError: If the experiment doesn't exist in the config
        ValueError: If neither config_path nor config is provided, or if both are provided
    """
    operation_name = "get_experiment_parameters"
    
    # Initialize dependency provider for testability
    if _deps is None:
        _deps = get_dependency_provider()
    
    logger.info(f"Getting parameters for experiment '{experiment_name}'")
    
    # Enhanced parameter validation with detailed error messages
    _validate_config_parameters(config_path, config, operation_name)
    
    # Validate experiment_name parameter
    if not experiment_name or not isinstance(experiment_name, str):
        error_msg = (
            f"Invalid experiment_name for {operation_name}: '{experiment_name}'. "
            "experiment_name must be a non-empty string representing the experiment identifier."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)
    
    # Load and validate configuration with enhanced error handling
    config_dict = _load_and_validate_config(config_path, config, operation_name, _deps)
    
    # Get experiment info with enhanced error handling
    try:
        experiment_info = _deps.config.get_experiment_info(config_dict, experiment_name)
        logger.debug(f"Retrieved experiment info for '{experiment_name}'")
    except KeyError as e:
        available_experiments = list(config_dict.get("experiments", {}).keys())
        error_msg = (
            f"Experiment '{experiment_name}' not found in configuration. "
            f"Available experiments: {available_experiments}. "
            "Please check the experiment name and ensure it's defined in your configuration."
        )
        logger.error(error_msg)
        raise KeyError(error_msg) from e
    
    # Extract parameters with logging
    parameters = experiment_info.get("parameters", {})
    param_count = len(parameters) if isinstance(parameters, dict) else 0
    logger.info(f"Retrieved {param_count} parameters for experiment '{experiment_name}'")
    logger.debug(f"Parameter keys: {list(parameters.keys()) if isinstance(parameters, dict) else 'N/A'}")
    
    return parameters


def get_dataset_parameters(
    config_path: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    dataset_name: str = "",
    _deps: Optional[DefaultDependencyProvider] = None
) -> Dict[str, Any]:
    """
    Get parameters for a specific dataset with enhanced testability.
    
    This function supports comprehensive dependency injection for testing scenarios
    through the _deps parameter, enabling pytest.monkeypatch patterns.
    
    Args:
        config_path: Path to the YAML configuration file
        config: Pre-loaded configuration dictionary (can be a Kedro-style parameters dictionary)
        dataset_name: Name of the dataset to get parameters for
        _deps: Optional dependency provider for testing injection (internal parameter)
        
    Returns:
        Dictionary containing dataset parameters
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        KeyError: If the dataset doesn't exist in the config
        ValueError: If neither config_path nor config is provided, or if both are provided
    """
    operation_name = "get_dataset_parameters"
    
    # Initialize dependency provider for testability
    if _deps is None:
        _deps = get_dependency_provider()
    
    logger.info(f"Getting parameters for dataset '{dataset_name}'")
    
    # Enhanced parameter validation with detailed error messages
    _validate_config_parameters(config_path, config, operation_name)
    
    # Validate dataset_name parameter
    if not dataset_name or not isinstance(dataset_name, str):
        error_msg = (
            f"Invalid dataset_name for {operation_name}: '{dataset_name}'. "
            "dataset_name must be a non-empty string representing the dataset identifier."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)
    
    # Load and validate configuration with enhanced error handling
    config_dict = _load_and_validate_config(config_path, config, operation_name, _deps)
    
    # Get dataset info with enhanced error handling
    try:
        dataset_info = _deps.config.get_dataset_info(config_dict, dataset_name)
        logger.debug(f"Retrieved dataset info for '{dataset_name}'")
    except KeyError as e:
        available_datasets = list(config_dict.get("datasets", {}).keys())
        error_msg = (
            f"Dataset '{dataset_name}' not found in configuration. "
            f"Available datasets: {available_datasets}. "
            "Please check the dataset name and ensure it's defined in your configuration."
        )
        logger.error(error_msg)
        raise KeyError(error_msg) from e
    
    # Extract parameters with logging
    parameters = dataset_info.get("parameters", {})
    param_count = len(parameters) if isinstance(parameters, dict) else 0
    logger.info(f"Retrieved {param_count} parameters for dataset '{dataset_name}'")
    logger.debug(f"Parameter keys: {list(parameters.keys()) if isinstance(parameters, dict) else 'N/A'}")
    
    return parameters


def process_experiment_data(
    data_path: Union[str, Path],
    *,
    column_config_path: Optional[Union[str, Path, Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    strict_schema: bool = False,
    _deps: Optional[DefaultDependencyProvider] = None
) -> pd.DataFrame:
    """
    Process experimental data and return a pandas DataFrame with a guaranteed `file_path` column that stores the absolute source path.
    
    This function supports comprehensive dependency injection for testing scenarios
    through the _deps parameter, enabling pytest.monkeypatch patterns.
    
    Args:
        data_path: Path to the pickle file containing experimental data
        column_config_path: Path to column configuration file, configuration dictionary,
                            or ColumnConfigDict instance. If None, uses default configuration.
        metadata: Optional dictionary of metadata to add to the DataFrame
        strict_schema: If True, drop any columns not present in the provided column
            configuration. Requires that ``column_config_path`` is supplied. This
            option is useful for downstream pipelines that rely on a strict
            schema definition (e.g. Kedro parameters).
        _deps: Optional dependency provider for testing injection (internal parameter)
        
    Returns:
        pd.DataFrame: Processed experimental data. Always contains a `file_path` column with the absolute path to the source pickle.
        
    Raises:
        FileNotFoundError: If the data or config file doesn't exist
        ValueError: If required columns are missing from the data or path is invalid
    """
    operation_name = "process_experiment_data"

    # Initialize dependency provider for testability
    if _deps is None:
        _deps = get_dependency_provider()

    logger.info(f"Processing experimental data from: {data_path}")

    # Validate data_path parameter
    if not data_path:
        error_msg = (
            f"Invalid data_path for {operation_name}: '{data_path}'. "
            "data_path must be a non-empty string or Path object pointing to the data file."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)

    # Convert to Path for validation
    data_path_obj = Path(data_path)
    if not data_path_obj.exists():
        error_msg = (
            f"Data file not found for {operation_name}: {data_path}. "
            "Please ensure the file exists and the path is correct."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)

    # Log metadata information
    if metadata:
        logger.debug(f"Processing with metadata: {list(metadata.keys())}")
    else:
        logger.debug("Processing without additional metadata")

    # Read the experimental data with enhanced error handling
    try:
        logger.debug(f"Reading experimental data from: {data_path}")
        exp_matrix = _deps.io.read_pickle_any_format(data_path)

        if not isinstance(exp_matrix, dict):
            logger.warning(f"Expected dictionary data structure, got {type(exp_matrix).__name__}")
        else:
            matrix_keys = list(exp_matrix.keys())
            logger.debug(f"Loaded experimental matrix with keys: {matrix_keys}")

    except Exception as e:
        error_msg = (
            f"Failed to read experimental data for {operation_name} from {data_path}: {e}. "
            "Please check the file format and ensure it's a valid pickle file."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg) from e

    # Create DataFrame using column configuration with enhanced error handling
    try:
        logger.debug("Creating DataFrame from experimental matrix")
        result = _deps.io.make_dataframe_from_config(
            exp_matrix=exp_matrix,
            config_source=column_config_path,
            metadata=metadata
        )

        # Ensure we return a pandas DataFrame
        if isinstance(result, dict):
            df = pd.DataFrame(result)
        else:
            df = result if isinstance(result, pd.DataFrame) else pd.DataFrame(result)

        # Add absolute file path column for downstream joins
        df["file_path"] = str(data_path_obj.resolve())

        logger.info(f"Successfully created DataFrame with shape: {df.shape}")
        logger.debug(f"DataFrame columns: {list(df.columns)}")

        # Optional strict-schema filtering
        if strict_schema:
            if column_config_path is None:
                raise FlyRigLoaderError(
                    "strict_schema=True requires a column_config_path (schema) to be provided"
                )
            try:
                schema_model = _get_config_from_source(column_config_path)
            except Exception as e:
                raise FlyRigLoaderError(
                    f"Failed to load column configuration for strict schema filtering: {e}"
                ) from e
            allowed_cols = set(schema_model.columns.keys())
            if extra_cols := [c for c in df.columns if c not in allowed_cols]:
                logger.debug(
                    "Dropping %d columns not in schema: %s",
                    len(extra_cols),
                    ", ".join(extra_cols),
                )
                df = df[list(allowed_cols & set(df.columns))]
        return df

    except Exception as e:
        error_msg = (
            f"Failed to create DataFrame for {operation_name}: {e}. "
            "Please check the column configuration and data structure compatibility."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg) from e


def get_default_column_config(
    _deps: Optional[DefaultDependencyProvider] = None
) -> ColumnConfigDict:
    """
    Get the default column configuration with enhanced testability.
    
    This function supports comprehensive dependency injection for testing scenarios
    through the _deps parameter, enabling pytest.monkeypatch patterns.
    
    Args:
        _deps: Optional dependency provider for testing injection (internal parameter)
    
    Returns:
        ColumnConfigDict with the default configuration
        
    Raises:
        FileNotFoundError: If the default configuration file doesn't exist
        ValueError: If the configuration is invalid
    """
    operation_name = "get_default_column_config"
    
    # Initialize dependency provider for testability
    if _deps is None:
        _deps = get_dependency_provider()
    
    logger.info("Loading default column configuration")
    
    try:
        # Load the default configuration with dependency injection
        result = _deps.io.get_config_from_source(None)
        
        if hasattr(result, 'columns'):
            column_count = len(result.columns) if hasattr(result.columns, '__len__') else 0
            logger.info(f"Successfully loaded default configuration with {column_count} columns")
            logger.debug(f"Column names: {list(result.columns.keys()) if hasattr(result.columns, 'keys') else 'N/A'}")
        else:
            logger.info(f"Successfully loaded default configuration, type: {type(result).__name__}")
        
        return result
        
    except Exception as e:
        error_msg = (
            f"Failed to load default column configuration for {operation_name}: {e}. "
            "Please ensure the default configuration file exists and is valid."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg) from e


#
# File and path utilities
#
# Note: For standard path operations, consider using Python's pathlib directly:
#  - Path.name - Get filename without directory (instead of get_file_name)
#  - Path.suffix - Get file extension with dot (instead of get_extension)
#  - Path.parent - Get parent directory (instead of get_parent_dir)
#  - Path.resolve() - Normalize path (instead of normalize_file_path)
#

def get_file_statistics(
    path: Union[str, Path],
    _deps: Optional[DefaultDependencyProvider] = None
) -> Dict[str, Any]:
    """
    Get comprehensive statistics for a file with enhanced testability.
    
    This function supports comprehensive dependency injection for testing scenarios
    through the _deps parameter, enabling pytest.monkeypatch patterns.
    
    Args:
        path: Path to the file
        _deps: Optional dependency provider for testing injection (internal parameter)
        
    Returns:
        Dictionary containing file statistics (size, modification time, etc.)
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the path is invalid
    """
    operation_name = "get_file_statistics"
    
    # Initialize dependency provider for testability
    if _deps is None:
        _deps = get_dependency_provider()
    
    logger.debug(f"Getting file statistics for: {path}")
    
    # Validate path parameter
    if not path:
        error_msg = (
            f"Invalid path for {operation_name}: '{path}'. "
            "path must be a non-empty string or Path object."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)
    
    try:
        result = _deps.utils.get_file_stats(path)
        logger.debug(f"Successfully retrieved file statistics for: {path}")
        return result
    except FileNotFoundError as e:
        error_msg = (
            f"File not found for {operation_name}: {path}. "
            "Please ensure the file exists and the path is correct."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg) from e
    except Exception as e:
        error_msg = (
            f"Failed to get file statistics for {operation_name} from {path}: {e}. "
            "Please check the file path and permissions."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg) from e


def ensure_dir_exists(
    path: Union[str, Path],
    _deps: Optional[DefaultDependencyProvider] = None
) -> Path:
    """
    Ensure a directory exists, creating it if necessary, with enhanced testability.
    
    Args:
        path: Directory path
        _deps: Optional dependency provider for testing injection (internal parameter)
        
    Returns:
        Path object pointing to the directory
        
    Raises:
        ValueError: If the path is invalid
    """
    operation_name = "ensure_dir_exists"
    
    # Initialize dependency provider for testability
    if _deps is None:
        _deps = get_dependency_provider()
    
    logger.debug(f"Ensuring directory exists: {path}")
    
    # Validate path parameter
    if not path:
        error_msg = (
            f"Invalid path for {operation_name}: '{path}'. "
            "path must be a non-empty string or Path object."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)
    
    try:
        result = _deps.utils.ensure_directory_exists(path)
        logger.debug(f"Successfully ensured directory exists: {result}")
        return result
    except Exception as e:
        error_msg = (
            f"Failed to ensure directory exists for {operation_name} at {path}: {e}. "
            "Please check the path and permissions."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg) from e


def check_if_file_exists(
    path: Union[str, Path],
    _deps: Optional[DefaultDependencyProvider] = None
) -> bool:
    """
    Check if a file exists with enhanced testability.
    
    Args:
        path: Path to check
        _deps: Optional dependency provider for testing injection (internal parameter)
        
    Returns:
        True if the file exists, False otherwise
        
    Raises:
        ValueError: If the path is invalid
    """
    operation_name = "check_if_file_exists"
    
    # Initialize dependency provider for testability
    if _deps is None:
        _deps = get_dependency_provider()
    
    logger.debug(f"Checking if file exists: {path}")
    
    # Validate path parameter
    if not path:
        error_msg = (
            f"Invalid path for {operation_name}: '{path}'. "
            "path must be a non-empty string or Path object."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)
    
    try:
        result = _deps.utils.check_file_exists(path)
        logger.debug(f"File existence check for {path}: {result}")
        return result
    except Exception as e:
        error_msg = (
            f"Failed to check file existence for {operation_name} at {path}: {e}. "
            "Please check the path format."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg) from e


def get_path_relative_to(
    path: Union[str, Path], 
    base_dir: Union[str, Path],
    _deps: Optional[DefaultDependencyProvider] = None
) -> Path:
    """
    Get a path relative to a base directory with enhanced testability.
    
    Args:
        path: Path to convert
        base_dir: Base directory
        _deps: Optional dependency provider for testing injection (internal parameter)
        
    Returns:
        Relative path
        
    Raises:
        ValueError: If the path is not within the base directory or parameters are invalid
    """
    operation_name = "get_path_relative_to"

    # Initialize dependency provider for testability
    if _deps is None:
        _deps = get_dependency_provider()

    logger.debug(f"Getting relative path: {path} relative to {base_dir}")

    # Validate parameters
    if not path:
        _extracted_from_get_path_relative_to_30(
            'Invalid path for ',
            operation_name,
            path,
            "'. path must be a non-empty string or Path object.",
        )
    if not base_dir:
        _extracted_from_get_path_relative_to_30(
            'Invalid base_dir for ',
            operation_name,
            base_dir,
            "'. base_dir must be a non-empty string or Path object.",
        )
    try:
        result = _deps.utils.get_relative_path(path, base_dir)
        logger.debug(f"Relative path result: {result}")
        return result
    except ValueError as e:
        error_msg = (
            f"Path {path} is not within base directory {base_dir} for {operation_name}. "
            "Please ensure the path is within the specified base directory."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg) from e
    except Exception as e:
        error_msg = (
            f"Failed to get relative path for {operation_name}: {e}. "
            "Please check the path formats."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg) from e


# TODO Rename this here and in `get_path_relative_to`
def _extracted_from_get_path_relative_to_30(arg0, operation_name, arg2, arg3):
    error_msg = f"{arg0}{operation_name}: '{arg2}{arg3}"
    logger.error(error_msg)
    raise FlyRigLoaderError(error_msg)


def get_path_absolute(
    path: Union[str, Path], 
    base_dir: Union[str, Path],
    _deps: Optional[DefaultDependencyProvider] = None
) -> Path:
    """
    Convert a relative path to an absolute path with enhanced testability.
    
    Args:
        path: Relative path
        base_dir: Base directory
        _deps: Optional dependency provider for testing injection (internal parameter)
        
    Returns:
        Absolute path
        
    Raises:
        ValueError: If parameters are invalid
    """
    operation_name = "get_path_absolute"

    # Initialize dependency provider for testability
    if _deps is None:
        _deps = get_dependency_provider()

    logger.debug(f"Getting absolute path: {path} with base {base_dir}")

    # Validate parameters
    if not path:
        _extracted_from_get_path_absolute_30(
            'Invalid path for ',
            operation_name,
            path,
            "'. path must be a non-empty string or Path object.",
        )
    if not base_dir:
        _extracted_from_get_path_absolute_30(
            'Invalid base_dir for ',
            operation_name,
            base_dir,
            "'. base_dir must be a non-empty string or Path object.",
        )
    try:
        result = _deps.utils.get_absolute_path(path, base_dir)
        logger.debug(f"Absolute path result: {result}")
        return result
    except Exception as e:
        error_msg = (
            f"Failed to get absolute path for {operation_name}: {e}. "
            "Please check the path formats."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg) from e


# TODO Rename this here and in `get_path_absolute`
def _extracted_from_get_path_absolute_30(arg0, operation_name, arg2, arg3):
    error_msg = f"{arg0}{operation_name}: '{arg2}{arg3}"
    logger.error(error_msg)
    raise FlyRigLoaderError(error_msg)


def get_common_base_dir(
    paths: List[Union[str, Path]],
    _deps: Optional[DefaultDependencyProvider] = None
) -> Optional[Path]:
    """
    Find the common base directory for a list of paths with enhanced testability.
    
    Args:
        paths: List of paths to analyze
        _deps: Optional dependency provider for testing injection (internal parameter)
        
    Returns:
        Common base directory or None if no common base can be found
        
    Raises:
        ValueError: If the paths parameter is invalid
    """
    operation_name = "get_common_base_dir"
    
    # Initialize dependency provider for testability
    if _deps is None:
        _deps = get_dependency_provider()
    
    logger.debug(f"Finding common base directory for {len(paths) if paths else 0} paths")
    
    # Validate paths parameter
    if not isinstance(paths, list):
        error_msg = (
            f"Invalid paths parameter for {operation_name}: {type(paths).__name__}. "
            "paths must be a list of path strings or Path objects."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)
    
    if not paths:
        logger.debug("Empty paths list provided, returning None")
        return None
    
    # Validate individual paths
    for i, path in enumerate(paths):
        if not path:
            error_msg = (
                f"Invalid path at index {i} for {operation_name}: '{path}'. "
                "All paths must be non-empty strings or Path objects."
            )
            logger.error(error_msg)
            raise FlyRigLoaderError(error_msg)
    
    try:
        result = _deps.utils.find_common_base_directory(paths)
        if result:
            logger.debug(f"Found common base directory: {result}")
        else:
            logger.debug("No common base directory found")
        return result
    except Exception as e:
        error_msg = (
            f"Failed to find common base directory for {operation_name}: {e}. "
            "Please check the path formats."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg) from e


# Test-specific entry points for comprehensive testing scenarios

def _create_test_dependency_provider(
    config_provider: Optional[ConfigProvider] = None,
    discovery_provider: Optional[DiscoveryProvider] = None,
    io_provider: Optional[IOProvider] = None,
    utils_provider: Optional[UtilsProvider] = None
) -> DefaultDependencyProvider:
    """
    Create a test dependency provider with optional mock providers.
    
    This function supports comprehensive testing scenarios by allowing individual
    providers to be mocked while maintaining the overall dependency structure.
    
    Args:
        config_provider: Optional mock configuration provider
        discovery_provider: Optional mock discovery provider
        io_provider: Optional mock I/O provider
        utils_provider: Optional mock utilities provider
        
    Returns:
        DefaultDependencyProvider instance configured for testing
        
    Example:
        >>> from unittest.mock import Mock
        >>> mock_config = Mock(spec=ConfigProvider)
        >>> test_deps = _create_test_dependency_provider(config_provider=mock_config)
        >>> set_dependency_provider(test_deps)
    """
    logger.debug("Creating test dependency provider")
    
    test_provider = DefaultDependencyProvider()
    
    # Override individual providers if specified
    if config_provider is not None:
        test_provider._config_module = config_provider
        logger.debug("Injected custom config provider")
    if discovery_provider is not None:
        test_provider._discovery_module = discovery_provider
        logger.debug("Injected custom discovery provider")
    if io_provider is not None:
        test_provider._io_module = io_provider
        logger.debug("Injected custom I/O provider")
    if utils_provider is not None:
        test_provider._utils_module = utils_provider
        logger.debug("Injected custom utils provider")
    
    return test_provider