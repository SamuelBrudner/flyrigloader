"""
High-level API for flyrigloader.

This module provides simple entry points for external projects to use flyrigloader
functionality without having to directly import from multiple submodules.

Enhanced with comprehensive dependency injection patterns for improved testability
and configurable dependency providers supporting pytest.monkeypatch scenarios.

Implements the new decoupled data loading architecture with manifest-based discovery,
selective loading, and optional DataFrame transformation for improved memory usage
and scientific reproducibility through comprehensive audit logging.
"""
from pathlib import Path
import copy
import os
from typing import Dict, List, Any, Optional, Union, Protocol, Callable
from abc import ABC, abstractmethod
import pandas as pd
from flyrigloader.io.pickle import (
    read_pickle_any_format as _read_pickle_any_format,
)
from flyrigloader.io.transformers import (
    make_dataframe_from_config as _make_dataframe_from_config,
    transform_to_dataframe,
)
from flyrigloader.io.column_models import get_config_from_source as _get_config_from_source

# New imports for refactored architecture
from flyrigloader.config.models import LegacyConfigAdapter

# Re-export helpers for convenience
read_pickle_any_format = _read_pickle_any_format
make_dataframe_from_config = _make_dataframe_from_config
get_config_from_source = _get_config_from_source

__all__ = [
    # New decoupled architecture functions
    "discover_experiment_manifest",
    "load_data_file", 
    "transform_to_dataframe",
    # Existing API functions (maintained for backward compatibility)
    "load_experiment_files",
    "load_dataset_files",
    "process_experiment_data",
    "get_experiment_parameters",
    "get_dataset_parameters",
    "_resolve_base_directory",
    # Utility functions
    "read_pickle_any_format",
    "make_dataframe_from_config",
    "get_config_from_source",
    # Exception
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
    """
    Protocol for configuration providers supporting dependency injection.
    
    Updated to support both dictionary and Pydantic model configurations
    through LegacyConfigAdapter for backward compatibility during the
    configuration system transition.
    """
    
    def load_config(self, config_path: Union[str, Path]) -> Union[Dict[str, Any], LegacyConfigAdapter]:
        """
        Load configuration from path.
        
        Returns either a dictionary (legacy mode) or LegacyConfigAdapter (new mode)
        that wraps validated Pydantic models while providing dict-like access.
        """
        ...
    
    def get_ignore_patterns(self, config: Union[Dict[str, Any], LegacyConfigAdapter], experiment: Optional[str] = None) -> List[str]:
        """Get ignore patterns from configuration (supports both dict and LegacyConfigAdapter)."""
        ...
    
    def get_mandatory_substrings(self, config: Union[Dict[str, Any], LegacyConfigAdapter], experiment: Optional[str] = None) -> List[str]:
        """Get mandatory substrings from configuration (supports both dict and LegacyConfigAdapter)."""
        ...
    
    def get_dataset_info(self, config: Union[Dict[str, Any], LegacyConfigAdapter], dataset_name: str) -> Dict[str, Any]:
        """Get dataset information (supports both dict and LegacyConfigAdapter)."""
        ...
    
    def get_experiment_info(self, config: Union[Dict[str, Any], LegacyConfigAdapter], experiment_name: str) -> Dict[str, Any]:
        """Get experiment information (supports both dict and LegacyConfigAdapter)."""
        ...


class DiscoveryProvider(Protocol):
    """
    Protocol for file discovery providers supporting dependency injection.
    
    Updated to support both dictionary and Pydantic model configurations
    through LegacyConfigAdapter for enhanced validation and type safety.
    """
    
    def discover_files_with_config(
        self,
        config: Union[Dict[str, Any], LegacyConfigAdapter],
        directory: Union[str, List[str]],
        pattern: str,
        recursive: bool = False,
        extensions: Optional[List[str]] = None,
        experiment: Optional[str] = None,
        extract_metadata: bool = False,
        parse_dates: bool = False
    ) -> Union[List[str], Dict[str, Dict[str, Any]]]:
        """Discover files using configuration-aware filtering (supports both dict and LegacyConfigAdapter)."""
        ...
    
    def discover_experiment_files(
        self,
        config: Union[Dict[str, Any], LegacyConfigAdapter],
        experiment_name: str,
        base_directory: Union[str, Path],
        pattern: str = "*.*",
        recursive: bool = True,
        extensions: Optional[List[str]] = None,
        extract_metadata: bool = False,
        parse_dates: bool = False
    ) -> Union[List[str], Dict[str, Dict[str, Any]]]:
        """Discover files related to a specific experiment (supports both dict and LegacyConfigAdapter)."""
        ...
    
    def discover_dataset_files(
        self,
        config: Union[Dict[str, Any], LegacyConfigAdapter],
        dataset_name: str,
        base_directory: Union[str, Path],
        pattern: str = "*.*",
        recursive: bool = True,
        extensions: Optional[List[str]] = None,
        extract_metadata: bool = False,
        parse_dates: bool = False
    ) -> Union[List[str], Dict[str, Dict[str, Any]]]:
        """Discover files related to a specific dataset (supports both dict and LegacyConfigAdapter)."""
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
        _raise_config_validation_error(
            operation_name,
            ": Either 'config_path' or 'config' must be provided, but both are None. Please provide either a path to a YAML configuration file or a pre-loaded configuration dictionary.",
        )
    if config_path is not None and config is not None:
        _raise_config_validation_error(
            operation_name,
            ": Both 'config_path' and 'config' are provided, but only one is allowed. Please provide either a path to a configuration file OR a configuration dictionary, not both.",
        )
    logger.debug(f"Config parameter validation successful for {operation_name}")


def _raise_config_validation_error(operation_name: str, error_details: str) -> None:
    """
    Raise a configuration validation error with standardized formatting.
    
    This function centralizes error formatting for configuration validation failures
    and ensures consistent error messaging across the API.
    
    Args:
        operation_name: Name of the operation that failed validation
        error_details: Detailed error message explaining the validation failure
        
    Raises:
        FlyRigLoaderError: Always raises with formatted error message
    """
    error_msg = f"Invalid configuration parameters for {operation_name}{error_details}"
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
    config: Union[Dict[str, Any], LegacyConfigAdapter], 
    base_directory: Optional[Union[str, Path]],
    operation_name: str
) -> Union[str, Path]:
    """
    Resolve base directory following documented precedence order with comprehensive audit logging.
    
    Precedence order (first match wins):
    1. Explicit base_directory parameter (function argument)
    2. Configuration-defined major_data_directory 
    3. Environment variable FLYRIGLOADER_DATA_DIR (for CI/CD scenarios)
    
    Each resolution step is logged at appropriate levels for scientific reproducibility
    and debugging workflows. This ensures transparent data path resolution with
    clear audit trails.
    
    Args:
        config: Configuration dictionary or LegacyConfigAdapter
        base_directory: Optional explicit override for base directory (highest precedence)
        operation_name: Name of operation for error context and logging
        
    Returns:
        Resolved base directory path
        
    Raises:
        FlyRigLoaderError: If no valid base directory can be resolved
    """
    logger.debug(f"Starting base directory resolution for {operation_name}")
    logger.debug(f"Resolution precedence: 1) explicit parameter, 2) config major_data_directory, 3) FLYRIGLOADER_DATA_DIR env var")
    
    resolved_directory = None
    resolution_source = None
    
    # Precedence 1: Explicit base_directory parameter (highest priority)
    if base_directory is not None:
        resolved_directory = base_directory
        resolution_source = "explicit function parameter"
        logger.info(f"Using explicit base_directory parameter: {base_directory}")
        logger.debug(f"Resolution source: {resolution_source} (precedence 1)")
    
    # Precedence 2: Configuration-defined major_data_directory
    if resolved_directory is None:
        logger.debug("No explicit base_directory provided, checking config for major_data_directory")
        
        # Support both dict and LegacyConfigAdapter
        if hasattr(config, 'get'):
            # Dict-like access (works for both dict and LegacyConfigAdapter)
            config_directory = (
                config.get("project", {})
                .get("directories", {})
                .get("major_data_directory")
            )
        else:
            logger.warning(f"Unexpected config type for {operation_name}: {type(config)}")
            config_directory = None
        
        if config_directory:
            resolved_directory = config_directory
            resolution_source = "configuration major_data_directory"
            logger.info(f"Using major_data_directory from config: {config_directory}")
            logger.debug(f"Resolution source: {resolution_source} (precedence 2)")
        else:
            logger.debug("No major_data_directory found in config")
    
    # Precedence 3: Environment variable FLYRIGLOADER_DATA_DIR (lowest priority)
    if resolved_directory is None:
        logger.debug("No config major_data_directory found, checking FLYRIGLOADER_DATA_DIR environment variable")
        env_directory = os.environ.get("FLYRIGLOADER_DATA_DIR")
        
        if env_directory:
            resolved_directory = env_directory
            resolution_source = "FLYRIGLOADER_DATA_DIR environment variable"
            logger.info(f"Using FLYRIGLOADER_DATA_DIR environment variable: {env_directory}")
            logger.debug(f"Resolution source: {resolution_source} (precedence 3)")
        else:
            logger.debug("No FLYRIGLOADER_DATA_DIR environment variable found")

    # Validation: Ensure we have a resolved directory
    if not resolved_directory:
        error_msg = (
            f"No data directory specified for {operation_name}. "
            "Tried all resolution methods in precedence order:\n"
            "1. Explicit 'base_directory' parameter (not provided)\n"
            "2. Configuration 'project.directories.major_data_directory' (not found)\n"
            "3. Environment variable 'FLYRIGLOADER_DATA_DIR' (not set)\n\n"
            "Please provide a data directory using one of these methods:\n"
            "- Pass base_directory parameter to the function\n"
            "- Set 'major_data_directory' in your config:\n"
            "  project:\n    directories:\n      major_data_directory: /path/to/data\n"
            "- Set environment variable: export FLYRIGLOADER_DATA_DIR=/path/to/data"
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)
    
    # Convert to Path for validation and final logging
    resolved_path = Path(resolved_directory)
    
    # Existence check with appropriate logging level
    if resolved_path.exists():
        logger.debug(f"Resolved directory exists: {resolved_path}")
    else:
        logger.warning(f"Resolved directory does not exist: {resolved_path}")
        logger.warning(f"This may cause file discovery failures in {operation_name}")
    
    # Final audit log entry
    logger.info(f"✓ Base directory resolution complete for {operation_name}")
    logger.info(f"  Resolved path: {resolved_directory}")
    logger.info(f"  Resolution source: {resolution_source}")
    logger.debug(f"  Absolute path: {resolved_path.resolve()}")
    
    return resolved_directory


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


# ====================================================================================
# NEW DECOUPLED DATA LOADING ARCHITECTURE
# ====================================================================================
# The following functions implement the new manifest-based workflow that separates
# file discovery, data loading, and DataFrame transformation for improved memory usage,
# selective processing, and enhanced scientific reproducibility.


def discover_experiment_manifest(
    config_path: Optional[Union[str, Path]] = None,
    config: Optional[Union[Dict[str, Any], LegacyConfigAdapter]] = None,
    experiment_name: str = "",
    base_directory: Optional[Union[str, Path]] = None,
    pattern: str = "*.*",
    recursive: bool = True,
    extensions: Optional[List[str]] = None,
    extract_metadata: bool = True,
    parse_dates: bool = True,
    _deps: Optional[DefaultDependencyProvider] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Discover experiment files and return a comprehensive manifest without loading data.
    
    This function implements the first step of the new decoupled architecture, providing
    file discovery with metadata extraction but without data loading. This enables
    selective processing, better memory management, and manifest-based workflows.
    
    Args:
        config_path: Path to the YAML configuration file
        config: Pre-loaded configuration dictionary or LegacyConfigAdapter
        experiment_name: Name of the experiment to discover files for
        base_directory: Optional override for the data directory
        pattern: File pattern to search for in glob format (e.g., "*.pkl", "data_*.pkl")
        recursive: Whether to search recursively (defaults to True)
        extensions: Optional list of file extensions to filter by
        extract_metadata: Extract metadata from filenames using config patterns (default True)
        parse_dates: Attempt to parse dates from filenames (default True)
        _deps: Optional dependency provider for testing injection (internal parameter)
        
    Returns:
        Dictionary mapping file paths to metadata dictionaries containing:
        - 'path': Absolute path to the file
        - 'size': File size in bytes
        - 'modified': Last modification timestamp
        - 'metadata': Extracted metadata (if extract_metadata=True)
        - 'parsed_dates': Parsed date information (if parse_dates=True)
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        KeyError: If the experiment doesn't exist in the config
        ValueError: If neither config_path nor config is provided, or if both are provided
        
    Example:
        >>> manifest = discover_experiment_manifest(
        ...     config_path="config.yaml",
        ...     experiment_name="plume_navigation_analysis"
        ... )
        >>> print(f"Found {len(manifest)} files")
        >>> for file_path, metadata in manifest.items():
        ...     print(f"File: {file_path}, Size: {metadata['size']} bytes")
    """
    operation_name = "discover_experiment_manifest"
    
    # Initialize dependency provider for testability
    if _deps is None:
        _deps = get_dependency_provider()
    
    logger.info(f"🔍 Discovering experiment manifest for '{experiment_name}'")
    logger.debug(f"Discovery parameters: pattern={pattern}, recursive={recursive}, "
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
    
    # Determine the data directory with enhanced validation and logging
    base_directory = _resolve_base_directory(config_dict, base_directory, operation_name)
    
    # Validate experiment exists in configuration
    try:
        experiment_info = _deps.config.get_experiment_info(config_dict, experiment_name)
        logger.debug(f"Found experiment configuration for '{experiment_name}'")
        logger.debug(f"Experiment datasets: {experiment_info.get('datasets', [])}")
    except KeyError as e:
        available_experiments = list(config_dict.get("experiments", {}).keys())
        error_msg = (
            f"Experiment '{experiment_name}' not found in configuration. "
            f"Available experiments: {available_experiments}. "
            "Please check the experiment name and ensure it's defined in your configuration."
        )
        logger.error(error_msg)
        raise KeyError(error_msg) from e
    
    # Discover experiment files with comprehensive metadata
    try:
        logger.debug(f"Starting file discovery for experiment '{experiment_name}'")
        logger.debug(f"Using base directory: {base_directory}")
        
        manifest = _deps.discovery.discover_experiment_files(
            config=config_dict,
            experiment_name=experiment_name,
            base_directory=base_directory,
            pattern=pattern,
            recursive=recursive,
            extensions=extensions,
            extract_metadata=extract_metadata,
            parse_dates=parse_dates
        )
        
        # Ensure we return a dictionary with metadata (manifest format)
        if isinstance(manifest, list):
            # Convert simple file list to manifest format
            logger.debug("Converting file list to manifest format")
            manifest_dict = {}
            for file_path in manifest:
                manifest_dict[file_path] = {
                    'path': str(Path(file_path).resolve()),
                    'size': Path(file_path).stat().st_size if Path(file_path).exists() else 0,
                    'metadata': {},
                    'parsed_dates': {}
                }
            manifest = manifest_dict
        
        file_count = len(manifest)
        total_size = sum(item.get('size', 0) for item in manifest.values())
        logger.info(f"✓ Discovered {file_count} files for experiment '{experiment_name}'")
        logger.info(f"  Total data size: {total_size:,} bytes ({total_size / (1024**2):.1f} MB)")
        logger.debug(f"  Sample files: {list(manifest.keys())[:3]}{'...' if file_count > 3 else ''}")
        
        return manifest
        
    except Exception as e:
        error_msg = (
            f"Failed to discover experiment manifest for '{experiment_name}': {e}. "
            "Please check the experiment configuration and data directory structure."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def load_data_file(
    file_path: Union[str, Path],
    validate_format: bool = True,
    _deps: Optional[DefaultDependencyProvider] = None
) -> Dict[str, Any]:
    """
    Load raw data from a single file without DataFrame transformation.
    
    This function implements the second step of the new decoupled architecture,
    providing selective data loading from individual files. This enables
    memory-efficient processing of large datasets and selective analysis workflows.
    
    Args:
        file_path: Path to the data file to load
        validate_format: Whether to validate the loaded data format (default True)
        _deps: Optional dependency provider for testing injection (internal parameter)
        
    Returns:
        Dictionary containing raw experimental data from the file
        
    Raises:
        FileNotFoundError: If the data file doesn't exist
        ValueError: If the file format is invalid or data cannot be loaded
        
    Example:
        >>> # Load individual files from manifest
        >>> manifest = discover_experiment_manifest(...)
        >>> for file_path in manifest.keys():
        ...     raw_data = load_data_file(file_path)
        ...     print(f"Loaded {len(raw_data)} data columns from {file_path}")
    """
    operation_name = "load_data_file"
    
    # Initialize dependency provider for testability
    if _deps is None:
        _deps = get_dependency_provider()
    
    logger.debug(f"📁 Loading data file: {file_path}")
    
    # Validate file_path parameter
    if not file_path:
        error_msg = (
            f"Invalid file_path for {operation_name}: '{file_path}'. "
            "file_path must be a non-empty string or Path object pointing to the data file."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)
    
    # Convert to Path for validation
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        error_msg = (
            f"Data file not found for {operation_name}: {file_path}. "
            "Please ensure the file exists and the path is correct."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)
    
    # Load the raw data with enhanced error handling
    try:
        logger.debug(f"Reading raw data from: {file_path}")
        raw_data = _deps.io.read_pickle_any_format(file_path)
        
        # Validate format if requested
        if validate_format:
            if not isinstance(raw_data, dict):
                logger.warning(f"Expected dictionary data structure, got {type(raw_data).__name__}")
                if validate_format:
                    raise ValueError(f"Invalid data format: expected dict, got {type(raw_data).__name__}")
            else:
                data_keys = list(raw_data.keys())
                logger.debug(f"Loaded data with {len(data_keys)} columns: {data_keys}")
                
                # Basic validation for common required keys
                if 't' not in raw_data:
                    logger.warning("Time column 't' not found in data - may cause issues in downstream processing")
        
        file_size = file_path_obj.stat().st_size
        logger.debug(f"✓ Successfully loaded {len(raw_data) if isinstance(raw_data, dict) else 'N/A'} data columns from {file_path}")
        logger.debug(f"  File size: {file_size:,} bytes")
        
        return raw_data
        
    except Exception as e:
        error_msg = (
            f"Failed to load data from {file_path} for {operation_name}: {e}. "
            "Please check the file format and ensure it's a valid data file."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg) from e


def transform_to_dataframe(
    raw_data: Dict[str, Any],
    column_config_path: Optional[Union[str, Path, Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    add_file_path: bool = True,
    file_path: Optional[Union[str, Path]] = None,
    strict_schema: bool = False,
    _deps: Optional[DefaultDependencyProvider] = None
) -> pd.DataFrame:
    """
    Transform raw experimental data into a pandas DataFrame with optional configuration.
    
    This function implements the third step of the new decoupled architecture,
    providing optional DataFrame transformation from raw data. This enables
    selective processing and memory-efficient workflows where not all data
    needs to be converted to DataFrames.
    
    Args:
        raw_data: Dictionary containing raw experimental data
        column_config_path: Path to column configuration file, configuration dictionary,
                           or ColumnConfigDict instance. If None, uses default configuration.
        metadata: Optional dictionary of metadata to add to the DataFrame
        add_file_path: Whether to add a 'file_path' column with source file path (default True)
        file_path: Source file path to add to 'file_path' column (required if add_file_path=True)
        strict_schema: If True, drop any columns not present in the column configuration
        _deps: Optional dependency provider for testing injection (internal parameter)
        
    Returns:
        pd.DataFrame: Processed experimental data with configured columns and metadata
        
    Raises:
        ValueError: If required columns are missing from the data or configuration is invalid
        TypeError: If input types are invalid
        
    Example:
        >>> # Transform only selected files
        >>> manifest = discover_experiment_manifest(...)
        >>> dataframes = []
        >>> for file_path in list(manifest.keys())[:5]:  # Process only first 5 files
        ...     raw_data = load_data_file(file_path)
        ...     df = transform_to_dataframe(raw_data, file_path=file_path)
        ...     dataframes.append(df)
        >>> combined_df = pd.concat(dataframes, ignore_index=True)
    """
    operation_name = "transform_to_dataframe"
    
    # Initialize dependency provider for testability
    if _deps is None:
        _deps = get_dependency_provider()
    
    logger.debug(f"🔄 Transforming raw data to DataFrame")
    
    # Validate raw_data parameter
    if not isinstance(raw_data, dict):
        error_msg = (
            f"Invalid raw_data for {operation_name}: expected dict, got {type(raw_data).__name__}. "
            "raw_data must be a dictionary containing experimental data columns."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)
    
    if not raw_data:
        error_msg = (
            f"Empty raw_data for {operation_name}. "
            "raw_data must contain at least one data column."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)
    
    # Validate file_path requirement when add_file_path=True
    if add_file_path and not file_path:
        error_msg = (
            f"file_path parameter required when add_file_path=True for {operation_name}. "
            "Please provide the source file path or set add_file_path=False."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)
    
    # Log transformation details
    data_columns = list(raw_data.keys())
    logger.debug(f"Transforming {len(data_columns)} data columns: {data_columns}")
    if metadata:
        logger.debug(f"Adding metadata: {list(metadata.keys())}")
    
    # Use the imported transform_to_dataframe function from transformers module
    try:
        logger.debug("Calling DataFrame transformation utility")
        from flyrigloader.io.transformers import transform_to_dataframe as _transform_to_dataframe
        df = _transform_to_dataframe(
            exp_matrix=raw_data,
            config_source=column_config_path,
            metadata=metadata
        )
        
        # Add file path column if requested
        if add_file_path and file_path:
            file_path_obj = Path(file_path)
            df["file_path"] = str(file_path_obj.resolve())
            logger.debug(f"Added file_path column: {file_path}")
        
        # Apply strict schema filtering if requested
        if strict_schema:
            if column_config_path is None:
                raise FlyRigLoaderError(
                    "strict_schema=True requires a column_config_path (schema) to be provided"
                )
            try:
                schema_model = _get_config_from_source(column_config_path)
                allowed_cols = set(schema_model.columns.keys())
                if add_file_path:
                    allowed_cols.add("file_path")  # Always allow file_path column
                
                if extra_cols := [c for c in df.columns if c not in allowed_cols]:
                    logger.debug(f"Dropping {len(extra_cols)} columns not in schema: {extra_cols}")
                    df = df[list(allowed_cols & set(df.columns))]
            except Exception as e:
                raise FlyRigLoaderError(
                    f"Failed to load column configuration for strict schema filtering: {e}"
                ) from e
        
        logger.debug(f"✓ Successfully transformed to DataFrame with shape: {df.shape}")
        logger.debug(f"  DataFrame columns: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        error_msg = (
            f"Failed to transform raw data to DataFrame for {operation_name}: {e}. "
            "Please check the data structure and column configuration compatibility."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg) from e


# ====================================================================================
# EXISTING API FUNCTIONS (BACKWARD COMPATIBILITY)
# ====================================================================================
# The following functions maintain the existing API surface for backward compatibility
# while internally using the new decoupled architecture where appropriate.


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
    Process experimental data and return a pandas DataFrame with a guaranteed `file_path` column.
    
    This function maintains backward compatibility while internally using the new decoupled
    architecture (load_data_file + transform_to_dataframe). It provides the same API surface
    as before but with enhanced logging and validation.
    
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
        
    Example:
        >>> df = process_experiment_data("experiment_data.pkl")
        >>> print(f"Processed {len(df)} rows with columns: {list(df.columns)}")
    """
    operation_name = "process_experiment_data"

    # Initialize dependency provider for testability
    if _deps is None:
        _deps = get_dependency_provider()

    logger.info(f"📊 Processing experimental data from: {data_path}")
    logger.debug(f"Using backward-compatible API with new decoupled architecture")

    # Use the new decoupled architecture internally
    try:
        # Step 1: Load raw data (replaces direct pickle loading)
        logger.debug("Step 1: Loading raw data using decoupled architecture")
        raw_data = load_data_file(
            file_path=data_path,
            validate_format=True,
            _deps=_deps
        )
        
        # Step 2: Transform to DataFrame (replaces make_dataframe_from_config)
        logger.debug("Step 2: Transforming to DataFrame using decoupled architecture")
        df = transform_to_dataframe(
            raw_data=raw_data,
            column_config_path=column_config_path,
            metadata=metadata,
            add_file_path=True,
            file_path=data_path,
            strict_schema=strict_schema,
            _deps=_deps
        )
        
        logger.info(f"✓ Successfully processed experimental data")
        logger.info(f"  DataFrame shape: {df.shape}")
        logger.info(f"  Source file: {data_path}")
        logger.debug(f"  DataFrame columns: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        # Re-raise with original API error format for backward compatibility
        error_msg = (
            f"Failed to process experimental data from {data_path}: {e}. "
            "Please check the file format and column configuration compatibility."
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
        _raise_path_validation_error(
            'Invalid path for ',
            operation_name,
            path,
            "'. path must be a non-empty string or Path object.",
        )
    if not base_dir:
        _raise_path_validation_error(
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


def _raise_path_validation_error(prefix: str, operation_name: str, path_value: Any, suffix: str) -> None:
    """
    Raise a path validation error with standardized formatting.
    
    This function centralizes error formatting for path validation failures
    and ensures consistent error messaging across path utility functions.
    
    Args:
        prefix: Error message prefix
        operation_name: Name of the operation that failed validation
        path_value: The invalid path value
        suffix: Error message suffix
        
    Raises:
        FlyRigLoaderError: Always raises with formatted error message
    """
    error_msg = f"{prefix}{operation_name}: '{path_value}{suffix}"
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
        _raise_path_validation_error(
            'Invalid path for ',
            operation_name,
            path,
            "'. path must be a non-empty string or Path object.",
        )
    if not base_dir:
        _raise_path_validation_error(
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