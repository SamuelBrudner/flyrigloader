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
import importlib.util
from typing import Dict, List, Any, Optional, Union, Protocol, Callable
from collections.abc import MutableMapping
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
from flyrigloader.config.models import LegacyConfigAdapter, create_config
from flyrigloader.discovery.files import discover_experiment_manifest as _discover_experiment_manifest
from flyrigloader.io.loaders import load_data_file as _load_data_file
from flyrigloader.io.transformers import transform_to_dataframe as _transform_to_dataframe
from flyrigloader.exceptions import FlyRigLoaderError

# New imports for enhanced refactoring per Section 0.2.1
from flyrigloader.registries import get_loader_capabilities as _get_loader_capabilities
from flyrigloader.config.validators import validate_config_version
from flyrigloader.config.versioning import CURRENT_SCHEMA_VERSION
from semantic_version import Version

import warnings
import functools

_KEDRO_IMPORT_ERROR: Optional[ModuleNotFoundError] = None
if importlib.util.find_spec("kedro") is not None:
    try:  # pragma: no branch - executed once at import
        from flyrigloader.kedro.datasets import FlyRigLoaderDataSet
    except ModuleNotFoundError as exc:  # pragma: no cover - environment specific
        FlyRigLoaderDataSet = None  # type: ignore[assignment]
        _KEDRO_IMPORT_ERROR = exc
else:  # pragma: no cover - environment specific
    FlyRigLoaderDataSet = None  # type: ignore[assignment]
    _KEDRO_IMPORT_ERROR = ModuleNotFoundError(
        "kedro is not installed; FlyRigLoader Kedro integration is unavailable."
    )

# Re-export helpers for convenience
read_pickle_any_format = _read_pickle_any_format
make_dataframe_from_config = _make_dataframe_from_config
get_config_from_source = _get_config_from_source

__all__ = [
    # New decoupled architecture functions
    "discover_experiment_manifest",
    "load_data_file", 
    "transform_to_dataframe",
    # Configuration builders
    "create_config",
    # Enhanced refactoring functions per Section 0.2.1
    "validate_manifest",
    "create_kedro_dataset", 
    "get_registered_loaders",
    "get_loader_capabilities",
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


def _ensure_kedro_available() -> None:
    """Ensure optional Kedro integration is available before use."""

    if FlyRigLoaderDataSet is None:
        message = (
            "Kedro integration requires the 'kedro' package. "
            "Install flyrigloader with the 'kedro' extra or add kedro to your environment."
        )
        if _KEDRO_IMPORT_ERROR is not None:
            logger.error("Kedro integration unavailable: %s", _KEDRO_IMPORT_ERROR)
        raise FlyRigLoaderError(message) from _KEDRO_IMPORT_ERROR


if FlyRigLoaderDataSet is None:
    logger.warning(
        "FlyRigLoader Kedro integration disabled because 'kedro' could not be imported."
    )


MISSING_DATA_DIR_ERROR = (
    "No data directory specified. Either provide base_directory parameter "
    "or ensure 'major_data_directory' is set in config."
)


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
            from flyrigloader.io.pickle import read_pickle_any_format
            from flyrigloader.io.transformers import make_dataframe_from_config
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
            from flyrigloader.discovery.stats import get_file_stats as _get_file_stats
            from flyrigloader.utils.paths import (
                get_relative_path as _get_relative_path,
                get_absolute_path as _get_absolute_path,
                check_file_exists as _check_file_exists,
                ensure_directory_exists as _ensure_directory_exists,
                find_common_base_directory as _find_common_base_directory
            )
            
            class UtilsModule:
                get_file_stats = staticmethod(_get_file_stats)
                get_relative_path = staticmethod(_get_relative_path)
                get_absolute_path = staticmethod(_get_absolute_path)
                check_file_exists = staticmethod(_check_file_exists)
                ensure_directory_exists = staticmethod(_ensure_directory_exists)
                find_common_base_directory = staticmethod(_find_common_base_directory)
            
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


CONFIG_SOURCE_ERROR_MESSAGE = "Exactly one of 'config_path' or 'config' must be provided"


def _validate_config_parameters(
    config_path: Optional[Union[str, Path]],
    config: Optional[Dict[str, Any]],
    operation_name: str
) -> None:
    """Ensure callers provide exactly one configuration source."""
    logger.debug(f"Validating config parameters for {operation_name}")

    both_missing = config_path is None and config is None
    both_provided = config_path is not None and config is not None

    if both_missing or both_provided:
        if both_missing:
            logger.error(
                "Configuration source validation failed for %s: neither config nor config_path was provided",
                operation_name,
            )
        else:
            logger.error(
                "Configuration source validation failed for %s: both config and config_path were provided",
                operation_name,
            )
        raise ValueError(CONFIG_SOURCE_ERROR_MESSAGE)

    logger.debug(f"Config parameter validation successful for {operation_name}")


def _resolve_config_source(
    config: Optional[Union[Dict[str, Any], Any]],
    config_path: Optional[Union[str, Path]],
    operation_name: str,
    deps: Optional[DefaultDependencyProvider]
) -> Dict[str, Any]:
    """Return the concrete configuration dictionary for the requested operation."""
    _validate_config_parameters(config_path, config, operation_name)

    if config is not None:
        logger.debug(
            "Using provided configuration object for %s", operation_name
        )
        return config  # type: ignore[return-value]

    # At this point validation guarantees config_path is not None.
    assert config_path is not None
    logger.debug(
        "Loading configuration from file source %s for %s", config_path, operation_name
    )
    return _load_and_validate_config(config_path, None, operation_name, deps)


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


def _coerce_config_for_version_validation(config_obj: Any) -> Union[Dict[str, Any], str]:
    """Normalize configuration objects before schema version validation."""

    if isinstance(config_obj, (dict, str)):
        return config_obj

    if isinstance(config_obj, MutableMapping):
        logger.debug(
            "Converted MutableMapping configuration of type %s for version validation",
            type(config_obj).__name__,
        )
        return dict(config_obj)

    model_dump = getattr(config_obj, "model_dump", None)
    if callable(model_dump):
        try:
            dumped_config = model_dump()
            logger.debug(
                "Converted Pydantic model %s to dictionary via model_dump for version validation",
                type(config_obj).__name__,
            )
            return dumped_config
        except Exception as exc:
            logger.debug(
                "Failed to convert configuration %s using model_dump(): %s",
                type(config_obj).__name__,
                exc,
            )

    to_dict = getattr(config_obj, "to_dict", None)
    if callable(to_dict):
        try:
            dict_config = to_dict()
            logger.debug(
                "Converted configuration %s using to_dict() for version validation",
                type(config_obj).__name__,
            )
            return dict_config
        except Exception as exc:
            logger.debug(
                "Failed to convert configuration %s using to_dict(): %s",
                type(config_obj).__name__,
                exc,
            )

    raise TypeError(
        f"Configuration data must be dict-like or convertible, got {type(config_obj)}"
    )


def _attach_metadata_bucket(
    discovery_result: Union[List[str], Dict[str, Any]]
) -> Union[List[str], Dict[str, Dict[str, Any]]]:
    """Ensure discovery results provide a nested ``metadata`` dictionary."""

    if not isinstance(discovery_result, dict):
        return discovery_result

    normalised: Dict[str, Dict[str, Any]] = {}

    for path, payload in discovery_result.items():
        path_str = str(path)

        if not isinstance(payload, dict):
            normalised[path_str] = {"path": path_str, "metadata": {}}
            continue

        flattened = dict(payload)
        flattened.setdefault("path", path_str)

        existing_metadata = flattened.get("metadata")
        metadata_bucket: Dict[str, Any] = {}
        if isinstance(existing_metadata, dict):
            metadata_bucket.update(existing_metadata)

        for key, value in flattened.items():
            if key in {"metadata", "path"}:
                continue
            metadata_bucket.setdefault(key, value)

        flattened["metadata"] = metadata_bucket
        normalised[path_str] = flattened

    return normalised


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
    config: Optional[Union[Dict[str, Any], LegacyConfigAdapter, Any]] = None,
    experiment_name: str = "",
    config_path: Optional[Union[str, Path]] = None,
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
        config: Pre-loaded configuration dictionary, LegacyConfigAdapter, or Pydantic model
        experiment_name: Name of the experiment to discover files for
        config_path: Path to the YAML configuration file (alternative to config)
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
        >>> # Using Pydantic model directly
        >>> config = create_config(
        ...     project_name="fly_behavior",
        ...     base_directory="/data/experiments"
        ... )
        >>> manifest = discover_experiment_manifest(
        ...     config=config,
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
    
    # Validate experiment_name parameter
    if not experiment_name or not isinstance(experiment_name, str):
        error_msg = (
            f"Invalid experiment_name for {operation_name}: '{experiment_name}'. "
            "experiment_name must be a non-empty string representing the experiment identifier."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)
    
    config_dict = _resolve_config_source(config, config_path, operation_name, _deps)
    
    # Use the new decoupled discovery function
    try:
        logger.debug(f"Starting decoupled file discovery for experiment '{experiment_name}'")
        
        # Call the new discovery function from discovery/files.py
        file_manifest = _discover_experiment_manifest(
            config=config_dict,
            experiment_name=experiment_name,
            patterns=None,  # Use config patterns
            parse_dates=parse_dates,
            include_stats=extract_metadata,
            test_mode=False
        )
        
        # Convert FileManifest to dictionary format for backward compatibility
        manifest_dict = {}
        for file_info in file_manifest.files:
            manifest_dict[file_info.path] = {
                'path': file_info.path,
                'size': file_info.size or 0,
                'metadata': file_info.extracted_metadata if file_info.extracted_metadata is not None else {},
                'parsed_dates': {'parsed_date': file_info.parsed_date} if file_info.parsed_date else {}
            }
        
        file_count = len(manifest_dict)
        total_size = sum(item.get('size', 0) for item in manifest_dict.values())
        logger.info(f"✓ Discovered {file_count} files for experiment '{experiment_name}'")
        logger.info(f"  Total data size: {total_size:,} bytes ({total_size / (1024**2):.1f} MB)")
        logger.debug(f"  Sample files: {list(manifest_dict.keys())[:3]}{'...' if file_count > 3 else ''}")
        
        return manifest_dict
        
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
    loader: Optional[str] = None,
    _deps: Optional[DefaultDependencyProvider] = None
) -> Dict[str, Any]:
    """
    Load raw data from a single file without DataFrame transformation.
    
    This function implements the second step of the new decoupled architecture,
    providing selective data loading from individual files using the registry-based
    loader system. This enables memory-efficient processing of large datasets and
    selective analysis workflows.
    
    Args:
        file_path: Path to the data file to load
        validate_format: Whether to validate the loaded data format (default True)
        loader: Optional loader identifier for explicit loader selection
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
    
    # Use the new decoupled loading function
    try:
        logger.debug(f"Loading raw data using decoupled loader: {file_path}")
        raw_data = _load_data_file(file_path, loader)
        
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
        
        file_path_obj = Path(file_path)
        file_size = file_path_obj.stat().st_size if file_path_obj.exists() else 0
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
    
    # Use the new decoupled transformation function
    try:
        logger.debug("Calling DataFrame transformation utility from decoupled architecture")
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
# ENHANCED REFACTORING API FUNCTIONS (SECTION 0.2.1)
# ====================================================================================
# The following functions implement the enhanced refactoring requirements per Section 0.2.1
# providing comprehensive testing capabilities, Kedro integration, and registry introspection.


def validate_manifest(
    manifest: Dict[str, Dict[str, Any]],
    config: Optional[Union[Dict[str, Any], Any]] = None,
    config_path: Optional[Union[str, Path]] = None,
    strict_validation: bool = False,
    _deps: Optional[DefaultDependencyProvider] = None
) -> Dict[str, Any]:
    """
    Validate an experiment manifest for pre-flight validation without side effects.
    
    This function provides comprehensive testing capabilities by validating file manifests
    against configuration requirements and ensuring all files are accessible and properly
    formatted before data loading operations commence. This enables fail-fast validation
    for robust research workflows.
    
    Args:
        manifest: File manifest dictionary from discover_experiment_manifest()
        config: Pre-loaded configuration dictionary, Pydantic model, or LegacyConfigAdapter
        config_path: Path to the YAML configuration file (alternative to config)
        strict_validation: If True, perform comprehensive file existence and format checks
        _deps: Optional dependency provider for testing injection (internal parameter)
        
    Returns:
        Dict[str, Any]: Validation report containing:
        - 'valid': Boolean indicating overall validation success
        - 'file_count': Number of files validated
        - 'errors': List of validation errors encountered
        - 'warnings': List of validation warnings
        - 'metadata': Additional validation metadata
        
    Raises:
        ValueError: If manifest format is invalid or validation parameters are incorrect
        FlyRigLoaderError: For configuration-related validation failures
        
    Example:
        >>> manifest = discover_experiment_manifest(config, "exp1")
        >>> validation_report = validate_manifest(manifest, config)
        >>> if not validation_report['valid']:
        ...     print(f"Validation errors: {validation_report['errors']}")
        >>> else:
        ...     print(f"✓ {validation_report['file_count']} files validated successfully")
    """
    operation_name = "validate_manifest"
    
    # Initialize dependency provider for testability
    if _deps is None:
        _deps = get_dependency_provider()
    
    logger.info(f"🔍 Validating experiment manifest with {len(manifest)} files")
    logger.debug(f"Validation parameters: strict_validation={strict_validation}")
    
    # Validate manifest parameter
    if not isinstance(manifest, dict):
        error_msg = (
            f"Invalid manifest parameter for {operation_name}: expected dict, got {type(manifest).__name__}. "
            "manifest must be a dictionary from discover_experiment_manifest()."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)
    
    if not manifest:
        logger.warning("Empty manifest provided for validation")
        return {
            'valid': True,
            'file_count': 0,
            'errors': [],
            'warnings': ['Empty manifest - no files to validate'],
            'metadata': {'validation_type': 'empty_manifest'}
        }
    
    # Initialize validation report
    validation_report = {
        'valid': True,
        'file_count': len(manifest),
        'errors': [],
        'warnings': [],
        'metadata': {
            'validation_type': 'strict' if strict_validation else 'basic',
            'validated_files': [],
            'failed_files': [],
            'total_size_bytes': 0
        }
    }
    
    try:
        # Load configuration if needed for validation rules
        config_dict = None
        if config is not None or config_path is not None:
            try:
                if config is not None:
                    config_dict = config
                    logger.debug("Using provided configuration for validation rules")
                else:
                    config_dict = _load_and_validate_config(config_path, None, operation_name, _deps)
                    logger.debug(f"Loaded configuration from {config_path} for validation")
            except Exception as e:
                validation_report['warnings'].append(f"Failed to load configuration for validation: {e}")
                logger.warning(f"Configuration loading failed, proceeding with basic validation: {e}")
        
        # Validate each file in the manifest
        for file_path, file_metadata in manifest.items():
            try:
                logger.debug(f"Validating file: {file_path}")
                
                # Basic structure validation
                if not isinstance(file_metadata, dict):
                    error_msg = f"Invalid metadata for file {file_path}: expected dict, got {type(file_metadata).__name__}"
                    validation_report['errors'].append(error_msg)
                    validation_report['metadata']['failed_files'].append(file_path)
                    continue
                
                # Check required metadata fields
                required_fields = ['path', 'size']
                for field in required_fields:
                    if field not in file_metadata:
                        validation_report['warnings'].append(f"Missing metadata field '{field}' for file {file_path}")
                
                # File existence check if strict validation is enabled
                if strict_validation:
                    file_path_obj = Path(file_path)
                    if not file_path_obj.exists():
                        error_msg = f"File does not exist: {file_path}"
                        validation_report['errors'].append(error_msg)
                        validation_report['metadata']['failed_files'].append(file_path)
                        continue
                    
                    # Check file accessibility
                    if not file_path_obj.is_file():
                        error_msg = f"Path is not a regular file: {file_path}"
                        validation_report['errors'].append(error_msg)
                        validation_report['metadata']['failed_files'].append(file_path)
                        continue
                    
                    # Validate file size consistency
                    actual_size = file_path_obj.stat().st_size
                    reported_size = file_metadata.get('size', 0)
                    if abs(actual_size - reported_size) > 1024:  # Allow 1KB tolerance
                        validation_report['warnings'].append(
                            f"Size mismatch for {file_path}: reported {reported_size}, actual {actual_size}"
                        )
                
                # Accumulate total size
                file_size = file_metadata.get('size', 0)
                if isinstance(file_size, (int, float)):
                    validation_report['metadata']['total_size_bytes'] += file_size
                
                validation_report['metadata']['validated_files'].append(file_path)
                
            except Exception as e:
                error_msg = f"Validation failed for file {file_path}: {e}"
                validation_report['errors'].append(error_msg)
                validation_report['metadata']['failed_files'].append(file_path)
                logger.error(error_msg)
        
        # Check configuration compatibility if available
        if config_dict is not None:
            try:
                normalized_config = _coerce_config_for_version_validation(config_dict)
                is_valid, detected_version, message = validate_config_version(normalized_config)
                validation_report['metadata']['config_version'] = str(detected_version)
                logger.debug("Detected configuration version: %s", detected_version)

                current_version = Version(CURRENT_SCHEMA_VERSION)
                config_version = Version(str(detected_version))

                if not is_valid:
                    validation_report['errors'].append(message)
                elif config_version < current_version:
                    validation_report['warnings'].append(
                        f"Configuration version {detected_version} is older than supported version {CURRENT_SCHEMA_VERSION}."
                    )
                elif config_version > current_version:
                    validation_report['errors'].append(
                        f"Configuration version {detected_version} is newer than supported version {CURRENT_SCHEMA_VERSION}. "
                        "Please upgrade FlyRigLoader."
                    )
                
            except Exception as e:
                validation_report['warnings'].append(f"Configuration version validation failed: {e}")
                logger.warning(f"Configuration version validation error: {e}")
        
        # Determine overall validation status
        validation_report['valid'] = len(validation_report['errors']) == 0
        
        # Log validation summary
        if validation_report['valid']:
            logger.info(f"✓ Manifest validation successful: {validation_report['file_count']} files validated")
            if validation_report['warnings']:
                logger.info(f"  Warnings: {len(validation_report['warnings'])}")
        else:
            logger.error(f"✗ Manifest validation failed: {len(validation_report['errors'])} errors, {len(validation_report['warnings'])} warnings")
        
        logger.debug(f"Validation summary: {validation_report['metadata']['total_size_bytes']:,} bytes total")
        
        return validation_report
        
    except Exception as e:
        error_msg = (
            f"Manifest validation failed for {operation_name}: {e}. "
            "Please check the manifest structure and validation parameters."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg) from e


def create_kedro_dataset(
    config_path: Union[str, Path],
    experiment_name: str,
    *,
    recursive: bool = True,
    extract_metadata: bool = True,
    parse_dates: bool = True,
    dataset_options: Optional[Dict[str, Any]] = None,
    _deps: Optional[DefaultDependencyProvider] = None
) -> "FlyRigLoaderDataSet":
    """
    Factory function for creating Kedro dataset instances with proper lifecycle management.
    
    This function enables seamless Kedro catalog integration by providing a standardized
    factory method for FlyRigLoaderDataSet creation. It handles configuration validation,
    parameter normalization, and proper dataset initialization following Kedro best practices.
    
    Args:
        config_path: Path to the FlyRigLoader configuration file
        experiment_name: Name of the experiment to load data for
        recursive: Whether to search recursively in directories (default: True)
        extract_metadata: Whether to extract metadata from filenames (default: True)
        parse_dates: Whether to parse dates from filenames (default: True)
        dataset_options: Additional options to pass to the dataset constructor
        _deps: Optional dependency provider for testing injection (internal parameter)
        
    Returns:
        FlyRigLoaderDataSet: Configured Kedro dataset instance ready for catalog use
        
    Raises:
        ValueError: If parameters are invalid or configuration is missing
        FileNotFoundError: If configuration file doesn't exist
        FlyRigLoaderError: For configuration validation or dataset creation failures
        
    Example:
        >>> # For direct usage
        >>> dataset = create_kedro_dataset(
        ...     config_path="experiment_config.yaml",
        ...     experiment_name="baseline_study",
        ...     recursive=True,
        ...     extract_metadata=True
        ... )
        >>> 
        >>> # For Kedro catalog.yml integration
        >>> # my_experiment_data:
        >>> #   type: flyrigloader.api.create_kedro_dataset
        >>> #   config_path: "${base_dir}/config/experiment_config.yaml"
        >>> #   experiment_name: "baseline_study"
        >>> #   recursive: true
        >>> #   extract_metadata: true
    """
    _ensure_kedro_available()

    operation_name = "create_kedro_dataset"
    
    # Initialize dependency provider for testability
    if _deps is None:
        _deps = get_dependency_provider()
    
    logger.info(f"🏗️ Creating Kedro dataset for experiment '{experiment_name}'")
    logger.debug(f"Dataset parameters: config_path={config_path}, recursive={recursive}, "
                f"extract_metadata={extract_metadata}, parse_dates={parse_dates}")
    
    # Validate parameters
    if not config_path:
        error_msg = (
            f"Invalid config_path for {operation_name}: '{config_path}'. "
            "config_path must be a non-empty string or Path object pointing to the configuration file."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)
    
    if not experiment_name or not isinstance(experiment_name, str):
        error_msg = (
            f"Invalid experiment_name for {operation_name}: '{experiment_name}'. "
            "experiment_name must be a non-empty string representing the experiment identifier."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)
    
    # Validate configuration file exists
    config_path_obj = Path(config_path)
    if not config_path_obj.exists():
        error_msg = (
            f"Configuration file not found for {operation_name}: {config_path}. "
            "Please ensure the configuration file exists and the path is correct."
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        # Pre-validate configuration by loading it
        logger.debug("Pre-validating configuration for dataset creation")
        config_dict = _load_and_validate_config(str(config_path), None, operation_name, _deps)
        
        # Validate experiment exists in configuration
        if 'experiments' not in config_dict or not config_dict['experiments']:
            error_msg = (
                f"No experiments found in configuration for {operation_name}. "
                "Configuration must contain an 'experiments' section."
            )
            logger.error(error_msg)
            raise FlyRigLoaderError(error_msg)
        
        if experiment_name not in config_dict['experiments']:
            available_experiments = list(config_dict['experiments'].keys())
            error_msg = (
                f"Experiment '{experiment_name}' not found in configuration. "
                f"Available experiments: {available_experiments}. "
                "Please check the experiment name and ensure it's defined in your configuration."
            )
            logger.error(error_msg)
            raise FlyRigLoaderError(error_msg)
        
        # Prepare dataset options
        dataset_kwargs = {
            'config_path': str(config_path_obj.resolve()),
            'experiment_name': experiment_name,
            'recursive': recursive,
            'extract_metadata': extract_metadata,
            'parse_dates': parse_dates
        }
        
        # Add additional options if provided
        if dataset_options:
            logger.debug(f"Adding custom dataset options: {list(dataset_options.keys())}")
            dataset_kwargs.update(dataset_options)
        
        # Create the Kedro dataset instance
        logger.debug("Creating FlyRigLoaderDataSet instance")
        dataset = FlyRigLoaderDataSet(
            filepath=dataset_kwargs['config_path'],
            experiment_name=dataset_kwargs['experiment_name'],
            recursive=dataset_kwargs['recursive'],
            extract_metadata=dataset_kwargs['extract_metadata'],
            parse_dates=dataset_kwargs['parse_dates']
        )
        
        logger.info(f"✓ Successfully created Kedro dataset for experiment '{experiment_name}'")
        logger.debug(f"  Dataset configuration: {config_path}")
        logger.debug(f"  Dataset options: {list(dataset_kwargs.keys())}")
        
        return dataset
        
    except Exception as e:
        # Re-raise known exceptions as-is
        if isinstance(e, (ValueError, FileNotFoundError, FlyRigLoaderError)):
            raise
        
        # Wrap unexpected exceptions
        error_msg = (
            f"Failed to create Kedro dataset for {operation_name}: {e}. "
            "Please check the configuration file and experiment parameters."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg) from e


def get_registered_loaders(_deps: Optional[DefaultDependencyProvider] = None) -> Dict[str, Dict[str, Any]]:
    """
    Get comprehensive information about all registered file loaders for plugin discovery.
    
    This function provides registry introspection capabilities by returning detailed
    information about all registered file loaders, including their supported extensions,
    capabilities, and metadata. This enables plugin discovery and system introspection
    for debugging and development purposes.
    
    Args:
        _deps: Optional dependency provider for testing injection (internal parameter)
        
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping extensions to loader information:
        - 'loader_class': Name of the loader class
        - 'supported_extensions': List of file extensions handled
        - 'priority': Loader priority level
        - 'capabilities': Dictionary of loader capabilities
        - 'metadata': Additional loader metadata
        
    Example:
        >>> loaders = get_registered_loaders()
        >>> for ext, info in loaders.items():
        ...     print(f"Extension {ext}: {info['loader_class']}")
        ...     print(f"  Capabilities: {info['capabilities']}")
        >>> 
        >>> # Check if specific extension is supported
        >>> if '.pkl' in loaders:
        ...     print("Pickle files are supported")
    """
    operation_name = "get_registered_loaders"
    
    # Initialize dependency provider for testability
    if _deps is None:
        _deps = get_dependency_provider()
    
    logger.debug("🔍 Retrieving registered loader information")
    
    try:
        # Import registry functions locally to avoid circular imports
        from flyrigloader.registries import LoaderRegistry
        
        # Get the singleton registry instance
        registry = LoaderRegistry()
        
        # Build comprehensive loader information
        loader_info = {}
        
        # Get all registered loaders and extract extensions
        all_loaders = registry.get_all_loaders()
        registered_extensions = list(all_loaders.keys())
        logger.debug(f"Found {len(registered_extensions)} registered extensions")
        
        for extension in registered_extensions:
            try:
                # Get loader for this extension
                loader_class = registry.get_loader_for_extension(extension)
                
                # Get loader capabilities using the imported function
                capabilities = _get_loader_capabilities(extension)
                
                # Build comprehensive information
                loader_info[extension] = {
                    'loader_class': loader_class.__name__ if hasattr(loader_class, '__name__') else str(loader_class),
                    'supported_extensions': [extension],  # Primary extension
                    'priority': getattr(loader_class, 'priority', 'BUILTIN'),
                    'capabilities': capabilities,
                    'metadata': {
                        'module': getattr(loader_class, '__module__', 'unknown'),
                        'registered': True,
                        'extension_primary': extension
                    }
                }
                
                # Add additional supported extensions if available
                if hasattr(loader_class, 'supported_extensions'):
                    additional_extensions = [
                        ext for ext in loader_class.supported_extensions 
                        if ext != extension and ext in registered_extensions
                    ]
                    if additional_extensions:
                        loader_info[extension]['supported_extensions'].extend(additional_extensions)
                
                logger.debug(f"Retrieved information for loader: {extension}")
                
            except Exception as e:
                logger.warning(f"Failed to get information for extension {extension}: {e}")
                # Add minimal information for problematic loaders
                loader_info[extension] = {
                    'loader_class': 'unknown',
                    'supported_extensions': [extension],
                    'priority': 'unknown',
                    'capabilities': {},
                    'metadata': {
                        'error': str(e),
                        'registered': True,
                        'extension_primary': extension
                    }
                }
        
        logger.info(f"✓ Retrieved information for {len(loader_info)} registered loaders")
        logger.debug(f"  Extensions: {list(loader_info.keys())}")
        
        return loader_info
        
    except Exception as e:
        error_msg = (
            f"Failed to retrieve registered loaders for {operation_name}: {e}. "
            "Please check the registry system and loader registrations."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg) from e


def get_loader_capabilities(
    extension: Optional[str] = None,
    _deps: Optional[DefaultDependencyProvider] = None
) -> Dict[str, Any]:
    """
    Get detailed capability metadata for file loaders supporting plugin discovery.
    
    This function provides comprehensive capability introspection for file loaders,
    returning detailed metadata about loader features, performance characteristics,
    and compatibility information. This enables intelligent loader selection and
    system optimization based on data requirements.
    
    Args:
        extension: Specific file extension to query (e.g., '.pkl'). If None, returns
                  capabilities for all registered loaders.
        _deps: Optional dependency provider for testing injection (internal parameter)
        
    Returns:
        Dict[str, Any]: Capability information dictionary:
        - If extension specified: Returns capabilities for that specific loader
        - If extension is None: Returns dict mapping extensions to their capabilities
        
        Capability information includes:
        - 'streaming_support': Whether loader supports streaming large files
        - 'compression_support': List of supported compression formats
        - 'metadata_extraction': Whether loader can extract file metadata
        - 'performance_profile': Performance characteristics and benchmarks
        - 'memory_efficiency': Memory usage characteristics
        - 'thread_safety': Thread safety information
        
    Raises:
        ValueError: If specified extension is not registered
        FlyRigLoaderError: For registry access or capability retrieval failures
        
    Example:
        >>> # Get capabilities for specific extension
        >>> pkl_caps = get_loader_capabilities('.pkl')
        >>> if pkl_caps['streaming_support']:
        ...     print("Pickle loader supports streaming")
        >>> 
        >>> # Get all loader capabilities
        >>> all_caps = get_loader_capabilities()
        >>> for ext, caps in all_caps.items():
        ...     print(f"{ext}: streaming={caps['streaming_support']}")
    """
    operation_name = "get_loader_capabilities"
    
    # Initialize dependency provider for testability
    if _deps is None:
        _deps = get_dependency_provider()
    
    logger.debug(f"🔍 Retrieving loader capabilities for extension: {extension or 'all'}")
    
    try:
        # Single extension query
        if extension is not None:
            if not extension or not isinstance(extension, str):
                error_msg = (
                    f"Invalid extension for {operation_name}: '{extension}'. "
                    "extension must be a non-empty string (e.g., '.pkl')."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Get capabilities for specific extension using imported function
            capabilities = _get_loader_capabilities(extension)
            
            if not capabilities:
                error_msg = (
                    f"No loader registered for extension '{extension}'. "
                    "Please check the extension format and ensure a loader is registered."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            logger.debug(f"Retrieved capabilities for extension {extension}")
            return capabilities
        
        # All extensions query
        else:
            from flyrigloader.registries import LoaderRegistry
            
            registry = LoaderRegistry()
            all_loaders = registry.get_all_loaders()
            registered_extensions = list(all_loaders.keys())
            
            all_capabilities = {}
            for ext in registered_extensions:
                try:
                    capabilities = _get_loader_capabilities(ext)
                    all_capabilities[ext] = capabilities
                    logger.debug(f"Retrieved capabilities for extension {ext}")
                except Exception as e:
                    logger.warning(f"Failed to get capabilities for extension {ext}: {e}")
                    # Add minimal capability info for problematic loaders
                    all_capabilities[ext] = {
                        'streaming_support': False,
                        'compression_support': [],
                        'metadata_extraction': False,
                        'performance_profile': {'status': 'unknown'},
                        'memory_efficiency': {'rating': 'unknown'},
                        'thread_safety': {'safe': False},
                        'error': str(e)
                    }
            
            logger.info(f"✓ Retrieved capabilities for {len(all_capabilities)} loaders")
            return all_capabilities
            
    except Exception as e:
        # Re-raise known exceptions as-is
        if isinstance(e, ValueError):
            raise
        
        # Wrap unexpected exceptions
        error_msg = (
            f"Failed to retrieve loader capabilities for {operation_name}: {e}. "
            "Please check the registry system and loader implementations."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg) from e


# ====================================================================================
# EXISTING API FUNCTIONS (BACKWARD COMPATIBILITY)
# ====================================================================================
# The following functions maintain the existing API surface for backward compatibility
# while internally using the new decoupled architecture where appropriate.


def load_experiment_files(
    config: Optional[Union[Dict[str, Any], Any]] = None,
    experiment_name: str = "",
    config_path: Optional[Union[str, Path]] = None,
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
    through the _deps parameter, enabling pytest.monkeypatch patterns. It now accepts
    Pydantic models directly for improved type safety.
    
    Args:
        config: Pre-loaded configuration dictionary, Pydantic model, or LegacyConfigAdapter
        experiment_name: Name of the experiment to load files for
        config_path: Path to the YAML configuration file (alternative to config)
        base_directory: Optional override for the data directory (if not specified, uses config)
        pattern: File pattern to search for in glob format (e.g., "*.csv", "data_*.pkl")
        recursive: Whether to search recursively (defaults to True)
        extensions: Optional list of file extensions to filter by
        extract_metadata: If True, extract metadata from filenames using config patterns
        parse_dates: If True, attempt to parse dates from filenames
        _deps: Optional dependency provider for testing injection (internal parameter)
        
    Returns:
        If extract_metadata or parse_dates is True: Dictionary mapping file paths to metadata.
        Each entry contains the flattened metadata fields and a ``metadata`` bucket with the
        same information for convenient downstream access. Otherwise: List of file paths for
        the experiment.
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        KeyError: If the experiment doesn't exist in the config
        ValueError: If neither config_path nor config is provided, or if both are provided
        
    Example:
        >>> # Using Pydantic model directly
        >>> config = create_config(
        ...     project_name="fly_behavior",
        ...     base_directory="/data/experiments"
        ... )
        >>> files = load_experiment_files(
        ...     config=config,
        ...     experiment_name="plume_navigation"
        ... )
        >>> print(f"Found {len(files)} files")
    """
    operation_name = "load_experiment_files"
    
    # Initialize dependency provider for testability
    if _deps is None:
        _deps = get_dependency_provider()
    
    logger.info(f"Loading experiment files for experiment '{experiment_name}'")
    logger.debug(f"Parameters: pattern={pattern}, recursive={recursive}, "
                f"extensions={extensions}, extract_metadata={extract_metadata}, "
                f"parse_dates={parse_dates}")
    
    # Validate experiment_name parameter
    if not experiment_name or not isinstance(experiment_name, str):
        error_msg = (
            f"Invalid experiment_name for {operation_name}: '{experiment_name}'. "
            "experiment_name must be a non-empty string representing the experiment identifier."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)
    
    config_dict = _resolve_config_source(config, config_path, operation_name, _deps)
    
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

        if extract_metadata or parse_dates:
            result = _attach_metadata_bucket(result)

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
    config: Optional[Union[Dict[str, Any], Any]] = None,
    dataset_name: str = "",
    config_path: Optional[Union[str, Path]] = None,
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
    through the _deps parameter, enabling pytest.monkeypatch patterns. It now accepts
    Pydantic models directly for improved type safety.
    
    Args:
        config: Pre-loaded configuration dictionary, Pydantic model, or LegacyConfigAdapter
        dataset_name: Name of the dataset to load files for
        config_path: Path to the YAML configuration file (alternative to config)
        base_directory: Optional override for the data directory (if not specified, uses config)
        pattern: File pattern to search for in glob format (e.g., "*.csv", "data_*.pkl")
        recursive: Whether to search recursively (defaults to True)
        extensions: Optional list of file extensions to filter by
        extract_metadata: If True, extract metadata from filenames using config patterns
        parse_dates: If True, attempt to parse dates from filenames
        _deps: Optional dependency provider for testing injection (internal parameter)
        
    Returns:
        If extract_metadata or parse_dates is True: Dictionary mapping file paths to metadata.
        Each entry contains the flattened metadata fields and a ``metadata`` bucket with the
        same information for convenient downstream access. Otherwise: List of file paths for
        the dataset.
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        KeyError: If the dataset doesn't exist in the config
        ValueError: If neither config_path nor config is provided, or if both are provided
        
    Example:
        >>> # Using Pydantic model directly
        >>> config = create_config(
        ...     project_name="fly_behavior",
        ...     base_directory="/data/experiments"
        ... )
        >>> files = load_dataset_files(
        ...     config=config,
        ...     dataset_name="plume_tracking"
        ... )
        >>> print(f"Found {len(files)} files")
    """
    operation_name = "load_dataset_files"
    
    # Initialize dependency provider for testability
    if _deps is None:
        _deps = get_dependency_provider()
    
    logger.info(f"Loading dataset files for dataset '{dataset_name}'")
    logger.debug(f"Parameters: pattern={pattern}, recursive={recursive}, "
                f"extensions={extensions}, extract_metadata={extract_metadata}, "
                f"parse_dates={parse_dates}")
    
    # Validate dataset_name parameter
    if not dataset_name or not isinstance(dataset_name, str):
        error_msg = (
            f"Invalid dataset_name for {operation_name}: '{dataset_name}'. "
            "dataset_name must be a non-empty string representing the dataset identifier."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)
    
    config_dict = _resolve_config_source(config, config_path, operation_name, _deps)
    
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

        if extract_metadata or parse_dates:
            result = _attach_metadata_bucket(result)

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
    config: Optional[Union[Dict[str, Any], Any]] = None,
    experiment_name: str = "",
    config_path: Optional[Union[str, Path]] = None,
    _deps: Optional[DefaultDependencyProvider] = None
) -> Dict[str, Any]:
    """
    Get parameters for a specific experiment with enhanced testability.
    
    This function supports comprehensive dependency injection for testing scenarios
    through the _deps parameter, enabling pytest.monkeypatch patterns. It now accepts
    Pydantic models directly for improved type safety.
    
    Args:
        config: Pre-loaded configuration dictionary, Pydantic model, or LegacyConfigAdapter
        experiment_name: Name of the experiment to get parameters for
        config_path: Path to the YAML configuration file (alternative to config)
        _deps: Optional dependency provider for testing injection (internal parameter)
        
    Returns:
        Dictionary containing experiment parameters
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        KeyError: If the experiment doesn't exist in the config
        ValueError: If neither config_path nor config is provided, or if both are provided
        
    Example:
        >>> # Using Pydantic model directly
        >>> config = create_config(
        ...     project_name="fly_behavior",
        ...     base_directory="/data/experiments"
        ... )
        >>> params = get_experiment_parameters(
        ...     config=config,
        ...     experiment_name="plume_navigation"
        ... )
        >>> print(f"Found {len(params)} parameters")
    """
    operation_name = "get_experiment_parameters"
    
    # Initialize dependency provider for testability
    if _deps is None:
        _deps = get_dependency_provider()
    
    logger.info(f"Getting parameters for experiment '{experiment_name}'")
    
    # Validate experiment_name parameter
    if not experiment_name or not isinstance(experiment_name, str):
        error_msg = (
            f"Invalid experiment_name for {operation_name}: '{experiment_name}'. "
            "experiment_name must be a non-empty string representing the experiment identifier."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)
    
    config_dict = _resolve_config_source(config, config_path, operation_name, _deps)
    
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
    config: Optional[Union[Dict[str, Any], Any]] = None,
    dataset_name: str = "",
    config_path: Optional[Union[str, Path]] = None,
    _deps: Optional[DefaultDependencyProvider] = None
) -> Dict[str, Any]:
    """
    Get parameters for a specific dataset with enhanced testability.
    
    This function supports comprehensive dependency injection for testing scenarios
    through the _deps parameter, enabling pytest.monkeypatch patterns. It now accepts
    Pydantic models directly for improved type safety.
    
    Args:
        config: Pre-loaded configuration dictionary, Pydantic model, or LegacyConfigAdapter
        dataset_name: Name of the dataset to get parameters for
        config_path: Path to the YAML configuration file (alternative to config)
        _deps: Optional dependency provider for testing injection (internal parameter)
        
    Returns:
        Dictionary containing dataset parameters
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        KeyError: If the dataset doesn't exist in the config
        ValueError: If neither config_path nor config is provided, or if both are provided
        
    Example:
        >>> # Using Pydantic model directly
        >>> config = create_config(
        ...     project_name="fly_behavior",
        ...     base_directory="/data/experiments"
        ... )
        >>> params = get_dataset_parameters(
        ...     config=config,
        ...     dataset_name="plume_tracking"
        ... )
        >>> print(f"Found {len(params)} parameters")
    """
    operation_name = "get_dataset_parameters"
    
    # Initialize dependency provider for testability
    if _deps is None:
        _deps = get_dependency_provider()
    
    logger.info(f"Getting parameters for dataset '{dataset_name}'")
    
    # Validate dataset_name parameter
    if not dataset_name or not isinstance(dataset_name, str):
        error_msg = (
            f"Invalid dataset_name for {operation_name}: '{dataset_name}'. "
            "dataset_name must be a non-empty string representing the dataset identifier."
        )
        logger.error(error_msg)
        raise FlyRigLoaderError(error_msg)
    
    config_dict = _resolve_config_source(config, config_path, operation_name, _deps)
    
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


def deprecated(reason: str, alternative: str):
    """Decorator for marking deprecated functions."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} is deprecated. {reason}. Use {alternative} instead.",
                DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator


@deprecated(
    reason="Monolithic approach is less flexible. Use the new decoupled architecture",
    alternative="load_data_file() + transform_to_dataframe()"
)
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
    
    .. deprecated:: 
        This function is deprecated. The monolithic approach is less flexible. 
        Use the new decoupled architecture with load_data_file() + transform_to_dataframe() instead.
    
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
        >>> # Deprecated approach
        >>> df = process_experiment_data("experiment_data.pkl")
        >>> print(f"Processed {len(df)} rows with columns: {list(df.columns)}")
        
        >>> # New decoupled approach (recommended)
        >>> raw_data = load_data_file("experiment_data.pkl")
        >>> df = transform_to_dataframe(raw_data, file_path="experiment_data.pkl")
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