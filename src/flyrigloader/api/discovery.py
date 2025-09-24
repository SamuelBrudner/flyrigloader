"""Discovery-related public API entry points."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterator, List, Optional, Union

from semantic_version import Version

from flyrigloader import logger
from flyrigloader._api_registry import get_facade_modules
from flyrigloader.config.models import LegacyConfigAdapter
from flyrigloader.config.validators import validate_config_version
from flyrigloader.config.versioning import CURRENT_SCHEMA_VERSION
from flyrigloader.discovery.files import (
    discover_experiment_manifest as _default_discover_experiment_manifest,
)
from flyrigloader.exceptions import FlyRigLoaderError
from flyrigloader.registries import get_loader_capabilities as _get_loader_capabilities

from .configuration import (
    coerce_config_for_version_validation as _coerce_config_for_version_validation,
    load_and_validate_config as _load_and_validate_config,
    resolve_config_source as _resolve_config_source,
)
from .dependencies import DefaultDependencyProvider, get_dependency_provider

def _iter_known_facade_modules() -> Iterator[ModuleType]:
    """Yield every facade module instance that might carry test patches."""
    seen: set[int] = set()

    active_module = sys.modules.get("flyrigloader.api")
    if active_module is not None:
        seen.add(id(active_module))
        yield active_module

    for module in get_facade_modules():
        module_id = id(module)
        if module_id in seen:
            continue
        seen.add(module_id)
        yield module


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
    _deps: Optional[DefaultDependencyProvider] = None,
    _manifest_discoverer: Optional[Any] = None,
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
        manifest_callable = _manifest_discoverer
        if manifest_callable is None:
            active_module = sys.modules.get("flyrigloader.api")
            for module in _iter_known_facade_modules():
                patched = getattr(module, "_discover_experiment_manifest", None)
                if not callable(patched):
                    continue

                api_facade_callable = getattr(module, "discover_experiment_manifest", None)
                if patched is api_facade_callable or patched is _default_discover_experiment_manifest:
                    continue

                origin = "active" if module is active_module else "cached"
                logger.debug(
                    "Using patched discover_experiment_manifest from {} API facade: {}",
                    origin,
                    patched,
                )
                manifest_callable = patched
                break

        if manifest_callable is None:
            manifest_callable = _default_discover_experiment_manifest

        file_manifest = manifest_callable(
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
            metadata_payload: Dict[str, Any]
            if isinstance(file_info.extracted_metadata, dict):
                metadata_payload = dict(file_info.extracted_metadata)
            else:
                metadata_payload = {}

            manifest_entry: Dict[str, Any] = {
                'path': file_info.path,
                'size': file_info.size if file_info.size is not None else 0,
                'metadata': metadata_payload,
                'parsed_dates': {'parsed_date': file_info.parsed_date} if file_info.parsed_date else {}
            }

            if file_info.mtime is not None:
                manifest_entry['mtime'] = file_info.mtime
            if file_info.ctime is not None:
                manifest_entry['ctime'] = file_info.ctime
            if file_info.creation_time is not None:
                manifest_entry['creation_time'] = file_info.creation_time

            manifest_dict[file_info.path] = manifest_entry
        
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
                logger.debug(f"Detected configuration version: {detected_version}")

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
