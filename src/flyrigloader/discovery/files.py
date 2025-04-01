"""
File discovery based on configuration.

This module provides utilities for finding and organizing files based on
configuration settings and pattern matching.
"""
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import os
import glob
from pathlib import Path
from loguru import logger
import fnmatch
import re
from datetime import datetime

from ..core.utils import PathLike, ensure_path, create_metadata, create_error_metadata
from ..core.utils.file_utils import list_directory_with_pattern
from ..config_utils.filter import filter_config_by_experiment
from .patterns import PatternMatcher, match_files_to_patterns


def discover_files(
    base_dirs: Union[str, List[str]],
    patterns: List[str],
    recursive: bool = True
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Any]]:
    """
    Discover files matching patterns in the specified directories.
    
    Args:
        base_dirs: Base directory or list of directories to search in
        patterns: List of file patterns to match
        recursive: Whether to search recursively
        
    Returns:
        Tuple of (matched_files, metadata) where matched_files is a dictionary 
        of matched files with extracted information and metadata contains
        status information and any error details
    """
    try:
        metadata = create_metadata()
        
        if isinstance(base_dirs, str):
            base_dirs = [base_dirs]
            
        all_files = []
        
        for base_dir in base_dirs:
            # Ensure base_dir is a Path object
            base_path = Path(base_dir)
            
            # Skip if base directory doesn't exist
            if not base_path.exists():
                logger.warning(f"Base directory does not exist: {base_dir}")
                continue
                
            # Find all files
            if recursive:
                for root, _, files in os.walk(base_path):
                    all_files.extend([os.path.join(root, file) for file in files])
            else:
                all_files.extend([
                    os.path.join(base_path, f) 
                    for f in os.listdir(base_path) 
                    if os.path.isfile(os.path.join(base_path, f))
                ])
        
        # Match files against patterns
        result = match_files_to_patterns(all_files, patterns)
        
        metadata["success"] = True
        metadata["files_found"] = len(result)
        
        return result, metadata
    except Exception as e:
        logger.error(f"Failed to discover files: {str(e)}")
        return {}, create_error_metadata(e)


def discover_by_experiment(
    config: Dict[str, Any],
    experiment_name: str,
    base_dirs: Optional[Union[str, List[str]]] = None
) -> Tuple[Dict[str, Dict[str, Dict[str, str]]], Dict[str, Any]]:
    """
    Discover files for a specific experiment based on config.
    
    Args:
        config: Configuration dictionary
        experiment_name: Name of experiment to discover files for
        base_dirs: Optional override for base directories
        
    Returns:
        Tuple of (discovered_files, metadata) where discovered_files is a dictionary 
        of file categories to matched files with extracted info and metadata contains
        status information and any error details
    """
    try:
        metadata = create_metadata()
        
        # Filter config to specific experiment
        try:
            exp_config = filter_config_by_experiment(config, experiment_name)
            if not exp_config:
                metadata["error"] = f"No configuration found for experiment: {experiment_name}"
                return {}, metadata
        except ValueError as e:
            metadata["error"] = f"Invalid experiment configuration: {str(e)}"
            return {}, metadata
        
        # Get experiment-specific patterns
        file_patterns = exp_config.get('file_patterns', {})
        
        # Use provided base_dirs or get from config
        if base_dirs is None:
            base_dirs = exp_config.get('data_directories', [])
        elif isinstance(base_dirs, str):
            base_dirs = [base_dirs]
        
        # Discover files for each pattern category
        result = {}
        for category, patterns in file_patterns.items():
            if not patterns:
                continue
            
            category_files, cat_metadata = discover_files(base_dirs, patterns, recursive=True)
            if not cat_metadata["success"]:
                metadata["category_errors"] = metadata.get("category_errors", {})
                metadata["category_errors"][category] = cat_metadata.get("error", "Unknown error")
                continue
                
            if category_files:
                result[category] = category_files
        
        metadata["success"] = True
        metadata["categories_found"] = len(result)
        
        return result, metadata
    except Exception as e:
        logger.error(f"Failed to discover files by experiment: {str(e)}")
        return {}, create_error_metadata(e)


def organize_by_dataset(
    files: Dict[str, Dict[str, str]],
    dataset_field: str = 'dataset'
) -> Tuple[Dict[str, Dict[str, Dict[str, str]]], Dict[str, Any]]:
    """
    Organize discovered files by dataset.
    
    Args:
        files: Dictionary of matched files with extracted info
        dataset_field: Field name in extracted info that contains dataset name
        
    Returns:
        Tuple of (organized_files, metadata) where organized_files is a dictionary 
        mapping dataset names to their matched files and metadata contains
        status information and any error details
    """
    try:
        metadata = create_metadata()
        result = {}
        missing_dataset_count = 0
        
        for file_path, info in files.items():
            if dataset_field not in info:
                logger.warning(f"Dataset field '{dataset_field}' not found in file info: {file_path}")
                missing_dataset_count += 1
                continue
                
            dataset = info[dataset_field]
            if dataset not in result:
                result[dataset] = {}
                
            result[dataset][file_path] = info
        
        if missing_dataset_count > 0:
            metadata["missing_dataset_count"] = missing_dataset_count
            
        metadata["success"] = True
        metadata["datasets_found"] = len(result)
        
        return result, metadata
    except Exception as e:
        logger.error(f"Failed to organize files by dataset: {str(e)}")
        return {}, create_error_metadata(e)


def get_unique_values(
    files: Dict[str, Dict[str, str]],
    field: str
) -> Tuple[Set[str], Dict[str, Any]]:
    """
    Get set of unique values for a specific field across all matched files.
    
    Args:
        files: Dictionary of matched files with extracted info
        field: Field name to extract unique values for
        
    Returns:
        Tuple of (unique_values, metadata) where unique_values is a set of 
        unique values for the specified field and metadata contains
        status information and any error details
    """
    try:
        metadata = create_metadata()
        values = set()
        missing_field_count = 0
        
        for file_path, info in files.items():
            if field in info:
                values.add(info[field])
            else:
                missing_field_count += 1
        
        if missing_field_count > 0:
            metadata["missing_field_count"] = missing_field_count
            
        metadata["success"] = True
        metadata["unique_values_count"] = len(values)
        
        return values, metadata
    except Exception as e:
        logger.error(f"Failed to get unique values: {str(e)}")
        return set(), create_error_metadata(e)


def get_latest_files(
    files: Dict[str, Dict[str, str]],
    date_field: str = 'date',
    date_format: str = '%Y%m%d',
    alternative_formats: List[str] = None,
    include_unparseable: bool = False,
    unparseable_date_behavior: str = 'oldest'  # 'oldest', 'newest', or 'exclude'
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Any]]:
    """
    Filter the files to only include the latest version of each file.
    
    Args:
        files: Dictionary of matched files with extracted info
        date_field: Field name that contains the date information
        date_format: Primary format string for parsing the date
        alternative_formats: List of alternative date formats to try if primary format fails
        include_unparseable: Whether to include files with unparseable dates in results
        unparseable_date_behavior: How to treat unparseable dates:
            - 'oldest': Treat unparseable dates as oldest (will be excluded if newer versions exist)
            - 'newest': Treat unparseable dates as newest (will be included over dated versions)
            - 'exclude': Exclude files with unparseable dates (default)
        
    Returns:
        Tuple of (latest_files, metadata) where latest_files is a dictionary 
        of latest files with their extracted info and metadata contains
        status information and any error details
    """
    try:
        metadata = create_metadata()
        from datetime import datetime
        import sys
        
        # Set default alternative formats if none provided
        if alternative_formats is None:
            alternative_formats = [
                '%Y-%m-%d',      # ISO format: 2023-01-15
                '%m/%d/%Y',      # US format: 01/15/2023
                '%d/%m/%Y',      # EU format: 15/01/2023
                '%d-%m-%Y',      # EU dash format: 15-01-2023
                '%m-%d-%Y',      # US dash format: 01-15-2023
                '%Y%m%d%H%M%S',  # Full timestamp: 20230115123045
                '%Y%m%d_%H%M%S', # Timestamp with underscore: 20230115_123045
            ]
        
        # Group files by all fields except date
        groups = {}
        files_with_unparseable_dates = []
        
        # Get min/max dates for handling unparseable dates
        min_date = datetime.min
        max_date = datetime.max
        
        for filepath, info in files.items():
            if date_field not in info:
                # Skip files without date info
                logger.warning(f"Missing date field '{date_field}' for file: {filepath}")
                continue
                
            # Create a key from all fields except date
            key_parts = [f"{field}:{value}" for field, value in sorted(info.items()) if field != date_field]
            group_key = '|'.join(key_parts)
            
            if group_key not in groups:
                groups[group_key] = []
            
            # Try to parse the date with the primary format first, then alternatives
            date_str = info[date_field]
            date_parsed = False
            date = None
            
            # Try the primary format
            try:
                date = datetime.strptime(date_str, date_format)
                date_parsed = True
            except ValueError:
                # Try alternative formats
                for alt_format in alternative_formats:
                    try:
                        date = datetime.strptime(date_str, alt_format)
                        date_parsed = True
                        
                        # Log that we used an alternative format for visibility
                        logger.info(f"Parsed date '{date_str}' using alternative format '{alt_format}' for file: {filepath}")
                        break
                    except ValueError:
                        continue
            
            if date_parsed:
                groups[group_key].append((date, filepath, info))
            else:
                logger.warning(f"Could not parse date '{date_str}' for file: {filepath} using any known format")
                if include_unparseable or unparseable_date_behavior in ('oldest', 'newest'):
                    # Handle unparseable dates according to the specified behavior
                    if unparseable_date_behavior == 'oldest':
                        groups[group_key].append((min_date, filepath, info))
                    elif unparseable_date_behavior == 'newest':
                        groups[group_key].append((max_date, filepath, info))
                    files_with_unparseable_dates.append(filepath)
        
        # Report summary of parsing issues
        if files_with_unparseable_dates:
            logger.warning(f"Found {len(files_with_unparseable_dates)} files with unparseable dates")
            if include_unparseable or unparseable_date_behavior in ('oldest', 'newest'):
                logger.info(f"Including files with unparseable dates as '{unparseable_date_behavior}'")
        
        # Get the latest file from each group
        result = {}
        for group_key, group_files in groups.items():
            if not group_files:
                continue
                
            # Sort by date (newest first) and get the first one
            sorted_files = sorted(group_files, key=lambda x: x[0], reverse=True)
            _, filepath, info = sorted_files[0]
            
            # Include the parsed date in the info for downstream usage
            if hasattr(sorted_files[0][0], 'isoformat'):  # Check if it's a real date, not min/max
                info['parsed_date'] = sorted_files[0][0].isoformat()
                
            result[filepath] = info
        
        metadata["success"] = True
        metadata["files_found"] = len(result)
        
        return result, metadata
    except Exception as e:
        logger.error(f"Failed to get latest files: {str(e)}")
        return {}, create_error_metadata(e)


def discover_dataset_files(
    dataset_name: str,
    dataset_config: Dict[str, Any],
    global_config: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Discover files for a specific dataset based on its configuration.
    
    Args:
        dataset_name: Name of the dataset to discover files for
        dataset_config: Configuration for the specific dataset
        global_config: Complete configuration dictionary
        
    Returns:
        Tuple of (dataset_files, metadata) where dataset_files is a list of 
        dictionaries containing file info and metadata, and metadata contains
        status information and any error details
    """
    try:
        metadata = create_metadata()
        
        # Get base directories from dataset or global config
        base_dirs = dataset_config.get('data_directories') or global_config.get('data_directories', [])
        
        if not base_dirs:
            metadata["error"] = f"No data directories specified for dataset: {dataset_name}"
            return [], metadata
        
        # Get file patterns from dataset config
        if not (patterns := dataset_config.get('file_patterns', [])):
            metadata["error"] = f"No file patterns specified for dataset: {dataset_name}"
            return [], metadata
        
        # Discover matching files
        matched_files, match_metadata = discover_files(base_dirs, patterns)
        
        if not match_metadata["success"]:
            metadata["error"] = match_metadata.get("error", "Unknown error")
            return [], metadata
        
        # Convert to list of items with metadata
        result = []
        metadata["files_found"] = len(matched_files)
        
        for filepath, extracted_info in matched_files.items():
            item = {
                'path': Path(filepath),
                'file_name': Path(filepath).name,
                'dataset': dataset_name,
                **extracted_info,  # Include all extracted pattern info
            }
            
            result.append(item)
        
        metadata["success"] = True
        
        return result, metadata
    except Exception as e:
        logger.error(f"Failed to discover dataset files: {str(e)}")
        return [], create_error_metadata(e)


def discover_files_in_folder(
    folder_path: PathLike,
    pattern: str = "*",
    recursive: bool = False
) -> Tuple[List[Path], Dict[str, Any]]:
    """
    Discover files matching a pattern in a folder.
    
    Args:
        folder_path: Path to folder to search in
        pattern: Glob pattern to match (e.g., "*.pkl")
        recursive: Whether to search recursively
        
    Returns:
        Tuple of (matched_files, metadata) where matched_files is a list of 
        Path objects for matched files and metadata contains status information 
        and any error details
    """
    try:
        metadata = create_metadata()
        
        return list_directory_with_pattern(folder_path, pattern, recursive), metadata
    except Exception as e:
        logger.error(f"Failed to discover files in folder: {str(e)}")
        return [], create_error_metadata(e)


def build_file_items(
    files: List[Path],
    metadata: Optional[Dict[str, Any]] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Build a list of item dictionaries for files, with common metadata.
    
    Args:
        files: List of file paths
        metadata: Optional common metadata to apply to all files
        
    Returns:
        Tuple of (file_items, metadata) where file_items is a list of 
        dictionaries with path and metadata, and metadata contains status 
        information and any error details
    """
    try:
        metadata = metadata or {}
        result_metadata = create_metadata()
        
        items = []
        for file_path in files:
            # Create a dictionary for this file
            item = {
                "path": str(file_path),
                "filename": file_path.name,
                "extension": file_path.suffix.lstrip('.')
            }
            
            # Add file stats
            try:
                stats = file_path.stat()
                item.update({
                    "size": stats.st_size,
                    "modified": stats.st_mtime,
                    "created": stats.st_ctime
                })
            except Exception as e:
                logger.warning(f"Error getting stats for {file_path}: {e}")
            
            # Add common metadata
            item.update(metadata)
            
            items.append(item)
        
        result_metadata["success"] = True
        result_metadata["files_processed"] = len(items)
        
        return items, result_metadata
    except Exception as e:
        logger.error(f"Failed to build file items: {str(e)}")
        return [], create_error_metadata(e)
