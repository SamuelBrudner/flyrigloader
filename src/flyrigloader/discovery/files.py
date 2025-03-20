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

from ..config_utils.filter import filter_config_by_experiment
from .patterns import PatternMatcher, match_files_to_patterns


def discover_files(
    base_dirs: Union[str, List[str]],
    patterns: List[str],
    recursive: bool = True
) -> Dict[str, Dict[str, str]]:
    """
    Discover files matching patterns in the specified directories.
    
    Args:
        base_dirs: Base directory or list of directories to search in
        patterns: List of file patterns to match
        recursive: Whether to search recursively
        
    Returns:
        Dictionary of matched files with extracted information
    """
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
    return match_files_to_patterns(all_files, patterns)


def discover_by_experiment(
    config: Dict[str, Any],
    experiment_name: str,
    base_dirs: Optional[Union[str, List[str]]] = None
) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Discover files for a specific experiment based on config.
    
    Args:
        config: Configuration dictionary
        experiment_name: Name of experiment to discover files for
        base_dirs: Optional override for base directories
        
    Returns:
        Dictionary of file categories to matched files with extracted info
    """
    # Filter config to specific experiment
    exp_config = filter_config_by_experiment(config, experiment_name)
    
    if not exp_config:
        logger.warning(f"No configuration found for experiment: {experiment_name}")
        return {}
    
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
            
        if matched_files := discover_files(base_dirs, patterns):
            result[category] = matched_files
    
    return result


def organize_by_dataset(
    files: Dict[str, Dict[str, str]],
    dataset_field: str = 'dataset'
) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Organize discovered files by dataset.
    
    Args:
        files: Dictionary of matched files with extracted info
        dataset_field: Field name in extracted info that contains dataset name
        
    Returns:
        Dictionary mapping dataset names to their matched files
    """
    result = {}
    
    for filepath, info in files.items():
        if dataset_field not in info:
            # Skip files that don't have dataset info
            continue
            
        dataset = info[dataset_field]
        if dataset not in result:
            result[dataset] = {}
            
        result[dataset][filepath] = info
    
    return result


def get_unique_values(
    files: Dict[str, Dict[str, str]],
    field: str
) -> Set[str]:
    """
    Get set of unique values for a specific field across all matched files.
    
    Args:
        files: Dictionary of matched files with extracted info
        field: Field name to extract unique values for
        
    Returns:
        Set of unique values
    """
    return {info[field] for info in files.values() if field in info}


def get_latest_files(
    files: Dict[str, Dict[str, str]],
    date_field: str = 'date',
    date_format: str = '%Y%m%d'
) -> Dict[str, Dict[str, str]]:
    """
    Filter the files to only include the latest version of each file.
    
    Args:
        files: Dictionary of matched files with extracted info
        date_field: Field name that contains the date information
        date_format: Format string for parsing the date
        
    Returns:
        Dictionary of latest files with their extracted info
    """
    from datetime import datetime
    
    # Group files by all fields except date
    groups = {}
    
    for filepath, info in files.items():
        if date_field not in info:
            # Skip files without date info
            continue
            
        # Create a key from all fields except date
        key_parts = [f"{field}:{value}" for field, value in sorted(info.items()) if field != date_field]
        group_key = '|'.join(key_parts)
        
        if group_key not in groups:
            groups[group_key] = []
            
        # Add to group with parsed date and filepath
        try:
            date = datetime.strptime(info[date_field], date_format)
            groups[group_key].append((date, filepath, info))
        except ValueError:
            logger.warning(f"Could not parse date '{info[date_field]}' for file: {filepath}")
    
    # Get the latest file from each group
    return {
        filepath: info
        for group_files in groups.values()
        if group_files
        for _, filepath, info in [sorted(group_files, key=lambda x: x[0], reverse=True)[0]]
    }


def discover_dataset_files(
    dataset_name: str,
    dataset_config: Dict[str, Any],
    global_config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Discover files for a specific dataset based on its configuration.
    
    Args:
        dataset_name: Name of the dataset to discover files for
        dataset_config: Configuration for the specific dataset
        global_config: Complete configuration dictionary
        
    Returns:
        List of dictionaries containing file info and metadata
    """
    # Get base directories from dataset or global config
    base_dirs = dataset_config.get('data_directories') or global_config.get('data_directories', [])
        
    if not base_dirs:
        logger.warning(f"No data directories specified for dataset: {dataset_name}")
        return []
    
    # Get file patterns from dataset config
    if not (patterns := dataset_config.get('file_patterns', [])):
        logger.warning(f"No file patterns specified for dataset: {dataset_name}")
        return []
    
    # Discover matching files
    matched_files = discover_files(base_dirs, patterns)
    
    # Convert to list of items with metadata
    result = []
    metadata = dataset_config.get('metadata', {})
    
    for filepath, extracted_info in matched_files.items():
        item = {
            'path': Path(filepath),
            'file_name': Path(filepath).name,
            'dataset': dataset_name,
            **extracted_info,  # Include all extracted pattern info
            **(metadata or {})  # Add dataset-specific metadata if available
        }
        
        result.append(item)
    
    logger.info(f"Discovered {len(result)} files for dataset {dataset_name}")
    return result
