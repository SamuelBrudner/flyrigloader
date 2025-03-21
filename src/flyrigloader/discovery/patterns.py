"""
Pattern matching for vials and experiments.

This module provides utilities for matching and extracting information 
from file names and paths based on patterns.
"""
from typing import Any, Dict, List, Optional, Pattern, Union
import re
import os
from pathlib import Path
from loguru import logger


class PatternMatcher:
    """
    Pattern matcher for identifying and extracting components from filenames.
    """
    def __init__(self, patterns: List[str]):
        """
        Initialize with a list of regex patterns.
        
        Args:
            patterns: List of regex pattern strings
        """
        self.compiled_patterns = [re.compile(pattern) for pattern in patterns]
        
    def match(self, filename: str) -> Optional[Dict[str, str]]:
        """
        Match filename against patterns and return captured groups.
        
        Args:
            filename: Filename to match
            
        Returns:
            Dictionary of captured named groups, or None if no match
        """
        basename = os.path.basename(filename)
        
        for pattern in self.compiled_patterns:
            if match := pattern.match(basename):
                return match.groupdict()
        
        return None
    
    def filter_files(self, files: List[str]) -> Dict[str, Dict[str, str]]:
        """
        Filter list of files that match patterns and return matches.
        
        Args:
            files: List of file paths or names
            
        Returns:
            Dictionary mapping matched files to their extracted components
        """
        result = {}
        
        for file in files:
            if match_dict := self.match(file):
                result[file] = match_dict
                
        return result


def create_experiment_matcher(experiment_patterns: List[str]) -> PatternMatcher:
    """
    Create a PatternMatcher for experiment files.
    
    Args:
        experiment_patterns: List of experiment regex patterns
        
    Returns:
        PatternMatcher for experiments
    """
    return PatternMatcher(experiment_patterns)


def create_vial_matcher(vial_patterns: List[str]) -> PatternMatcher:
    """
    Create a PatternMatcher for vial files.
    
    Args:
        vial_patterns: List of vial regex patterns
        
    Returns:
        PatternMatcher for vials
    """
    return PatternMatcher(vial_patterns)


def extract_experiment_info(
    filepath: str, 
    patterns: Union[List[str], PatternMatcher]
) -> Optional[Dict[str, str]]:
    """
    Extract experiment information from a filepath.
    
    Args:
        filepath: Path to extract information from
        patterns: List of patterns or a PatternMatcher
        
    Returns:
        Dictionary of extracted information or None if no match
    """
    matcher = PatternMatcher(patterns) if isinstance(patterns, list) else patterns
    return matcher.match(filepath)


def extract_vial_info(
    filepath: str,
    patterns: Union[List[str], PatternMatcher]
) -> Optional[Dict[str, str]]:
    """
    Extract vial information from a filepath.
    
    Args:
        filepath: Path to extract information from
        patterns: List of patterns or a PatternMatcher
        
    Returns:
        Dictionary of extracted information or None if no match
    """
    matcher = PatternMatcher(patterns) if isinstance(patterns, list) else patterns
    return matcher.match(filepath)


def match_files_to_patterns(
    files: List[str],
    patterns: List[str]
) -> Dict[str, Dict[str, str]]:
    """
    Match a list of files against patterns and return matching files with extracted info.
    
    Args:
        files: List of file paths
        patterns: List of regex patterns
        
    Returns:
        Dict mapping matched files to their extracted information
    """
    matcher = PatternMatcher(patterns)
    return matcher.filter_files(files)


def generate_pattern_from_template(template: str) -> str:
    """
    Convert a template with placeholders to a regex pattern with named capture groups.
    
    Example:
        Template: "exp_{experiment_id}_{date}.csv"
        Result: "exp_(?P<experiment_id>[^_]+)_(?P<date>[^.]+)\.csv"
    
    Args:
        template: Template string with placeholders in curly braces
        
    Returns:
        Regex pattern string with named capture groups
    """
    # Find all placeholders in curly braces
    placeholders = re.findall(r'{([^}]+)}', template)
    
    # First, escape special regex characters in the template
    # But preserve the curly braces for placeholder replacement
    escaped_template = template
    for char in ['.', '/', '(', ')', '[', ']', '+', '*', '?', '|', '^', '$']:
        # Don't replace inside curly braces
        parts = []
        last_end = 0
        for match in re.finditer(r'{([^}]+)}', escaped_template):
            start, end = match.span()
            # Add parts before match and the placeholder
            parts.extend([
                escaped_template[last_end:start].replace(char, '\\' + char),
                escaped_template[start:end]
            ])
            last_end = end
        # Add the remaining part
        parts.append(escaped_template[last_end:].replace(char, '\\' + char))
        escaped_template = ''.join(parts)
    
    # Replace each placeholder with a named capture group
    pattern = escaped_template
    for placeholder in placeholders:
        # Determine what characters to match based on context
        capture_pattern = (
            r'[A-Za-z0-9]+' if placeholder.endswith('_id') or placeholder == 'id'
            else r'[^_.]+' if placeholder == 'date' or placeholder.endswith('_date')
            else r'[^_.]+'
        )
            
        pattern = pattern.replace(
            f'{{{placeholder}}}', 
            f'(?P<{placeholder}>{capture_pattern})'
        )
    
    return pattern
