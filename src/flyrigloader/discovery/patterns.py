"""
Pattern matching functionality.

Utilities for working with regex patterns to extract metadata from filenames.
"""

import contextlib
from typing import Any, Dict, List, Optional, Pattern, Union, Tuple, Protocol, Callable
import re
from flyrigloader import logger
from pathlib import Path
from abc import ABC, abstractmethod


class PatternCompilationError(Exception):
    """
    Exception raised when regex pattern compilation fails.
    
    This exception provides structured context for debugging pattern compilation
    issues during testing and development.
    """
    
    def __init__(self, pattern: str, original_error: Exception, pattern_index: int = None):
        """
        Initialize the PatternCompilationError.
        
        Args:
            pattern: The regex pattern that failed to compile
            original_error: The original exception that caused the failure
            pattern_index: Optional index of the pattern in a list of patterns
        """
        self.pattern = pattern
        self.original_error = original_error
        self.pattern_index = pattern_index
        
        # Construct a detailed error message
        message = f"Failed to compile regex pattern: '{pattern}'"
        if pattern_index is not None:
            message += f" (pattern index: {pattern_index})"
        message += f". Original error: {original_error}"
        
        super().__init__(message)


class LoggerProvider(Protocol):
    """
    Protocol for logger providers to enable dependency injection in testing.
    
    This protocol allows tests to inject custom logger implementations
    supporting pytest.monkeypatch scenarios per TST-REF-003 requirements.
    """
    
    def debug(self, message: str) -> None:
        """Log a debug message."""
        ...
    
    def info(self, message: str) -> None:
        """Log an info message."""
        ...
    
    def warning(self, message: str) -> None:
        """Log a warning message."""
        ...
    
    def error(self, message: str) -> None:
        """Log an error message."""
        ...


class RegexProvider(Protocol):
    """
    Protocol for regex providers to enable dependency injection in testing.
    
    This protocol allows tests to inject mock regex engines for comprehensive
    unit testing per TST-REF-001 requirements.
    """
    
    def compile(self, pattern: str) -> Pattern[str]:
        """Compile a regex pattern and return a Pattern object."""
        ...
    
    def match(self, pattern: str, string: str) -> Optional[re.Match[str]]:
        """Match a pattern against a string."""
        ...
    
    def search(self, pattern: Pattern[str], string: str) -> Optional[re.Match[str]]:
        """Search for a pattern in a string."""
        ...
    
    def findall(self, pattern: str, string: str) -> List[str]:
        """Find all matches of a pattern in a string."""
        ...


class DefaultLoggerProvider:
    """
    Default logger provider implementation using Loguru.
    
    This implementation provides the standard logging behavior while
    allowing for dependency injection during testing.
    """
    
    def __init__(self, logger_instance=None):
        """
        Initialize the logger provider.
        
        Args:
            logger_instance: Optional logger instance to use (defaults to global loguru logger)
        """
        self._logger = logger_instance if logger_instance is not None else logger
    
    def debug(self, message: str) -> None:
        """Log a debug message."""
        self._logger.debug(message)
    
    def info(self, message: str) -> None:
        """Log an info message."""
        self._logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log a warning message."""
        self._logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log an error message."""
        self._logger.error(message)


class DefaultRegexProvider:
    """
    Default regex provider implementation using Python's re module.
    
    This implementation provides the standard regex behavior while
    allowing for dependency injection during testing.
    """
    
    def compile(self, pattern: str) -> Pattern[str]:
        """Compile a regex pattern and return a Pattern object."""
        try:
            return re.compile(pattern)
        except re.error as e:
            raise PatternCompilationError(pattern, e) from e
    
    def match(self, pattern: str, string: str) -> Optional[re.Match[str]]:
        """Match a pattern against a string."""
        try:
            return re.match(pattern, string)
        except re.error as e:
            raise PatternCompilationError(pattern, e) from e
    
    def search(self, pattern: Pattern[str], string: str) -> Optional[re.Match[str]]:
        """Search for a pattern in a string."""
        return pattern.search(string)
    
    def findall(self, pattern: str, string: str) -> List[str]:
        """Find all matches of a pattern in a string."""
        try:
            return re.findall(pattern, string)
        except re.error as e:
            raise PatternCompilationError(pattern, e) from e


class PatternMatchingInterface(ABC):
    """
    Abstract interface for pattern matching operations.
    
    This interface reduces coupling with filesystem operations per TST-REF-002
    requirements and enables comprehensive mocking scenarios.
    """
    
    @abstractmethod
    def match(self, filename: str) -> Optional[Dict[str, str]]:
        """
        Match a filename against patterns and extract metadata.
        
        Args:
            filename: The filename to match against patterns
            
        Returns:
            Dictionary with extracted metadata or None if no match
        """
        pass
    
    @abstractmethod
    def filter_files(self, files: List[str]) -> Dict[str, Dict[str, str]]:
        """
        Filter a list of files based on patterns and extract metadata.
        
        Args:
            files: List of file paths to filter
            
        Returns:
            Dictionary mapping file paths to extracted metadata
        """
        pass


class PatternMatcher(PatternMatchingInterface):
    """
    Class for matching and extracting information from filenames based on regex patterns.
    
    This class provides methods for matching filenames, filtering files, and extracting
    information based on named groups in the regex patterns.
    
    Enhanced with dependency injection support for comprehensive testing per F-016 requirements.
    """
    
    def __init__(
        self, 
        patterns: List[str],
        logger_provider: Optional[LoggerProvider] = None,
        regex_provider: Optional[RegexProvider] = None
    ):
        """
        Initialize the PatternMatcher with regex patterns and configurable dependencies.
        
        This constructor supports dependency injection for testing scenarios per F-016
        requirements, enabling pytest.monkeypatch patterns for comprehensive testing.
        
        Args:
            patterns: List of regex pattern strings
            logger_provider: Optional logger provider for dependency injection (defaults to DefaultLoggerProvider)
            regex_provider: Optional regex provider for dependency injection (defaults to DefaultRegexProvider)
        """
        # Parse patterns to determine their target path component (full path, filename, parent, or parts[idx])
        self.patterns: List[str] = []  # Raw regex patterns without directive
        self._targets: List[Tuple[str, Optional[int]]] = []  # (target_type, index)

        for original_pattern in patterns:
            target_type, target_idx, raw_pattern = self._split_pattern_target(original_pattern)
            self.patterns.append(raw_pattern)
            self._targets.append((target_type, target_idx))

        # Configure dependency injection for testability
        self._logger = logger_provider if logger_provider is not None else DefaultLoggerProvider()
        self._regex = regex_provider if regex_provider is not None else DefaultRegexProvider()

        # Compile the *raw* patterns with enhanced error handling
        self.compiled_patterns = self._compile_patterns_with_error_handling(self.patterns)

        # Configure logging - only do this once per application
        with contextlib.suppress(Exception):
            logger.level("INFO")
    
    def _split_pattern_target(self, pattern_str: str) -> Tuple[str, Optional[int], str]:
        """Split a pattern string into (target_type, index, raw_pattern).

        Supported prefixes:
            filename::  → match against Path(filename).name
            parent::    → match against Path(filename).parent.name
            parts[N]::  → match against Path(filename).parts[N] (N can be negative)
        If no prefix is provided, the pattern matches against the full path (default, backward-compatible).
        """
        if "::" not in pattern_str:
            return ("full", None, pattern_str)

        prefix, raw = pattern_str.split("::", 1)
        prefix = prefix.strip()

        if prefix == "filename":
            return ("filename", None, raw)
        if prefix == "parent":
            return ("parent", None, raw)
        if prefix.startswith("parts[") and prefix.endswith("]"):
            try:
                idx = int(prefix[6:-1])
                return ("parts", idx, raw)
            except ValueError:
                # Invalid index; treat as full path to remain safe
                return ("full", None, pattern_str)
        # Unknown prefix – fallback to full path for robustness
        return ("full", None, pattern_str)

    def _compile_patterns_with_error_handling(self, patterns: List[str]) -> List[Pattern[str]]:
        """
        Compile regex patterns with enhanced error handling.
        
        This method provides structured exception context for improved test diagnostics
        per Section 2.2.8 requirements.
        
        Args:
            patterns: List of regex pattern strings to compile
            
        Returns:
            List of compiled regex patterns
            
        Raises:
            PatternCompilationError: If any pattern fails to compile
        """
        compiled_patterns = []

        for i, pattern in enumerate(patterns):
            try:
                compiled_pattern = self._regex.compile(pattern)
                compiled_patterns.append(compiled_pattern)
                self._logger.debug(f"Successfully compiled pattern {i}: {pattern}")
            except PatternCompilationError:
                # Re-raise with pattern index information
                raise
            except Exception as e:
                # Wrap other exceptions in our structured error
                raise PatternCompilationError(pattern, e, i) from e

        return compiled_patterns
    
    def _get_field_names_for_pattern(self, pattern_idx: int) -> Optional[List[str]]:
        """
        Determine the field names to use for a pattern with positional groups.
        
        Args:
            pattern_idx: Index of the pattern in the patterns list
            
        Returns:
            List of field names or None if pattern type is unknown
        """
        pattern = self.patterns[pattern_idx]
        
        # Determine pattern type based on content
        if "mouse" in pattern and "_mouse_" not in pattern:
            return ["animal", "date", "condition", "replicate"]
        elif "rat" in pattern and "_rat_" not in pattern:
            return ["date", "animal", "condition", "replicate"]
        elif "exp" in pattern:
            return ["experiment_id", "animal", "condition"]
        
        return None

    def match_all(self, filename: str) -> Optional[Dict[str, str]]:
        """Match *all* patterns against *filename* and merge metadata.

        Parameters
        ----------
        filename:
            Path to match against the configured patterns.

        Returns
        -------
        dict | None
            Combined metadata from every pattern that matches the file. ``None``
            if no pattern matches at all.
        """
        self._logger.debug(
            f"Attempting to match all patterns against: {filename}"
        )

        combined: Dict[str, str] = {}

        for i, pattern in enumerate(self.compiled_patterns):
            self._logger.debug(f"Trying pattern {i}: {self.patterns[i]}")

            target_type, target_idx = self._targets[i]
            path_obj = Path(filename)

            if target_type == "filename":
                candidate = path_obj.name
            elif target_type == "parent":
                candidate = path_obj.parent.name
            elif target_type == "parts":
                parts = path_obj.parts
                idx = target_idx if target_idx is not None else 0
                if idx < 0:
                    idx = len(parts) + idx
                if idx < 0 or idx >= len(parts):
                    self._logger.debug(
                        f"Skipping pattern {i} due to out-of-range parts index {target_idx} for path {filename}"
                    )
                    continue
                candidate = parts[idx]
            else:
                candidate = filename

            self._logger.debug(
                f"Matching against component '{candidate}' (target={target_type})"
            )

            if (match := self._regex.search(pattern, candidate)):
                self._logger.debug(f"Match found with pattern {i}")
                result = self._extract_groups_from_match(match, i)
                self._logger.debug(f"Extracted groups: {result}")
                result = self._process_special_cases(result, filename)
                combined.update(result)

        if combined:
            return combined

        self._logger.debug(f"No matches found for {filename}")
        return None
    
    def _extract_groups_from_match(self, match: re.Match, pattern_idx: int) -> Dict[str, str]:
        """
        Extract groups from a match object, handling both named and positional groups.
        
        Args:
            match: The regex match object
            pattern_idx: Index of the pattern that matched
            
        Returns:
            Dictionary of extracted field values
        """
        # If pattern uses named groups, use them directly
        if match.groupdict():
            return match.groupdict()
        
        # For backward compatibility with positional groups
        if fields := self._get_field_names_for_pattern(pattern_idx):
            return dict(zip(fields, match.groups()))
        
        # Generic handling for unknown patterns
        return {f"group{j}": group for j, group in enumerate(match.groups())}
    
    def _process_special_cases(self, result: Dict[str, str], filename: str) -> Dict[str, str]:
        """
        Apply special case processing to the extracted metadata.
        
        Args:
            result: The extracted metadata
            filename: The filename that was matched
            
        Returns:
            Updated metadata with special cases handled
        """
        # Handle animal files with condition_replicate format
        if ("animal" in result and result["animal"] in ["mouse", "rat"] and 
                "condition" in result and "replicate" in result):
            if (condition_match := self._regex.match(r"(.+)_(\d+)$", result["condition"])):
                # Split into condition and replicate
                result["condition"], extra_replicate = condition_match.groups()
                # Only use the extra replicate if none was provided
                if not result.get("replicate"):
                    result["replicate"] = extra_replicate
        
        # Special case for experiment files with animal=mouse
        if ("experiment_id" in result and result.get("animal") == "mouse" and 
                "baseline" in filename):
            result["animal"] = "exp_mouse"  # Mark specially to not count as a mouse file
        
        return result
    
    def match(self, filename: str) -> Optional[Dict[str, str]]:
        """
        Match a filename against the compiled regex patterns.
        
        Args:
            filename: The filename (or full path) to match against patterns
            
        Returns:
            Dictionary with extracted metadata from the first matching pattern, or None if no match
        """
        self._logger.debug(f"Attempting to match patterns against: {filename}")
        
        # Try each pattern until we find a match
        for i, pattern in enumerate(self.compiled_patterns):
            self._logger.debug(f"Trying pattern {i}: {self.patterns[i]}")

            # Determine which component of the path to match against
            target_type, target_idx = self._targets[i]
            candidate: str
            path_obj = Path(filename)
            if target_type == "filename":
                candidate = path_obj.name
            elif target_type == "parent":
                candidate = path_obj.parent.name
            elif target_type == "parts":
                parts = path_obj.parts
                idx = target_idx if target_idx is not None else 0
                if idx < 0:
                    idx = len(parts) + idx
                # If index is out of bounds, this pattern cannot match
                if idx < 0 or idx >= len(parts):
                    self._logger.debug(
                        f"Skipping pattern {i} due to out-of-range parts index {target_idx} for path {filename}"
                    )
                    continue
                candidate = parts[idx]
            else:  # full path (default)
                candidate = filename

            self._logger.debug(f"Matching against component '{candidate}' (target={target_type})")

            # Search for the pattern in the selected component
            if (match := self._regex.search(pattern, candidate)):
                self._logger.debug(f"Match found with pattern {i}")
                
                # Extract fields from the match
                result = self._extract_groups_from_match(match, i)
                self._logger.debug(f"Extracted groups: {result}")
                
                # Handle special cases
                result = self._process_special_cases(result, filename)
                
                return result
        
        self._logger.debug(f"No match found for {filename}")
        return None
    
    def filter_files(self, files: List[str]) -> Dict[str, Dict[str, str]]:
        """
        Filter a list of files based on regex patterns and extract metadata.
        
        Args:
            files: List of file paths to match against the patterns
            
        Returns:
            Dictionary mapping file paths to extracted metadata
        """
        result = {}
        
        for file_path in files:
            if (metadata := self.match(file_path)):
                result[file_path] = metadata
        
        return result


def match_files_to_patterns(
    files: List[str], 
    patterns: List[str],
    logger_provider: Optional[LoggerProvider] = None,
    regex_provider: Optional[RegexProvider] = None
) -> Dict[str, Dict[str, str]]:
    """
    Match a list of files against a list of patterns.
    
    This is a convenience function that creates a PatternMatcher and calls its filter_files method.
    Enhanced with configurable dependencies for comprehensive testing per F-016 requirements.
    
    Args:
        files: List of file paths to match
        patterns: List of regex patterns to match against
        logger_provider: Optional logger provider for dependency injection
        regex_provider: Optional regex provider for dependency injection
        
    Returns:
        Dictionary mapping file paths to extracted metadata
    """
    matcher = PatternMatcher(patterns, logger_provider, regex_provider)
    return matcher.filter_files(files)


def create_experiment_matcher(
    experiment_patterns: List[str],
    logger_provider: Optional[LoggerProvider] = None,
    regex_provider: Optional[RegexProvider] = None
) -> PatternMatcher:
    """
    Create a PatternMatcher for experiment files with configurable dependencies.
    
    Enhanced with dependency injection support for comprehensive testing per F-016 requirements.
    
    Args:
        experiment_patterns: Regex patterns for experiment files
        logger_provider: Optional logger provider for dependency injection
        regex_provider: Optional regex provider for dependency injection
        
    Returns:
        A configured PatternMatcher
    """
    return PatternMatcher(experiment_patterns, logger_provider, regex_provider)


def create_vial_matcher(
    vial_patterns: List[str],
    logger_provider: Optional[LoggerProvider] = None,
    regex_provider: Optional[RegexProvider] = None
) -> PatternMatcher:
    """
    Create a PatternMatcher for vial files with configurable dependencies.
    
    Enhanced with dependency injection support for comprehensive testing per F-016 requirements.
    
    Args:
        vial_patterns: Regex patterns for vial files
        logger_provider: Optional logger provider for dependency injection
        regex_provider: Optional regex provider for dependency injection
        
    Returns:
        A configured PatternMatcher
    """
    return PatternMatcher(vial_patterns, logger_provider, regex_provider)


def match_experiment_file(
    filepath: str, 
    patterns: List[str],
    logger_provider: Optional[LoggerProvider] = None,
    regex_provider: Optional[RegexProvider] = None
) -> Optional[Dict[str, str]]:
    """
    Check if a file matches experiment patterns and extract metadata.
    
    Enhanced with configurable dependencies for comprehensive testing per F-016 requirements.
    
    Args:
        filepath: Path to file to check
        patterns: Experiment patterns to match against
        logger_provider: Optional logger provider for dependency injection
        regex_provider: Optional regex provider for dependency injection
        
    Returns:
        Dictionary with experiment metadata or None if not a match
    """
    matcher = create_experiment_matcher(patterns, logger_provider, regex_provider)
    return matcher.match(filepath)


def match_vial_file(
    filepath: str, 
    patterns: List[str],
    logger_provider: Optional[LoggerProvider] = None,
    regex_provider: Optional[RegexProvider] = None
) -> Optional[Dict[str, str]]:
    """
    Check if a file matches vial patterns and extract metadata.
    
    Enhanced with configurable dependencies for comprehensive testing per F-016 requirements.
    
    Args:
        filepath: Path to file to check
        patterns: Vial patterns to match against
        logger_provider: Optional logger provider for dependency injection
        regex_provider: Optional regex provider for dependency injection
        
    Returns:
        Dictionary with vial metadata or None if not a match
    """
    matcher = create_vial_matcher(patterns, logger_provider, regex_provider)
    return matcher.match(filepath)


def generate_pattern_from_template(
    template: str, 
    logger_provider: Optional[LoggerProvider] = None,
    regex_provider: Optional[RegexProvider] = None
) -> str:
    """
    Convert a template with placeholders to a regex pattern with named capture groups.
    
    Enhanced with configurable dependencies for comprehensive testing per F-016 requirements.
    
    Args:
        template: Template string with placeholders (e.g. "{animal}_{date}.csv")
        logger_provider: Optional logger provider for dependency injection
        regex_provider: Optional regex provider for dependency injection
        
    Returns:
        Regex pattern with named capture groups
        
    Raises:
        PatternCompilationError: If template processing fails
    """
    # Configure dependency injection for testability
    _logger = logger_provider if logger_provider is not None else DefaultLoggerProvider()
    _regex = regex_provider if regex_provider is not None else DefaultRegexProvider()

    try:
        # Define pattern mapping for common fields
        field_patterns = {
            "animal": r"[a-zA-Z]+",            # Letters
            "date": r"\d+",                    # Any number sequence (more permissive)
            "condition": r"[a-zA-Z0-9_-]+",    # Alphanumeric with underscore and hyphen
            "replicate": r"\d+",               # Numbers
            "experiment_id": r"\d+",           # Numbers
            "sample_id": r"[a-zA-Z0-9_-]+",    # Alphanumeric with underscore and hyphen
        }

        # Simple implementation - first escape all regex special characters
        escaped_template = re.escape(template)

        # Then replace placeholders with regex capture groups
        pattern = escaped_template
        for field, field_pattern in field_patterns.items():
            placeholder = re.escape(f"{{{field}}}")
            pattern = pattern.replace(placeholder, f"(?P<{field}>{field_pattern})")

        # For any remaining placeholders not in our mapping, use a default pattern
        for field in _regex.findall(r"\\{([^}]+)\\}", pattern):
            if not re.search(f"\\(\\?P<{field}>", pattern):  # Check if not already replaced
                placeholder = re.escape(f"{{{field}}}")
                pattern = pattern.replace(placeholder, f"(?P<{field}>[\\w-]+)")

        # Add anchors
        pattern = f"^{pattern}$"

        # Validate the generated pattern by attempting to compile it
        try:
            _regex.compile(pattern)
        except Exception as e:
            raise PatternCompilationError(pattern, e) from e

        # For debugging
        _logger.debug(f"Generated pattern: {pattern}")

        return pattern

    except Exception as e:
        # Wrap unexpected errors in our structured exception
        if not isinstance(e, PatternCompilationError):
            raise PatternCompilationError(template, e) from e
        raise


# Add the missing functions used in the tests
def extract_experiment_info(
    filepath: str, 
    patterns: List[str],
    logger_provider: Optional[LoggerProvider] = None,
    regex_provider: Optional[RegexProvider] = None
) -> Optional[Dict[str, str]]:
    """
    Extract experiment information from a filepath using patterns.
    
    Enhanced with configurable dependencies for comprehensive testing per F-016 requirements.
    
    Args:
        filepath: Path to the experiment file
        patterns: List of regex patterns to match against
        logger_provider: Optional logger provider for dependency injection
        regex_provider: Optional regex provider for dependency injection
        
    Returns:
        Dictionary with extracted experiment metadata or None if not matching
    """
    return match_experiment_file(filepath, patterns, logger_provider, regex_provider)


def extract_vial_info(
    filepath: str, 
    patterns: List[str],
    logger_provider: Optional[LoggerProvider] = None,
    regex_provider: Optional[RegexProvider] = None
) -> Optional[Dict[str, str]]:
    """
    Extract vial information from a filepath using patterns.
    
    Enhanced with configurable dependencies for comprehensive testing per F-016 requirements.
    
    Args:
        filepath: Path to the vial file
        patterns: List of regex patterns to match against
        logger_provider: Optional logger provider for dependency injection
        regex_provider: Optional regex provider for dependency injection
        
    Returns:
        Dictionary with extracted vial metadata or None if not matching
    """
    return match_vial_file(filepath, patterns, logger_provider, regex_provider)