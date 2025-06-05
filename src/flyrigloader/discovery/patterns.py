"""
Pattern matching functionality.

Utilities for working with regex patterns to extract metadata from filenames.
Enhanced with dependency injection patterns for comprehensive testing support.
"""
from typing import Any, Dict, List, Optional, Pattern, Union, Tuple, Protocol, runtime_checkable
import re
from loguru import logger
from pathlib import Path
from abc import ABC, abstractmethod


@runtime_checkable
class RegexEngine(Protocol):
    """Protocol for regex compilation and matching operations."""
    
    def compile(self, pattern: str, flags: int = 0) -> Pattern[str]:
        """Compile a regex pattern."""
        ...
    
    def search(self, pattern: Pattern[str], string: str) -> Optional[re.Match[str]]:
        """Search for pattern in string."""
        ...


@runtime_checkable
class Logger(Protocol):
    """Protocol for logging operations."""
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        ...
    
    def level(self, level: str) -> None:
        """Set logging level."""
        ...


class StandardRegexEngine:
    """Standard implementation of RegexEngine using Python's re module."""
    
    def compile(self, pattern: str, flags: int = 0) -> Pattern[str]:
        """Compile a regex pattern using the standard re module."""
        try:
            return re.compile(pattern, flags)
        except re.error as e:
            raise PatternCompilationError(
                pattern=pattern,
                original_error=str(e),
                error_type="regex_compilation_error"
            ) from e
    
    def search(self, pattern: Pattern[str], string: str) -> Optional[re.Match[str]]:
        """Search for pattern in string using the standard re module."""
        try:
            return pattern.search(string)
        except Exception as e:
            raise PatternMatchingError(
                pattern=pattern.pattern,
                text=string,
                original_error=str(e),
                error_type="pattern_matching_error"
            ) from e


class LoguruLogger:
    """Standard implementation of Logger using Loguru."""
    
    def debug(self, message: str) -> None:
        """Log debug message using Loguru."""
        logger.debug(message)
    
    def level(self, level: str) -> None:
        """Set logging level using Loguru."""
        logger.level(level)


class PatternCompilationError(Exception):
    """Structured exception for pattern compilation failures."""
    
    def __init__(self, pattern: str, original_error: str, error_type: str):
        self.pattern = pattern
        self.original_error = original_error
        self.error_type = error_type
        super().__init__(f"Failed to compile pattern '{pattern}': {original_error}")


class PatternMatchingError(Exception):
    """Structured exception for pattern matching failures."""
    
    def __init__(self, pattern: str, text: str, original_error: str, error_type: str):
        self.pattern = pattern
        self.text = text
        self.original_error = original_error
        self.error_type = error_type
        super().__init__(f"Failed to match pattern '{pattern}' against text '{text}': {original_error}")


class PatternMatcher:
    """
    Class for matching and extracting information from filenames based on regex patterns.
    
    This class provides methods for matching filenames, filtering files, and extracting
    information based on named groups in the regex patterns. Enhanced with dependency
    injection for comprehensive testing support.
    """
    
    def __init__(
        self, 
        patterns: List[str],
        regex_engine: Optional[RegexEngine] = None,
        logger_instance: Optional[Logger] = None
    ):
        """
        Initialize the PatternMatcher with regex patterns and configurable dependencies.
        
        Args:
            patterns: List of regex pattern strings
            regex_engine: Optional regex engine for pattern compilation and matching.
                         Defaults to StandardRegexEngine for production use.
            logger_instance: Optional logger instance for logging operations.
                           Defaults to LoguruLogger for production use.
        """
        # Store original patterns for debugging and error context
        self.patterns = patterns
        
        # Inject dependencies with sensible defaults
        self._regex_engine = regex_engine or StandardRegexEngine()
        self._logger = logger_instance or LoguruLogger()
        
        # Compile patterns using the injected regex engine with enhanced error handling
        self.compiled_patterns = []
        self._compile_patterns()
        
        # Configure logging through dependency injection
        try:
            self._logger.level("INFO")
        except Exception:
            # Gracefully handle logger configuration failures in test environments
            pass
    
    def _compile_patterns(self) -> None:
        """
        Compile all patterns using the injected regex engine with comprehensive error context.
        
        Raises:
            PatternCompilationError: If any pattern fails to compile with detailed context
        """
        compilation_errors = []
        
        for i, pattern in enumerate(self.patterns):
            try:
                compiled_pattern = self._regex_engine.compile(pattern)
                self.compiled_patterns.append(compiled_pattern)
                self._logger.debug(f"Successfully compiled pattern {i}: {pattern}")
            except Exception as e:
                error_context = {
                    "pattern_index": i,
                    "pattern": pattern,
                    "total_patterns": len(self.patterns),
                    "error_details": str(e)
                }
                compilation_errors.append(error_context)
                self._logger.debug(f"Failed to compile pattern {i}: {pattern} - {e}")
        
        # If any patterns failed to compile, raise with comprehensive context
        if compilation_errors:
            raise PatternCompilationError(
                pattern=f"{len(compilation_errors)} patterns failed",
                original_error=f"Compilation errors: {compilation_errors}",
                error_type="batch_compilation_error"
            )
    
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
            if (condition_match := re.match(r"(.+)_(\d+)$", result["condition"])):
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
        Match a filename against the compiled regex patterns using the injected regex engine.
        
        Args:
            filename: The filename (or full path) to match against patterns
            
        Returns:
            Dictionary with extracted metadata from the first matching pattern, or None if no match
            
        Raises:
            PatternMatchingError: If pattern matching fails with structured error context
        """
        try:
            self._logger.debug(f"Attempting to match patterns against: {filename}")
            
            # Try each pattern until we find a match
            for i, pattern in enumerate(self.compiled_patterns):
                self._logger.debug(f"Trying pattern {i}: {self.patterns[i]}")
                
                try:
                    # Use injected regex engine for pattern matching
                    match = self._regex_engine.search(pattern, filename)
                    
                    if match:
                        self._logger.debug(f"Match found with pattern {i}")
                        
                        # Extract fields from the match
                        result = self._extract_groups_from_match(match, i)
                        self._logger.debug(f"Extracted groups: {result}")
                        
                        # Handle special cases
                        result = self._process_special_cases(result, filename)
                        
                        return result
                        
                except Exception as e:
                    # Log pattern-specific failures but continue with next pattern
                    self._logger.debug(f"Pattern {i} matching failed: {e}")
                    continue
            
            self._logger.debug(f"No match found for {filename}")
            return None
            
        except Exception as e:
            # Provide structured error context for debugging
            raise PatternMatchingError(
                pattern=f"batch of {len(self.compiled_patterns)} patterns",
                text=filename,
                original_error=str(e),
                error_type="batch_matching_error"
            ) from e
    
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
    regex_engine: Optional[RegexEngine] = None,
    logger_instance: Optional[Logger] = None
) -> Dict[str, Dict[str, str]]:
    """
    Match a list of files against a list of patterns with configurable dependencies.
    
    This is a convenience function that creates a PatternMatcher and calls its filter_files method.
    Enhanced with dependency injection for testing support.
    
    Args:
        files: List of file paths to match
        patterns: List of regex patterns to match against
        regex_engine: Optional regex engine for pattern operations
        logger_instance: Optional logger for operation logging
        
    Returns:
        Dictionary mapping file paths to extracted metadata
    """
    matcher = PatternMatcher(patterns, regex_engine=regex_engine, logger_instance=logger_instance)
    return matcher.filter_files(files)


def create_experiment_matcher(
    experiment_patterns: List[str],
    regex_engine: Optional[RegexEngine] = None,
    logger_instance: Optional[Logger] = None
) -> PatternMatcher:
    """
    Create a PatternMatcher for experiment files with configurable dependencies.
    
    Args:
        experiment_patterns: Regex patterns for experiment files
        regex_engine: Optional regex engine for pattern operations
        logger_instance: Optional logger for operation logging
        
    Returns:
        A configured PatternMatcher with dependency injection support
    """
    return PatternMatcher(experiment_patterns, regex_engine=regex_engine, logger_instance=logger_instance)


def create_vial_matcher(
    vial_patterns: List[str],
    regex_engine: Optional[RegexEngine] = None,
    logger_instance: Optional[Logger] = None
) -> PatternMatcher:
    """
    Create a PatternMatcher for vial files with configurable dependencies.
    
    Args:
        vial_patterns: Regex patterns for vial files
        regex_engine: Optional regex engine for pattern operations
        logger_instance: Optional logger for operation logging
        
    Returns:
        A configured PatternMatcher with dependency injection support
    """
    return PatternMatcher(vial_patterns, regex_engine=regex_engine, logger_instance=logger_instance)


def match_experiment_file(
    filepath: str, 
    patterns: List[str],
    regex_engine: Optional[RegexEngine] = None,
    logger_instance: Optional[Logger] = None
) -> Optional[Dict[str, str]]:
    """
    Check if a file matches experiment patterns and extract metadata with configurable dependencies.
    
    Args:
        filepath: Path to file to check
        patterns: Experiment patterns to match against
        regex_engine: Optional regex engine for pattern operations
        logger_instance: Optional logger for operation logging
        
    Returns:
        Dictionary with experiment metadata or None if not a match
    """
    matcher = create_experiment_matcher(patterns, regex_engine=regex_engine, logger_instance=logger_instance)
    return matcher.match(filepath)


def match_vial_file(
    filepath: str, 
    patterns: List[str],
    regex_engine: Optional[RegexEngine] = None,
    logger_instance: Optional[Logger] = None
) -> Optional[Dict[str, str]]:
    """
    Check if a file matches vial patterns and extract metadata with configurable dependencies.
    
    Args:
        filepath: Path to file to check
        patterns: Vial patterns to match against
        regex_engine: Optional regex engine for pattern operations
        logger_instance: Optional logger for operation logging
        
    Returns:
        Dictionary with vial metadata or None if not a match
    """
    matcher = create_vial_matcher(patterns, regex_engine=regex_engine, logger_instance=logger_instance)
    return matcher.match(filepath)


def generate_pattern_from_template(
    template: str,
    regex_engine: Optional[RegexEngine] = None,
    logger_instance: Optional[Logger] = None
) -> str:
    """
    Convert a template with placeholders to a regex pattern with named capture groups.
    Enhanced with dependency injection for testing support.
    
    Args:
        template: Template string with placeholders (e.g. "{animal}_{date}.csv")
        regex_engine: Optional regex engine for pattern validation
        logger_instance: Optional logger for operation logging
        
    Returns:
        Regex pattern with named capture groups
        
    Raises:
        PatternCompilationError: If the generated pattern is invalid
    """
    # Inject dependencies with sensible defaults
    _regex_engine = regex_engine or StandardRegexEngine()
    _logger = logger_instance or LoguruLogger()
    
    # Define pattern mapping for common fields
    field_patterns = {
        "animal": r"[a-zA-Z]+",            # Letters
        "date": r"\d+",                    # Any number sequence (more permissive)
        "condition": r"[a-zA-Z0-9_-]+",    # Alphanumeric with underscore and hyphen
        "replicate": r"\d+",               # Numbers
        "experiment_id": r"\d+",           # Numbers
        "sample_id": r"[a-zA-Z0-9_-]+",    # Alphanumeric with underscore and hyphen
    }
    
    try:
        # Simple implementation - first escape all regex special characters
        escaped_template = re.escape(template)
        
        # Then replace placeholders with regex capture groups
        pattern = escaped_template
        for field, field_pattern in field_patterns.items():
            placeholder = re.escape(f"{{{field}}}")
            pattern = pattern.replace(placeholder, f"(?P<{field}>{field_pattern})")
        
        # For any remaining placeholders not in our mapping, use a default pattern
        for field in re.findall(r"\\{([^}]+)\\}", pattern):
            if not re.search(f"\\(\\?P<{field}>", pattern):  # Check if not already replaced
                placeholder = re.escape(f"{{{field}}}")
                pattern = pattern.replace(placeholder, f"(?P<{field}>[\\w-]+)")
        
        # Add anchors
        pattern = f"^{pattern}$"
        
        # Validate the generated pattern by attempting to compile it
        _regex_engine.compile(pattern)
        
        # For debugging
        _logger.debug(f"Generated pattern: {pattern}")
        
        return pattern
        
    except Exception as e:
        raise PatternCompilationError(
            pattern=f"template:{template}",
            original_error=str(e),
            error_type="template_generation_error"
        ) from e


# Enhanced functions with dependency injection support for comprehensive testing
def extract_experiment_info(
    filepath: str, 
    patterns: List[str],
    regex_engine: Optional[RegexEngine] = None,
    logger_instance: Optional[Logger] = None
) -> Optional[Dict[str, str]]:
    """
    Extract experiment information from a filepath using patterns with configurable dependencies.
    
    Args:
        filepath: Path to the experiment file
        patterns: List of regex patterns to match against
        regex_engine: Optional regex engine for pattern operations
        logger_instance: Optional logger for operation logging
        
    Returns:
        Dictionary with extracted experiment metadata or None if not matching
    """
    return match_experiment_file(filepath, patterns, regex_engine=regex_engine, logger_instance=logger_instance)


def extract_vial_info(
    filepath: str, 
    patterns: List[str],
    regex_engine: Optional[RegexEngine] = None,
    logger_instance: Optional[Logger] = None
) -> Optional[Dict[str, str]]:
    """
    Extract vial information from a filepath using patterns with configurable dependencies.
    
    Args:
        filepath: Path to the vial file
        patterns: List of regex patterns to match against
        regex_engine: Optional regex engine for pattern operations
        logger_instance: Optional logger for operation logging
        
    Returns:
        Dictionary with extracted vial metadata or None if not matching
    """
    return match_vial_file(filepath, patterns, regex_engine=regex_engine, logger_instance=logger_instance)


# Additional testing support functions for comprehensive test scenarios
def create_pattern_matcher_with_dependencies(
    patterns: List[str],
    regex_engine: Optional[RegexEngine] = None,
    logger_instance: Optional[Logger] = None
) -> PatternMatcher:
    """
    Factory function for creating PatternMatcher instances with full dependency injection.
    
    This function provides explicit dependency injection support for test scenarios
    requiring complete control over regex compilation and logging behavior.
    
    Args:
        patterns: List of regex pattern strings
        regex_engine: Regex engine implementation for pattern operations
        logger_instance: Logger implementation for operation logging
        
    Returns:
        PatternMatcher instance with injected dependencies
    """
    return PatternMatcher(patterns, regex_engine=regex_engine, logger_instance=logger_instance)


class MockablePatternMatcher(PatternMatcher):
    """
    Extended PatternMatcher class with additional test hooks for pytest.monkeypatch scenarios.
    
    This class provides additional entry points for testing complex pattern matching
    scenarios with mock regex engines and custom logging behavior.
    """
    
    def get_compiled_patterns(self) -> List[Pattern[str]]:
        """
        Expose compiled patterns for testing and validation.
        
        Returns:
            List of compiled regex patterns
        """
        return self.compiled_patterns
    
    def get_original_patterns(self) -> List[str]:
        """
        Expose original pattern strings for testing and debugging.
        
        Returns:
            List of original pattern strings
        """
        return self.patterns
    
    def get_regex_engine(self) -> RegexEngine:
        """
        Expose the injected regex engine for testing verification.
        
        Returns:
            The current regex engine instance
        """
        return self._regex_engine
    
    def get_logger(self) -> Logger:
        """
        Expose the injected logger for testing verification.
        
        Returns:
            The current logger instance
        """
        return self._logger
    
    def set_regex_engine(self, regex_engine: RegexEngine) -> None:
        """
        Replace the regex engine at runtime for dynamic testing scenarios.
        
        Args:
            regex_engine: New regex engine to use for pattern operations
        """
        self._regex_engine = regex_engine
        # Recompile patterns with the new engine
        self.compiled_patterns = []
        self._compile_patterns()
    
    def set_logger(self, logger_instance: Logger) -> None:
        """
        Replace the logger at runtime for dynamic testing scenarios.
        
        Args:
            logger_instance: New logger to use for operation logging
        """
        self._logger = logger_instance