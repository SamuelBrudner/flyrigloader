"""
DiscoveryOptions - Immutable configuration for file discovery operations.

This module provides a type-safe, immutable dataclass for configuring file
discovery behavior, consolidating multiple parameters into a single object.

Following our project standards:
- Frozen dataclass for immutability and thread safety
- Comprehensive validation with recovery hints
- Factory methods for common patterns
- Full type annotations
"""

from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path


@dataclass(frozen=True)
class DiscoveryOptions:
    """
    Immutable configuration for file discovery operations.
    
    This dataclass consolidates all file discovery parameters into a single,
    type-safe object. Being frozen makes it immutable and thread-safe.
    
    Attributes:
        pattern: Glob pattern for file matching (default: "*.*")
        recursive: Search subdirectories recursively (default: True)
        extensions: Filter by file extensions, e.g., ['.pkl', '.csv'] (default: None)
        extract_metadata: Extract metadata from filenames using config patterns (default: False)
        parse_dates: Parse dates from filename components (default: False)
    
    Examples:
        >>> # Use defaults
        >>> options = DiscoveryOptions.defaults()
        
        >>> # Simple pattern
        >>> options = DiscoveryOptions.minimal("*.pkl")
        
        >>> # With metadata extraction
        >>> options = DiscoveryOptions.with_metadata("exp_*.pkl", parse_dates=True)
        
        >>> # With extension filtering
        >>> options = DiscoveryOptions.with_filtering(extensions=['.pkl', '.csv'])
        
        >>> # Custom configuration
        >>> options = DiscoveryOptions(
        ...     pattern="data_*.pkl",
        ...     recursive=False,
        ...     extract_metadata=True
        ... )
    """
    
    pattern: str = "*.*"
    recursive: bool = True
    extensions: Optional[List[str]] = None
    extract_metadata: bool = False
    parse_dates: bool = False
    
    def __post_init__(self):
        """
        Validate options after initialization.
        
        Raises:
            ValueError: If any option is invalid, with recovery hint
            TypeError: If types are incorrect, with recovery hint
        """
        # Pattern validation
        if not isinstance(self.pattern, str):
            raise ValueError(
                f"pattern must be a string, got {type(self.pattern).__name__}. "
                f"Provide a glob pattern string. Example: '*.pkl' or 'data_*.csv'"
            )
        
        if not self.pattern.strip():
            raise ValueError(
                "pattern cannot be empty. "
                "Provide a valid glob pattern. Example: '*.*' for all files"
            )
        
        # Recursive validation
        if not isinstance(self.recursive, bool):
            raise TypeError(
                f"recursive must be a boolean, got {type(self.recursive).__name__}. "
                f"Use True or False for recursive parameter. Example: recursive=True"
            )
        
        # Extensions validation
        if self.extensions is not None:
            if not isinstance(self.extensions, list):
                raise ValueError(
                    f"extensions must be a list, got {type(self.extensions).__name__}. "
                    f"Provide a list of extension strings. Example: ['.pkl', '.csv']"
                )
            
            for ext in self.extensions:
                if not isinstance(ext, str):
                    raise ValueError(
                        f"Each extension must be a string, got {type(ext).__name__}. "
                        f"Use string extensions. Example: ['.pkl', '.csv'] not [.pkl, .csv]"
                    )
        
        # extract_metadata validation
        if not isinstance(self.extract_metadata, bool):
            raise TypeError(
                f"extract_metadata must be a boolean, got {type(self.extract_metadata).__name__}. "
                f"Use True or False for extract_metadata. Example: extract_metadata=True"
            )
        
        # parse_dates validation
        if not isinstance(self.parse_dates, bool):
            raise TypeError(
                f"parse_dates must be a boolean, got {type(self.parse_dates).__name__}. "
                f"Use True or False for parse_dates. Example: parse_dates=True"
            )
    
    @classmethod
    def defaults(cls) -> 'DiscoveryOptions':
        """
        Create default discovery options.
        
        Returns:
            DiscoveryOptions with pattern='*.*', recursive=True, no metadata extraction
            
        Example:
            >>> options = DiscoveryOptions.defaults()
            >>> options.pattern
            '*.*'
            >>> options.recursive
            True
            >>> options.extract_metadata
            False
        """
        return cls()
    
    @classmethod
    def minimal(cls, pattern: str = "*.*") -> 'DiscoveryOptions':
        """
        Create minimal discovery options with just a pattern.
        
        Args:
            pattern: Glob pattern for file matching (default: "*.*")
            
        Returns:
            DiscoveryOptions with specified pattern, recursive=True, no metadata
            
        Example:
            >>> options = DiscoveryOptions.minimal("*.pkl")
            >>> options.pattern
            '*.pkl'
            >>> options.extract_metadata
            False
        """
        return cls(pattern=pattern, recursive=True)
    
    @classmethod
    def with_metadata(
        cls, 
        pattern: str = "*.*",
        parse_dates: bool = True,
        recursive: bool = True
    ) -> 'DiscoveryOptions':
        """
        Create options configured for metadata extraction.
        
        Args:
            pattern: Glob pattern for file matching (default: "*.*")
            parse_dates: Whether to parse dates from filenames (default: True)
            recursive: Search recursively (default: True)
            
        Returns:
            DiscoveryOptions with metadata extraction enabled
            
        Example:
            >>> options = DiscoveryOptions.with_metadata("*.pkl", parse_dates=True)
            >>> options.extract_metadata
            True
            >>> options.parse_dates
            True
        """
        return cls(
            pattern=pattern,
            recursive=recursive,
            extract_metadata=True,
            parse_dates=parse_dates
        )
    
    @classmethod
    def with_filtering(
        cls,
        pattern: str = "*.*",
        extensions: Optional[List[str]] = None,
        recursive: bool = True
    ) -> 'DiscoveryOptions':
        """
        Create options configured for file type filtering.
        
        Args:
            pattern: Glob pattern for file matching (default: "*.*")
            extensions: List of file extensions to include (default: None)
            recursive: Search recursively (default: True)
            
        Returns:
            DiscoveryOptions with extension filtering
            
        Example:
            >>> options = DiscoveryOptions.with_filtering("*.*", ['.pkl', '.csv'])
            >>> options.extensions
            ['.pkl', '.csv']
        """
        return cls(
            pattern=pattern,
            recursive=recursive,
            extensions=extensions
        )
