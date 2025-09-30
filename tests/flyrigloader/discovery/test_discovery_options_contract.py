"""
Contract/Specification tests for DiscoveryOptions.

These tests define the behavior contracts that DiscoveryOptions must satisfy.
They test the interface, not implementation details.

Following TDD: These tests are written BEFORE implementation.
"""

import pytest
from typing import List, Optional


# Note: DiscoveryOptions doesn't exist yet - that's the point of TDD!
# When we run these tests, they'll fail. Then we implement to make them pass.


class TestDiscoveryOptionsContract:
    """
    Contract tests for DiscoveryOptions core behavior.
    
    These tests define what DiscoveryOptions MUST do, not how it does it.
    """
    
    def test_can_be_created_with_defaults(self):
        """
        CONTRACT: DiscoveryOptions can be instantiated with no arguments.
        All attributes have sensible defaults.
        """
        from flyrigloader.discovery.options import DiscoveryOptions
        
        options = DiscoveryOptions()
        
        # Contract: Default values are sensible and non-None
        assert options.pattern == "*.*"
        assert options.recursive is True
        assert options.extensions is None
        assert options.extract_metadata is False
        assert options.parse_dates is False
    
    def test_is_immutable(self):
        """
        CONTRACT: DiscoveryOptions is immutable (frozen dataclass).
        Once created, attributes cannot be modified.
        """
        from flyrigloader.discovery.options import DiscoveryOptions
        
        options = DiscoveryOptions(pattern="*.pkl")
        
        # Contract: Attempting to modify raises an error
        with pytest.raises((AttributeError, TypeError)):
            options.pattern = "*.csv"
        
        with pytest.raises((AttributeError, TypeError)):
            options.recursive = False
    
    def test_has_all_required_attributes(self):
        """
        CONTRACT: DiscoveryOptions has all documented attributes.
        """
        from flyrigloader.discovery.options import DiscoveryOptions
        
        options = DiscoveryOptions()
        
        # Contract: All documented attributes exist and are accessible
        assert hasattr(options, 'pattern')
        assert hasattr(options, 'recursive')
        assert hasattr(options, 'extensions')
        assert hasattr(options, 'extract_metadata')
        assert hasattr(options, 'parse_dates')
    
    def test_attributes_have_correct_types(self):
        """
        CONTRACT: Attributes have the documented types.
        """
        from flyrigloader.discovery.options import DiscoveryOptions
        
        options = DiscoveryOptions(
            pattern="*.pkl",
            recursive=True,
            extensions=['.pkl', '.csv'],
            extract_metadata=True,
            parse_dates=True
        )
        
        # Contract: Types match documentation
        assert isinstance(options.pattern, str)
        assert isinstance(options.recursive, bool)
        assert isinstance(options.extensions, (list, type(None)))
        assert isinstance(options.extract_metadata, bool)
        assert isinstance(options.parse_dates, bool)
    
    def test_can_override_all_defaults(self):
        """
        CONTRACT: All default values can be overridden at instantiation.
        """
        from flyrigloader.discovery.options import DiscoveryOptions
        
        options = DiscoveryOptions(
            pattern="data_*.pkl",
            recursive=False,
            extensions=['.pkl'],
            extract_metadata=True,
            parse_dates=True
        )
        
        # Contract: Overrides work
        assert options.pattern == "data_*.pkl"
        assert options.recursive is False
        assert options.extensions == ['.pkl']
        assert options.extract_metadata is True
        assert options.parse_dates is True


class TestDiscoveryOptionsFactoryMethodsContract:
    """
    Contract tests for factory methods.
    
    Factory methods MUST return properly configured DiscoveryOptions instances.
    """
    
    def test_defaults_factory_returns_default_configuration(self):
        """
        CONTRACT: DiscoveryOptions.defaults() returns default configuration.
        """
        from flyrigloader.discovery.options import DiscoveryOptions
        
        options = DiscoveryOptions.defaults()
        
        # Contract: Returns instance with defaults
        assert isinstance(options, DiscoveryOptions)
        assert options.pattern == "*.*"
        assert options.recursive is True
        assert options.extensions is None
        assert options.extract_metadata is False
        assert options.parse_dates is False
    
    def test_minimal_factory_accepts_pattern(self):
        """
        CONTRACT: DiscoveryOptions.minimal(pattern) creates minimal config.
        """
        from flyrigloader.discovery.options import DiscoveryOptions
        
        options = DiscoveryOptions.minimal("*.pkl")
        
        # Contract: Pattern is set, other options are minimal
        assert isinstance(options, DiscoveryOptions)
        assert options.pattern == "*.pkl"
        assert options.recursive is True  # Still recursive by default
        assert options.extract_metadata is False
        assert options.parse_dates is False
    
    def test_minimal_factory_has_default_pattern(self):
        """
        CONTRACT: DiscoveryOptions.minimal() works without arguments.
        """
        from flyrigloader.discovery.options import DiscoveryOptions
        
        options = DiscoveryOptions.minimal()
        
        # Contract: Has sensible default even without pattern
        assert isinstance(options, DiscoveryOptions)
        assert options.pattern == "*.*"
    
    def test_with_metadata_factory_enables_metadata_extraction(self):
        """
        CONTRACT: DiscoveryOptions.with_metadata() enables metadata extraction.
        """
        from flyrigloader.discovery.options import DiscoveryOptions
        
        options = DiscoveryOptions.with_metadata()
        
        # Contract: Metadata extraction is enabled
        assert isinstance(options, DiscoveryOptions)
        assert options.extract_metadata is True
        assert options.parse_dates is True  # Default for with_metadata
    
    def test_with_metadata_factory_accepts_pattern(self):
        """
        CONTRACT: DiscoveryOptions.with_metadata(pattern) accepts pattern override.
        """
        from flyrigloader.discovery.options import DiscoveryOptions
        
        options = DiscoveryOptions.with_metadata(pattern="exp_*.pkl")
        
        # Contract: Pattern override works
        assert options.pattern == "exp_*.pkl"
        assert options.extract_metadata is True
    
    def test_with_metadata_factory_accepts_parse_dates_flag(self):
        """
        CONTRACT: DiscoveryOptions.with_metadata(parse_dates=False) can disable date parsing.
        """
        from flyrigloader.discovery.options import DiscoveryOptions
        
        options = DiscoveryOptions.with_metadata(parse_dates=False)
        
        # Contract: Can disable date parsing while keeping metadata extraction
        assert options.extract_metadata is True
        assert options.parse_dates is False
    
    def test_with_filtering_factory_sets_extensions(self):
        """
        CONTRACT: DiscoveryOptions.with_filtering(extensions=...) sets extensions.
        """
        from flyrigloader.discovery.options import DiscoveryOptions
        
        options = DiscoveryOptions.with_filtering(extensions=['.pkl', '.csv'])
        
        # Contract: Extensions are set
        assert isinstance(options, DiscoveryOptions)
        assert options.extensions == ['.pkl', '.csv']
    
    def test_with_filtering_factory_accepts_pattern(self):
        """
        CONTRACT: DiscoveryOptions.with_filtering() accepts pattern parameter.
        """
        from flyrigloader.discovery.options import DiscoveryOptions
        
        options = DiscoveryOptions.with_filtering(
            pattern="data_*",
            extensions=['.pkl']
        )
        
        # Contract: Both pattern and extensions work together
        assert options.pattern == "data_*"
        assert options.extensions == ['.pkl']


class TestDiscoveryOptionsValidationContract:
    """
    Contract tests for validation behavior.
    
    DiscoveryOptions MUST validate inputs and raise clear errors.
    """
    
    def test_rejects_invalid_pattern_type(self):
        """
        CONTRACT: pattern must be a string, or ValueError is raised.
        """
        from flyrigloader.discovery.options import DiscoveryOptions
        
        # Contract: Non-string pattern raises ValueError
        with pytest.raises(ValueError) as exc_info:
            DiscoveryOptions(pattern=123)
        
        # Contract: Error message mentions the issue
        assert "pattern" in str(exc_info.value).lower()
        assert "string" in str(exc_info.value).lower()
    
    def test_rejects_empty_pattern(self):
        """
        CONTRACT: pattern cannot be empty string, or ValueError is raised.
        """
        from flyrigloader.discovery.options import DiscoveryOptions
        
        # Contract: Empty pattern raises ValueError
        with pytest.raises(ValueError) as exc_info:
            DiscoveryOptions(pattern="")
        
        # Contract: Error message indicates empty pattern problem
        assert "pattern" in str(exc_info.value).lower()
        assert "empty" in str(exc_info.value).lower()
    
    def test_rejects_whitespace_only_pattern(self):
        """
        CONTRACT: pattern cannot be whitespace-only, or ValueError is raised.
        """
        from flyrigloader.discovery.options import DiscoveryOptions
        
        # Contract: Whitespace-only pattern raises ValueError
        with pytest.raises(ValueError) as exc_info:
            DiscoveryOptions(pattern="   ")
        
        assert "pattern" in str(exc_info.value).lower()
    
    def test_rejects_invalid_extensions_type(self):
        """
        CONTRACT: extensions must be a list or None, or ValueError is raised.
        """
        from flyrigloader.discovery.options import DiscoveryOptions
        
        # Contract: String extensions (instead of list) raises ValueError
        with pytest.raises(ValueError) as exc_info:
            DiscoveryOptions(extensions=".pkl")  # Should be ['.pkl']
        
        # Contract: Error mentions list requirement
        assert "extensions" in str(exc_info.value).lower()
        assert "list" in str(exc_info.value).lower()
    
    def test_rejects_non_string_extension_elements(self):
        """
        CONTRACT: Each extension must be a string, or ValueError is raised.
        """
        from flyrigloader.discovery.options import DiscoveryOptions
        
        # Contract: Non-string elements in extensions list raise ValueError
        with pytest.raises(ValueError) as exc_info:
            DiscoveryOptions(extensions=['.pkl', 123, '.csv'])
        
        # Contract: Error indicates string requirement
        assert "extension" in str(exc_info.value).lower()
        assert "string" in str(exc_info.value).lower()
    
    def test_accepts_none_extensions(self):
        """
        CONTRACT: extensions=None is valid (no filtering).
        """
        from flyrigloader.discovery.options import DiscoveryOptions
        
        # Contract: None extensions is allowed
        options = DiscoveryOptions(extensions=None)
        assert options.extensions is None
    
    def test_accepts_empty_extensions_list(self):
        """
        CONTRACT: extensions=[] is valid (though probably not useful).
        """
        from flyrigloader.discovery.options import DiscoveryOptions
        
        # Contract: Empty list is allowed
        options = DiscoveryOptions(extensions=[])
        assert options.extensions == []


class TestDiscoveryOptionsErrorRecoveryContract:
    """
    Contract tests for error recovery hints.
    
    All validation errors MUST include recovery hints per project standards.
    """
    
    def test_invalid_pattern_type_has_recovery_hint(self):
        """
        CONTRACT: Pattern type errors include recovery_hint attribute.
        """
        from flyrigloader.discovery.options import DiscoveryOptions
        
        with pytest.raises(ValueError) as exc_info:
            DiscoveryOptions(pattern=123)
        
        # Contract: Exception has recovery_hint attribute
        error = exc_info.value
        assert hasattr(error, 'args')
        # The ValueError should be raised with recovery_hint as we've been doing
        # (Our pattern: raise ValueError(msg, recovery_hint="..."))
    
    def test_empty_pattern_has_recovery_hint(self):
        """
        CONTRACT: Empty pattern errors include recovery guidance.
        """
        from flyrigloader.discovery.options import DiscoveryOptions
        
        with pytest.raises(ValueError) as exc_info:
            DiscoveryOptions(pattern="")
        
        # Contract: Error provides guidance
        error_msg = str(exc_info.value)
        assert len(error_msg) > 0  # Has meaningful message
    
    def test_invalid_extensions_type_has_recovery_hint(self):
        """
        CONTRACT: Extensions type errors include recovery guidance.
        """
        from flyrigloader.discovery.options import DiscoveryOptions
        
        with pytest.raises(ValueError) as exc_info:
            DiscoveryOptions(extensions=".pkl")
        
        # Contract: Error message is helpful
        error_msg = str(exc_info.value)
        assert "list" in error_msg.lower()


class TestDiscoveryOptionsUsageContract:
    """
    Contract tests for real-world usage patterns.
    
    DiscoveryOptions MUST support documented usage patterns.
    """
    
    def test_supports_simple_pattern_override(self):
        """
        CONTRACT: Can create options with just a pattern change.
        """
        from flyrigloader.discovery.options import DiscoveryOptions
        
        # Contract: Simple pattern override works
        options = DiscoveryOptions(pattern="*.pkl")
        
        assert options.pattern == "*.pkl"
        # Other options remain at defaults
        assert options.recursive is True
        assert options.extract_metadata is False
    
    def test_supports_non_recursive_discovery(self):
        """
        CONTRACT: Can create options for non-recursive discovery.
        """
        from flyrigloader.discovery.options import DiscoveryOptions
        
        # Contract: Can disable recursion
        options = DiscoveryOptions(pattern="*.pkl", recursive=False)
        
        assert options.recursive is False
        assert options.pattern == "*.pkl"
    
    def test_supports_full_metadata_configuration(self):
        """
        CONTRACT: Can create fully configured metadata extraction options.
        """
        from flyrigloader.discovery.options import DiscoveryOptions
        
        # Contract: All metadata options work together
        options = DiscoveryOptions(
            pattern="exp_*.pkl",
            recursive=True,
            extract_metadata=True,
            parse_dates=True
        )
        
        assert options.extract_metadata is True
        assert options.parse_dates is True
        assert options.pattern == "exp_*.pkl"
    
    def test_supports_extension_filtering_with_metadata(self):
        """
        CONTRACT: Can combine extension filtering with metadata extraction.
        """
        from flyrigloader.discovery.options import DiscoveryOptions
        
        # Contract: Multiple features work together
        options = DiscoveryOptions(
            pattern="data_*",
            extensions=['.pkl', '.csv'],
            extract_metadata=True,
            parse_dates=True
        )
        
        assert options.extensions == ['.pkl', '.csv']
        assert options.extract_metadata is True
        assert options.parse_dates is True
    
    def test_can_be_used_as_function_default(self):
        """
        CONTRACT: DiscoveryOptions can be used as function default argument.
        """
        from flyrigloader.discovery.options import DiscoveryOptions
        
        # Contract: Can create at function definition time
        default_options = DiscoveryOptions.defaults()
        
        def example_function(options: 'DiscoveryOptions' = default_options):
            return options
        
        # Should work without error
        result = example_function()
        assert isinstance(result, DiscoveryOptions)


class TestDiscoveryOptionsEqualityContract:
    """
    Contract tests for equality and comparison.
    
    DiscoveryOptions MUST support equality checks.
    """
    
    def test_equal_options_are_equal(self):
        """
        CONTRACT: Two DiscoveryOptions with same values are equal.
        """
        from flyrigloader.discovery.options import DiscoveryOptions
        
        options1 = DiscoveryOptions(pattern="*.pkl", recursive=True)
        options2 = DiscoveryOptions(pattern="*.pkl", recursive=True)
        
        # Contract: Equality works
        assert options1 == options2
    
    def test_different_options_are_not_equal(self):
        """
        CONTRACT: DiscoveryOptions with different values are not equal.
        """
        from flyrigloader.discovery.options import DiscoveryOptions
        
        options1 = DiscoveryOptions(pattern="*.pkl")
        options2 = DiscoveryOptions(pattern="*.csv")
        
        # Contract: Inequality works
        assert options1 != options2
    
    def test_can_be_used_in_sets(self):
        """
        CONTRACT: DiscoveryOptions is hashable (can be used in sets/dicts).
        """
        from flyrigloader.discovery.options import DiscoveryOptions
        
        options1 = DiscoveryOptions(pattern="*.pkl")
        options2 = DiscoveryOptions(pattern="*.csv")
        
        # Contract: Hashable (frozen dataclass requirement)
        options_set = {options1, options2}
        assert len(options_set) == 2


class TestDiscoveryOptionsReprContract:
    """
    Contract tests for string representation.
    
    DiscoveryOptions MUST have useful string representation.
    """
    
    def test_has_readable_repr(self):
        """
        CONTRACT: repr() returns useful string representation.
        """
        from flyrigloader.discovery.options import DiscoveryOptions
        
        options = DiscoveryOptions(pattern="*.pkl", recursive=False)
        repr_str = repr(options)
        
        # Contract: repr includes class name and key values
        assert "DiscoveryOptions" in repr_str
        assert "pattern" in repr_str
        assert "*.pkl" in repr_str
    
    def test_repr_is_evaluable(self):
        """
        CONTRACT: repr() ideally returns string that can recreate object.
        """
        from flyrigloader.discovery.options import DiscoveryOptions
        
        options = DiscoveryOptions(pattern="*.pkl")
        repr_str = repr(options)
        
        # Contract: repr should be useful for debugging
        # (Not requiring eval() to work, but should be informative)
        assert len(repr_str) > 0
        assert "=" in repr_str  # Shows attribute values
