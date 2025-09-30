"""
Unit tests for DiscoveryOptions implementation.

These tests focus on implementation details, edge cases, and specific behaviors.
They complement the contract tests with more granular testing.

Following TDD: These tests are written BEFORE implementation.
"""

import pytest
from pathlib import Path
from typing import List, Optional


class TestDiscoveryOptionsInstantiation:
    """Unit tests for DiscoveryOptions instantiation edge cases."""
    
    def test_instantiation_with_no_args_sets_all_defaults(self):
        """All attributes should have documented default values."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        options = DiscoveryOptions()
        
        assert options.pattern == "*.*"
        assert options.recursive is True
        assert options.extensions is None
        assert options.extract_metadata is False
        assert options.parse_dates is False
    
    def test_instantiation_with_all_args_overrides_all_defaults(self):
        """Can override every single default at once."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        options = DiscoveryOptions(
            pattern="custom_*.pkl",
            recursive=False,
            extensions=['.pkl', '.csv', '.txt'],
            extract_metadata=True,
            parse_dates=True
        )
        
        assert options.pattern == "custom_*.pkl"
        assert options.recursive is False
        assert options.extensions == ['.pkl', '.csv', '.txt']
        assert options.extract_metadata is True
        assert options.parse_dates is True
    
    def test_instantiation_with_keyword_args(self):
        """Can use keyword arguments in any order."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        options = DiscoveryOptions(
            parse_dates=True,
            pattern="*.pkl",
            extract_metadata=True
        )
        
        assert options.pattern == "*.pkl"
        assert options.extract_metadata is True
        assert options.parse_dates is True
    
    def test_instantiation_preserves_exact_values(self):
        """Values are stored exactly as provided (no transformation)."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        # Complex pattern with special chars
        complex_pattern = "data_[0-9]{4}_*.pkl"
        
        options = DiscoveryOptions(pattern=complex_pattern)
        
        assert options.pattern == complex_pattern
        # Verify no stripping or transformation occurred
        assert len(options.pattern) == len(complex_pattern)


class TestDiscoveryOptionsPatternValidation:
    """Unit tests for pattern attribute validation details."""
    
    def test_pattern_must_be_string_not_bytes(self):
        """Pattern must be str, not bytes."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        with pytest.raises(ValueError) as exc_info:
            DiscoveryOptions(pattern=b"*.pkl")
        
        assert "string" in str(exc_info.value).lower()
    
    def test_pattern_must_be_string_not_pathlib_path(self):
        """Pattern must be str, not Path object."""
        from flyrigloader.discovery.options import DiscoveryOptions
        from pathlib import Path
        
        with pytest.raises(ValueError) as exc_info:
            DiscoveryOptions(pattern=Path("*.pkl"))
        
        assert "string" in str(exc_info.value).lower()
    
    def test_pattern_rejects_list_of_patterns(self):
        """Pattern is single string, not list."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        with pytest.raises(ValueError):
            DiscoveryOptions(pattern=["*.pkl", "*.csv"])
    
    def test_pattern_empty_string_raises_error(self):
        """Empty string is invalid pattern."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        with pytest.raises(ValueError) as exc_info:
            DiscoveryOptions(pattern="")
        
        error_msg = str(exc_info.value).lower()
        assert "pattern" in error_msg
        assert "empty" in error_msg
    
    def test_pattern_whitespace_only_raises_error(self):
        """Whitespace-only pattern is invalid."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        for whitespace in ["   ", "\t", "\n", "  \t\n  "]:
            with pytest.raises(ValueError) as exc_info:
                DiscoveryOptions(pattern=whitespace)
            
            assert "pattern" in str(exc_info.value).lower()
    
    def test_pattern_accepts_complex_glob_patterns(self):
        """Complex glob patterns are valid."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        complex_patterns = [
            "*.pkl",
            "data_*.csv",
            "exp_[0-9]*.pkl",
            "**/subdir/*.pkl",
            "prefix_*_suffix.txt",
            "?.pkl",
            "[abc]*.pkl"
        ]
        
        for pattern in complex_patterns:
            options = DiscoveryOptions(pattern=pattern)
            assert options.pattern == pattern
    
    def test_pattern_with_leading_trailing_spaces_preserved(self):
        """Leading/trailing spaces in pattern are preserved (not stripped)."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        # Note: This might be a design decision - preserve or strip?
        # Let's test that we preserve them (user responsibility)
        pattern_with_spaces = "  *.pkl  "
        
        options = DiscoveryOptions(pattern=pattern_with_spaces)
        
        # If implementation strips, this test will fail and we'll adjust
        assert options.pattern == pattern_with_spaces


class TestDiscoveryOptionsExtensionsValidation:
    """Unit tests for extensions attribute validation details."""
    
    def test_extensions_none_is_valid(self):
        """None means no extension filtering."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        options = DiscoveryOptions(extensions=None)
        assert options.extensions is None
    
    def test_extensions_empty_list_is_valid(self):
        """Empty list is technically valid (though not useful)."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        options = DiscoveryOptions(extensions=[])
        assert options.extensions == []
    
    def test_extensions_single_element_list(self):
        """Single extension in list works."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        options = DiscoveryOptions(extensions=['.pkl'])
        assert options.extensions == ['.pkl']
        assert len(options.extensions) == 1
    
    def test_extensions_multiple_elements_list(self):
        """Multiple extensions in list work."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        extensions_list = ['.pkl', '.csv', '.txt', '.json']
        options = DiscoveryOptions(extensions=extensions_list)
        
        assert options.extensions == extensions_list
        assert len(options.extensions) == 4
    
    def test_extensions_rejects_string_instead_of_list(self):
        """Common mistake: passing string instead of list."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        with pytest.raises(ValueError) as exc_info:
            DiscoveryOptions(extensions=".pkl")
        
        error_msg = str(exc_info.value).lower()
        assert "list" in error_msg
        assert "extensions" in error_msg
    
    def test_extensions_rejects_tuple(self):
        """Extensions must be list, not tuple."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        # Tuples aren't accepted (design choice - could change)
        with pytest.raises(ValueError):
            DiscoveryOptions(extensions=('.pkl', '.csv'))
    
    def test_extensions_rejects_set(self):
        """Extensions must be list, not set."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        with pytest.raises(ValueError):
            DiscoveryOptions(extensions={'.pkl', '.csv'})
    
    def test_extensions_rejects_non_string_elements(self):
        """Each extension must be a string."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        invalid_lists = [
            ['.pkl', 123],
            ['.pkl', None],
            ['.pkl', b'.csv'],
            ['.pkl', Path('.csv')],
            [1, 2, 3]
        ]
        
        for invalid_list in invalid_lists:
            with pytest.raises(ValueError) as exc_info:
                DiscoveryOptions(extensions=invalid_list)
            
            error_msg = str(exc_info.value).lower()
            assert "string" in error_msg
    
    def test_extensions_preserves_list_order(self):
        """Extension list order is preserved."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        ordered_exts = ['.txt', '.pkl', '.csv', '.json']
        options = DiscoveryOptions(extensions=ordered_exts)
        
        assert options.extensions == ordered_exts
        assert options.extensions[0] == '.txt'
        assert options.extensions[-1] == '.json'
    
    def test_extensions_allows_duplicates(self):
        """Duplicate extensions are allowed (user's choice)."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        exts_with_dups = ['.pkl', '.csv', '.pkl']
        options = DiscoveryOptions(extensions=exts_with_dups)
        
        assert options.extensions == exts_with_dups
        assert len(options.extensions) == 3
    
    def test_extensions_allows_with_and_without_leading_dot(self):
        """Extensions can have or omit leading dot."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        # Both should be valid (implementation may normalize)
        options1 = DiscoveryOptions(extensions=['.pkl', '.csv'])
        options2 = DiscoveryOptions(extensions=['pkl', 'csv'])
        
        # Both should work without error
        assert options1.extensions is not None
        assert options2.extensions is not None


class TestDiscoveryOptionsBooleanFlags:
    """Unit tests for boolean flag attributes."""
    
    def test_recursive_must_be_bool(self):
        """recursive must be boolean."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        # These should work
        DiscoveryOptions(recursive=True)
        DiscoveryOptions(recursive=False)
        
        # These should fail (truthy/falsy but not bool)
        with pytest.raises((ValueError, TypeError)):
            DiscoveryOptions(recursive=1)
        
        with pytest.raises((ValueError, TypeError)):
            DiscoveryOptions(recursive="True")
        
        with pytest.raises((ValueError, TypeError)):
            DiscoveryOptions(recursive=None)
    
    def test_extract_metadata_must_be_bool(self):
        """extract_metadata must be boolean."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        DiscoveryOptions(extract_metadata=True)
        DiscoveryOptions(extract_metadata=False)
        
        with pytest.raises((ValueError, TypeError)):
            DiscoveryOptions(extract_metadata=1)
    
    def test_parse_dates_must_be_bool(self):
        """parse_dates must be boolean."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        DiscoveryOptions(parse_dates=True)
        DiscoveryOptions(parse_dates=False)
        
        with pytest.raises((ValueError, TypeError)):
            DiscoveryOptions(parse_dates="yes")


class TestDiscoveryOptionsFactoryMethodDefaults:
    """Unit tests for factory method implementation details."""
    
    def test_defaults_returns_new_instance_each_time(self):
        """defaults() creates a new instance each call."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        opts1 = DiscoveryOptions.defaults()
        opts2 = DiscoveryOptions.defaults()
        
        # Different instances
        assert opts1 is not opts2
        # But equal values
        assert opts1 == opts2
    
    def test_minimal_with_default_pattern_equals_defaults(self):
        """minimal() with default pattern == defaults()."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        minimal = DiscoveryOptions.minimal()
        defaults = DiscoveryOptions.defaults()
        
        assert minimal == defaults
    
    def test_minimal_with_custom_pattern_differs_from_defaults(self):
        """minimal("*.pkl") differs from defaults only in pattern."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        minimal = DiscoveryOptions.minimal("*.pkl")
        defaults = DiscoveryOptions.defaults()
        
        assert minimal.pattern != defaults.pattern
        assert minimal.recursive == defaults.recursive
        assert minimal.extract_metadata == defaults.extract_metadata
    
    def test_with_metadata_default_parse_dates_is_true(self):
        """with_metadata() enables parse_dates by default."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        opts = DiscoveryOptions.with_metadata()
        
        assert opts.extract_metadata is True
        assert opts.parse_dates is True
    
    def test_with_metadata_can_disable_parse_dates(self):
        """with_metadata(parse_dates=False) allows date parsing to be off."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        opts = DiscoveryOptions.with_metadata(parse_dates=False)
        
        assert opts.extract_metadata is True
        assert opts.parse_dates is False
    
    def test_with_metadata_accepts_all_common_params(self):
        """with_metadata accepts pattern, parse_dates, recursive."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        opts = DiscoveryOptions.with_metadata(
            pattern="exp_*.pkl",
            parse_dates=False,
            recursive=False
        )
        
        assert opts.pattern == "exp_*.pkl"
        assert opts.parse_dates is False
        assert opts.recursive is False
        assert opts.extract_metadata is True
    
    def test_with_filtering_default_pattern_is_wildcard(self):
        """with_filtering() uses "*.*" pattern by default."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        opts = DiscoveryOptions.with_filtering(extensions=['.pkl'])
        
        assert opts.pattern == "*.*"
        assert opts.extensions == ['.pkl']
    
    def test_with_filtering_accepts_custom_pattern(self):
        """with_filtering accepts pattern parameter."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        opts = DiscoveryOptions.with_filtering(
            pattern="data_*",
            extensions=['.pkl', '.csv']
        )
        
        assert opts.pattern == "data_*"
        assert opts.extensions == ['.pkl', '.csv']
    
    def test_with_filtering_accepts_recursive_param(self):
        """with_filtering accepts recursive parameter."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        opts = DiscoveryOptions.with_filtering(
            extensions=['.pkl'],
            recursive=False
        )
        
        assert opts.recursive is False


class TestDiscoveryOptionsImmutability:
    """Unit tests for immutability implementation details."""
    
    def test_cannot_modify_pattern_after_creation(self):
        """Attempting to modify pattern raises error."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        opts = DiscoveryOptions(pattern="*.pkl")
        
        with pytest.raises((AttributeError, TypeError)):
            opts.pattern = "*.csv"
    
    def test_cannot_modify_recursive_after_creation(self):
        """Attempting to modify recursive raises error."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        opts = DiscoveryOptions(recursive=True)
        
        with pytest.raises((AttributeError, TypeError)):
            opts.recursive = False
    
    def test_cannot_modify_extensions_after_creation(self):
        """Attempting to modify extensions raises error."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        opts = DiscoveryOptions(extensions=['.pkl'])
        
        with pytest.raises((AttributeError, TypeError)):
            opts.extensions = ['.csv']
    
    def test_cannot_add_new_attributes(self):
        """Cannot add new attributes to frozen dataclass."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        opts = DiscoveryOptions()
        
        with pytest.raises((AttributeError, TypeError)):
            opts.new_attribute = "value"
    
    def test_cannot_delete_attributes(self):
        """Cannot delete attributes from frozen dataclass."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        opts = DiscoveryOptions()
        
        with pytest.raises((AttributeError, TypeError)):
            del opts.pattern


class TestDiscoveryOptionsEquality:
    """Unit tests for equality implementation details."""
    
    def test_same_values_are_equal(self):
        """Two instances with same values are equal."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        opts1 = DiscoveryOptions(pattern="*.pkl", recursive=True)
        opts2 = DiscoveryOptions(pattern="*.pkl", recursive=True)
        
        assert opts1 == opts2
    
    def test_different_pattern_not_equal(self):
        """Different patterns make instances unequal."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        opts1 = DiscoveryOptions(pattern="*.pkl")
        opts2 = DiscoveryOptions(pattern="*.csv")
        
        assert opts1 != opts2
    
    def test_different_recursive_not_equal(self):
        """Different recursive values make instances unequal."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        opts1 = DiscoveryOptions(recursive=True)
        opts2 = DiscoveryOptions(recursive=False)
        
        assert opts1 != opts2
    
    def test_different_extensions_not_equal(self):
        """Different extensions make instances unequal."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        opts1 = DiscoveryOptions(extensions=['.pkl'])
        opts2 = DiscoveryOptions(extensions=['.csv'])
        
        assert opts1 != opts2
    
    def test_none_vs_empty_list_extensions_not_equal(self):
        """None extensions != [] extensions."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        opts1 = DiscoveryOptions(extensions=None)
        opts2 = DiscoveryOptions(extensions=[])
        
        assert opts1 != opts2
    
    def test_equality_with_non_discovery_options_is_false(self):
        """DiscoveryOptions != non-DiscoveryOptions objects."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        opts = DiscoveryOptions()
        
        assert opts != "not a DiscoveryOptions"
        assert opts != 123
        assert opts != None
        assert opts != {}


class TestDiscoveryOptionsHashing:
    """Unit tests for hashing implementation details."""
    
    def test_is_hashable(self):
        """DiscoveryOptions instances are hashable."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        opts = DiscoveryOptions(pattern="*.pkl")
        
        # Should not raise
        hash_value = hash(opts)
        assert isinstance(hash_value, int)
    
    def test_equal_instances_have_same_hash(self):
        """Equal instances have equal hashes."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        opts1 = DiscoveryOptions(pattern="*.pkl", recursive=True)
        opts2 = DiscoveryOptions(pattern="*.pkl", recursive=True)
        
        assert opts1 == opts2
        assert hash(opts1) == hash(opts2)
    
    def test_can_be_used_in_set(self):
        """Can add DiscoveryOptions to a set."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        opts1 = DiscoveryOptions(pattern="*.pkl")
        opts2 = DiscoveryOptions(pattern="*.csv")
        opts3 = DiscoveryOptions(pattern="*.pkl")  # Duplicate of opts1
        
        opts_set = {opts1, opts2, opts3}
        
        # Should have 2 unique options (opts1 and opts3 are equal)
        assert len(opts_set) == 2
    
    def test_can_be_used_as_dict_key(self):
        """Can use DiscoveryOptions as dictionary key."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        opts1 = DiscoveryOptions(pattern="*.pkl")
        opts2 = DiscoveryOptions(pattern="*.csv")
        
        config_map = {
            opts1: "pkl_config",
            opts2: "csv_config"
        }
        
        assert config_map[opts1] == "pkl_config"
        assert config_map[opts2] == "csv_config"


class TestDiscoveryOptionsStringRepresentation:
    """Unit tests for string representation details."""
    
    def test_repr_includes_class_name(self):
        """repr includes 'DiscoveryOptions'."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        opts = DiscoveryOptions()
        repr_str = repr(opts)
        
        assert "DiscoveryOptions" in repr_str
    
    def test_repr_includes_non_default_values(self):
        """repr shows non-default attribute values."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        opts = DiscoveryOptions(pattern="*.pkl", recursive=False)
        repr_str = repr(opts)
        
        assert "pattern" in repr_str
        assert "*.pkl" in repr_str
        assert "recursive" in repr_str
    
    def test_str_is_readable(self):
        """str() returns human-readable string."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        opts = DiscoveryOptions(pattern="*.pkl")
        str_repr = str(opts)
        
        # Should be informative
        assert len(str_repr) > 0
    
    def test_repr_different_for_different_instances(self):
        """Different instances have different repr (shows different values)."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        opts1 = DiscoveryOptions(pattern="*.pkl")
        opts2 = DiscoveryOptions(pattern="*.csv")
        
        assert repr(opts1) != repr(opts2)
        assert "*.pkl" in repr(opts1)
        assert "*.csv" in repr(opts2)


class TestDiscoveryOptionsEdgeCases:
    """Unit tests for edge cases and corner scenarios."""
    
    def test_pattern_with_unicode_characters(self):
        """Pattern can contain unicode characters."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        unicode_pattern = "データ_*.pkl"  # Japanese characters
        opts = DiscoveryOptions(pattern=unicode_pattern)
        
        assert opts.pattern == unicode_pattern
    
    def test_pattern_with_special_regex_chars(self):
        """Pattern can contain regex special characters."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        special_patterns = [
            "data_[0-9]{4}.pkl",
            "exp_(a|b|c)*.csv",
            "file?.txt",
            "**/*.pkl"
        ]
        
        for pattern in special_patterns:
            opts = DiscoveryOptions(pattern=pattern)
            assert opts.pattern == pattern
    
    def test_extensions_with_very_long_list(self):
        """Extensions can have many elements."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        long_list = [f'.ext{i}' for i in range(100)]
        opts = DiscoveryOptions(extensions=long_list)
        
        assert len(opts.extensions) == 100
    
    def test_extensions_with_unusual_extension_formats(self):
        """Extensions can have unusual but valid formats."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        unusual_exts = [
            '.tar.gz',  # Double extension
            '.pkl.bak',
            '.data.v2',
            '.',  # Just dot
            '.123',  # Numbers only
        ]
        
        opts = DiscoveryOptions(extensions=unusual_exts)
        assert opts.extensions == unusual_exts
    
    def test_all_boolean_combinations(self):
        """All 2^3 = 8 combinations of boolean flags work."""
        from flyrigloader.discovery.options import DiscoveryOptions
        
        for recursive in [True, False]:
            for extract_metadata in [True, False]:
                for parse_dates in [True, False]:
                    opts = DiscoveryOptions(
                        recursive=recursive,
                        extract_metadata=extract_metadata,
                        parse_dates=parse_dates
                    )
                    
                    assert opts.recursive == recursive
                    assert opts.extract_metadata == extract_metadata
                    assert opts.parse_dates == parse_dates
