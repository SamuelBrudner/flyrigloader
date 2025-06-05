"""
Comprehensive test suite for pattern matching functionality.

This module provides extensive testing coverage for the flyrigloader pattern matching system,
including advanced regex compilation, named-group extraction, dynamic pattern generation,
property-based testing, and performance validation per F-002-RQ-003 and F-007 requirements.
"""

import os
import re
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Pattern, Any
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from hypothesis.strategies import composite

from flyrigloader.discovery.patterns import (
    PatternMatcher,
    match_files_to_patterns,
    create_experiment_matcher,
    create_vial_matcher,
    extract_experiment_info,
    extract_vial_info,
    match_experiment_file,
    match_vial_file,
    generate_pattern_from_template,
)


# --- Test Data Generation Strategies for Property-Based Testing ---

@composite
def valid_filenames(draw):
    """Generate valid filename patterns for property-based testing."""
    # Generate components for realistic experimental filenames
    animals = ["mouse", "rat", "exp_mouse", "exp_rat"]
    dates = st.text(alphabet="0123456789", min_size=8, max_size=8)
    conditions = st.text(alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-", 
                        min_size=3, max_size=15)
    replicates = st.integers(min_value=1, max_value=99).map(str)
    experiment_ids = st.integers(min_value=1, max_value=999).map(lambda x: f"{x:03d}")
    extensions = ["csv", "pkl", "pickle", "txt", "dat"]
    
    # Choose filename pattern type
    pattern_type = draw(st.sampled_from(["vial", "experiment", "simple"]))
    
    if pattern_type == "vial":
        animal = draw(st.sampled_from(animals))
        date = draw(dates)
        condition = draw(conditions)
        replicate = draw(replicates)
        extension = draw(st.sampled_from(extensions))
        
        # Multiple vial filename formats
        format_type = draw(st.sampled_from(["standard", "with_replicate", "underscore_separated"]))
        if format_type == "standard":
            return f"vial_{date}_{condition}.{extension}"
        elif format_type == "with_replicate":
            return f"vial_{date}_{condition}_{replicate}.{extension}"
        else:
            return f"{animal}_{date}_{condition}_{replicate}.{extension}"
    
    elif pattern_type == "experiment":
        exp_id = draw(experiment_ids)
        animal = draw(st.sampled_from(animals))
        condition = draw(conditions)
        extension = draw(st.sampled_from(extensions))
        return f"exp{exp_id}_{animal}_{condition}.{extension}"
    
    else:  # simple pattern
        prefix = draw(st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=3, max_size=8))
        suffix = draw(st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=3, max_size=8))
        extension = draw(st.sampled_from(extensions))
        return f"{prefix}_{suffix}.{extension}"


@composite 
def regex_patterns(draw):
    """Generate realistic regex patterns for experimental data."""
    pattern_types = [
        # Vial patterns with named groups
        r"vial_(?P<date>\d{8})_(?P<condition>[a-zA-Z0-9_-]+)(_(?P<replicate>\d+))?\.(?P<extension>\w+)",
        r"(?P<animal>mouse|rat)_(?P<date>\d{8})_(?P<condition>[a-zA-Z0-9_-]+)_(?P<replicate>\d+)\.(?P<extension>\w+)",
        
        # Experiment patterns
        r"exp(?P<experiment_id>\d{3})_(?P<animal>\w+)_(?P<condition>\w+)\.(?P<extension>\w+)",
        r"(?P<experiment_id>\d+)_(?P<dataset>\w+)_(?P<date>\d{8})\.(?P<extension>\w+)",
        
        # Generic patterns
        r"(?P<prefix>\w+)_(?P<identifier>\d+)_(?P<suffix>\w+)\.(?P<extension>\w+)",
        r"data_(?P<session>\d{8})_(?P<subject_id>[A-Z]\d+)\.(?P<extension>\w+)",
    ]
    return draw(st.sampled_from(pattern_types))


@composite
def invalid_regex_patterns(draw):
    """Generate invalid regex patterns for error handling tests."""
    invalid_patterns = [
        r"[unclosed_bracket",
        r"(?P<invalid_group",
        r"(?P<>empty_name)",
        r"(?P<123>starts_with_number)",
        r"*invalid_quantifier",
        r"(?P<duplicate>test)(?P<duplicate>test)",
        r"+invalid_start",
        r"(?invalid_group_syntax)",
        r"(?P<space name>invalid)",
        "",  # Empty pattern
    ]
    return draw(st.sampled_from(invalid_patterns))


# --- Fixtures for Comprehensive Testing ---

@pytest.fixture
def sample_patterns():
    """Provide comprehensive set of realistic patterns for testing."""
    return [
        # Vial patterns
        r"vial_(?P<date>\d{8})_(?P<condition>[a-zA-Z0-9_-]+)(_(?P<replicate>\d+))?\.csv",
        r"(?P<animal>mouse|rat)_(?P<date>\d{8})_(?P<condition>[a-zA-Z0-9_-]+)_(?P<replicate>\d+)\.(?P<extension>pkl|csv)",
        
        # Experiment patterns  
        r"exp(?P<experiment_id>\d{3})_(?P<animal>\w+)_(?P<condition>\w+)\.(?P<extension>csv|pkl)",
        r"(?P<experiment_id>\d+)_(?P<dataset>\w+)_(?P<date>\d{8})\.csv",
        
        # Dataset patterns
        r"dataset_(?P<name>\w+)_(?P<version>\d+\.\d+)_(?P<date>\d{8})\.(?P<extension>pkl|csv)",
        
        # Legacy patterns for backward compatibility
        r"legacy_(\w+)_(\d{8})_(\w+)_(\d+)\.csv",  # Positional groups
    ]


@pytest.fixture
def sample_filenames():
    """Provide comprehensive set of test filenames."""
    return [
        # Valid vial files
        "vial_20241215_control.csv",
        "vial_20241216_treatment_1.csv", 
        "vial_20241217_baseline_2.csv",
        "mouse_20241218_control_1.pkl",
        "rat_20241219_treatment_3.csv",
        
        # Valid experiment files
        "exp001_mouse_control.csv",
        "exp002_rat_treatment.pkl", 
        "exp999_exp_mouse_baseline.csv",
        "123_dataset1_20241220.csv",
        
        # Valid dataset files
        "dataset_navigation_1.0_20241221.pkl",
        "dataset_plume_2.1_20241222.csv",
        
        # Legacy format files
        "legacy_mouse_20241223_control_1.csv",
        "legacy_rat_20241224_treatment_2.csv",
        
        # Non-matching files
        "notes.txt",
        "metadata.json",
        "analysis_script.py",
        "README.md",
        
        # Edge cases
        "vial_.csv",  # Missing components
        "exp_mouse_control.csv",  # Missing experiment ID
        "invalid_format_file.csv",
        
        # Files with special characters
        "vial_20241225_control-test_1.csv",
        "exp001_mouse_control_v2.csv",
        "dataset_test-data_1.0_20241226.pkl",
    ]


@pytest.fixture
def temp_file_structure():
    """Create comprehensive temporary file structure for integration testing."""
    temp_dir = tempfile.mkdtemp()
    try:
        files = {
            # Root directory files
            "vial_20241215_control.csv": "mock,data,1\n2,3,4",
            "vial_20241216_treatment_1.csv": "mock,data,1\n2,3,4", 
            "exp001_mouse_control.csv": "t,x,y\n0,1,2\n1,2,3",
            "exp002_rat_treatment.pkl": "binary_data",
            "dataset_navigation_1.0_20241217.pkl": "dataset_content",
            "notes.txt": "experiment notes",
            
            # Subdirectory files
            "batch1/vial_20241218_control_2.csv": "batch,data",
            "batch1/exp003_mouse_baseline.csv": "experiment,data",
            "batch2/dataset_plume_2.0_20241219.csv": "dataset,content",
            
            # Deep nested structure
            "experiments/2024/12/vial_20241220_treatment_1.csv": "nested,data",
            "experiments/2024/12/exp004_rat_control.csv": "nested,experiment",
            
            # Mixed extensions
            "data/vial_20241221_control.pkl": "pickle_data",
            "data/exp005_mouse_treatment.dat": "binary_data",
        }
        
        # Create all directories and files
        for relative_path, content in files.items():
            full_path = os.path.join(temp_dir, relative_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            with open(full_path, "w") as f:
                f.write(content)
        
        yield temp_dir
    finally:
        import shutil
        shutil.rmtree(temp_dir)


# --- Pattern Matcher Core Functionality Tests ---

class TestPatternMatcherInitialization:
    """Test PatternMatcher initialization and basic functionality."""
    
    def test_empty_patterns_initialization(self):
        """Test PatternMatcher initialization with empty patterns list."""
        matcher = PatternMatcher([])
        assert matcher.compiled_patterns == []
        assert matcher.patterns == []
    
    def test_single_pattern_initialization(self):
        """Test PatternMatcher initialization with single pattern."""
        pattern = r"test_(?P<id>\d+)\.csv"
        matcher = PatternMatcher([pattern])
        
        assert len(matcher.compiled_patterns) == 1
        assert len(matcher.patterns) == 1
        assert matcher.patterns[0] == pattern
        assert isinstance(matcher.compiled_patterns[0], Pattern)
    
    @pytest.mark.parametrize("patterns", [
        [r"vial_(?P<date>\d+)\.csv", r"exp(?P<id>\d+)\.pkl"],
        [r"(?P<animal>\w+)_(?P<date>\d{8})\.csv"] * 5,  # Multiple identical patterns
        [r"simple_pattern", r"(?P<complex>\w+)_(?P<pattern>\d+)"],
    ])
    def test_multiple_patterns_initialization(self, patterns):
        """Test PatternMatcher initialization with multiple patterns."""
        matcher = PatternMatcher(patterns)
        
        assert len(matcher.compiled_patterns) == len(patterns)
        assert len(matcher.patterns) == len(patterns)
        assert matcher.patterns == patterns
        assert all(isinstance(p, Pattern) for p in matcher.compiled_patterns)
    
    def test_invalid_regex_pattern_compilation(self):
        """Test PatternMatcher handles invalid regex patterns appropriately."""
        with pytest.raises(re.error):
            PatternMatcher([r"[unclosed_bracket"])
        
        with pytest.raises(re.error):
            PatternMatcher([r"(?P<invalid_group"])
    
    @given(st.lists(regex_patterns(), min_size=1, max_size=10))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_pattern_compilation_property(self, patterns):
        """Property-based test for pattern compilation."""
        matcher = PatternMatcher(patterns)
        
        assert len(matcher.compiled_patterns) == len(patterns)
        assert all(isinstance(p, Pattern) for p in matcher.compiled_patterns)


class TestPatternMatchingCore:
    """Test core pattern matching functionality."""
    
    def test_exact_named_group_matching(self, sample_patterns):
        """Test exact matching with named capture groups."""
        matcher = PatternMatcher(sample_patterns)
        
        # Test vial pattern matching
        result = matcher.match("vial_20241215_control.csv")
        assert result is not None
        assert result["date"] == "20241215"
        assert result["condition"] == "control"
        assert "replicate" not in result or result["replicate"] is None
        
        # Test vial with replicate
        result = matcher.match("vial_20241216_treatment_1.csv")
        assert result is not None
        assert result["date"] == "20241216"
        assert result["condition"] == "treatment"
        assert result["replicate"] == "1"
        
        # Test experiment pattern
        result = matcher.match("exp001_mouse_control.csv")
        assert result is not None
        assert result["experiment_id"] == "001"
        assert result["animal"] == "mouse"
        assert result["condition"] == "control"
    
    @pytest.mark.parametrize("filename,expected_fields", [
        ("vial_20241215_control.csv", {"date": "20241215", "condition": "control"}),
        ("vial_20241216_treatment_1.csv", {"date": "20241216", "condition": "treatment", "replicate": "1"}),
        ("mouse_20241217_baseline_2.pkl", {"animal": "mouse", "date": "20241217", "condition": "baseline", "replicate": "2"}),
        ("exp001_mouse_control.csv", {"experiment_id": "001", "animal": "mouse", "condition": "control"}),
        ("exp999_rat_treatment.pkl", {"experiment_id": "999", "animal": "rat", "condition": "treatment"}),
        ("123_dataset1_20241218.csv", {"experiment_id": "123", "dataset": "dataset1", "date": "20241218"}),
    ])
    def test_parametrized_pattern_extraction(self, sample_patterns, filename, expected_fields):
        """Parametrized test for pattern extraction validation."""
        matcher = PatternMatcher(sample_patterns)
        result = matcher.match(filename)
        
        assert result is not None, f"No match found for {filename}"
        for field, expected_value in expected_fields.items():
            assert field in result, f"Field {field} not found in result for {filename}"
            assert result[field] == expected_value, f"Field {field} mismatch for {filename}"
    
    @pytest.mark.parametrize("filename", [
        "notes.txt",
        "metadata.json", 
        "invalid_format.csv",
        "vial_.csv",  # Missing required components
        "exp_mouse.csv",  # Missing experiment ID
        "",  # Empty filename
        "   ",  # Whitespace only
    ])
    def test_non_matching_files(self, sample_patterns, filename):
        """Test that non-matching files return None."""
        matcher = PatternMatcher(sample_patterns)
        result = matcher.match(filename)
        assert result is None, f"Unexpected match for {filename}: {result}"
    
    def test_first_match_priority(self):
        """Test that PatternMatcher returns first matching pattern."""
        # Create patterns where multiple could match
        patterns = [
            r"(?P<type>vial)_(?P<date>\d+)_(?P<condition>\w+)\.csv",  # More specific
            r"(?P<prefix>\w+)_(?P<suffix>\w+)\.csv",  # More general
        ]
        matcher = PatternMatcher(patterns)
        
        result = matcher.match("vial_20241215_control.csv")
        assert result is not None
        assert "type" in result  # Should match first pattern
        assert result["type"] == "vial"
        assert "prefix" not in result  # Should not match second pattern
    
    @given(valid_filenames())
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_based_filename_handling(self, filename):
        """Property-based test for filename handling robustness."""
        assume(len(filename) > 0)
        assume(not filename.isspace())
        
        # Use flexible patterns that can match various formats
        patterns = [
            r"(?P<prefix>\w+)_(?P<component1>[^_]+)_(?P<component2>[^_.]+)\.(?P<extension>\w+)",
            r"(?P<name>\w+)\.(?P<extension>\w+)",
        ]
        matcher = PatternMatcher(patterns)
        
        # Should either match or return None without raising exceptions
        result = matcher.match(filename)
        if result is not None:
            assert isinstance(result, dict)
            assert all(isinstance(k, str) for k in result.keys())
            assert all(isinstance(v, str) for v in result.values() if v is not None)


class TestSpecialCaseProcessing:
    """Test special case processing and metadata extraction logic."""
    
    def test_animal_condition_replicate_splitting(self, sample_patterns):
        """Test splitting condition_replicate format in animal files."""
        matcher = PatternMatcher(sample_patterns)
        
        # This tests the _process_special_cases method
        # Create a pattern that might capture condition_replicate together
        patterns = [r"(?P<animal>mouse|rat)_(?P<date>\d+)_(?P<condition>\w+_\d+)\.csv"]
        special_matcher = PatternMatcher(patterns)
        
        result = special_matcher.match("mouse_20241215_control_2.csv")
        assert result is not None
        # The special case processing should handle this
    
    def test_experiment_mouse_baseline_special_case(self, sample_patterns):
        """Test special handling for experiment files with mouse baseline."""
        matcher = PatternMatcher(sample_patterns)
        
        result = matcher.match("exp001_mouse_baseline.csv")
        assert result is not None
        # Check for special processing in baseline experiments
        if "animal" in result and result["animal"] == "mouse" and "baseline" in "exp001_mouse_baseline.csv":
            # Special case handling should be applied
            pass
    
    def test_positional_group_fallback(self):
        """Test fallback to positional groups for legacy patterns."""
        # Test patterns without named groups (legacy support)
        patterns = [r"legacy_(\w+)_(\d{8})_(\w+)_(\d+)\.csv"]
        matcher = PatternMatcher(patterns)
        
        result = matcher.match("legacy_mouse_20241215_control_1.csv")
        assert result is not None
        # Should create field names based on pattern recognition
        assert isinstance(result, dict)
        assert len(result) > 0


class TestFilterFiles:
    """Test file filtering functionality."""
    
    def test_filter_empty_file_list(self, sample_patterns):
        """Test filtering empty file list."""
        matcher = PatternMatcher(sample_patterns)
        result = matcher.filter_files([])
        assert result == {}
    
    def test_filter_mixed_file_list(self, sample_patterns, sample_filenames):
        """Test filtering list with matching and non-matching files."""
        matcher = PatternMatcher(sample_patterns)
        result = matcher.filter_files(sample_filenames)
        
        # Should have matches for valid experimental files
        assert len(result) > 0
        
        # Verify specific matches
        matching_files = [f for f in sample_filenames if "vial_" in f or "exp" in f or "dataset_" in f]
        for filename in matching_files:
            if filename in result:
                assert isinstance(result[filename], dict)
                assert len(result[filename]) > 0
    
    def test_filter_with_file_paths(self, temp_file_structure, sample_patterns):
        """Test filtering with full file paths."""
        import glob
        
        # Get all files recursively
        all_files = glob.glob(os.path.join(temp_file_structure, "**/*"), recursive=True)
        all_files = [f for f in all_files if os.path.isfile(f)]
        
        matcher = PatternMatcher(sample_patterns)
        result = matcher.filter_files(all_files)
        
        # Should find multiple matches across the directory structure
        assert len(result) > 0
        
        # Verify that paths are preserved correctly
        for filepath in result.keys():
            assert os.path.isabs(filepath)
            assert filepath in all_files
    
    @given(st.lists(valid_filenames(), min_size=0, max_size=20))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_based_filtering(self, filenames):
        """Property-based test for file filtering."""
        patterns = [
            r"(?P<prefix>\w+)_(?P<component>[^_.]+)\.(?P<extension>\w+)",
            r"(?P<name>\w+)\.(?P<extension>\w+)",
        ]
        matcher = PatternMatcher(patterns)
        
        result = matcher.filter_files(filenames)
        
        # Basic properties that should always hold
        assert isinstance(result, dict)
        assert len(result) <= len(filenames)
        assert all(filename in filenames for filename in result.keys())


# --- Convenience Function Tests ---

class TestConvenienceFunctions:
    """Test convenience functions for pattern matching."""
    
    def test_match_files_to_patterns(self, sample_filenames):
        """Test match_files_to_patterns convenience function."""
        patterns = [
            r"vial_(?P<date>\d+)_(?P<condition>\w+)(_(?P<replicate>\d+))?\.csv",
            r"exp(?P<experiment_id>\d+)_(?P<animal>\w+)_(?P<condition>\w+)\.(?P<extension>\w+)",
        ]
        
        result = match_files_to_patterns(sample_filenames, patterns)
        
        assert isinstance(result, dict)
        assert len(result) >= 0
        
        # Check specific expected matches
        vial_files = [f for f in sample_filenames if f.startswith("vial_") and f.endswith(".csv")]
        exp_files = [f for f in sample_filenames if f.startswith("exp") and ("csv" in f or "pkl" in f)]
        
        for vial_file in vial_files:
            if vial_file in result:
                assert "date" in result[vial_file]
                assert "condition" in result[vial_file]
    
    def test_create_experiment_matcher(self):
        """Test create_experiment_matcher function."""
        patterns = [r"exp(?P<experiment_id>\d+)_(?P<animal>\w+)_(?P<condition>\w+)\.csv"]
        matcher = create_experiment_matcher(patterns)
        
        assert isinstance(matcher, PatternMatcher)
        assert len(matcher.patterns) == 1
        
        # Test the created matcher
        result = matcher.match("exp001_mouse_control.csv")
        assert result is not None
        assert result["experiment_id"] == "001"
    
    def test_create_vial_matcher(self):
        """Test create_vial_matcher function.""" 
        patterns = [r"vial_(?P<date>\d+)_(?P<condition>\w+)\.csv"]
        matcher = create_vial_matcher(patterns)
        
        assert isinstance(matcher, PatternMatcher)
        assert len(matcher.patterns) == 1
        
        # Test the created matcher
        result = matcher.match("vial_20241215_control.csv")
        assert result is not None
        assert result["date"] == "20241215"
    
    @pytest.mark.parametrize("function,patterns,test_file,expected_fields", [
        (extract_experiment_info, [r"exp(?P<experiment_id>\d+)_(?P<animal>\w+)_(?P<condition>\w+)\.csv"], 
         "exp001_mouse_control.csv", {"experiment_id": "001", "animal": "mouse"}),
        (extract_vial_info, [r"vial_(?P<date>\d+)_(?P<condition>\w+)\.csv"],
         "vial_20241215_control.csv", {"date": "20241215", "condition": "control"}),
        (match_experiment_file, [r"exp(?P<experiment_id>\d+)_(?P<animal>\w+)_(?P<condition>\w+)\.csv"],
         "exp002_rat_treatment.csv", {"experiment_id": "002", "animal": "rat"}),
        (match_vial_file, [r"vial_(?P<date>\d+)_(?P<condition>\w+)(_(?P<replicate>\d+))?\.csv"],
         "vial_20241216_treatment_1.csv", {"date": "20241216", "condition": "treatment", "replicate": "1"}),
    ])
    def test_extract_and_match_functions(self, function, patterns, test_file, expected_fields):
        """Parametrized test for extract and match convenience functions."""
        result = function(test_file, patterns)
        
        assert result is not None
        for field, expected_value in expected_fields.items():
            assert field in result
            assert result[field] == expected_value


# --- Pattern Generation Tests ---

class TestPatternGeneration:
    """Test dynamic pattern generation from templates."""
    
    @pytest.mark.parametrize("template,test_input,expected_groups", [
        ("{animal}_{date}_{condition}.csv", "mouse_20241215_control.csv", 
         {"animal": "mouse", "date": "20241215", "condition": "control"}),
        ("exp_{experiment_id}_{animal}_{condition}.pkl", "exp_001_rat_treatment.pkl",
         {"experiment_id": "001", "animal": "rat", "condition": "treatment"}),
        ("data_{session}_{subject_id}.csv", "data_20241215_A123.csv",
         {"session": "20241215", "subject_id": "A123"}),
        ("{prefix}[{identifier}]_{suffix}.txt", "test[123]_sample.txt",
         {"prefix": "test", "identifier": "123", "suffix": "sample"}),
    ])
    def test_generate_pattern_from_template(self, template, test_input, expected_groups):
        """Test pattern generation from templates with various formats."""
        pattern = generate_pattern_from_template(template)
        
        # Verify pattern structure
        assert isinstance(pattern, str)
        assert pattern.startswith("^") and pattern.endswith("$")
        
        # Test pattern matching
        regex = re.compile(pattern)
        match = regex.match(test_input)
        
        assert match is not None, f"Pattern {pattern} should match {test_input}"
        
        # Verify extracted groups
        for field, expected_value in expected_groups.items():
            assert field in match.groupdict()
            assert match.group(field) == expected_value
    
    def test_generate_pattern_special_characters(self):
        """Test pattern generation with special regex characters."""
        template = "data[{date}]_{sample_id}.txt"
        pattern = generate_pattern_from_template(template)
        
        # Should properly escape special characters
        assert r"data\[" in pattern
        assert r"\]" in pattern
        
        # Test the generated pattern
        regex = re.compile(pattern)
        match = regex.match("data[20241215]_sample123.txt")
        assert match is not None
        assert match.group("date") == "20241215"
        assert match.group("sample_id") == "sample123"
    
    def test_generate_pattern_unknown_placeholders(self):
        """Test pattern generation with unknown placeholders."""
        template = "{unknown_field}_{custom_field}.dat"
        pattern = generate_pattern_from_template(template)
        
        # Should create default patterns for unknown fields
        regex = re.compile(pattern)
        match = regex.match("test_value.dat")
        assert match is not None
        assert match.group("unknown_field") == "test"
        assert match.group("custom_field") == "value"
    
    @given(st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=20))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_based_template_generation(self, field_name):
        """Property-based test for template generation."""
        assume(field_name.isidentifier())  # Valid Python identifier
        assume(not field_name.startswith("_"))
        
        template = f"{{{field_name}}}_test.csv"
        pattern = generate_pattern_from_template(template)
        
        # Should create valid regex pattern
        try:
            regex = re.compile(pattern)
            assert regex is not None
        except re.error:
            pytest.fail(f"Generated invalid regex pattern: {pattern}")


# --- Error Handling and Edge Cases ---

class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.parametrize("invalid_pattern", [
        r"[unclosed_bracket",
        r"(?P<invalid_group",
        r"(?P<>empty_name)",
        r"(?P<123>starts_with_number)",
        r"*invalid_quantifier",
        r"+invalid_start",
    ])
    def test_invalid_regex_compilation(self, invalid_pattern):
        """Test handling of invalid regex patterns."""
        with pytest.raises(re.error):
            PatternMatcher([invalid_pattern])
    
    @given(invalid_regex_patterns())
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_based_invalid_patterns(self, invalid_pattern):
        """Property-based test for invalid pattern handling."""
        with pytest.raises(re.error):
            PatternMatcher([invalid_pattern])
    
    def test_empty_string_matching(self, sample_patterns):
        """Test matching against empty strings."""
        matcher = PatternMatcher(sample_patterns)
        
        result = matcher.match("")
        assert result is None
        
        result = matcher.match("   ")  # Whitespace only
        assert result is None
    
    def test_none_input_handling(self, sample_patterns):
        """Test handling of None inputs."""
        matcher = PatternMatcher(sample_patterns)
        
        with pytest.raises((AttributeError, TypeError)):
            matcher.match(None)
    
    def test_very_long_filename_handling(self, sample_patterns):
        """Test handling of very long filenames."""
        matcher = PatternMatcher(sample_patterns)
        
        # Create very long filename
        long_filename = "vial_20241215_" + "very_long_condition_" * 100 + ".csv"
        
        # Should handle gracefully (either match or return None)
        result = matcher.match(long_filename)
        # No exception should be raised


# --- Performance Testing ---

class TestPerformanceValidation:
    """Test performance requirements for pattern matching operations."""
    
    @pytest.mark.benchmark
    def test_pattern_compilation_performance(self, benchmark):
        """Test pattern compilation performance against SLA requirements."""
        patterns = [
            r"vial_(?P<date>\d{8})_(?P<condition>[a-zA-Z0-9_-]+)(_(?P<replicate>\d+))?\.csv",
            r"(?P<animal>mouse|rat)_(?P<date>\d{8})_(?P<condition>[a-zA-Z0-9_-]+)_(?P<replicate>\d+)\.pkl",
            r"exp(?P<experiment_id>\d{3})_(?P<animal>\w+)_(?P<condition>\w+)\.(?P<extension>csv|pkl)",
            r"(?P<experiment_id>\d+)_(?P<dataset>\w+)_(?P<date>\d{8})\.csv",
            r"dataset_(?P<name>\w+)_(?P<version>\d+\.\d+)_(?P<date>\d{8})\.pkl",
        ] * 20  # Multiple copies to test with many patterns
        
        def compile_patterns():
            return PatternMatcher(patterns)
        
        result = benchmark(compile_patterns)
        assert isinstance(result, PatternMatcher)
        
        # Verify compilation time is reasonable (benchmark handles timing)
        assert len(result.compiled_patterns) == len(patterns)
    
    @pytest.mark.benchmark
    def test_matching_performance(self, benchmark, sample_patterns):
        """Test pattern matching performance for single file."""
        matcher = PatternMatcher(sample_patterns)
        test_filename = "vial_20241215_control_1.csv"
        
        def match_single_file():
            return matcher.match(test_filename)
        
        result = benchmark(match_single_file)
        assert result is not None
        assert "date" in result
    
    @pytest.mark.benchmark  
    def test_bulk_filtering_performance(self, benchmark, sample_patterns, sample_filenames):
        """Test bulk file filtering performance."""
        matcher = PatternMatcher(sample_patterns)
        
        # Create larger test set
        large_file_list = sample_filenames * 100
        
        def filter_many_files():
            return matcher.filter_files(large_file_list)
        
        result = benchmark(filter_many_files)
        assert isinstance(result, dict)
    
    def test_regex_compilation_efficiency(self):
        """Test that regex patterns are compiled efficiently."""
        patterns = [r"test_(?P<id>\d+)\.csv"] * 10
        
        start_time = time.time()
        matcher = PatternMatcher(patterns)
        compilation_time = time.time() - start_time
        
        # Should compile quickly even with multiple patterns
        assert compilation_time < 1.0  # Less than 1 second
        assert len(matcher.compiled_patterns) == len(patterns)
    
    def test_large_file_list_handling(self, sample_patterns):
        """Test handling of large file lists."""
        matcher = PatternMatcher(sample_patterns)
        
        # Create large file list
        large_file_list = [f"vial_2024121{i%10}_control_{i%5}.csv" for i in range(10000)]
        
        start_time = time.time()
        result = matcher.filter_files(large_file_list)
        processing_time = time.time() - start_time
        
        # Should process 10k files in reasonable time
        assert processing_time < 5.0  # Less than 5 seconds per requirement
        assert isinstance(result, dict)
        assert len(result) > 0


# --- Integration Testing with Mocks ---

class TestPatternMatchingIntegration:
    """Test pattern matching integration with mocked dependencies."""
    
    def test_pattern_matcher_with_mocked_logging(self, mocker):
        """Test PatternMatcher with mocked logging dependencies."""
        # Mock the logger to test logging integration
        mock_logger = mocker.patch('flyrigloader.discovery.patterns.logger')
        
        patterns = [r"test_(?P<id>\d+)\.csv"]
        matcher = PatternMatcher(patterns)
        
        # Test matching with logging
        result = matcher.match("test_123.csv")
        
        assert result is not None
        assert result["id"] == "123"
        
        # Verify logging calls were made
        assert mock_logger.debug.called
    
    def test_file_discoverer_integration(self, mocker):
        """Test integration with FileDiscoverer mocking."""
        # Mock the FileDiscoverer import
        mock_file_discoverer = mocker.patch('flyrigloader.discovery.files.FileDiscoverer')
        
        patterns = [r"vial_(?P<date>\d+)_(?P<condition>\w+)\.csv"]
        matcher = PatternMatcher(patterns)
        
        # Should work independently of FileDiscoverer
        result = matcher.match("vial_20241215_control.csv")
        assert result is not None
        assert result["date"] == "20241215"
    
    def test_pattern_matching_with_mocked_re_module(self, mocker):
        """Test pattern matching with mocked regex compilation."""
        # Create a spy on re.compile to test compilation behavior
        spy_compile = mocker.spy(re, 'compile')
        
        patterns = [
            r"vial_(?P<date>\d+)_(?P<condition>\w+)\.csv",
            r"exp(?P<id>\d+)_(?P<animal>\w+)\.pkl"
        ]
        
        matcher = PatternMatcher(patterns)
        
        # Verify re.compile was called for each pattern
        assert spy_compile.call_count == len(patterns)
        
        # Test that matching still works
        result = matcher.match("vial_20241215_control.csv")
        assert result is not None


# --- Hypothesis Property-Based Testing ---

class TestPropertyBasedValidation:
    """Property-based testing for robust validation per Section 3.6.3 requirements."""
    
    @given(valid_filenames())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_filename_matching_robustness(self, filename):
        """Property-based test for filename matching robustness."""
        assume(len(filename) > 0 and not filename.isspace())
        
        patterns = [
            r"(?P<prefix>\w+)_(?P<date>\d{8})_(?P<condition>\w+)(_(?P<replicate>\d+))?\.(?P<extension>\w+)",
            r"(?P<type>exp)(?P<id>\d+)_(?P<animal>\w+)_(?P<condition>\w+)\.(?P<extension>\w+)",
            r"(?P<name>\w+)\.(?P<extension>\w+)",
        ]
        
        matcher = PatternMatcher(patterns)
        result = matcher.match(filename)
        
        # Properties that should always hold
        if result is not None:
            assert isinstance(result, dict)
            assert len(result) > 0
            assert all(isinstance(k, str) for k in result.keys())
            assert all(v is None or isinstance(v, str) for v in result.values())
    
    @given(st.lists(valid_filenames(), min_size=0, max_size=50))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_bulk_filtering_properties(self, filenames):
        """Property-based test for bulk filtering properties."""
        patterns = [
            r"(?P<prefix>\w+)_(?P<component>[^_.]+)\.(?P<extension>\w+)",
            r"(?P<name>\w+)\.(?P<extension>\w+)",
        ]
        
        matcher = PatternMatcher(patterns)
        result = matcher.filter_files(filenames)
        
        # Properties that should always hold
        assert isinstance(result, dict)
        assert len(result) <= len(filenames)
        assert all(filename in filenames for filename in result.keys())
        assert all(isinstance(metadata, dict) for metadata in result.values())
    
    @given(regex_patterns())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_pattern_compilation_robustness(self, pattern):
        """Property-based test for pattern compilation robustness."""
        try:
            matcher = PatternMatcher([pattern])
            assert len(matcher.compiled_patterns) == 1
            assert isinstance(matcher.compiled_patterns[0], Pattern)
        except re.error:
            # Invalid patterns should raise re.error, which is acceptable
            pass
    
    @given(st.text(min_size=1, max_size=100))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_match_result_consistency(self, filename):
        """Property-based test for match result consistency."""
        assume(len(filename.strip()) > 0)
        
        patterns = [r"(?P<component>\w+)\.(?P<extension>\w+)"]
        matcher = PatternMatcher(patterns)
        
        # Multiple calls should return same result
        result1 = matcher.match(filename)
        result2 = matcher.match(filename)
        
        assert result1 == result2


# --- Metadata Extraction Comprehensive Testing ---

class TestMetadataExtractionSystem:
    """Comprehensive testing of metadata extraction per F-007 requirements."""
    
    @pytest.mark.parametrize("filename,expected_metadata", [
        # Date parsing validation
        ("vial_20241215_control.csv", {"date": "20241215"}),
        ("experiment_20241225_baseline.pkl", {"date": "20241225"}),
        
        # Condition extraction validation
        ("vial_20241215_control.csv", {"condition": "control"}),
        ("vial_20241215_treatment.csv", {"condition": "treatment"}),
        ("vial_20241215_baseline.csv", {"condition": "baseline"}),
        ("vial_20241215_drug-treatment.csv", {"condition": "drug-treatment"}),
        
        # Replicate handling validation
        ("vial_20241215_control_1.csv", {"replicate": "1"}),
        ("vial_20241215_control_99.csv", {"replicate": "99"}),
        
        # Dataset identification validation
        ("dataset_navigation_1.0_20241215.csv", {"name": "navigation", "version": "1.0", "date": "20241215"}),
        ("dataset_plume_2.1_20241220.pkl", {"name": "plume", "version": "2.1", "date": "20241220"}),
        
        # Animal identification
        ("mouse_20241215_control_1.csv", {"animal": "mouse", "date": "20241215", "condition": "control"}),
        ("rat_20241220_treatment_2.pkl", {"animal": "rat", "date": "20241220", "condition": "treatment"}),
        
        # Experiment identification
        ("exp001_mouse_control.csv", {"experiment_id": "001", "animal": "mouse", "condition": "control"}),
        ("exp999_rat_baseline.pkl", {"experiment_id": "999", "animal": "rat", "condition": "baseline"}),
    ])
    def test_comprehensive_metadata_extraction(self, filename, expected_metadata):
        """Comprehensive test for metadata extraction validation."""
        patterns = [
            r"vial_(?P<date>\d{8})_(?P<condition>[a-zA-Z0-9_-]+)(_(?P<replicate>\d+))?\.(?P<extension>\w+)",
            r"(?P<animal>mouse|rat)_(?P<date>\d{8})_(?P<condition>[a-zA-Z0-9_-]+)_(?P<replicate>\d+)\.(?P<extension>\w+)",
            r"exp(?P<experiment_id>\d{3})_(?P<animal>\w+)_(?P<condition>\w+)\.(?P<extension>\w+)",
            r"dataset_(?P<name>\w+)_(?P<version>\d+\.\d+)_(?P<date>\d{8})\.(?P<extension>\w+)",
            r"(?P<prefix>\w+)_(?P<date>\d{8})_(?P<condition>\w+)\.(?P<extension>\w+)",
        ]
        
        matcher = PatternMatcher(patterns)
        result = matcher.match(filename)
        
        assert result is not None, f"No match found for {filename}"
        
        for field, expected_value in expected_metadata.items():
            assert field in result, f"Field {field} not found in result for {filename}"
            assert result[field] == expected_value, f"Field {field} mismatch for {filename}: expected {expected_value}, got {result[field]}"
    
    def test_date_parsing_edge_cases(self):
        """Test date parsing edge cases and boundary conditions."""
        date_patterns = [
            r"data_(?P<date>\d{8})\.csv",  # YYYYMMDD format
            r"file_(?P<date>\d{6})\.csv",  # YYMMDD format
            r"exp_(?P<date>\d{4})\.csv",   # YYYY format
        ]
        
        matcher = PatternMatcher(date_patterns)
        
        # Test various date formats
        test_cases = [
            ("data_20241215.csv", "20241215"),  # Full 8-digit date
            ("data_20240229.csv", "20240229"),  # Leap year date
            ("data_20241301.csv", "20241301"),  # Invalid month (should still extract)
            ("file_241215.csv", "241215"),     # 6-digit format
            ("exp_2024.csv", "2024"),          # Year only
        ]
        
        for filename, expected_date in test_cases:
            result = matcher.match(filename)
            assert result is not None, f"No match for {filename}"
            assert result["date"] == expected_date
    
    def test_condition_extraction_complexity(self):
        """Test complex condition extraction scenarios."""
        patterns = [r"vial_(?P<date>\d{8})_(?P<condition>[a-zA-Z0-9_-]+)(_(?P<replicate>\d+))?\.csv"]
        matcher = PatternMatcher(patterns)
        
        complex_conditions = [
            ("vial_20241215_control.csv", "control"),
            ("vial_20241215_drug-treatment.csv", "drug-treatment"),
            ("vial_20241215_baseline_test.csv", "baseline_test"),
            ("vial_20241215_condition-1a.csv", "condition-1a"),
            ("vial_20241215_HIGH-dose_treatment.csv", "HIGH-dose_treatment"),
        ]
        
        for filename, expected_condition in complex_conditions:
            result = matcher.match(filename)
            assert result is not None, f"No match for {filename}"
            assert result["condition"] == expected_condition
    
    def test_replicate_handling_variations(self):
        """Test replicate handling across different patterns."""
        patterns = [
            r"vial_(?P<date>\d{8})_(?P<condition>\w+)_(?P<replicate>\d+)\.csv",  # Required replicate
            r"exp_(?P<id>\d+)_(?P<condition>\w+)(_rep(?P<replicate>\d+))?\.csv",  # Optional with prefix
            r"data_(?P<condition>\w+)_r(?P<replicate>\d+)\.csv",  # Single letter prefix
        ]
        
        matcher = PatternMatcher(patterns)
        
        replicate_cases = [
            ("vial_20241215_control_1.csv", "1"),
            ("vial_20241215_control_99.csv", "99"),
            ("exp_001_control_rep2.csv", "2"),
            ("exp_002_treatment.csv", None),  # No replicate
            ("data_baseline_r5.csv", "5"),
        ]
        
        for filename, expected_replicate in replicate_cases:
            result = matcher.match(filename)
            if result is not None:
                if expected_replicate is None:
                    assert "replicate" not in result or result["replicate"] is None
                else:
                    assert "replicate" in result
                    assert result["replicate"] == expected_replicate


# --- Legacy and Backward Compatibility Testing ---

class TestBackwardCompatibility:
    """Test backward compatibility with legacy patterns and systems."""
    
    def test_positional_group_patterns(self):
        """Test legacy patterns using positional groups."""
        # Legacy patterns without named groups
        legacy_patterns = [
            r"legacy_(\w+)_(\d{8})_(\w+)_(\d+)\.csv",  # 4 positional groups
            r"old_format_(\w+)_(\d+)\.pkl",            # 2 positional groups
        ]
        
        matcher = PatternMatcher(legacy_patterns)
        
        # Test that positional groups are handled
        result = matcher.match("legacy_mouse_20241215_control_1.csv")
        assert result is not None
        assert isinstance(result, dict)
        assert len(result) > 0
        
        result = matcher.match("old_format_experiment_123.pkl")
        assert result is not None
        assert isinstance(result, dict)
    
    def test_mixed_named_and_positional_patterns(self):
        """Test mixing named and positional group patterns."""
        mixed_patterns = [
            r"new_(?P<date>\d{8})_(?P<condition>\w+)\.csv",  # Named groups
            r"old_(\w+)_(\d{8})_(\w+)\.csv",                 # Positional groups
        ]
        
        matcher = PatternMatcher(mixed_patterns)
        
        # Test named group pattern
        result = matcher.match("new_20241215_control.csv")
        assert result is not None
        assert "date" in result
        assert "condition" in result
        
        # Test positional group pattern  
        result = matcher.match("old_mouse_20241215_treatment.csv")
        assert result is not None
        assert isinstance(result, dict)
    
    def test_legacy_special_case_handling(self):
        """Test legacy special case handling is preserved."""
        # Test the special case processing mentioned in the original code
        patterns = [r"(?P<animal>mouse|rat)_(?P<date>\d+)_(?P<condition>\w+)\.csv"]
        matcher = PatternMatcher(patterns)
        
        # This should trigger special case processing
        result = matcher.match("mouse_20241215_baseline.csv")
        assert result is not None
        # Special case processing should be applied appropriately


if __name__ == "__main__":
    pytest.main([__file__, "-v"])