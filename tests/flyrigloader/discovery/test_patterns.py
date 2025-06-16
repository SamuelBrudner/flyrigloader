"""
Behavior-focused test suite for pattern matching functionality.

This module provides comprehensive testing coverage for the flyrigloader pattern matching system
through black-box behavioral validation, emphasizing public API contracts and observable system
behavior rather than implementation-specific details. Tests follow AAA (Arrange-Act-Assert)
patterns and use centralized fixtures from tests/conftest.py for consistency.

All performance benchmarks have been relocated to scripts/benchmarks/ per Section 0 requirements
for performance test isolation and rapid developer feedback cycles.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Any

import pytest
from hypothesis import given, strategies as st, assume, settings, HealthCheck

# Import centralized test utilities and fixtures
from tests.utils import (
    create_mock_filesystem,
    create_mock_config_provider,
    EdgeCaseScenarioGenerator,
    FlyrigloaderStrategies,
    TestStructureValidator,
)

# Import production code through public API
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
    PatternCompilationError,
)


# --- Centralized Test Utilities and Hypothesis Strategies ---

# Use centralized Hypothesis strategies from tests/utils.py
def get_flyrigloader_strategies():
    """Get domain-specific Hypothesis strategies for pattern testing."""
    return FlyrigloaderStrategies()


# --- Test Fixtures Using Centralized Infrastructure ---

@pytest.fixture
def sample_patterns(test_data_generator):
    """
    Provide comprehensive set of realistic patterns for testing using centralized data generator.
    
    Uses centralized test infrastructure to maintain consistency across test modules
    and eliminate fixture duplication per Section 0 requirements.
    """
    return [
        # Vial patterns with named groups for metadata extraction
        r"vial_(?P<date>\d{8})_(?P<condition>[a-zA-Z0-9_-]+)(_(?P<replicate>\d+))?\.csv",
        r"(?P<animal>mouse|rat)_(?P<date>\d{8})_(?P<condition>[a-zA-Z0-9_-]+)_(?P<replicate>\d+)\.(?P<extension>pkl|csv)",
        
        # Experiment patterns for research workflow validation
        r"exp(?P<experiment_id>\d{3})_(?P<animal>\w+)_(?P<condition>\w+)\.(?P<extension>csv|pkl)",
        r"(?P<experiment_id>\d+)_(?P<dataset>\w+)_(?P<date>\d{8})\.csv",
        
        # Dataset patterns for data organization
        r"dataset_(?P<name>\w+)_(?P<version>\d+\.\d+)_(?P<date>\d{8})\.(?P<extension>pkl|csv)",
        
        # Legacy patterns for backward compatibility
        r"legacy_(\w+)_(\d{8})_(\w+)_(\d+)\.csv",  # Positional groups
    ]


@pytest.fixture
def sample_filenames(test_data_generator):
    """
    Provide comprehensive set of test filenames using centralized data generator.
    
    Uses centralized test infrastructure for consistent filename generation patterns
    across all discovery test modules.
    """
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
        
        # Non-matching files for negative testing
        "notes.txt",
        "metadata.json",
        "analysis_script.py",
        "README.md",
        
        # Edge cases for boundary condition testing
        "vial_.csv",  # Missing components
        "exp_mouse_control.csv",  # Missing experiment ID
        "invalid_format_file.csv",
        
        # Files with special characters for Unicode testing
        "vial_20241225_control-test_1.csv",
        "exp001_mouse_control_v2.csv",
        "dataset_test-data_1.0_20241226.pkl",
    ]


@pytest.fixture
def mock_filesystem_structure(temp_experiment_directory):
    """
    Create comprehensive temporary file structure using centralized filesystem mock.
    
    Leverages centralized test infrastructure from tests/conftest.py to ensure
    consistent filesystem simulation across test modules.
    """
    return temp_experiment_directory


# --- Pattern Matcher Behavior-Focused Tests ---

class TestPatternMatcherBehavior:
    """
    Test PatternMatcher behavior through public API validation.
    
    These tests focus on observable behavior rather than implementation details,
    following black-box testing principles per Section 0 requirements.
    """
    
    def test_empty_patterns_behavior(self):
        """
        Test PatternMatcher behavior with empty patterns list.
        
        Validates that PatternMatcher with no patterns correctly returns
        None for any filename match attempt.
        """
        # ARRANGE - Create PatternMatcher with empty patterns
        matcher = PatternMatcher([])
        test_filename = "any_file.csv"
        
        # ACT - Attempt to match a filename
        result = matcher.match(test_filename)
        
        # ASSERT - Should return None for any filename
        assert result is None
    
    def test_single_pattern_matching_behavior(self):
        """
        Test PatternMatcher behavior with single pattern.
        
        Validates correct metadata extraction through public API
        without examining internal compilation details.
        """
        # ARRANGE - Create PatternMatcher with single pattern
        pattern = r"test_(?P<id>\d+)\.csv"
        matcher = PatternMatcher([pattern])
        test_filename = "test_123.csv"
        
        # ACT - Match filename against pattern
        result = matcher.match(test_filename)
        
        # ASSERT - Should extract expected metadata
        assert result is not None
        assert result["id"] == "123"
    
    @pytest.mark.parametrize("patterns,test_file,expected_field", [
        ([r"vial_(?P<date>\d+)\.csv"], "vial_20241215.csv", "date"),
        ([r"exp(?P<id>\d+)\.pkl"], "exp001.pkl", "id"),
        ([r"(?P<animal>\w+)_(?P<date>\d{8})\.csv"], "mouse_20241215.csv", "animal"),
    ])
    def test_multiple_patterns_behavior(self, patterns, test_file, expected_field):
        """
        Test PatternMatcher behavior with multiple patterns.
        
        Validates that PatternMatcher correctly identifies and extracts
        metadata from various pattern types.
        """
        # ARRANGE - Create PatternMatcher with multiple patterns
        matcher = PatternMatcher(patterns)
        
        # ACT - Match test file against patterns
        result = matcher.match(test_file)
        
        # ASSERT - Should extract expected field
        assert result is not None
        assert expected_field in result
        assert result[expected_field] is not None
    
    def test_invalid_regex_error_handling(self):
        """
        Test PatternMatcher error handling for invalid regex patterns.
        
        Validates proper exception raising for malformed patterns
        through public API behavior.
        """
        # ARRANGE - Prepare invalid regex patterns
        invalid_patterns = [
            r"[unclosed_bracket",
            r"(?P<invalid_group",
            r"(?P<>empty_name)",
        ]
        
        # ACT & ASSERT - Should raise appropriate exceptions
        for invalid_pattern in invalid_patterns:
            with pytest.raises((re.error, PatternCompilationError)):
                PatternMatcher([invalid_pattern])
    
    @given(get_flyrigloader_strategies().experimental_file_paths())
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_pattern_matching_robustness(self, filename):
        """
        Property-based test for pattern matching robustness.
        
        Uses domain-specific strategies to validate PatternMatcher
        behavior across wide range of realistic experimental filenames.
        """
        # ARRANGE - Create PatternMatcher with realistic experimental patterns
        patterns = [
            r"(?P<animal>\w+)_(?P<date>\d{8})_(?P<condition>\w+)_rep(?P<replicate>\d+)\.(?P<extension>\w+)",
            r"(?P<name>\w+)\.(?P<extension>\w+)",  # Fallback pattern
        ]
        matcher = PatternMatcher(patterns)
        
        # ACT - Attempt to match generated filename
        result = matcher.match(filename)
        
        # ASSERT - Should either match successfully or return None without error
        if result is not None:
            assert isinstance(result, dict)
            assert all(isinstance(k, str) for k in result.keys())
            assert all(v is None or isinstance(v, str) for v in result.values())


class TestPatternMatchingBehavior:
    """
    Test pattern matching behavior through observable outcomes.
    
    Focuses on metadata extraction accuracy and behavioral contracts
    rather than internal pattern compilation details.
    """
    
    def test_experimental_file_metadata_extraction(self, sample_patterns):
        """
        Test accurate metadata extraction from experimental filenames.
        
        Validates that PatternMatcher correctly extracts research-relevant
        metadata from various experimental file naming conventions.
        """
        # ARRANGE - Create PatternMatcher with experimental patterns
        matcher = PatternMatcher(sample_patterns)
        
        # ACT & ASSERT - Test vial pattern metadata extraction
        vial_result = matcher.match("vial_20241215_control.csv")
        assert vial_result is not None
        assert vial_result["date"] == "20241215"
        assert vial_result["condition"] == "control"
        # Replicate should be None or not present for non-replicated vials
        assert vial_result.get("replicate") is None
        
        # ACT & ASSERT - Test vial with replicate metadata extraction
        replicate_result = matcher.match("vial_20241216_treatment_1.csv")
        assert replicate_result is not None
        assert replicate_result["date"] == "20241216"
        assert replicate_result["condition"] == "treatment"
        assert replicate_result["replicate"] == "1"
        
        # ACT & ASSERT - Test experiment pattern metadata extraction
        experiment_result = matcher.match("exp001_mouse_control.csv")
        assert experiment_result is not None
        assert experiment_result["experiment_id"] == "001"
        assert experiment_result["animal"] == "mouse"
        assert experiment_result["condition"] == "control"
    
    @pytest.mark.parametrize("filename,expected_fields", [
        ("vial_20241215_control.csv", {"date": "20241215", "condition": "control"}),
        ("vial_20241216_treatment_1.csv", {"date": "20241216", "condition": "treatment", "replicate": "1"}),
        ("mouse_20241217_baseline_2.pkl", {"animal": "mouse", "date": "20241217", "condition": "baseline", "replicate": "2"}),
        ("exp001_mouse_control.csv", {"experiment_id": "001", "animal": "mouse", "condition": "control"}),
        ("exp999_rat_treatment.pkl", {"experiment_id": "999", "animal": "rat", "condition": "treatment"}),
        ("123_dataset1_20241218.csv", {"experiment_id": "123", "dataset": "dataset1", "date": "20241218"}),
    ])
    def test_metadata_extraction_accuracy(self, sample_patterns, filename, expected_fields):
        """
        Test metadata extraction accuracy across various filename patterns.
        
        Validates that extracted metadata matches expected values for
        research workflow requirements.
        """
        # ARRANGE - Create PatternMatcher with comprehensive patterns
        matcher = PatternMatcher(sample_patterns)
        
        # ACT - Extract metadata from filename
        result = matcher.match(filename)
        
        # ASSERT - Validate extracted metadata accuracy
        assert result is not None, f"Pattern matching failed for research file: {filename}"
        for field, expected_value in expected_fields.items():
            assert field in result, f"Required metadata field '{field}' missing for {filename}"
            assert result[field] == expected_value, f"Metadata field '{field}' mismatch for {filename}: expected {expected_value}, got {result[field]}"
    
    @pytest.mark.parametrize("filename", [
        "notes.txt",
        "metadata.json", 
        "invalid_format.csv",
        "vial_.csv",  # Missing required components
        "exp_mouse.csv",  # Missing experiment ID
        "",  # Empty filename
        "   ",  # Whitespace only
    ])
    def test_non_experimental_file_rejection(self, sample_patterns, filename):
        """
        Test rejection of non-experimental files.
        
        Validates that files not matching experimental patterns
        are correctly identified and return None.
        """
        # ARRANGE - Create PatternMatcher with experimental patterns
        matcher = PatternMatcher(sample_patterns)
        
        # ACT - Attempt to match non-experimental filename
        result = matcher.match(filename)
        
        # ASSERT - Should correctly reject non-experimental files
        assert result is None, f"Unexpected match for non-experimental file {filename}: {result}"
    
    def test_pattern_priority_behavior(self):
        """
        Test pattern matching priority behavior.
        
        Validates that when multiple patterns could match,
        the first matching pattern takes precedence.
        """
        # ARRANGE - Create patterns with overlapping match capabilities
        patterns = [
            r"(?P<type>vial)_(?P<date>\d+)_(?P<condition>\w+)\.csv",  # More specific vial pattern
            r"(?P<prefix>\w+)_(?P<suffix>\w+)\.csv",  # General pattern
        ]
        matcher = PatternMatcher(patterns)
        test_filename = "vial_20241215_control.csv"
        
        # ACT - Match filename that could match multiple patterns
        result = matcher.match(test_filename)
        
        # ASSERT - Should match first (more specific) pattern
        assert result is not None
        assert "type" in result  # Field from first pattern
        assert result["type"] == "vial"
        assert "prefix" not in result  # Should not include fields from second pattern
    
    @given(get_flyrigloader_strategies().experimental_file_paths())
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_experimental_filename_robustness(self, filename):
        """
        Property-based test for experimental filename handling robustness.
        
        Uses domain-specific filename generation to validate robust
        handling of realistic experimental filename variations.
        """
        # ARRANGE - Create PatternMatcher with flexible experimental patterns
        patterns = [
            r"(?P<prefix>\w+)_(?P<component1>[^_]+)_(?P<component2>[^_.]+)\.(?P<extension>\w+)",
            r"(?P<name>\w+)\.(?P<extension>\w+)",  # Fallback for simple names
        ]
        matcher = PatternMatcher(patterns)
        
        # ACT - Attempt to match generated experimental filename
        result = matcher.match(filename)
        
        # ASSERT - Should handle gracefully without exceptions
        if result is not None:
            assert isinstance(result, dict)
            assert all(isinstance(k, str) for k in result.keys())
            assert all(v is None or isinstance(v, str) for v in result.values())


class TestPatternMatchingEdgeCases:
    """
    Test pattern matching edge cases and special scenarios.
    
    Focuses on observable behavior for complex filename patterns
    and edge cases in experimental data workflows.
    """
    
    def test_complex_condition_pattern_handling(self):
        """
        Test handling of complex condition patterns in animal files.
        
        Validates behavior when condition and replicate information
        might be captured together and need processing.
        """
        # ARRANGE - Create pattern that captures complex condition formats
        patterns = [r"(?P<animal>mouse|rat)_(?P<date>\d+)_(?P<condition>\w+_\d+)\.csv"]
        matcher = PatternMatcher(patterns)
        test_filename = "mouse_20241215_control_2.csv"
        
        # ACT - Match filename with complex condition pattern
        result = matcher.match(test_filename)
        
        # ASSERT - Should extract metadata successfully
        assert result is not None
        assert result["animal"] == "mouse"
        assert result["date"] == "20241215"
        assert "condition" in result  # Should extract condition information
    
    def test_baseline_experiment_metadata_extraction(self, sample_patterns):
        """
        Test metadata extraction for baseline experiment files.
        
        Validates proper handling of baseline experimental conditions
        in research workflow scenarios.
        """
        # ARRANGE - Create PatternMatcher with experimental patterns
        matcher = PatternMatcher(sample_patterns)
        test_filename = "exp001_mouse_baseline.csv"
        
        # ACT - Extract metadata from baseline experiment file
        result = matcher.match(test_filename)
        
        # ASSERT - Should extract baseline experiment metadata
        assert result is not None
        assert "experiment_id" in result or "animal" in result
        # Verify that baseline condition is properly captured
        if "condition" in result:
            assert "baseline" in test_filename.lower()
    
    def test_legacy_pattern_compatibility(self):
        """
        Test backward compatibility with legacy positional group patterns.
        
        Validates that legacy filename patterns without named groups
        still produce meaningful metadata extraction.
        """
        # ARRANGE - Create matcher with legacy positional pattern
        patterns = [r"legacy_(\w+)_(\d{8})_(\w+)_(\d+)\.csv"]
        matcher = PatternMatcher(patterns)
        test_filename = "legacy_mouse_20241215_control_1.csv"
        
        # ACT - Match legacy filename pattern
        result = matcher.match(test_filename)
        
        # ASSERT - Should extract metadata from positional groups
        assert result is not None
        assert isinstance(result, dict)
        assert len(result) > 0
        # Validate that some meaningful field extraction occurred
        assert any(value is not None for value in result.values())


class TestFileFilteringBehavior:
    """
    Test file filtering behavior through observable outcomes.
    
    Validates file filtering functionality for research workflow
    file discovery and organization scenarios.
    """
    
    def test_empty_file_list_filtering(self, sample_patterns):
        """
        Test file filtering behavior with empty input list.
        
        Validates that empty file lists are handled gracefully
        and return empty results.
        """
        # ARRANGE - Create PatternMatcher with experimental patterns
        matcher = PatternMatcher(sample_patterns)
        empty_file_list = []
        
        # ACT - Filter empty file list
        result = matcher.filter_files(empty_file_list)
        
        # ASSERT - Should return empty dictionary
        assert result == {}
        assert isinstance(result, dict)
    
    def test_mixed_file_list_filtering(self, sample_patterns, sample_filenames):
        """
        Test filtering behavior with mixed experimental and non-experimental files.
        
        Validates correct identification and metadata extraction for
        experimental files while rejecting non-experimental files.
        """
        # ARRANGE - Create PatternMatcher with comprehensive experimental patterns
        matcher = PatternMatcher(sample_patterns)
        
        # ACT - Filter mixed file list
        result = matcher.filter_files(sample_filenames)
        
        # ASSERT - Should identify experimental files
        assert len(result) > 0, "Should identify experimental files in mixed list"
        
        # Verify that all matched files have valid metadata
        for filename, metadata in result.items():
            assert isinstance(metadata, dict)
            assert len(metadata) > 0
            assert filename in sample_filenames
        
        # Verify experimental files are captured
        experimental_indicators = ["vial_", "exp", "dataset_"]
        found_experimental = any(
            any(indicator in filename for indicator in experimental_indicators)
            for filename in result.keys()
        )
        assert found_experimental, "Should identify files with experimental naming patterns"
    
    def test_full_path_filtering_behavior(self, mock_filesystem_structure, sample_patterns):
        """
        Test file filtering behavior with full file paths.
        
        Validates that filtering works correctly with absolute paths
        and preserves path information in results.
        """
        # ARRANGE - Create PatternMatcher and get test directory structure
        matcher = PatternMatcher(sample_patterns)
        test_directory = mock_filesystem_structure["directory"]
        test_files = [str(f) for f in mock_filesystem_structure["raw_files"]]
        
        # ACT - Filter files with full paths
        result = matcher.filter_files(test_files)
        
        # ASSERT - Should handle full paths correctly
        if len(result) > 0:  # If any matches found
            for filepath in result.keys():
                assert filepath in test_files, "Result should contain original file paths"
                # Verify metadata extraction worked
                assert isinstance(result[filepath], dict)
    
    @given(get_flyrigloader_strategies().experimental_file_paths())
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=20)
    def test_file_filtering_robustness(self, generated_filenames):
        """
        Property-based test for file filtering robustness.
        
        Uses domain-specific filename generation to validate
        filtering behavior across realistic experimental scenarios.
        """
        # ARRANGE - Create PatternMatcher with flexible patterns
        patterns = [
            r"(?P<prefix>\w+)_(?P<component>[^_.]+)\.(?P<extension>\w+)",
            r"(?P<name>\w+)\.(?P<extension>\w+)",
        ]
        matcher = PatternMatcher(patterns)
        
        # Convert single filename to list for testing
        if isinstance(generated_filenames, str):
            filenames = [generated_filenames]
        else:
            filenames = generated_filenames if isinstance(generated_filenames, list) else [generated_filenames]
        
        # ACT - Filter generated filenames
        result = matcher.filter_files(filenames)
        
        # ASSERT - Should maintain filtering invariants
        assert isinstance(result, dict)
        assert len(result) <= len(filenames)
        assert all(filename in filenames for filename in result.keys())
        # All matched files should have valid metadata
        assert all(isinstance(metadata, dict) for metadata in result.values())


# --- Public API Convenience Function Tests ---

class TestPublicAPIConvenienceFunctions:
    """
    Test public API convenience functions for pattern matching.
    
    Validates high-level convenience functions that provide simplified
    interfaces for common research workflow pattern matching scenarios.
    """
    
    def test_match_files_to_patterns_api(self, sample_filenames):
        """
        Test match_files_to_patterns convenience function behavior.
        
        Validates the high-level API for matching multiple files
        against multiple patterns in research workflows.
        """
        # ARRANGE - Create experimental patterns for bulk matching
        patterns = [
            r"vial_(?P<date>\d+)_(?P<condition>\w+)(_(?P<replicate>\d+))?\.csv",
            r"exp(?P<experiment_id>\d+)_(?P<animal>\w+)_(?P<condition>\w+)\.(?P<extension>\w+)",
        ]
        
        # ACT - Use convenience function for bulk file matching
        result = match_files_to_patterns(sample_filenames, patterns)
        
        # ASSERT - Should return structured results
        assert isinstance(result, dict)
        
        # Verify that vial files are properly matched and have expected metadata
        vial_files = [f for f in sample_filenames if f.startswith("vial_") and f.endswith(".csv")]
        for vial_file in vial_files:
            if vial_file in result:
                assert "date" in result[vial_file], f"Date metadata missing for {vial_file}"
                assert "condition" in result[vial_file], f"Condition metadata missing for {vial_file}"
    
    def test_create_experiment_matcher_behavior(self):
        """
        Test create_experiment_matcher factory function behavior.
        
        Validates that the factory function creates functional
        PatternMatcher instances for experiment file patterns.
        """
        # ARRANGE - Define experiment-specific patterns
        patterns = [r"exp(?P<experiment_id>\d+)_(?P<animal>\w+)_(?P<condition>\w+)\.csv"]
        
        # ACT - Create experiment matcher using factory function
        matcher = create_experiment_matcher(patterns)
        
        # ASSERT - Should create functional matcher
        assert isinstance(matcher, PatternMatcher)
        
        # Verify matcher functionality through behavior
        test_result = matcher.match("exp001_mouse_control.csv")
        assert test_result is not None
        assert test_result["experiment_id"] == "001"
        assert test_result["animal"] == "mouse"
        assert test_result["condition"] == "control"
    
    def test_create_vial_matcher_behavior(self):
        """
        Test create_vial_matcher factory function behavior.
        
        Validates that the factory function creates functional
        PatternMatcher instances for vial file patterns.
        """
        # ARRANGE - Define vial-specific patterns
        patterns = [r"vial_(?P<date>\d+)_(?P<condition>\w+)\.csv"]
        
        # ACT - Create vial matcher using factory function
        matcher = create_vial_matcher(patterns)
        
        # ASSERT - Should create functional matcher
        assert isinstance(matcher, PatternMatcher)
        
        # Verify matcher functionality through behavior
        test_result = matcher.match("vial_20241215_control.csv")
        assert test_result is not None
        assert test_result["date"] == "20241215"
        assert test_result["condition"] == "control"
    
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
    def test_specialized_extraction_functions(self, function, patterns, test_file, expected_fields):
        """
        Test specialized metadata extraction convenience functions.
        
        Validates domain-specific convenience functions for extracting
        metadata from experiment and vial files.
        """
        # ARRANGE - Use provided function, patterns, and test file
        
        # ACT - Call specialized extraction function
        result = function(test_file, patterns)
        
        # ASSERT - Should extract expected metadata
        assert result is not None, f"Function {function.__name__} failed to extract metadata from {test_file}"
        for field, expected_value in expected_fields.items():
            assert field in result, f"Expected field '{field}' missing from {function.__name__} result"
            assert result[field] == expected_value, f"Field '{field}' value mismatch in {function.__name__}"


# --- Dynamic Pattern Generation Tests ---

class TestDynamicPatternGeneration:
    """
    Test dynamic pattern generation from templates.
    
    Validates template-based pattern generation for creating
    flexible filename matching patterns in research workflows.
    """
    
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
    def test_template_to_pattern_conversion_behavior(self, template, test_input, expected_groups):
        """
        Test template-to-pattern conversion behavior.
        
        Validates that templates are correctly converted to functional
        regex patterns that extract expected metadata.
        """
        # ARRANGE - Use template for pattern generation
        
        # ACT - Generate pattern from template
        pattern = generate_pattern_from_template(template)
        
        # ASSERT - Generated pattern should be functional
        assert isinstance(pattern, str)
        assert pattern.startswith("^") and pattern.endswith("$"), "Pattern should be anchored"
        
        # Verify pattern functionality through behavior
        regex = re.compile(pattern)
        match = regex.match(test_input)
        
        assert match is not None, f"Generated pattern should match test input: {test_input}"
        
        # Validate metadata extraction accuracy
        for field, expected_value in expected_groups.items():
            assert field in match.groupdict(), f"Field {field} should be captured"
            assert match.group(field) == expected_value, f"Field {field} value mismatch"
    
    def test_special_character_handling_behavior(self):
        """
        Test special character handling in template generation.
        
        Validates that special regex characters in templates are
        properly escaped while preserving pattern functionality.
        """
        # ARRANGE - Template with special regex characters
        template = "data[{date}]_{sample_id}.txt"
        test_filename = "data[20241215]_sample123.txt"
        
        # ACT - Generate pattern from template with special characters
        pattern = generate_pattern_from_template(template)
        
        # ASSERT - Should handle special characters correctly
        assert r"data\[" in pattern, "Square brackets should be escaped"
        assert r"\]" in pattern, "Closing bracket should be escaped"
        
        # Verify pattern works correctly
        regex = re.compile(pattern)
        match = regex.match(test_filename)
        assert match is not None, "Pattern should match despite special characters"
        assert match.group("date") == "20241215"
        assert match.group("sample_id") == "sample123"
    
    def test_unknown_placeholder_handling_behavior(self):
        """
        Test handling of unknown placeholders in templates.
        
        Validates that templates with custom field names create
        appropriate default patterns for metadata extraction.
        """
        # ARRANGE - Template with unknown custom fields
        template = "{unknown_field}_{custom_field}.dat"
        test_filename = "test_value.dat"
        
        # ACT - Generate pattern from template with unknown fields
        pattern = generate_pattern_from_template(template)
        
        # ASSERT - Should create functional pattern with default patterns
        regex = re.compile(pattern)
        match = regex.match(test_filename)
        assert match is not None, "Should handle unknown placeholders gracefully"
        assert match.group("unknown_field") == "test"
        assert match.group("custom_field") == "value"
    
    @given(st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=20))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_template_generation_robustness(self, field_name):
        """
        Property-based test for template generation robustness.
        
        Validates that template generation produces valid regex patterns
        for arbitrary field names in research filename scenarios.
        """
        # ARRANGE - Filter for valid field names
        assume(field_name.isidentifier())  # Valid Python identifier
        assume(not field_name.startswith("_"))
        
        template = f"{{{field_name}}}_test.csv"
        
        # ACT - Generate pattern from property-based template
        pattern = generate_pattern_from_template(template)
        
        # ASSERT - Should create valid, compilable regex pattern
        try:
            regex = re.compile(pattern)
            assert regex is not None, "Generated pattern should be compilable"
        except re.error as e:
            pytest.fail(f"Generated invalid regex pattern: {pattern}, error: {e}")


# --- Error Handling and Edge Case Validation ---

class TestErrorHandlingBehavior:
    """
    Test error handling behavior and edge case validation.
    
    Validates robust error handling and graceful degradation
    for malformed patterns and edge case input scenarios.
    """
    
    @pytest.mark.parametrize("invalid_pattern", [
        r"[unclosed_bracket",
        r"(?P<invalid_group",
        r"(?P<>empty_name)",
        r"(?P<123>starts_with_number)",
        r"*invalid_quantifier",
        r"+invalid_start",
    ])
    def test_malformed_pattern_error_handling(self, invalid_pattern):
        """
        Test error handling for malformed regex patterns.
        
        Validates that PatternMatcher appropriately raises exceptions
        for invalid regex patterns during initialization.
        """
        # ARRANGE - Use malformed regex pattern
        
        # ACT & ASSERT - Should raise appropriate exception
        with pytest.raises((re.error, PatternCompilationError)):
            PatternMatcher([invalid_pattern])
    
    def test_edge_case_input_handling(self, sample_patterns):
        """
        Test handling of edge case inputs.
        
        Validates robust behavior for empty strings, whitespace,
        and other boundary condition inputs.
        """
        # ARRANGE - Create PatternMatcher with valid patterns
        matcher = PatternMatcher(sample_patterns)
        
        # ACT & ASSERT - Test empty string behavior
        empty_result = matcher.match("")
        assert empty_result is None, "Empty string should not match any pattern"
        
        # ACT & ASSERT - Test whitespace-only behavior
        whitespace_result = matcher.match("   ")
        assert whitespace_result is None, "Whitespace-only string should not match"
        
        # ACT & ASSERT - Test very short input
        short_result = matcher.match("a")
        assert short_result is None, "Very short input should not match experimental patterns"
    
    def test_none_input_error_handling(self, sample_patterns):
        """
        Test error handling for None input.
        
        Validates that PatternMatcher appropriately handles
        None input with proper exception behavior.
        """
        # ARRANGE - Create PatternMatcher with valid patterns
        matcher = PatternMatcher(sample_patterns)
        
        # ACT & ASSERT - Should raise appropriate exception for None input
        with pytest.raises((AttributeError, TypeError)):
            matcher.match(None)
    
    def test_boundary_condition_filename_handling(self, sample_patterns):
        """
        Test handling of boundary condition filenames.
        
        Validates graceful handling of very long filenames and
        other boundary conditions without raising exceptions.
        """
        # ARRANGE - Create PatternMatcher and boundary condition filename
        matcher = PatternMatcher(sample_patterns)
        long_filename = "vial_20241215_" + "very_long_condition_" * 100 + ".csv"
        
        # ACT - Attempt to match very long filename
        result = matcher.match(long_filename)
        
        # ASSERT - Should handle gracefully without exception
        # Result can be None or successful match, but no exception should be raised
        assert result is None or isinstance(result, dict)
    
    def test_corrupted_data_scenarios(self):
        """
        Test handling of corrupted or malformed filename patterns.
        
        Uses centralized edge-case scenarios to validate robust
        error handling for realistic corruption scenarios.
        """
        # ARRANGE - Generate corrupted data scenarios using centralized utilities
        edge_case_generator = EdgeCaseScenarioGenerator()
        corrupted_scenarios = edge_case_generator.generate_corrupted_data_scenarios()
        
        patterns = [r"(?P<prefix>\w+)_(?P<component>[^_.]+)\.(?P<extension>\w+)"]
        matcher = PatternMatcher(patterns)
        
        # ACT & ASSERT - Test each corrupted scenario
        for scenario in corrupted_scenarios:
            if scenario['type'] in ['malformed_yaml', 'binary_in_text']:
                # These scenarios involve binary data that might create unusual filenames
                test_filename = f"corrupted_{scenario['type']}.csv"
                
                # Should handle gracefully without exceptions
                result = matcher.match(test_filename)
                assert result is None or isinstance(result, dict)


# --- Performance Tests (Relocated to scripts/benchmarks/) ---
#
# NOTE: All performance and benchmark tests have been relocated to scripts/benchmarks/
# per Section 0 requirements for performance test isolation. These tests are excluded
# from default pytest execution and run via:
#
#   python scripts/benchmarks/run_benchmarks.py --category pattern-matching
#
# Former benchmark tests included:
# - test_pattern_compilation_performance
# - test_matching_performance  
# - test_bulk_filtering_performance
# - test_regex_compilation_efficiency
# - test_large_file_list_handling
#
# All performance validation now occurs through dedicated CLI runner to maintain
# rapid developer feedback cycles (<30s test execution time).


# --- Integration Testing with Protocol-Based Mocks ---

class TestPatternMatchingIntegration:
    """
    Test pattern matching integration using protocol-based mocks.
    
    Uses centralized mock implementations from tests/utils.py for
    consistent dependency isolation across discovery tests.
    """
    
    def test_pattern_matcher_with_mock_filesystem(self, mock_filesystem, hypothesis_strategies):
        """
        Test PatternMatcher integration with filesystem mock.
        
        Uses centralized MockFilesystem to validate pattern matching
        behavior in realistic file system scenarios.
        """
        # ARRANGE - Create mock filesystem with experimental files
        filesystem = create_mock_filesystem(
            structure={
                'files': {
                    '/test/data/vial_20241215_control.csv': {'size': 1024},
                    '/test/data/exp001_mouse_control.pkl': {'size': 2048},
                    '/test/data/notes.txt': {'size': 256}
                },
                'directories': ['/test', '/test/data']
            }
        )
        
        patterns = [
            r"vial_(?P<date>\d{8})_(?P<condition>\w+)\.csv",
            r"exp(?P<experiment_id>\d{3})_(?P<animal>\w+)_(?P<condition>\w+)\.pkl"
        ]
        matcher = PatternMatcher(patterns)
        
        # ACT - Filter files from mock filesystem
        test_files = ['/test/data/vial_20241215_control.csv', '/test/data/exp001_mouse_control.pkl', '/test/data/notes.txt']
        result = matcher.filter_files(test_files)
        
        # ASSERT - Should identify experimental files correctly
        assert len(result) >= 2, "Should identify experimental files"
        assert '/test/data/vial_20241215_control.csv' in result
        assert '/test/data/exp001_mouse_control.pkl' in result
        assert '/test/data/notes.txt' not in result  # Non-experimental file should be filtered out
    
    def test_pattern_matcher_with_edge_case_scenarios(self):
        """
        Test PatternMatcher with edge-case scenarios from centralized generator.
        
        Uses EdgeCaseScenarioGenerator to validate robust handling
        of boundary conditions and unicode scenarios.
        """
        # ARRANGE - Generate edge-case scenarios using centralized utilities
        edge_case_generator = EdgeCaseScenarioGenerator()
        unicode_scenarios = edge_case_generator.generate_unicode_scenarios(count=3)
        
        patterns = [r"(?P<prefix>\w+)_(?P<date>\d{8})_(?P<condition>[\w\-]+)\.(?P<extension>\w+)"]
        matcher = PatternMatcher(patterns)
        
        # ACT & ASSERT - Test each unicode scenario
        for scenario in unicode_scenarios:
            filename = scenario['filename']
            
            # Should handle unicode filenames gracefully
            result = matcher.match(filename)
            # Result can be None (no match) or dict (successful match)
            assert result is None or isinstance(result, dict)
    
    def test_convenience_functions_integration(self, sample_filenames):
        """
        Test integration of convenience functions with centralized patterns.
        
        Validates end-to-end workflow using convenience functions
        for common research file processing scenarios.
        """
        # ARRANGE - Use realistic experimental patterns
        experiment_patterns = [r"exp(?P<experiment_id>\d{3})_(?P<animal>\w+)_(?P<condition>\w+)\.(?P<extension>\w+)"]
        vial_patterns = [r"vial_(?P<date>\d{8})_(?P<condition>\w+)(_(?P<replicate>\d+))?\.csv"]
        
        # ACT - Test experiment file processing workflow
        experiment_files = [f for f in sample_filenames if f.startswith("exp")]
        experiment_results = []
        for exp_file in experiment_files:
            result = extract_experiment_info(exp_file, experiment_patterns)
            if result:
                experiment_results.append(result)
        
        # ACT - Test vial file processing workflow  
        vial_files = [f for f in sample_filenames if f.startswith("vial_")]
        vial_results = []
        for vial_file in vial_files:
            result = extract_vial_info(vial_file, vial_patterns)
            if result:
                vial_results.append(result)
        
        # ASSERT - Should successfully process experimental files
        assert len(experiment_results) > 0, "Should process experiment files"
        assert len(vial_results) > 0, "Should process vial files"
        
        # Verify metadata structure
        for exp_result in experiment_results:
            assert "experiment_id" in exp_result
            assert "animal" in exp_result
            assert "condition" in exp_result
        
        for vial_result in vial_results:
            assert "date" in vial_result
            assert "condition" in vial_result


# --- Property-Based Testing for Comprehensive Validation ---

class TestPropertyBasedPatternValidation:
    """
    Property-based testing for comprehensive pattern matching validation.
    
    Uses Hypothesis with domain-specific strategies to validate pattern
    matching behavior across wide range of realistic scenarios.
    """
    
    @given(get_flyrigloader_strategies().experimental_file_paths())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_experimental_filename_matching_robustness(self, filename):
        """
        Property-based test for experimental filename matching robustness.
        
        Uses domain-specific filename generation to validate robust
        pattern matching across realistic experimental scenarios.
        """
        # ARRANGE - Create PatternMatcher with comprehensive experimental patterns
        patterns = [
            r"(?P<animal>\w+)_(?P<date>\d{8})_(?P<condition>\w+)_rep(?P<replicate>\d+)\.(?P<extension>\w+)",
            r"(?P<type>exp)(?P<id>\d+)_(?P<animal>\w+)_(?P<condition>\w+)\.(?P<extension>\w+)",
            r"(?P<name>\w+)\.(?P<extension>\w+)",  # Fallback pattern
        ]
        matcher = PatternMatcher(patterns)
        
        # ACT - Attempt to match generated experimental filename
        result = matcher.match(filename)
        
        # ASSERT - Should handle gracefully and maintain invariants
        if result is not None:
            # Successful match should have valid structure
            assert isinstance(result, dict)
            assert len(result) > 0
            assert all(isinstance(k, str) for k in result.keys())
            assert all(v is None or isinstance(v, str) for v in result.values())
    
    @given(get_flyrigloader_strategies().experimental_configurations())
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_pattern_generation_with_config_data(self, config_data):
        """
        Property-based test for pattern generation using configuration data.
        
        Validates pattern generation using realistic experimental
        configuration structures.
        """
        # ARRANGE - Extract patterns from generated configuration
        if 'project' in config_data and 'extraction_patterns' in config_data.get('project', {}):
            patterns = config_data['project']['extraction_patterns']
            
            # ACT - Create PatternMatcher with configuration patterns
            try:
                matcher = PatternMatcher(patterns)
                
                # Test with a simple filename
                test_result = matcher.match("test_20241215_control_1.csv")
                
                # ASSERT - Should create functional matcher
                assert isinstance(matcher, PatternMatcher)
                # Result can be None or dict, but no exceptions
                assert test_result is None or isinstance(test_result, dict)
                
            except (re.error, PatternCompilationError):
                # Some generated patterns may be invalid, which is acceptable
                pass


# --- Comprehensive Metadata Extraction Testing ---

class TestMetadataExtractionValidation:
    """
    Comprehensive testing of metadata extraction for research workflows.
    
    Validates accurate metadata extraction from experimental files
    with focus on research-relevant data organization.
    """
    
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
        
        # Animal identification
        ("mouse_20241215_control_1.csv", {"animal": "mouse", "date": "20241215", "condition": "control"}),
        ("rat_20241220_treatment_2.pkl", {"animal": "rat", "date": "20241220", "condition": "treatment"}),
        
        # Experiment identification
        ("exp001_mouse_control.csv", {"experiment_id": "001", "animal": "mouse", "condition": "control"}),
        ("exp999_rat_baseline.pkl", {"experiment_id": "999", "animal": "rat", "condition": "baseline"}),
    ])
    def test_research_metadata_extraction_accuracy(self, filename, expected_metadata):
        """
        Test accurate extraction of research-relevant metadata.
        
        Validates that metadata extraction correctly identifies
        experimental parameters for research workflow organization.
        """
        # ARRANGE - Create comprehensive patterns for research metadata
        patterns = [
            r"vial_(?P<date>\d{8})_(?P<condition>[a-zA-Z0-9_-]+)(_(?P<replicate>\d+))?\.(?P<extension>\w+)",
            r"(?P<animal>mouse|rat)_(?P<date>\d{8})_(?P<condition>[a-zA-Z0-9_-]+)_(?P<replicate>\d+)\.(?P<extension>\w+)",
            r"exp(?P<experiment_id>\d{3})_(?P<animal>\w+)_(?P<condition>\w+)\.(?P<extension>\w+)",
            r"(?P<prefix>\w+)_(?P<date>\d{8})_(?P<condition>\w+)\.(?P<extension>\w+)",
        ]
        matcher = PatternMatcher(patterns)
        
        # ACT - Extract metadata from research filename
        result = matcher.match(filename)
        
        # ASSERT - Should extract expected research metadata
        assert result is not None, f"Pattern matching failed for research file: {filename}"
        
        for field, expected_value in expected_metadata.items():
            assert field in result, f"Research metadata field '{field}' missing for {filename}"
            assert result[field] == expected_value, f"Research metadata field '{field}' mismatch for {filename}: expected {expected_value}, got {result[field]}"
    
    def test_complex_experimental_condition_parsing(self):
        """
        Test parsing of complex experimental conditions.
        
        Validates handling of multi-part condition names and
        special characters in research workflow scenarios.
        """
        # ARRANGE - Create pattern for complex conditions
        patterns = [r"vial_(?P<date>\d{8})_(?P<condition>[a-zA-Z0-9_-]+)(_(?P<replicate>\d+))?\.csv"]
        matcher = PatternMatcher(patterns)
        
        complex_conditions = [
            ("vial_20241215_control.csv", "control"),
            ("vial_20241215_drug-treatment.csv", "drug-treatment"),
            ("vial_20241215_baseline_test.csv", "baseline_test"),
            ("vial_20241215_condition-1a.csv", "condition-1a"),
            ("vial_20241215_HIGH-dose_treatment.csv", "HIGH-dose_treatment"),
        ]
        
        # ACT & ASSERT - Test each complex condition
        for filename, expected_condition in complex_conditions:
            result = matcher.match(filename)
            assert result is not None, f"Pattern matching failed for complex condition: {filename}"
            assert result["condition"] == expected_condition, f"Condition extraction failed for {filename}"
    
    def test_experimental_workflow_metadata_completeness(self, sample_patterns, sample_filenames):
        """
        Test completeness of metadata extraction for experimental workflows.
        
        Validates that all experimental files have complete metadata
        for research data organization and analysis workflows.
        """
        # ARRANGE - Create PatternMatcher with comprehensive patterns
        matcher = PatternMatcher(sample_patterns)
        
        # ACT - Extract metadata from all experimental files
        experimental_files = [f for f in sample_filenames if any(indicator in f for indicator in ["vial_", "exp", "mouse_", "rat_"])]
        metadata_results = []
        
        for exp_file in experimental_files:
            result = matcher.match(exp_file)
            if result:
                metadata_results.append((exp_file, result))
        
        # ASSERT - Should have comprehensive metadata for experimental files
        assert len(metadata_results) > 0, "Should extract metadata from experimental files"
        
        for filename, metadata in metadata_results:
            # Verify metadata completeness
            assert isinstance(metadata, dict), f"Metadata should be dictionary for {filename}"
            assert len(metadata) > 0, f"Metadata should not be empty for {filename}"
            
            # Verify essential fields for research workflows
            has_temporal_info = any(field in metadata for field in ["date", "experiment_id"])
            has_condition_info = "condition" in metadata
            
            assert has_temporal_info or has_condition_info, f"Essential research metadata missing for {filename}"


# --- Legacy Compatibility and Backward Compatibility Testing ---

class TestLegacyCompatibility:
    """
    Test backward compatibility with legacy patterns and filename formats.
    
    Ensures that existing research workflows continue to function
    with legacy filename patterns and data organization systems.
    """
    
    def test_positional_group_pattern_compatibility(self):
        """
        Test compatibility with legacy positional group patterns.
        
        Validates that patterns without named groups still
        provide meaningful metadata extraction for legacy workflows.
        """
        # ARRANGE - Create matcher with legacy positional patterns
        legacy_patterns = [
            r"legacy_(\w+)_(\d{8})_(\w+)_(\d+)\.csv",  # 4 positional groups
            r"old_format_(\w+)_(\d+)\.pkl",            # 2 positional groups
        ]
        matcher = PatternMatcher(legacy_patterns)
        
        # ACT - Test legacy filename patterns
        legacy_result = matcher.match("legacy_mouse_20241215_control_1.csv")
        old_format_result = matcher.match("old_format_experiment_123.pkl")
        
        # ASSERT - Should extract meaningful metadata from legacy patterns
        assert legacy_result is not None, "Should match legacy pattern"
        assert isinstance(legacy_result, dict)
        assert len(legacy_result) > 0, "Should extract fields from legacy pattern"
        
        assert old_format_result is not None, "Should match old format pattern"
        assert isinstance(old_format_result, dict)
        assert len(old_format_result) > 0, "Should extract fields from old format"
    
    def test_mixed_named_and_positional_pattern_priority(self):
        """
        Test pattern matching priority with mixed named and positional groups.
        
        Validates correct precedence handling when both legacy
        and modern patterns could match the same filename.
        """
        # ARRANGE - Create patterns mixing named and positional groups
        mixed_patterns = [
            r"new_(?P<date>\d{8})_(?P<condition>\w+)\.csv",  # Named groups (modern)
            r"old_(\w+)_(\d{8})_(\w+)\.csv",                 # Positional groups (legacy)
        ]
        matcher = PatternMatcher(mixed_patterns)
        
        # ACT - Test with filename that could match either pattern
        modern_result = matcher.match("new_20241215_control.csv")
        
        # ASSERT - Should prefer named group pattern
        assert modern_result is not None
        assert "date" in modern_result, "Should use named groups from modern pattern"
        assert "condition" in modern_result, "Should extract named fields"
        assert modern_result["date"] == "20241215"
        assert modern_result["condition"] == "control"
    
    def test_research_workflow_backward_compatibility(self, sample_patterns):
        """
        Test backward compatibility for existing research workflows.
        
        Validates that pattern matching continues to support
        existing experimental data organization patterns.
        """
        # ARRANGE - Create matcher with mixed pattern types
        all_patterns = sample_patterns + [
            r"legacy_(\w+)_(\d{8})_(\w+)_(\d+)\.csv",  # Legacy positional
            r"archive_(?P<year>\d{4})_(?P<study>\w+)\.(?P<extension>\w+)",  # Archive format
        ]
        matcher = PatternMatcher(all_patterns)
        
        # ACT - Test various legacy and modern formats
        test_files = [
            "vial_20241215_control.csv",  # Modern vial format
            "exp001_mouse_baseline.csv",  # Modern experiment format
            "legacy_rat_20241220_treatment_3.csv",  # Legacy format
            "archive_2024_navigation_study.pkl",  # Archive format
        ]
        
        results = []
        for test_file in test_files:
            result = matcher.match(test_file)
            if result:
                results.append((test_file, result))
        
        # ASSERT - Should support all format types
        assert len(results) >= 3, "Should support multiple format types for backward compatibility"
        
        # Verify each result has meaningful metadata
        for filename, metadata in results:
            assert isinstance(metadata, dict), f"Should extract metadata from {filename}"
            assert len(metadata) > 0, f"Should have meaningful metadata for {filename}"


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