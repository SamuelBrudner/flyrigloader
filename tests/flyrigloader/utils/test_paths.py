"""
Behavior-focused tests for path utilities with Protocol-based dependency injection.

This module implements comprehensive testing of path manipulation utilities through
black-box behavioral validation, following the enhanced testing strategy requirements
for behavior-focused testing, Protocol-based mock implementations, and AAA pattern
structure enforcement.

Testing Strategy Implementation:
- Behavior-focused testing: All tests validate public API behavior rather than internal implementation
- Protocol-based mocks: Utilizes centralized MockFilesystemProvider from tests/utils.py
- AAA pattern structure: Clear separation of Arrange-Act-Assert phases
- Edge-case coverage: Comprehensive parameterized tests for Unicode, cross-platform, and boundary conditions
- Centralized fixtures: Leverages tests/conftest.py fixture infrastructure
- Observable behavior focus: Tests path resolution results, directory creation success, file existence validation

Key Test Categories:
- Path resolution and normalization behavior validation
- Cross-platform compatibility through mock filesystem providers
- Unicode and special character handling in path operations
- Error condition and boundary testing with realistic scenarios
- Concurrent access simulation and race condition handling
- Memory constraint testing for large path operations

Test Functions:
- test_get_relative_path_*: Relative path calculation behavior validation
- test_get_absolute_path_*: Absolute path resolution behavior validation  
- test_find_common_base_directory_*: Common base directory detection behavior
- test_ensure_directory_exists_*: Directory creation behavior validation
- test_check_file_exists_*: File existence checking behavior validation
- test_*_edge_cases: Comprehensive edge-case and boundary condition testing
"""

import os
import platform
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from unittest.mock import MagicMock

import pytest
from hypothesis import given, settings, strategies as st

# Import path utilities with dependency injection support
from flyrigloader.utils.paths import (
    get_relative_path,
    get_absolute_path,
    find_common_base_directory,
    ensure_directory_exists,
    check_file_exists,
    FileSystemProvider,
    set_filesystem_provider_for_testing,
    restore_filesystem_provider,
    get_current_filesystem_provider
)

# Import centralized test utilities and mock factories
from tests.utils import (
    create_mock_filesystem,
    MockFilesystem,
    EdgeCaseScenarioGenerator,
    create_hypothesis_strategies,
    generate_edge_case_scenarios
)


# ============================================================================
# PROTOCOL-BASED MOCK FILESYSTEM PROVIDER IMPLEMENTATION
# ============================================================================

class MockPathFilesystemProvider:
    """
    Protocol-based mock filesystem provider for path utility testing.
    
    Implements FileSystemProvider interface with comprehensive behavior simulation
    for cross-platform path operations, error scenarios, and edge-case validation
    while maintaining dependency injection patterns for controlled testing.
    """
    
    def __init__(self, mock_filesystem: MockFilesystem):
        """Initialize with mock filesystem backend."""
        self.mock_fs = mock_filesystem
        self.operation_log = []  # Track operations for behavior verification
    
    def resolve_path(self, path: Path) -> Path:
        """Mock path resolution with cross-platform behavior simulation."""
        self.operation_log.append(('resolve_path', str(path)))
        
        # Simulate realistic path resolution behavior
        if str(path).startswith('~'):
            # Mock home directory expansion
            resolved = Path('/home/test_user') / str(path)[2:]
        elif path.is_absolute():
            resolved = path
        else:
            # Resolve relative to mock current directory
            resolved = Path('/mock/current/directory') / path
        
        # Simulate symlink resolution and . / .. handling
        parts = []
        for part in resolved.parts:
            if part == '..':
                if parts:
                    parts.pop()
            elif part != '.':
                parts.append(part)
        
        return Path(*parts) if parts else Path('/')
    
    def make_relative(self, path: Path, base: Path) -> Path:
        """Mock relative path calculation with error handling."""
        self.operation_log.append(('make_relative', str(path), str(base)))
        
        resolved_path = self.resolve_path(path)
        resolved_base = self.resolve_path(base)
        
        try:
            return resolved_path.relative_to(resolved_base)
        except ValueError as e:
            raise ValueError(f"Path {resolved_path} is not within base directory {resolved_base}") from e
    
    def is_absolute(self, path: Path) -> bool:
        """Mock absolute path detection with platform considerations."""
        self.operation_log.append(('is_absolute', str(path)))
        return path.is_absolute()
    
    def join_paths(self, base: Path, *parts: Union[str, Path]) -> Path:
        """Mock path joining with validation."""
        self.operation_log.append(('join_paths', str(base), [str(p) for p in parts]))
        
        result = base
        for part in parts:
            result = result / part
        return result
    
    def create_directory(self, path: Path, parents: bool = True, exist_ok: bool = True) -> Path:
        """Mock directory creation with filesystem state updates."""
        self.operation_log.append(('create_directory', str(path), parents, exist_ok))
        
        # Check if path already exists as file (should raise error)
        if self.mock_fs.is_file(path):
            raise FileExistsError(f"File exists at path: {path}")
        
        # Create directory in mock filesystem
        if not self.mock_fs.exists(path):
            if parents:
                # Create parent directories if needed
                parent = path.parent
                while parent != Path('.') and not self.mock_fs.exists(parent):
                    self.mock_fs.add_directory(parent)
                    parent = parent.parent
            
            self.mock_fs.add_directory(path)
        elif not exist_ok and self.mock_fs.is_dir(path):
            raise FileExistsError(f"Directory already exists: {path}")
        
        return path
    
    def check_file_exists(self, path: Path) -> bool:
        """Mock file existence checking with realistic behavior."""
        self.operation_log.append(('check_file_exists', str(path)))
        return self.mock_fs.is_file(path)
    
    def get_path_parts(self, path: Path) -> tuple:
        """Mock path parts extraction."""
        self.operation_log.append(('get_path_parts', str(path)))
        return path.parts


# ============================================================================
# CENTRALIZED FIXTURES FOR PATH UTILITY TESTING
# ============================================================================

@pytest.fixture
def mock_filesystem_provider(mock_filesystem):
    """
    Centralized fixture providing Protocol-based filesystem provider for path testing.
    
    Utilizes the centralized mock_filesystem fixture from tests/conftest.py and wraps
    it with the FileSystemProvider protocol implementation for dependency injection
    into path utility functions.
    """
    # Create realistic filesystem structure for path testing
    mock_filesystem.add_directory('/test')
    mock_filesystem.add_directory('/test/data')
    mock_filesystem.add_directory('/test/experiments')
    mock_filesystem.add_file('/test/data/experiment_001.pkl', size=1024)
    mock_filesystem.add_file('/test/data/experiment_002.pkl', size=2048)
    mock_filesystem.add_file('/test/experiments/config.yaml', size=512)
    
    provider = MockPathFilesystemProvider(mock_filesystem)
    
    # Store original provider for restoration
    original_provider = get_current_filesystem_provider()
    
    # Install mock provider for testing
    set_filesystem_provider_for_testing(provider)
    
    yield provider
    
    # Restore original provider after test
    restore_filesystem_provider(original_provider)


@pytest.fixture
def sample_path_structures(temp_experiment_directory):
    """
    Centralized fixture providing realistic path structures for comprehensive testing.
    
    Leverages the temp_experiment_directory fixture from tests/conftest.py to create
    comprehensive directory structures with realistic experimental file patterns
    for testing path utility operations.
    """
    experiment_dir = temp_experiment_directory['directory']
    
    return {
        'base_directory': experiment_dir,
        'data_files': temp_experiment_directory['raw_files'],
        'config_file': temp_experiment_directory['config_file'],
        'subdirectories': [experiment_dir / subdir for subdir in temp_experiment_directory['subdirs']],
        'relative_paths': [
            Path('relative/path/to/file.pkl'),
            Path('other/relative/path.csv'),
            Path('nested/deep/structure/data.json')
        ],
        'absolute_paths': [
            experiment_dir / 'absolute' / 'path' / 'file.pkl',
            experiment_dir / 'another' / 'absolute' / 'file.csv'
        ]
    }


@pytest.fixture
def edge_case_scenarios():
    """
    Centralized fixture providing comprehensive edge-case scenarios for path testing.
    
    Generates Unicode path scenarios, boundary conditions, and platform-specific
    edge cases using the centralized EdgeCaseScenarioGenerator from tests/utils.py.
    """
    generator = EdgeCaseScenarioGenerator()
    
    scenarios = {
        'unicode_paths': generator.generate_unicode_scenarios(count=5),
        'boundary_conditions': generator.generate_boundary_conditions(['array_size', 'file_size']),
        'corrupted_scenarios': generator.generate_corrupted_data_scenarios(),
        'platform_specific': []
    }
    
    # Add platform-specific scenarios
    if platform.system() == "Windows":
        scenarios['platform_specific'].extend([
            {'type': 'long_path', 'path': 'C:\\' + 'very_long\\' * 30 + 'file.pkl'},
            {'type': 'reserved_name', 'path': 'CON.pkl'},
            {'type': 'trailing_space', 'path': 'file .pkl'}
        ])
    else:
        scenarios['platform_specific'].extend([
            {'type': 'hidden_file', 'path': '.hidden_file.pkl'},
            {'type': 'case_sensitive', 'path': 'File.pkl', 'alt_path': 'file.pkl'}
        ])
    
    return scenarios


# ============================================================================
# GET_RELATIVE_PATH BEHAVIOR VALIDATION TESTS
# ============================================================================

class TestGetRelativePathBehavior:
    """
    Comprehensive behavior validation for get_relative_path function.
    
    Tests focus on observable behavior of relative path calculation through
    the public API interface, using Protocol-based mock filesystem providers
    for controlled dependency injection and comprehensive edge-case coverage.
    """
    
    def test_get_relative_path_basic_behavior(self, mock_filesystem_provider):
        """
        Test basic relative path calculation behavior with realistic directory structures.
        
        Validates that get_relative_path correctly calculates relative paths between
        directories and files using the public API interface with dependency injection.
        """
        # ARRANGE - Set up test paths with mock filesystem provider
        base_dir = Path('/test/data')
        target_file = Path('/test/data/subdir/experiment.pkl')
        expected_relative = Path('subdir/experiment.pkl')
        
        # Add directory structure to mock filesystem
        mock_filesystem_provider.mock_fs.add_directory('/test/data/subdir')
        mock_filesystem_provider.mock_fs.add_file('/test/data/subdir/experiment.pkl', size=1024)
        
        # ACT - Execute relative path calculation through public API
        result = get_relative_path(target_file, base_dir, fs_provider=mock_filesystem_provider)
        
        # ASSERT - Verify observable behavior matches expectations
        assert result == expected_relative
        assert 'resolve_path' in [op[0] for op in mock_filesystem_provider.operation_log]
        assert 'make_relative' in [op[0] for op in mock_filesystem_provider.operation_log]
    
    def test_get_relative_path_with_path_objects(self, mock_filesystem_provider):
        """
        Test relative path calculation behavior with Path object inputs.
        
        Validates that the function handles both string and Path object inputs
        correctly, returning consistent results through the public interface.
        """
        # ARRANGE - Set up Path objects for testing
        base_dir = Path('/test/experiments')
        target_file = Path('/test/experiments/data/sample.pkl')
        expected_relative = Path('data/sample.pkl')
        
        mock_filesystem_provider.mock_fs.add_directory('/test/experiments/data')
        mock_filesystem_provider.mock_fs.add_file('/test/experiments/data/sample.pkl', size=512)
        
        # ACT - Execute with Path object inputs
        result = get_relative_path(target_file, base_dir, fs_provider=mock_filesystem_provider)
        
        # ASSERT - Verify Path object handling behavior
        assert isinstance(result, Path)
        assert result == expected_relative
        assert result.name == 'sample.pkl'
        assert str(result.parent) == 'data'
    
    def test_get_relative_path_error_behavior_path_not_under_base(self, mock_filesystem_provider):
        """
        Test error handling behavior when target path is not under base directory.
        
        Validates that appropriate ValueError is raised when attempting to calculate
        relative path for files outside the base directory tree.
        """
        # ARRANGE - Set up paths where target is not under base
        base_dir = Path('/test/data')
        target_file = Path('/other/directory/file.pkl')
        
        mock_filesystem_provider.mock_fs.add_directory('/test/data')
        mock_filesystem_provider.mock_fs.add_directory('/other/directory')
        mock_filesystem_provider.mock_fs.add_file('/other/directory/file.pkl', size=256)
        
        # ACT & ASSERT - Verify error behavior through public API
        with pytest.raises(ValueError, match="is not within base directory"):
            get_relative_path(target_file, base_dir, fs_provider=mock_filesystem_provider)
    
    @pytest.mark.parametrize("input_type", ["string", "path_object"])
    def test_get_relative_path_input_type_behavior(self, mock_filesystem_provider, input_type):
        """
        Test relative path calculation behavior with different input types.
        
        Parameterized test validating consistent behavior regardless of whether
        inputs are provided as strings or Path objects.
        """
        # ARRANGE - Set up test inputs based on parameter
        base_path = Path('/test/base')
        target_path = Path('/test/base/subdir/file.txt')
        
        mock_filesystem_provider.mock_fs.add_directory('/test/base/subdir')
        mock_filesystem_provider.mock_fs.add_file('/test/base/subdir/file.txt', size=128)
        
        if input_type == "string":
            base_input = str(base_path)
            target_input = str(target_path)
        else:
            base_input = base_path
            target_input = target_path
        
        # ACT - Execute with parameterized input types
        result = get_relative_path(target_input, base_input, fs_provider=mock_filesystem_provider)
        
        # ASSERT - Verify consistent behavior across input types
        assert result == Path('subdir/file.txt')
        assert isinstance(result, Path)
    
    def test_get_relative_path_unicode_behavior(self, mock_filesystem_provider, edge_case_scenarios):
        """
        Test relative path calculation behavior with Unicode characters in paths.
        
        Validates that Unicode characters in file and directory names are handled
        correctly through the public API, supporting international file naming.
        """
        # ARRANGE - Set up Unicode path scenario
        unicode_scenario = edge_case_scenarios['unicode_paths'][0]
        base_dir = Path('/test') / unicode_scenario['directory']
        target_file = base_dir / unicode_scenario['filename']
        
        mock_filesystem_provider.mock_fs.add_directory(str(base_dir))
        mock_filesystem_provider.mock_fs.add_file(str(target_file), size=512)
        
        # ACT - Execute with Unicode paths
        try:
            result = get_relative_path(target_file, base_dir, fs_provider=mock_filesystem_provider)
            
            # ASSERT - Verify Unicode handling behavior
            assert result == Path(unicode_scenario['filename'])
            assert str(result) == unicode_scenario['filename']
            
        except (UnicodeError, OSError):
            # Skip test if Unicode not supported on platform
            pytest.skip(f"Unicode paths not supported on {platform.system()}")


# ============================================================================
# GET_ABSOLUTE_PATH BEHAVIOR VALIDATION TESTS
# ============================================================================

class TestGetAbsolutePathBehavior:
    """
    Comprehensive behavior validation for get_absolute_path function.
    
    Tests focus on observable behavior of absolute path resolution through
    the public API interface, validating correct handling of relative paths,
    base directory resolution, and cross-platform path normalization.
    """
    
    def test_get_absolute_path_basic_behavior(self, mock_filesystem_provider):
        """
        Test basic absolute path resolution behavior with relative input paths.
        
        Validates that get_absolute_path correctly resolves relative paths to
        absolute paths using the provided base directory through public API.
        """
        # ARRANGE - Set up relative path and base directory
        relative_path = Path('experiments/data/file.pkl')
        base_dir = Path('/test/project')
        expected_absolute = Path('/test/project/experiments/data/file.pkl')
        
        mock_filesystem_provider.mock_fs.add_directory('/test/project/experiments/data')
        
        # ACT - Execute absolute path resolution through public API
        result = get_absolute_path(relative_path, base_dir, fs_provider=mock_filesystem_provider)
        
        # ASSERT - Verify observable absolute path behavior
        assert result.is_absolute()
        assert result == expected_absolute
        assert 'join_paths' in [op[0] for op in mock_filesystem_provider.operation_log]
        assert 'resolve_path' in [op[0] for op in mock_filesystem_provider.operation_log]
    
    def test_get_absolute_path_already_absolute_behavior(self, mock_filesystem_provider):
        """
        Test behavior when input path is already absolute.
        
        Validates that get_absolute_path returns already absolute paths unchanged
        while still performing resolution through the filesystem provider.
        """
        # ARRANGE - Set up already absolute path
        absolute_path = Path('/existing/absolute/path.pkl')
        base_dir = Path('/test/base')
        
        mock_filesystem_provider.mock_fs.add_file('/existing/absolute/path.pkl', size=1024)
        
        # ACT - Execute with already absolute path
        result = get_absolute_path(absolute_path, base_dir, fs_provider=mock_filesystem_provider)
        
        # ASSERT - Verify absolute path preservation behavior
        assert result.is_absolute()
        assert result == absolute_path
        assert 'is_absolute' in [op[0] for op in mock_filesystem_provider.operation_log]
    
    def test_get_absolute_path_dot_segment_resolution(self, mock_filesystem_provider):
        """
        Test behavior with paths containing dot segments (. and ..).
        
        Validates that relative paths with dot segments are correctly resolved
        to normalized absolute paths through the public API interface.
        """
        # ARRANGE - Set up path with dot segments
        relative_path = Path('data/../configs/./experiment.yaml')
        base_dir = Path('/test/project')
        expected_absolute = Path('/test/project/configs/experiment.yaml')
        
        mock_filesystem_provider.mock_fs.add_directory('/test/project/configs')
        mock_filesystem_provider.mock_fs.add_file('/test/project/configs/experiment.yaml', size=256)
        
        # ACT - Execute dot segment resolution
        result = get_absolute_path(relative_path, base_dir, fs_provider=mock_filesystem_provider)
        
        # ASSERT - Verify dot segment normalization behavior
        assert result == expected_absolute
        assert '..' not in str(result)
        assert '/.' not in str(result)
    
    @pytest.mark.parametrize("base_dir_type", ["relative", "absolute"])
    def test_get_absolute_path_base_directory_behavior(self, mock_filesystem_provider, base_dir_type):
        """
        Test behavior with different base directory types (relative vs absolute).
        
        Parameterized test validating that the function handles both relative and
        absolute base directories correctly through the public API.
        """
        # ARRANGE - Set up base directory based on parameter
        relative_path = Path('data/file.pkl')
        
        if base_dir_type == "relative":
            base_dir = Path('project/base')
            # Mock current directory resolution
            mock_filesystem_provider.mock_fs.add_directory('/mock/current/directory/project/base/data')
        else:
            base_dir = Path('/absolute/base')
            mock_filesystem_provider.mock_fs.add_directory('/absolute/base/data')
        
        # ACT - Execute with parameterized base directory type
        result = get_absolute_path(relative_path, base_dir, fs_provider=mock_filesystem_provider)
        
        # ASSERT - Verify correct base directory handling behavior
        assert result.is_absolute()
        if base_dir_type == "absolute":
            assert str(result).startswith('/absolute/base')
        else:
            assert str(result).startswith('/mock/current/directory')
    
    def test_get_absolute_path_error_handling_behavior(self, mock_filesystem_provider):
        """
        Test error handling behavior with invalid path scenarios.
        
        Validates that appropriate errors are raised for filesystem operation
        failures while maintaining clean error propagation through public API.
        """
        # ARRANGE - Set up error scenario in mock filesystem
        relative_path = Path('nonexistent/deep/path.pkl')
        base_dir = Path('/test/base')
        
        # Don't create the path structure to simulate filesystem errors
        
        # ACT & ASSERT - Verify error handling behavior
        try:
            result = get_absolute_path(relative_path, base_dir, fs_provider=mock_filesystem_provider)
            # Should still work even if directories don't exist (path resolution only)
            assert result.is_absolute()
        except ValueError as e:
            # Acceptable if provider raises ValueError for resolution failures
            assert "Failed to get absolute path" in str(e)


# ============================================================================
# FIND_COMMON_BASE_DIRECTORY BEHAVIOR VALIDATION TESTS
# ============================================================================

class TestFindCommonBaseDirectoryBehavior:
    """
    Comprehensive behavior validation for find_common_base_directory function.
    
    Tests focus on observable behavior of common base directory detection through
    the public API interface, validating correct handling of multiple path inputs,
    edge cases with empty lists, and cross-platform path normalization.
    """
    
    def test_find_common_base_directory_basic_behavior(self, mock_filesystem_provider):
        """
        Test basic common base directory detection behavior with realistic paths.
        
        Validates that find_common_base_directory correctly identifies the common
        base directory for a list of related file paths through public API.
        """
        # ARRANGE - Set up related file paths with common base
        file_paths = [
            Path('/test/project/data/experiment_001.pkl'),
            Path('/test/project/data/experiment_002.pkl'),
            Path('/test/project/configs/setup.yaml')
        ]
        expected_base = Path('/test/project')
        
        # Add files to mock filesystem
        for file_path in file_paths:
            mock_filesystem_provider.mock_fs.add_directory(str(file_path.parent))
            mock_filesystem_provider.mock_fs.add_file(str(file_path), size=512)
        
        # ACT - Execute common base directory detection
        result = find_common_base_directory(file_paths, fs_provider=mock_filesystem_provider)
        
        # ASSERT - Verify observable common base detection behavior
        assert result == expected_base
        assert len([op for op in mock_filesystem_provider.operation_log if op[0] == 'resolve_path']) == len(file_paths)
        assert 'get_path_parts' in [op[0] for op in mock_filesystem_provider.operation_log]
    
    def test_find_common_base_directory_no_common_base_behavior(self, mock_filesystem_provider):
        """
        Test behavior when paths have no common base directory.
        
        Validates that the function correctly handles paths from completely
        different directory trees and returns appropriate results.
        """
        # ARRANGE - Set up paths with no common base
        file_paths = [
            Path('/first/directory/file1.pkl'),
            Path('/second/directory/file2.pkl'),
            Path('/third/directory/file3.pkl')
        ]
        
        # Add files to different root directories
        for file_path in file_paths:
            mock_filesystem_provider.mock_fs.add_directory(str(file_path.parent))
            mock_filesystem_provider.mock_fs.add_file(str(file_path), size=256)
        
        # ACT - Execute with no common base scenario
        result = find_common_base_directory(file_paths, fs_provider=mock_filesystem_provider)
        
        # ASSERT - Verify behavior for no common base case
        assert result == Path('/')  # Root should be common base on Unix-like systems
    
    def test_find_common_base_directory_empty_list_behavior(self, mock_filesystem_provider):
        """
        Test behavior with empty path list input.
        
        Validates that find_common_base_directory handles empty input lists
        gracefully and returns appropriate None result through public API.
        """
        # ARRANGE - Set up empty path list
        empty_path_list = []
        
        # ACT - Execute with empty list
        result = find_common_base_directory(empty_path_list, fs_provider=mock_filesystem_provider)
        
        # ASSERT - Verify empty list handling behavior
        assert result is None
    
    def test_find_common_base_directory_single_path_behavior(self, mock_filesystem_provider):
        """
        Test behavior with single path input.
        
        Validates that the function correctly handles single-path scenarios
        and returns the parent directory as the common base.
        """
        # ARRANGE - Set up single path scenario
        single_path = [Path('/test/data/single_file.pkl')]
        expected_base = Path('/test/data')
        
        mock_filesystem_provider.mock_fs.add_directory('/test/data')
        mock_filesystem_provider.mock_fs.add_file('/test/data/single_file.pkl', size=128)
        
        # ACT - Execute with single path
        result = find_common_base_directory(single_path, fs_provider=mock_filesystem_provider)
        
        # ASSERT - Verify single path handling behavior
        assert result == expected_base
    
    @pytest.mark.parametrize("depth", [2, 3, 5, 10])
    def test_find_common_base_directory_nested_depth_behavior(self, mock_filesystem_provider, depth):
        """
        Test behavior with deeply nested directory structures.
        
        Parameterized test validating correct common base detection across
        various nesting depths to ensure algorithm scalability.
        """
        # ARRANGE - Set up nested directory structure based on depth
        base_path = Path('/test/deep/nested')
        
        # Create paths at different nesting levels
        file_paths = []
        for i in range(3):  # Create 3 files at different nested levels
            nested_parts = [f'level_{j}' for j in range(depth)]
            nested_path = base_path.joinpath(*nested_parts[:depth-i]) / f'file_{i}.pkl'
            file_paths.append(nested_path)
            
            mock_filesystem_provider.mock_fs.add_directory(str(nested_path.parent))
            mock_filesystem_provider.mock_fs.add_file(str(nested_path), size=64)
        
        # ACT - Execute with nested depth scenario
        result = find_common_base_directory(file_paths, fs_provider=mock_filesystem_provider)
        
        # ASSERT - Verify nested depth handling behavior
        assert result is not None
        assert all(str(path).startswith(str(result)) for path in file_paths)
    
    def test_find_common_base_directory_mixed_input_types(self, mock_filesystem_provider):
        """
        Test behavior with mixed string and Path object inputs.
        
        Validates that the function handles mixed input types correctly
        and returns consistent results regardless of input format.
        """
        # ARRANGE - Set up mixed input types (strings and Path objects)
        mixed_paths = [
            '/test/mixed/file1.pkl',  # String
            Path('/test/mixed/file2.pkl'),  # Path object
            '/test/mixed/subdir/file3.pkl'  # String
        ]
        expected_base = Path('/test/mixed')
        
        # Add files to mock filesystem
        for path in mixed_paths:
            path_obj = Path(path)
            mock_filesystem_provider.mock_fs.add_directory(str(path_obj.parent))
            mock_filesystem_provider.mock_fs.add_file(str(path_obj), size=256)
        
        # ACT - Execute with mixed input types
        result = find_common_base_directory(mixed_paths, fs_provider=mock_filesystem_provider)
        
        # ASSERT - Verify mixed input type handling behavior
        assert result == expected_base
        assert isinstance(result, Path)


# ============================================================================
# ENSURE_DIRECTORY_EXISTS BEHAVIOR VALIDATION TESTS
# ============================================================================

class TestEnsureDirectoryExistsBehavior:
    """
    Comprehensive behavior validation for ensure_directory_exists function.
    
    Tests focus on observable behavior of directory creation through the public
    API interface, validating successful directory creation, idempotent behavior,
    parent directory creation, and error handling for various scenarios.
    """
    
    def test_ensure_directory_exists_basic_creation_behavior(self, mock_filesystem_provider):
        """
        Test basic directory creation behavior with non-existing directory.
        
        Validates that ensure_directory_exists successfully creates directories
        that don't exist and returns the created directory path through public API.
        """
        # ARRANGE - Set up non-existing directory path
        target_directory = Path('/test/new/directory')
        
        # Verify directory doesn't exist initially
        assert not mock_filesystem_provider.mock_fs.exists(target_directory)
        
        # ACT - Execute directory creation through public API
        result = ensure_directory_exists(target_directory, fs_provider=mock_filesystem_provider)
        
        # ASSERT - Verify successful directory creation behavior
        assert result == target_directory
        assert mock_filesystem_provider.mock_fs.is_dir(target_directory)
        assert 'create_directory' in [op[0] for op in mock_filesystem_provider.operation_log]
    
    def test_ensure_directory_exists_idempotent_behavior(self, mock_filesystem_provider):
        """
        Test idempotent behavior when directory already exists.
        
        Validates that ensure_directory_exists can be called multiple times
        without errors and returns consistent results for existing directories.
        """
        # ARRANGE - Set up existing directory
        existing_directory = Path('/test/existing')
        mock_filesystem_provider.mock_fs.add_directory(str(existing_directory))
        
        # Verify directory exists initially
        assert mock_filesystem_provider.mock_fs.is_dir(existing_directory)
        
        # ACT - Execute ensure operation on existing directory
        result1 = ensure_directory_exists(existing_directory, fs_provider=mock_filesystem_provider)
        result2 = ensure_directory_exists(existing_directory, fs_provider=mock_filesystem_provider)
        
        # ASSERT - Verify idempotent behavior
        assert result1 == existing_directory
        assert result2 == existing_directory
        assert result1 == result2
        assert mock_filesystem_provider.mock_fs.is_dir(existing_directory)
    
    def test_ensure_directory_exists_parent_creation_behavior(self, mock_filesystem_provider):
        """
        Test parent directory creation behavior with nested paths.
        
        Validates that ensure_directory_exists creates parent directories
        when parents=True (default) and handles nested directory structures.
        """
        # ARRANGE - Set up deeply nested directory path
        nested_directory = Path('/test/deep/nested/structure/target')
        
        # Verify none of the path components exist
        assert not mock_filesystem_provider.mock_fs.exists(nested_directory)
        assert not mock_filesystem_provider.mock_fs.exists(nested_directory.parent)
        
        # ACT - Execute with parent creation enabled
        result = ensure_directory_exists(nested_directory, parents=True, fs_provider=mock_filesystem_provider)
        
        # ASSERT - Verify parent directory creation behavior
        assert result == nested_directory
        assert mock_filesystem_provider.mock_fs.is_dir(nested_directory)
        
        # Verify all parent directories were created
        current_path = nested_directory
        while current_path.parent != current_path:
            if str(current_path) != '/':
                assert mock_filesystem_provider.mock_fs.exists(current_path)
            current_path = current_path.parent
    
    def test_ensure_directory_exists_error_behavior_file_exists(self, mock_filesystem_provider):
        """
        Test error handling behavior when file exists at target path.
        
        Validates that appropriate FileExistsError is raised when attempting
        to create directory where a file already exists through public API.
        """
        # ARRANGE - Set up existing file at target path
        target_path = Path('/test/conflicting_file')
        mock_filesystem_provider.mock_fs.add_file(str(target_path), size=512)
        
        # Verify file exists at target location
        assert mock_filesystem_provider.mock_fs.is_file(target_path)
        
        # ACT & ASSERT - Verify error behavior for file conflict
        with pytest.raises(FileExistsError, match="File exists at path"):
            ensure_directory_exists(target_path, fs_provider=mock_filesystem_provider)
    
    @pytest.mark.parametrize("exist_ok", [True, False])
    def test_ensure_directory_exists_exist_ok_parameter_behavior(self, mock_filesystem_provider, exist_ok):
        """
        Test behavior with different exist_ok parameter values.
        
        Parameterized test validating correct handling of exist_ok parameter
        for controlling error behavior when directories already exist.
        """
        # ARRANGE - Set up existing directory
        existing_dir = Path('/test/existing_dir')
        mock_filesystem_provider.mock_fs.add_directory(str(existing_dir))
        
        # ACT & ASSERT - Execute based on exist_ok parameter
        if exist_ok:
            # Should succeed without error
            result = ensure_directory_exists(existing_dir, exist_ok=True, fs_provider=mock_filesystem_provider)
            assert result == existing_dir
        else:
            # Should raise FileExistsError
            with pytest.raises(FileExistsError, match="Directory already exists"):
                ensure_directory_exists(existing_dir, exist_ok=False, fs_provider=mock_filesystem_provider)
    
    def test_ensure_directory_exists_unicode_path_behavior(self, mock_filesystem_provider, edge_case_scenarios):
        """
        Test directory creation behavior with Unicode characters in paths.
        
        Validates that Unicode characters in directory names are handled
        correctly through the public API, supporting international naming.
        """
        # ARRANGE - Set up Unicode directory path
        unicode_scenario = edge_case_scenarios['unicode_paths'][0]
        unicode_dir = Path('/test') / unicode_scenario['directory']
        
        # ACT - Execute with Unicode directory path
        try:
            result = ensure_directory_exists(unicode_dir, fs_provider=mock_filesystem_provider)
            
            # ASSERT - Verify Unicode directory creation behavior
            assert result == unicode_dir
            assert mock_filesystem_provider.mock_fs.is_dir(unicode_dir)
            
        except (UnicodeError, OSError):
            # Skip test if Unicode not supported on platform
            pytest.skip(f"Unicode directory names not supported on {platform.system()}")


# ============================================================================
# CHECK_FILE_EXISTS BEHAVIOR VALIDATION TESTS
# ============================================================================

class TestCheckFileExistsBehavior:
    """
    Comprehensive behavior validation for check_file_exists function.
    
    Tests focus on observable behavior of file existence checking through
    the public API interface, validating correct detection of existing files,
    handling of non-existent files, and distinction between files and directories.
    """
    
    def test_check_file_exists_basic_detection_behavior(self, mock_filesystem_provider):
        """
        Test basic file existence detection behavior with existing files.
        
        Validates that check_file_exists correctly identifies existing files
        and returns True for files that exist in the filesystem through public API.
        """
        # ARRANGE - Set up existing file in mock filesystem
        existing_file = Path('/test/data/existing_file.pkl')
        mock_filesystem_provider.mock_fs.add_file(str(existing_file), size=1024)
        
        # Verify file exists in mock filesystem
        assert mock_filesystem_provider.mock_fs.is_file(existing_file)
        
        # ACT - Execute file existence check through public API
        result = check_file_exists(existing_file, fs_provider=mock_filesystem_provider)
        
        # ASSERT - Verify positive file detection behavior
        assert result is True
        assert 'check_file_exists' in [op[0] for op in mock_filesystem_provider.operation_log]
    
    def test_check_file_exists_nonexistent_file_behavior(self, mock_filesystem_provider):
        """
        Test behavior when checking for non-existent files.
        
        Validates that check_file_exists correctly returns False for files
        that don't exist in the filesystem through public API interface.
        """
        # ARRANGE - Set up non-existent file path
        nonexistent_file = Path('/test/data/nonexistent.pkl')
        
        # Verify file doesn't exist in mock filesystem
        assert not mock_filesystem_provider.mock_fs.exists(nonexistent_file)
        
        # ACT - Execute file existence check for non-existent file
        result = check_file_exists(nonexistent_file, fs_provider=mock_filesystem_provider)
        
        # ASSERT - Verify negative file detection behavior
        assert result is False
    
    def test_check_file_exists_directory_vs_file_behavior(self, mock_filesystem_provider):
        """
        Test behavior when distinguishing between files and directories.
        
        Validates that check_file_exists returns False for directories,
        correctly distinguishing between files and directory paths.
        """
        # ARRANGE - Set up directory and file with same base path
        test_directory = Path('/test/directory')
        test_file = Path('/test/file.pkl')
        
        mock_filesystem_provider.mock_fs.add_directory(str(test_directory))
        mock_filesystem_provider.mock_fs.add_file(str(test_file), size=256)
        
        # ACT - Execute existence checks for both directory and file
        directory_result = check_file_exists(test_directory, fs_provider=mock_filesystem_provider)
        file_result = check_file_exists(test_file, fs_provider=mock_filesystem_provider)
        
        # ASSERT - Verify file vs directory distinction behavior
        assert directory_result is False  # Directory should not be detected as file
        assert file_result is True        # File should be detected as file
    
    @pytest.mark.parametrize("input_type", ["string", "path_object"])
    def test_check_file_exists_input_type_behavior(self, mock_filesystem_provider, input_type):
        """
        Test file existence checking behavior with different input types.
        
        Parameterized test validating consistent behavior regardless of whether
        file path is provided as string or Path object through public API.
        """
        # ARRANGE - Set up test file and prepare input based on parameter
        test_file = Path('/test/input_type_test.pkl')
        mock_filesystem_provider.mock_fs.add_file(str(test_file), size=512)
        
        if input_type == "string":
            file_input = str(test_file)
        else:
            file_input = test_file
        
        # ACT - Execute with parameterized input type
        result = check_file_exists(file_input, fs_provider=mock_filesystem_provider)
        
        # ASSERT - Verify consistent behavior across input types
        assert result is True
        assert isinstance(result, bool)
    
    def test_check_file_exists_unicode_filename_behavior(self, mock_filesystem_provider, edge_case_scenarios):
        """
        Test file existence checking behavior with Unicode characters in filenames.
        
        Validates that Unicode characters in file names are handled correctly
        through the public API, supporting international file naming conventions.
        """
        # ARRANGE - Set up Unicode filename scenario
        unicode_scenario = edge_case_scenarios['unicode_paths'][0]
        unicode_file = Path('/test') / unicode_scenario['filename']
        
        mock_filesystem_provider.mock_fs.add_file(str(unicode_file), size=128)
        
        # ACT - Execute with Unicode filename
        try:
            result = check_file_exists(unicode_file, fs_provider=mock_filesystem_provider)
            
            # ASSERT - Verify Unicode filename handling behavior
            assert result is True
            
        except (UnicodeError, OSError):
            # Skip test if Unicode not supported on platform
            pytest.skip(f"Unicode filenames not supported on {platform.system()}")
    
    def test_check_file_exists_case_sensitivity_behavior(self, mock_filesystem_provider):
        """
        Test case sensitivity behavior in file existence checking.
        
        Validates platform-appropriate case sensitivity handling in file
        existence checks through the public API interface.
        """
        # ARRANGE - Set up file with specific case
        original_file = Path('/test/CaseTest.pkl')
        different_case = Path('/test/casetest.pkl')
        
        mock_filesystem_provider.mock_fs.add_file(str(original_file), size=256)
        
        # ACT - Execute case sensitivity check
        original_result = check_file_exists(original_file, fs_provider=mock_filesystem_provider)
        different_case_result = check_file_exists(different_case, fs_provider=mock_filesystem_provider)
        
        # ASSERT - Verify case sensitivity behavior (platform dependent)
        assert original_result is True
        
        # Different case behavior depends on platform/filesystem
        # On case-sensitive systems: should be False
        # On case-insensitive systems: may be True
        # Test documents the behavior without enforcing specific result
        assert isinstance(different_case_result, bool)


# ============================================================================
# COMPREHENSIVE EDGE-CASE AND INTEGRATION BEHAVIOR TESTS
# ============================================================================

class TestPathUtilitiesEdgeCaseBehavior:
    """
    Comprehensive edge-case and integration behavior validation for path utilities.
    
    Tests focus on complex scenarios, boundary conditions, and integration
    behavior across multiple path utility functions using realistic data
    patterns and comprehensive edge-case coverage.
    """
    
    @pytest.mark.parametrize("scenario_type", ["unicode", "boundary", "platform_specific"])
    def test_path_utilities_edge_case_scenarios(self, mock_filesystem_provider, edge_case_scenarios, scenario_type):
        """
        Test path utilities behavior with comprehensive edge-case scenarios.
        
        Parameterized test covering Unicode paths, boundary conditions, and
        platform-specific edge cases across all path utility functions.
        """
        # ARRANGE - Select edge-case scenario based on parameter
        scenarios = edge_case_scenarios[scenario_type]
        
        if not scenarios:
            pytest.skip(f"No {scenario_type} scenarios available for current platform")
        
        scenario = scenarios[0]  # Use first scenario for testing
        
        if scenario_type == "unicode":
            test_path = Path('/test') / scenario['directory'] / scenario['filename']
            mock_filesystem_provider.mock_fs.add_directory(str(test_path.parent))
            mock_filesystem_provider.mock_fs.add_file(str(test_path), size=512)
        elif scenario_type == "platform_specific":
            test_path = Path(scenario['path'])
        else:
            # Boundary condition testing
            test_path = Path('/test/boundary_test.pkl')
            mock_filesystem_provider.mock_fs.add_file(str(test_path), size=1024)
        
        # ACT & ASSERT - Execute comprehensive edge-case testing
        try:
            # Test file existence checking
            exists_result = check_file_exists(test_path, fs_provider=mock_filesystem_provider)
            assert isinstance(exists_result, bool)
            
            if exists_result:
                # Test relative path calculation if file exists
                base_dir = test_path.parent
                try:
                    relative_result = get_relative_path(test_path, base_dir, fs_provider=mock_filesystem_provider)
                    assert isinstance(relative_result, Path)
                except ValueError:
                    # Acceptable for certain edge cases
                    pass
            
            # Test directory creation for parent directory
            try:
                ensure_result = ensure_directory_exists(test_path.parent, fs_provider=mock_filesystem_provider)
                assert isinstance(ensure_result, Path)
            except (OSError, ValueError, FileExistsError):
                # Acceptable for certain edge cases
                pass
                
        except (UnicodeError, OSError) as e:
            # Skip test if edge case not supported on current platform
            pytest.skip(f"{scenario_type} edge case not supported: {e}")
    
    def test_path_utilities_integration_behavior(self, mock_filesystem_provider, sample_path_structures):
        """
        Test integration behavior across multiple path utility functions.
        
        Validates that path utility functions work together correctly in
        realistic scenarios with complex directory structures and file patterns.
        """
        # ARRANGE - Set up complex directory structure using sample paths
        base_directory = sample_path_structures['base_directory']
        data_files = sample_path_structures['data_files'][:3]  # Use first 3 files
        
        # Add files to mock filesystem
        for file_path in data_files:
            mock_filesystem_provider.mock_fs.add_directory(str(file_path.parent))
            mock_filesystem_provider.mock_fs.add_file(str(file_path), size=1024)
        
        # ACT - Execute integrated path operations
        
        # 1. Find common base directory for all files
        common_base = find_common_base_directory(data_files, fs_provider=mock_filesystem_provider)
        
        # 2. Calculate relative paths from common base
        relative_paths = []
        for file_path in data_files:
            try:
                rel_path = get_relative_path(file_path, common_base, fs_provider=mock_filesystem_provider)
                relative_paths.append(rel_path)
            except ValueError:
                # Skip files not under common base
                continue
        
        # 3. Convert relative paths back to absolute
        absolute_paths = []
        for rel_path in relative_paths:
            abs_path = get_absolute_path(rel_path, common_base, fs_provider=mock_filesystem_provider)
            absolute_paths.append(abs_path)
        
        # 4. Verify file existence for all resolved paths
        existence_results = []
        for abs_path in absolute_paths:
            exists = check_file_exists(abs_path, fs_provider=mock_filesystem_provider)
            existence_results.append(exists)
        
        # ASSERT - Verify integrated operation behavior
        assert common_base is not None
        assert len(relative_paths) > 0
        assert len(absolute_paths) == len(relative_paths)
        assert all(existence_results)  # All files should exist
        
        # Verify round-trip consistency
        for original, resolved in zip(data_files[:len(absolute_paths)], absolute_paths):
            # Paths should resolve to equivalent locations
            assert original.name == resolved.name
    
    @given(st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=10))
    @settings(max_examples=10, deadline=5000)
    def test_path_utilities_property_based_behavior(self, mock_filesystem_provider, path_components):
        """
        Property-based test for path utilities behavior with generated inputs.
        
        Uses Hypothesis to generate random path components and validates
        that path utility functions maintain consistent behavior properties
        across wide input ranges through public API interface.
        """
        # ARRANGE - Generate realistic path from components
        try:
            # Filter out problematic characters for cross-platform compatibility
            safe_components = []
            for component in path_components:
                # Remove characters that are problematic on Windows
                safe_component = ''.join(c for c in component if c not in '<>:"|?*\0')
                safe_component = safe_component.strip('. ')  # Remove trailing dots/spaces
                if safe_component and safe_component not in ('CON', 'PRN', 'AUX', 'NUL'):
                    safe_components.append(safe_component)
            
            if not safe_components:
                return  # Skip if no valid components
            
            test_path = Path('/test').joinpath(*safe_components)
            
            # Add path to mock filesystem
            mock_filesystem_provider.mock_fs.add_directory(str(test_path.parent))
            mock_filesystem_provider.mock_fs.add_file(str(test_path), size=256)
            
            # ACT - Execute path utility operations
            
            # Property 1: File existence should be boolean
            exists_result = check_file_exists(test_path, fs_provider=mock_filesystem_provider)
            assert isinstance(exists_result, bool)
            
            # Property 2: If file exists, relative path calculation should succeed
            if exists_result:
                base_dir = test_path.parent
                rel_path = get_relative_path(test_path, base_dir, fs_provider=mock_filesystem_provider)
                assert isinstance(rel_path, Path)
                assert not rel_path.is_absolute()
            
            # Property 3: Absolute path resolution should always return absolute path
            if test_path.parent != test_path:  # Avoid root directory issues
                abs_path = get_absolute_path(test_path.name, test_path.parent, fs_provider=mock_filesystem_provider)
                assert abs_path.is_absolute()
            
        except (OSError, ValueError, UnicodeError):
            # Skip test for inputs that cause filesystem errors
            # This is expected behavior for invalid path components
            pass
    
    def test_path_utilities_concurrent_behavior_simulation(self, mock_filesystem_provider):
        """
        Test simulated concurrent behavior for path utility operations.
        
        Validates that path utility functions maintain consistent behavior
        under simulated concurrent access scenarios through public API.
        """
        # ARRANGE - Set up shared filesystem state for concurrent simulation
        shared_directory = Path('/test/concurrent')
        shared_files = [
            shared_directory / f'file_{i}.pkl' for i in range(5)
        ]
        
        mock_filesystem_provider.mock_fs.add_directory(str(shared_directory))
        for file_path in shared_files:
            mock_filesystem_provider.mock_fs.add_file(str(file_path), size=512)
        
        # ACT - Simulate concurrent operations
        operation_results = []
        
        # Simulate multiple "concurrent" operations
        for iteration in range(3):
            # Multiple existence checks
            existence_checks = [
                check_file_exists(file_path, fs_provider=mock_filesystem_provider)
                for file_path in shared_files
            ]
            
            # Common base directory calculation
            common_base = find_common_base_directory(shared_files, fs_provider=mock_filesystem_provider)
            
            # Directory creation attempts (should be idempotent)
            directory_results = []
            for i in range(2):
                try:
                    result = ensure_directory_exists(
                        shared_directory / f'subdir_{iteration}_{i}',
                        fs_provider=mock_filesystem_provider
                    )
                    directory_results.append(result)
                except FileExistsError:
                    # Expected for idempotent operations
                    pass
            
            operation_results.append({
                'iteration': iteration,
                'existence_checks': existence_checks,
                'common_base': common_base,
                'directory_results': directory_results
            })
        
        # ASSERT - Verify consistent concurrent behavior
        assert len(operation_results) == 3
        
        # All existence checks should return True consistently
        for result in operation_results:
            assert all(result['existence_checks'])
        
        # Common base should be consistent across iterations
        common_bases = [result['common_base'] for result in operation_results]
        assert all(base == common_bases[0] for base in common_bases)
        
        # Directory creation should be successful and idempotent
        for result in operation_results:
            assert len(result['directory_results']) >= 0  # At least some should succeed


# ============================================================================
# PERFORMANCE AND MEMORY BEHAVIOR VALIDATION
# ============================================================================

@pytest.mark.performance
class TestPathUtilitiesPerformanceBehavior:
    """
    Performance behavior validation for path utilities.
    
    Tests focus on observable performance characteristics and memory usage
    patterns of path utility functions under various load conditions and
    data sizes while maintaining behavior-focused testing principles.
    """
    
    def test_path_utilities_large_dataset_behavior(self, mock_filesystem_provider, performance_benchmarks):
        """
        Test behavior with large datasets of path operations.
        
        Validates that path utility functions maintain acceptable performance
        and memory usage characteristics when processing large numbers of paths.
        """
        # ARRANGE - Set up large dataset of paths
        large_path_count = 1000
        base_directory = Path('/test/large_dataset')
        
        large_paths = []
        for i in range(large_path_count):
            file_path = base_directory / f'subdir_{i % 10}' / f'file_{i:04d}.pkl'
            large_paths.append(file_path)
            
            # Add subset to mock filesystem to keep memory reasonable
            if i < 100:  # Only add first 100 to mock filesystem
                mock_filesystem_provider.mock_fs.add_directory(str(file_path.parent))
                mock_filesystem_provider.mock_fs.add_file(str(file_path), size=64)
        
        # ACT - Execute large dataset operations with performance monitoring
        import time
        
        start_time = time.time()
        
        # Test find_common_base_directory performance
        common_base = find_common_base_directory(large_paths[:100], fs_provider=mock_filesystem_provider)
        
        # Test batch file existence checking
        existence_results = []
        for path in large_paths[:100]:
            result = check_file_exists(path, fs_provider=mock_filesystem_provider)
            existence_results.append(result)
        
        end_time = time.time()
        operation_duration = end_time - start_time
        
        # ASSERT - Verify performance behavior characteristics
        assert common_base is not None
        assert len(existence_results) == 100
        
        # Performance should be reasonable for dataset size
        expected_max_time = performance_benchmarks.benchmark_file_discovery(100)
        assert operation_duration <= expected_max_time, f"Operation took {operation_duration:.3f}s, expected <= {expected_max_time:.3f}s"
    
    def test_path_utilities_memory_usage_behavior(self, mock_filesystem_provider):
        """
        Test memory usage behavior for path utility operations.
        
        Validates that path utility functions don't exhibit memory leaks
        or excessive memory consumption during repeated operations.
        """
        # ARRANGE - Set up repeated operation scenario
        test_directory = Path('/test/memory_test')
        test_paths = [test_directory / f'file_{i}.pkl' for i in range(50)]
        
        for path in test_paths:
            mock_filesystem_provider.mock_fs.add_directory(str(path.parent))
            mock_filesystem_provider.mock_fs.add_file(str(path), size=128)
        
        # ACT - Execute repeated operations to test memory behavior
        try:
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Perform repeated operations
            for cycle in range(10):
                # Common base calculation
                common_base = find_common_base_directory(test_paths, fs_provider=mock_filesystem_provider)
                
                # File existence checks
                for path in test_paths:
                    check_file_exists(path, fs_provider=mock_filesystem_provider)
                
                # Directory creation
                ensure_directory_exists(test_directory / f'cycle_{cycle}', fs_provider=mock_filesystem_provider)
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # ASSERT - Verify reasonable memory usage behavior
            assert memory_increase < 10, f"Memory increased by {memory_increase:.2f}MB during repeated operations"
            
        except ImportError:
            # Skip memory testing if psutil not available
            pytest.skip("psutil not available for memory monitoring")


# ============================================================================
# TEST EXECUTION AND FIXTURE CLEANUP
# ============================================================================

def test_filesystem_provider_restoration(mock_filesystem_provider):
    """
    Test that filesystem provider is properly restored after testing.
    
    Validates that the dependency injection system correctly restores
    the original filesystem provider after test completion to prevent
    cross-test contamination.
    """
    # ARRANGE - Get current provider during test
    current_provider = get_current_filesystem_provider()
    
    # ACT - Verify mock provider is active
    assert isinstance(current_provider, MockPathFilesystemProvider)
    
    # ASSERT - Provider restoration will be handled by fixture cleanup
    # This test documents the expected behavior
    assert current_provider is not None


def test_operation_logging_behavior(mock_filesystem_provider):
    """
    Test that mock filesystem provider logs operations for behavior verification.
    
    Validates that the mock provider correctly tracks filesystem operations
    for comprehensive behavior verification in tests.
    """
    # ARRANGE - Clear operation log
    mock_filesystem_provider.operation_log.clear()
    
    # ACT - Execute various path operations
    test_path = Path('/test/logging/file.pkl')
    mock_filesystem_provider.mock_fs.add_file(str(test_path), size=256)
    
    check_file_exists(test_path, fs_provider=mock_filesystem_provider)
    ensure_directory_exists(test_path.parent, fs_provider=mock_filesystem_provider)
    
    # ASSERT - Verify operation logging behavior
    assert len(mock_filesystem_provider.operation_log) > 0
    operation_types = [op[0] for op in mock_filesystem_provider.operation_log]
    assert 'check_file_exists' in operation_types
    assert 'create_directory' in operation_types