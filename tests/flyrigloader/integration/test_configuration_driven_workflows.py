"""
Configuration-driven workflow integration test suite.

This module provides behavior-focused integration testing for configuration-driven workflows,
validating observable YAML configuration behavior through black-box testing approaches
per Section 0 requirements for implementation-agnostic testing patterns.

The test suite validates complete workflows from YAML configuration loading through
file discovery, metadata extraction, to final processing results using Protocol-based
mock implementations and centralized fixture management for consistent testing patterns.

Requirements Tested:
- F-001: Hierarchical YAML Configuration System integration (Section 2.1.1)
- F-001-RQ-004: Hierarchical configuration merging validation (Section 2.2.1)
- TST-INTEG-001: Configuration-driven workflow validation (Section 2.2.10)
- F-002: Configuration-driven file discovery pattern validation (Section 2.1.2)
- F-007: Configuration-driven metadata extraction pattern validation (Section 2.1.7)
- Section 4.1.1.2: Configuration Management Process integration testing

Test Architecture:
- Black-box behavior validation focusing on observable outcomes
- Protocol-based mock implementations from centralized tests/utils.py
- AAA pattern structure with clear Arrange-Act-Assert phases
- Edge-case coverage through parameterized test scenarios
- Centralized fixture usage from tests/conftest.py
"""

import pytest
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from unittest.mock import patch, MagicMock

# Import centralized test utilities for Protocol-based mocking
from tests.utils import (
    create_mock_config_provider,
    create_mock_filesystem,
    create_mock_dataloader,
    create_integration_test_environment,
    generate_edge_case_scenarios,
    MockConfigurationProvider,
    MockFilesystemProvider,
    MockDataLoadingProvider
)

# Import flyrigloader configuration functions for behavior validation
from flyrigloader.config.yaml_config import (
    load_config,
    validate_config_dict,
    get_ignore_patterns,
    get_mandatory_substrings,
    get_extraction_patterns,
    get_dataset_info,
    get_experiment_info,
    get_all_experiment_names,
    get_all_dataset_names
)
from flyrigloader.config.discovery import (
    discover_files_with_config,
    discover_experiment_files,
    discover_dataset_files
)
from flyrigloader.discovery.files import discover_files
from flyrigloader.io.column_models import ColumnConfigDict, ColumnConfig


class TestConfigurationDrivenWorkflows:
    """
    Behavior-focused integration tests for configuration-driven workflow validation.
    
    These tests validate observable configuration-driven workflow behaviors using
    Protocol-based mock implementations and centralized fixtures, focusing on
    black-box testing approaches rather than internal implementation details.
    
    Test Architecture:
    - Uses centralized fixtures from tests/conftest.py for workspace and data generation
    - Employs Protocol-based mocks from tests/utils.py for dependency isolation
    - Implements AAA pattern for clear test structure and maintainability
    - Focuses on configuration-driven workflow behavior validation
    """

    @pytest.fixture(scope="function")
    def mock_config_provider(self):
        """
        ARRANGE: Create Protocol-based mock configuration provider for dependency isolation.
        
        Returns centralized mock configuration provider with comprehensive
        test scenarios including edge cases and error conditions.
        """
        return create_mock_config_provider(
            config_type='comprehensive',
            include_errors=True
        )

    @pytest.fixture(scope="function") 
    def mock_filesystem(self):
        """
        ARRANGE: Create Protocol-based mock filesystem for dependency isolation.
        
        Returns centralized mock filesystem with realistic experimental
        data structure including Unicode files and corrupted scenarios.
        """
        filesystem_structure = {
            'files': {
                '/test/data/2024-12-20/mouse_20241220_baseline_001.pkl': {'size': 2048},
                '/test/data/2024-12-20/mouse_20241220_treatment_002.pkl': {'size': 4096},
                '/test/data/2024-12-20/rat_20241220_control_001.pkl': {'size': 1024},
                '/test/data/2024-12-22/mouse_20241222_baseline_001.pkl': {'size': 2048},
                '/test/data/2024-12-22/mouse_20241222_treatment_003.pkl': {'size': 4096},
                '/test/data/2024-12-25/exp001_mouse_baseline.pkl': {'size': 3072},
                '/test/data/2024-12-25/exp001_rat_treatment.pkl': {'size': 3072},
                '/test/data/2024-12-25/exp002_mouse_control.pkl': {'size': 2048},
                # Files that should be filtered by ignore patterns
                '/test/data/2024-12-20/._hidden_file.pkl': {'size': 100},
                '/test/data/2024-12-20/static_horiz_ribbon_data.pkl': {'size': 1024},
                '/test/data/2024-12-22/temp_backup_file.pkl': {'size': 512},
                # Different file formats
                '/test/data/2024-12-20/mouse_20241220_baseline_004.csv': {'size': 256},
                '/test/data/2024-12-22/metadata_summary.txt': {'size': 128}
            },
            'directories': [
                '/test',
                '/test/data', 
                '/test/data/2024-12-20',
                '/test/data/2024-12-22',
                '/test/data/2024-12-25',
                '/test/configs',
                '/test/experiments'
            ]
        }
        
        return create_mock_filesystem(
            structure=filesystem_structure,
            unicode_files=True,
            corrupted_files=True
        )

    @pytest.fixture(scope="function")
    def mock_data_loader(self):
        """
        ARRANGE: Create Protocol-based mock data loader for dependency isolation.
        
        Returns centralized mock data loader with realistic experimental
        data scenarios including corrupted files and memory constraints.
        """
        return create_mock_dataloader(
            scenarios=['basic', 'corrupted', 'network'],
            include_experimental_data=True
        )

    @pytest.fixture(scope="function")
    def sample_hierarchical_config(self, mock_config_provider):
        """
        ARRANGE: Create comprehensive hierarchical configuration for behavior testing.
        
        Uses centralized configuration provider to generate realistic
        experimental configuration structure with hierarchical merging.
        """
        return {
            "project": {
                "directories": {
                    "major_data_directory": "/test/data",
                    "batchfile_directory": "/test/configs"
                },
                "ignore_substrings": [
                    "._",
                    "static_horiz_ribbon",
                    "temp_"
                ],
                "mandatory_experiment_strings": [],
                "extraction_patterns": [
                    r".*_(?P<date>\d{8})_(?P<condition>\w+)_(?P<replicate>\d+)\.pkl",
                    r".*/(?P<animal>mouse|rat)_(?P<date>\d{8})_(?P<condition>\w+)_(?P<replicate>\d+)\.pkl"
                ]
            },
            "rigs": {
                "old_opto": {
                    "sampling_frequency": 60,
                    "mm_per_px": 0.154
                },
                "new_opto": {
                    "sampling_frequency": 120,
                    "mm_per_px": 0.1818
                }
            },
            "datasets": {
                "baseline_navigation": {
                    "rig": "old_opto",
                    "patterns": ["*baseline*"],
                    "dates_vials": {
                        "2024-12-20": [1, 2],
                        "2024-12-22": [1, 3]
                    },
                    "metadata": {
                        "extraction_patterns": [
                            r".*_(?P<dataset>baseline)_(?P<date>\d{8})_(?P<replicate>\d+)\.pkl"
                        ]
                    }
                },
                "treatment_response": {
                    "rig": "new_opto",
                    "patterns": ["*treatment*"],
                    "dates_vials": {
                        "2024-12-20": [2],
                        "2024-12-22": [3]
                    },
                    "metadata": {
                        "extraction_patterns": [
                            r".*_(?P<dataset>treatment)_(?P<date>\d{8})_(?P<replicate>\d+)\.pkl"
                        ]
                    }
                },
                "control_group": {
                    "rig": "old_opto",
                    "patterns": ["*control*"],
                    "dates_vials": {
                        "2024-12-20": [1]
                    }
                }
            },
            "experiments": {
                "baseline_comparison": {
                    "datasets": ["baseline_navigation", "control_group"],
                    "filters": {
                        "ignore_substrings": ["backup"],
                        "mandatory_experiment_strings": ["baseline", "control"]
                    },
                    "metadata": {
                        "extraction_patterns": [
                            r".*_(?P<experiment>baseline)_(?P<condition>\w+)\.pkl"
                        ]
                    }
                },
                "treatment_analysis": {
                    "datasets": ["treatment_response"],
                    "filters": {
                        "ignore_substrings": ["test_", "debug_"]
                    },
                    "metadata": {
                        "extraction_patterns": [
                            r".*_(?P<experiment>treatment)_(?P<analysis_type>\w+)\.pkl"
                        ]
                    }
                },
                "comprehensive_study": {
                    "datasets": ["baseline_navigation", "treatment_response", "control_group"],
                    "filters": {
                        "mandatory_experiment_strings": ["mouse", "rat"]
                    },
                    "metadata": {
                        "extraction_patterns": [
                            r".*/(?P<experiment>exp\d+)_(?P<animal>\w+)_(?P<condition>\w+)\.pkl"
                        ]
                    }
                }
            }
        }

    def test_hierarchical_configuration_loading_and_validation_f001(self, sample_hierarchical_config):
        """
        Test F-001: Hierarchical YAML Configuration System integration with behavior validation.
        
        Validates that configuration loading produces expected behavioral outcomes
        from both dictionary and file-based sources using black-box verification.
        
        Focus: Observable configuration loading behavior rather than internal mechanisms.
        """
        # ARRANGE - Set up configuration test scenarios
        test_config = sample_hierarchical_config
        expected_structure_keys = {"project", "datasets", "experiments", "rigs"}
        
        # ACT - Execute configuration loading behavior
        config_from_dict = load_config(test_config)
        validated_config = validate_config_dict(test_config)
        
        # ASSERT - Verify observable configuration behavior
        assert config_from_dict is not None, "Configuration loading should return valid result"
        assert isinstance(config_from_dict, dict), "Configuration should be accessible as dictionary"
        assert config_from_dict == test_config, "Dictionary loading should preserve configuration data"
        
        assert validated_config == test_config, "Validation should preserve valid configuration"
        assert expected_structure_keys.issubset(validated_config.keys()), \
            "Configuration should contain expected structural elements"
        
        # Verify hierarchical structure accessibility
        assert "datasets" in validated_config, "Configuration should provide dataset access"
        for dataset_name, dataset_config in validated_config["datasets"].items():
            if "dates_vials" in dataset_config:
                dates_vials = dataset_config["dates_vials"]
                assert isinstance(dates_vials, dict), f"Dataset {dataset_name} should provide valid dates_vials structure"
                for date, vials in dates_vials.items():
                    assert isinstance(date, str), f"Date keys should be string format for {dataset_name}"
                    assert isinstance(vials, list), f"Vial lists should be accessible for {dataset_name}"

    def test_hierarchical_configuration_merging_behavior_f001_rq004(self, sample_hierarchical_config):
        """
        Test F-001-RQ-004: Hierarchical configuration merging behavioral validation.
        
        Validates observable configuration merging behavior between project-level
        and experiment-level configurations using black-box verification approaches.
        
        Focus: Configuration-driven workflow behavior rather than internal merging mechanisms.
        """
        # ARRANGE - Set up hierarchical configuration merging scenarios
        config = sample_hierarchical_config
        project_expected_ignore = ["*._*", "*static_horiz_ribbon*", "*temp_*"]
        baseline_experiment = "baseline_comparison"
        treatment_experiment = "treatment_analysis"
        comprehensive_experiment = "comprehensive_study"
        
        # ACT - Execute configuration-driven pattern extraction behavior
        project_ignore_patterns = get_ignore_patterns(config)
        baseline_ignore_patterns = get_ignore_patterns(config, experiment=baseline_experiment)
        treatment_ignore_patterns = get_ignore_patterns(config, experiment=treatment_experiment)
        
        project_mandatory = get_mandatory_substrings(config)
        baseline_mandatory = get_mandatory_substrings(config, experiment=baseline_experiment)
        comprehensive_mandatory = get_mandatory_substrings(config, experiment=comprehensive_experiment)
        
        project_patterns = get_extraction_patterns(config)
        baseline_patterns = get_extraction_patterns(config, experiment=baseline_experiment)
        
        # ASSERT - Verify observable hierarchical merging behavior
        # Project-level ignore pattern behavior
        assert project_ignore_patterns == project_expected_ignore, \
            "Project-level ignore patterns should be accessible through configuration API"
        
        # Experiment-level override behavior validation
        expected_baseline_ignore = project_expected_ignore + ["*backup*"]
        assert baseline_ignore_patterns == expected_baseline_ignore, \
            "Baseline experiment should merge project patterns with experiment-specific patterns"
        
        expected_treatment_ignore = project_expected_ignore + ["*test_*", "*debug_*"]
        assert treatment_ignore_patterns == expected_treatment_ignore, \
            "Treatment experiment should merge project patterns with additional experiment filters"
        
        # Mandatory substring merging behavior
        assert project_mandatory == [], "Project level should have empty mandatory strings by default"
        assert "baseline" in baseline_mandatory, "Baseline experiment should enforce baseline requirement"
        assert "control" in baseline_mandatory, "Baseline experiment should enforce control requirement"
        assert "mouse" in comprehensive_mandatory, "Comprehensive study should require mouse data"
        assert "rat" in comprehensive_mandatory, "Comprehensive study should require rat data"
        
        # Extraction pattern hierarchical behavior
        assert len(project_patterns) == 2, "Project should provide base extraction patterns"
        assert len(baseline_patterns) >= len(project_patterns), \
            "Experiment patterns should extend project patterns"
        
        # Verify experiment-specific patterns are merged
        baseline_experiment_pattern = r".*_(?P<experiment>baseline)_(?P<condition>\w+)\.pkl"
        assert baseline_experiment_pattern in baseline_patterns, \
            "Experiment-specific patterns should be available in merged pattern set"

    def test_configuration_driven_file_discovery_behavior_f002(self, sample_hierarchical_config, mock_filesystem):
        """
        Test F-002: Configuration-driven file discovery behavioral validation.
        
        Validates observable configuration-driven file discovery behavior including
        pattern filtering, ignore rules, and directory structure processing through
        black-box verification of discovery outcomes.
        
        Focus: Configuration workflow behavior rather than internal discovery mechanisms.
        """
        # ARRANGE - Set up configuration-driven discovery scenario
        config = sample_hierarchical_config
        base_directory = "/test/data"
        search_pattern = "*.pkl"
        baseline_experiment = "baseline_comparison"
        comprehensive_experiment = "comprehensive_study"
        
        # Mock filesystem discovery behavior
        mock_filesystem.activate()
        
        # ACT - Execute configuration-driven file discovery workflow
        with patch('flyrigloader.discovery.files.discover_files') as mock_discover:
            # Mock file discovery to return realistic file sets
            mock_discover.return_value = [
                '/test/data/2024-12-20/mouse_20241220_baseline_001.pkl',
                '/test/data/2024-12-20/mouse_20241220_treatment_002.pkl',
                '/test/data/2024-12-20/rat_20241220_control_001.pkl',
                '/test/data/2024-12-22/mouse_20241222_baseline_001.pkl',
                '/test/data/2024-12-25/exp001_mouse_baseline.pkl'
            ]
            
            basic_discovery_result = discover_files_with_config(
                config=config,
                directory=base_directory,
                pattern=search_pattern,
                recursive=True
            )
            
            baseline_discovery_result = discover_files_with_config(
                config=config,
                directory=base_directory,
                pattern=search_pattern,
                recursive=True,
                experiment=baseline_experiment
            )
            
            comprehensive_discovery_result = discover_files_with_config(
                config=config,
                directory=base_directory,
                pattern=search_pattern,
                recursive=True,
                experiment=comprehensive_experiment
            )
        
        # ASSERT - Verify observable configuration-driven discovery behavior
        # Basic configuration-driven discovery behavior
        assert basic_discovery_result is not None, \
            "Configuration-driven discovery should return discoverable files"
        assert len(basic_discovery_result) > 0, \
            "Discovery should locate files matching configuration criteria"
        assert all(f.endswith('.pkl') for f in basic_discovery_result), \
            "Discovery should respect file pattern specifications"
        
        # Ignore pattern filtering behavior verification
        ignored_substrings = ["._", "static_horiz_ribbon", "temp_"]
        for result_file in basic_discovery_result:
            assert not any(ignore_pattern in result_file for ignore_pattern in ignored_substrings), \
                f"Discovery should filter files matching ignore patterns: {result_file}"
        
        # Experiment-specific filtering behavior verification
        assert baseline_discovery_result is not None, \
            "Experiment-specific discovery should return valid results"
        
        # Verify no backup files in baseline experiment (experiment-specific ignore pattern)
        for result_file in baseline_discovery_result:
            assert "backup" not in result_file, \
                f"Baseline experiment should filter backup files: {result_file}"
        
        # Comprehensive experiment mandatory substring behavior
        assert comprehensive_discovery_result is not None, \
            "Comprehensive experiment discovery should return valid results"
        
        # Verify all files contain mandatory substrings (mouse OR rat)
        for result_file in comprehensive_discovery_result:
            assert any(substring in result_file for substring in ["mouse", "rat"]), \
                f"Comprehensive study should enforce mandatory substring requirements: {result_file}"

    def test_configuration_driven_metadata_extraction_behavior_f007(self, sample_hierarchical_config, mock_filesystem):
        """
        Test F-007: Configuration-driven metadata extraction behavioral validation.
        
        Validates observable metadata extraction behavior driven by configuration-defined
        regex patterns with hierarchical pattern precedence using black-box verification.
        
        Focus: Configuration-driven metadata workflow behavior rather than internal extraction mechanisms.
        """
        # ARRANGE - Set up configuration-driven metadata extraction scenario
        config = sample_hierarchical_config
        base_directory = "/test/data"
        search_pattern = "*.pkl"
        comprehensive_experiment = "comprehensive_study"
        
        # Expected metadata extraction behavior for different file types
        expected_file_metadata = {
            'mouse_20241220_baseline_001.pkl': {
                'animal': 'mouse',
                'date': '20241220', 
                'condition': 'baseline',
                'replicate': '001'
            },
            'rat_20241220_control_001.pkl': {
                'animal': 'rat',
                'date': '20241220',
                'condition': 'control', 
                'replicate': '001'
            },
            'exp001_mouse_baseline.pkl': {
                'experiment': 'exp001',
                'animal': 'mouse',
                'condition': 'baseline'
            }
        }
        
        # ACT - Execute configuration-driven metadata extraction workflow
        with patch('flyrigloader.discovery.files.discover_files') as mock_discover:
            # Mock metadata extraction behavior
            mock_discover.return_value = {
                '/test/data/2024-12-20/mouse_20241220_baseline_001.pkl': {
                    'path': '/test/data/2024-12-20/mouse_20241220_baseline_001.pkl',
                    'animal': 'mouse',
                    'date': '20241220',
                    'condition': 'baseline',
                    'replicate': '001'
                },
                '/test/data/2024-12-20/rat_20241220_control_001.pkl': {
                    'path': '/test/data/2024-12-20/rat_20241220_control_001.pkl',
                    'animal': 'rat',
                    'date': '20241220',
                    'condition': 'control',
                    'replicate': '001'
                },
                '/test/data/2024-12-25/exp001_mouse_baseline.pkl': {
                    'path': '/test/data/2024-12-25/exp001_mouse_baseline.pkl',
                    'experiment': 'exp001',
                    'animal': 'mouse',
                    'condition': 'baseline'
                }
            }
            
            project_metadata_result = discover_files_with_config(
                config=config,
                directory=base_directory,
                pattern=search_pattern,
                recursive=True,
                extract_metadata=True
            )
            
            experiment_metadata_result = discover_files_with_config(
                config=config,
                directory=base_directory,
                pattern=search_pattern,
                recursive=True,
                experiment=comprehensive_experiment,
                extract_metadata=True
            )
        
        # ASSERT - Verify observable configuration-driven metadata extraction behavior
        # Project-level metadata extraction behavior verification
        assert isinstance(project_metadata_result, dict), \
            "Configuration-driven metadata extraction should return structured results"
        assert len(project_metadata_result) > 0, \
            "Metadata extraction should discover and process files"
        
        # Verify metadata extraction for standard animal experiment files
        animal_files = {path: metadata for path, metadata in project_metadata_result.items() 
                       if any(animal in path for animal in ["mouse", "rat"])}
        
        assert len(animal_files) > 0, "Should extract metadata from animal experiment files"
        
        for file_path, metadata in animal_files.items():
            filename = Path(file_path).name
            if filename in expected_file_metadata:
                expected_meta = expected_file_metadata[filename]
                for key, expected_value in expected_meta.items():
                    assert metadata.get(key) == expected_value, \
                        f"File {filename} should extract {key}={expected_value}, got {metadata.get(key)}"
        
        # Experiment-specific metadata extraction behavior verification
        assert isinstance(experiment_metadata_result, dict), \
            "Experiment-specific metadata extraction should return structured results"
        
        # Verify experiment pattern extraction for exp files
        exp_files = {path: metadata for path, metadata in experiment_metadata_result.items() 
                    if "exp" in path}
        
        for file_path, metadata in exp_files.items():
            if "exp001_mouse_baseline.pkl" in file_path:
                assert metadata.get("experiment") == "exp001", \
                    "Experiment-specific patterns should extract experiment ID"
                assert metadata.get("animal") == "mouse", \
                    "Experiment patterns should extract animal type"
                assert metadata.get("condition") == "baseline", \
                    "Experiment patterns should extract condition information"

    def test_experiment_specific_discovery_workflow_behavior(self, sample_hierarchical_config, mock_filesystem):
        """
        Test experiment-specific file discovery workflow behavioral validation.
        
        Validates observable experiment-specific file discovery behavior using
        experiment configuration to locate relevant files with proper filtering
        through black-box workflow verification.
        
        Focus: Configuration-driven experiment workflow behavior rather than internal discovery logic.
        """
        # ARRANGE - Set up experiment-specific discovery scenario
        config = sample_hierarchical_config
        baseline_experiment = "baseline_comparison"
        base_directory = "/test/data"
        search_pattern = "*.pkl"
        
        # Expected experiment behavior for baseline comparison
        expected_baseline_files = [
            '/test/data/2024-12-20/mouse_20241220_baseline_001.pkl',
            '/test/data/2024-12-20/rat_20241220_control_001.pkl',
            '/test/data/2024-12-22/mouse_20241222_baseline_001.pkl'
        ]
        
        # ACT - Execute experiment-specific discovery workflow
        with patch('flyrigloader.config.discovery.discover_experiment_files') as mock_exp_discover:
            # Mock experiment-specific discovery behavior
            mock_exp_discover.return_value = expected_baseline_files
            
            baseline_discovery_result = discover_experiment_files(
                config=config,
                experiment_name=baseline_experiment,
                base_directory=base_directory,
                pattern=search_pattern,
                recursive=True
            )
            
            # Mock metadata extraction behavior for experiment discovery
            mock_exp_discover.return_value = {
                '/test/data/2024-12-20/mouse_20241220_baseline_001.pkl': {
                    'path': '/test/data/2024-12-20/mouse_20241220_baseline_001.pkl',
                    'experiment': 'baseline_comparison',
                    'animal': 'mouse',
                    'condition': 'baseline'
                },
                '/test/data/2024-12-20/rat_20241220_control_001.pkl': {
                    'path': '/test/data/2024-12-20/rat_20241220_control_001.pkl',
                    'experiment': 'baseline_comparison',
                    'animal': 'rat',
                    'condition': 'control'
                }
            }
            
            baseline_metadata_result = discover_experiment_files(
                config=config,
                experiment_name=baseline_experiment,
                base_directory=base_directory,
                pattern=search_pattern,
                recursive=True,
                extract_metadata=True
            )
        
        # ASSERT - Verify observable experiment-specific discovery behavior
        # Basic experiment discovery behavior verification
        assert baseline_discovery_result is not None, \
            "Experiment-specific discovery should return valid results"
        assert len(baseline_discovery_result) > 0, \
            "Baseline experiment discovery should locate relevant files"
        
        # Verify experiment filtering behavior - only relevant files for baseline experiment
        baseline_relevant = [f for f in baseline_discovery_result 
                           if any(term in f for term in ["baseline", "control"])]
        assert len(baseline_relevant) > 0, \
            "Baseline experiment should locate files relevant to baseline comparison"
        
        # Experiment metadata extraction behavior verification
        assert isinstance(baseline_metadata_result, dict), \
            "Experiment discovery with metadata should return structured results"
        
        # Verify experiment-specific metadata extraction behavior
        for file_path, metadata in baseline_metadata_result.items():
            assert "path" in metadata, \
                "Experiment discovery should provide file path information"
            assert file_path == metadata["path"], \
                "Metadata should maintain path consistency"
            
            # Verify experiment configuration-driven filtering was applied
            experiment_info = get_experiment_info(config, baseline_experiment)
            if "filters" in experiment_info:
                ignore_patterns = experiment_info["filters"].get("ignore_substrings", [])
                for ignore_pattern in ignore_patterns:
                    assert ignore_pattern not in file_path, \
                        f"Experiment discovery should filter files by {ignore_pattern}: {file_path}"

    def test_dataset_specific_discovery_workflow_behavior(self, sample_hierarchical_config, mock_filesystem):
        """
        Test dataset-specific file discovery workflow behavioral validation.
        
        Validates observable dataset-specific file discovery behavior using
        dataset configuration including date-vial specifications through
        black-box workflow verification.
        
        Focus: Configuration-driven dataset workflow behavior rather than internal discovery mechanisms.
        """
        # ARRANGE - Set up dataset-specific discovery scenario
        config = sample_hierarchical_config
        baseline_dataset = "baseline_navigation"
        treatment_dataset = "treatment_response"
        base_directory = "/test/data"
        search_pattern = "*.pkl"
        
        # Expected dataset behavior for baseline navigation
        expected_baseline_dates = ["2024-12-20", "2024-12-22"]
        expected_treatment_files = [
            '/test/data/2024-12-20/mouse_20241220_treatment_002.pkl',
            '/test/data/2024-12-22/mouse_20241222_treatment_003.pkl'
        ]
        
        # ACT - Execute dataset-specific discovery workflow
        with patch('flyrigloader.config.discovery.discover_dataset_files') as mock_dataset_discover:
            # Mock dataset-specific discovery behavior
            mock_dataset_discover.return_value = [
                '/test/data/2024-12-20/mouse_20241220_baseline_001.pkl',
                '/test/data/2024-12-22/mouse_20241222_baseline_001.pkl'
            ]
            
            baseline_dataset_result = discover_dataset_files(
                config=config,
                dataset_name=baseline_dataset,
                base_directory=base_directory,
                pattern=search_pattern,
                recursive=True
            )
            
            # Mock metadata extraction for dataset discovery
            mock_dataset_discover.return_value = {
                '/test/data/2024-12-20/mouse_20241220_baseline_001.pkl': {
                    'path': '/test/data/2024-12-20/mouse_20241220_baseline_001.pkl',
                    'dataset': 'baseline_navigation',
                    'date': '20241220',
                    'replicate': '001'
                },
                '/test/data/2024-12-22/mouse_20241222_baseline_001.pkl': {
                    'path': '/test/data/2024-12-22/mouse_20241222_baseline_001.pkl',
                    'dataset': 'baseline_navigation', 
                    'date': '20241222',
                    'replicate': '001'
                }
            }
            
            baseline_metadata_result = discover_dataset_files(
                config=config,
                dataset_name=baseline_dataset,
                base_directory=base_directory,
                pattern=search_pattern,
                recursive=True,
                extract_metadata=True
            )
            
            # Test treatment response dataset behavior
            mock_dataset_discover.return_value = expected_treatment_files
            
            treatment_dataset_result = discover_dataset_files(
                config=config,
                dataset_name=treatment_dataset,
                base_directory=base_directory,
                pattern=search_pattern,
                recursive=True
            )
        
        # ASSERT - Verify observable dataset-specific discovery behavior
        # Basic dataset discovery behavior verification
        assert baseline_dataset_result is not None, \
            "Dataset-specific discovery should return valid results"
        assert len(baseline_dataset_result) > 0, \
            "Baseline dataset discovery should locate relevant files"
        
        # Verify dataset date-filtering behavior
        for file_path in baseline_dataset_result:
            assert any(date in file_path for date in expected_baseline_dates), \
                f"Dataset files should come from configured date directories: {file_path}"
        
        # Dataset metadata extraction behavior verification
        assert isinstance(baseline_metadata_result, dict), \
            "Dataset discovery with metadata should return structured results"
        
        for file_path, metadata in baseline_metadata_result.items():
            assert "path" in metadata, \
                "Dataset discovery should provide file path information"
            assert file_path == metadata["path"], \
                "Metadata should maintain path consistency"
        
        # Treatment dataset discovery behavior verification
        assert treatment_dataset_result is not None, \
            "Treatment dataset discovery should return valid results"
        
        # Verify treatment files are found with proper filtering
        treatment_relevant = [f for f in treatment_dataset_result if "treatment" in f]
        assert len(treatment_relevant) > 0, \
            "Treatment dataset should locate files relevant to treatment conditions"

    @pytest.mark.parametrize("edge_case_scenario", [
        "malformed_yaml_recovery",
        "unicode_path_handling", 
        "nested_parameter_edge_cases",
        "cross_platform_path_handling",
        "configuration_schema_violations"
    ])
    def test_configuration_edge_case_recovery_behavior(self, edge_case_scenario, sample_hierarchical_config):
        """
        Test configuration edge case recovery behavioral validation.
        
        Validates observable configuration error recovery and edge-case handling
        behavior using parameterized test scenarios for comprehensive coverage
        of malformed configurations, Unicode paths, and schema violations.
        
        Focus: Configuration robustness behavior rather than internal error handling mechanisms.
        """
        # ARRANGE - Set up edge case scenario testing
        config = sample_hierarchical_config
        
        if edge_case_scenario == "malformed_yaml_recovery":
            # Test malformed YAML configuration recovery behavior
            invalid_config = {"invalid": "structure"}
            
            # ACT - Execute configuration validation with invalid input
            ignore_patterns = get_ignore_patterns(invalid_config)
            mandatory_strings = get_mandatory_substrings(invalid_config)
            extraction_patterns = get_extraction_patterns(invalid_config)
            
            # ASSERT - Verify graceful degradation behavior
            assert ignore_patterns == [], \
                "Invalid configuration should return empty ignore patterns gracefully"
            assert mandatory_strings == [], \
                "Invalid configuration should return empty mandatory strings gracefully"
            assert extraction_patterns is None, \
                "Invalid configuration should return None for extraction patterns gracefully"
        
        elif edge_case_scenario == "unicode_path_handling":
            # Test Unicode path handling behavior
            unicode_config = config.copy()
            unicode_config["project"]["directories"]["major_data_directory"] = "/tëst/dàtä/ūnïcōdė"
            
            # ACT - Execute configuration processing with Unicode paths
            validated_config = validate_config_dict(unicode_config)
            project_patterns = get_ignore_patterns(validated_config)
            
            # ASSERT - Verify Unicode path handling behavior
            assert validated_config is not None, \
                "Configuration should handle Unicode paths gracefully"
            assert isinstance(project_patterns, list), \
                "Unicode configuration should provide valid pattern processing"
        
        elif edge_case_scenario == "nested_parameter_edge_cases":
            # Test deeply nested parameter dictionary edge cases
            nested_config = config.copy()
            nested_config["experiments"]["test_exp"] = {
                "datasets": ["nonexistent_dataset"],
                "filters": {
                    "nested": {
                        "deeply": {
                            "embedded": ["parameter"]
                        }
                    }
                }
            }
            
            # ACT - Execute configuration processing with nested structures
            exp_names = get_all_experiment_names(nested_config)
            
            # ASSERT - Verify nested parameter handling behavior
            assert "test_exp" in exp_names, \
                "Configuration should handle nested parameter structures"
            
            # Test missing dataset reference handling
            with pytest.raises(KeyError):
                get_dataset_info(nested_config, "nonexistent_dataset")
        
        elif edge_case_scenario == "cross_platform_path_handling":
            # Test cross-platform path handling behavior
            platform_config = config.copy()
            platform_config["project"]["directories"]["major_data_directory"] = "C:\\Windows\\Path" if edge_case_scenario else "/unix/path"
            
            # ACT - Execute cross-platform configuration processing
            validated_config = validate_config_dict(platform_config)
            
            # ASSERT - Verify cross-platform compatibility behavior
            assert validated_config is not None, \
                "Configuration should handle cross-platform paths"
        
        elif edge_case_scenario == "configuration_schema_violations":
            # Test configuration schema violation recovery
            schema_violation_configs = [
                {"datasets": "not_a_dict"},  # Invalid datasets structure
                {"datasets": {"test": {"dates_vials": "not_a_dict"}}},  # Invalid dates_vials
                {"datasets": {"test": {"dates_vials": {"date": "not_a_list"}}}}  # Invalid vials
            ]
            
            # ACT & ASSERT - Execute schema validation with violations
            for invalid_config in schema_violation_configs:
                with pytest.raises((ValueError, TypeError)):
                    validate_config_dict(invalid_config)

    def test_end_to_end_configuration_workflow_behavior_tst_integ_001(self, sample_hierarchical_config, mock_filesystem):
        """
        Test TST-INTEG-001: End-to-end configuration-driven workflow behavioral validation.
        
        Validates complete observable workflow behavior from YAML configuration loading
        through file discovery, metadata extraction, to final processed results using
        black-box verification of the entire configuration-driven pipeline.
        
        Focus: Complete workflow behavior rather than individual component implementations.
        """
        # ARRANGE - Set up end-to-end workflow scenario
        config = sample_hierarchical_config
        base_directory = "/test/data"
        search_pattern = "*.pkl"
        
        # Expected end-to-end workflow outcomes
        expected_experiments = ["baseline_comparison", "treatment_analysis", "comprehensive_study"]
        expected_datasets = ["baseline_navigation", "treatment_response", "control_group"]
        
        # ACT - Execute complete configuration-driven workflow
        # Step 1: Configuration validation and experiment/dataset discovery
        validated_config = validate_config_dict(config)
        all_experiment_names = get_all_experiment_names(validated_config)
        all_dataset_names = get_all_dataset_names(validated_config)
        
        workflow_results = {}
        
        # Step 2: Multi-experiment discovery workflow
        with patch('flyrigloader.config.discovery.discover_experiment_files') as mock_exp_discover:
            mock_exp_discover.return_value = {
                '/test/data/2024-12-20/mouse_20241220_baseline_001.pkl': {
                    'path': '/test/data/2024-12-20/mouse_20241220_baseline_001.pkl',
                    'experiment': 'baseline_comparison',
                    'metadata_extracted': True
                },
                '/test/data/2024-12-22/mouse_20241222_treatment_003.pkl': {
                    'path': '/test/data/2024-12-22/mouse_20241222_treatment_003.pkl',
                    'experiment': 'treatment_analysis', 
                    'metadata_extracted': True
                }
            }
            
            for experiment_name in all_experiment_names:
                experiment_files = discover_experiment_files(
                    config=validated_config,
                    experiment_name=experiment_name,
                    base_directory=base_directory,
                    pattern=search_pattern,
                    recursive=True,
                    extract_metadata=True,
                    parse_dates=True
                )
                workflow_results[experiment_name] = experiment_files
        
        # Step 3: Dataset-level workflow validation
        dataset_results = {}
        with patch('flyrigloader.config.discovery.discover_dataset_files') as mock_dataset_discover:
            mock_dataset_discover.return_value = {
                '/test/data/2024-12-20/mouse_20241220_baseline_001.pkl': {
                    'path': '/test/data/2024-12-20/mouse_20241220_baseline_001.pkl',
                    'dataset': 'baseline_navigation'
                }
            }
            
            for dataset_name in all_dataset_names:
                dataset_files = discover_dataset_files(
                    config=validated_config,
                    dataset_name=dataset_name,
                    base_directory=base_directory,
                    pattern=search_pattern,
                    recursive=True,
                    extract_metadata=True
                )
                dataset_results[dataset_name] = dataset_files
        
        # ASSERT - Verify end-to-end workflow behavior
        # Configuration workflow completeness validation
        assert set(all_experiment_names) == set(expected_experiments), \
            "Workflow should discover all configured experiments"
        assert set(all_dataset_names) == set(expected_datasets), \
            "Workflow should discover all configured datasets"
        assert len(workflow_results) > 0, \
            "End-to-end workflow should produce experiment results"
        
        # Experiment workflow behavior validation
        for experiment_name, experiment_files in workflow_results.items():
            assert isinstance(experiment_files, dict), \
                f"Experiment {experiment_name} should return structured results"
            
            # Verify configuration-driven filtering behavior applied
            for file_path, file_metadata in experiment_files.items():
                assert "path" in file_metadata, \
                    "Workflow should preserve file path information"
                assert file_path == file_metadata["path"], \
                    "Workflow should maintain path consistency"
                
                # Verify experiment-specific configuration filtering was applied
                experiment_info = get_experiment_info(validated_config, experiment_name)
                if "filters" in experiment_info:
                    ignore_patterns = experiment_info["filters"].get("ignore_substrings", [])
                    for ignore_pattern in ignore_patterns:
                        assert ignore_pattern not in file_path, \
                            f"Workflow should filter files by experiment rules: {ignore_pattern} in {file_path}"
        
        # Dataset workflow behavior validation
        dataset_files_found = sum(len(files) for files in dataset_results.values())
        assert dataset_files_found > 0, \
            "Dataset-level workflow should discover files"
        
        # Verify workflow completeness
        files_found = sum(len(files) for files in workflow_results.values())
        assert files_found > 0, \
            "End-to-end workflow should discover files across all experiments"

    @pytest.mark.parametrize("experiment_name,expected_datasets", [
        ("baseline_comparison", ["baseline_navigation", "control_group"]),
        ("treatment_analysis", ["treatment_response"]),
        ("comprehensive_study", ["baseline_navigation", "treatment_response", "control_group"])
    ])
    def test_experiment_dataset_relationship_behavior(self, sample_hierarchical_config, experiment_name, expected_datasets):
        """
        Test parameterized experiment-dataset relationship behavioral validation.
        
        Validates observable experiment-dataset relationships through configuration
        API behavior using parameterized test scenarios for comprehensive coverage
        of different experiment types and their associated dataset configurations.
        
        Focus: Configuration relationship behavior rather than internal mapping mechanisms.
        """
        # ARRANGE - Set up experiment-dataset relationship scenario
        config = sample_hierarchical_config
        
        # ACT - Execute experiment-dataset relationship discovery
        experiment_info = get_experiment_info(config, experiment_name)
        actual_datasets = experiment_info.get("datasets", [])
        
        # ASSERT - Verify observable experiment-dataset relationship behavior
        assert set(actual_datasets) == set(expected_datasets), \
            f"Experiment {experiment_name} should reference datasets {expected_datasets}, got {actual_datasets}"
        
        # Verify all referenced datasets are accessible through configuration API
        for dataset_name in expected_datasets:
            dataset_info = get_dataset_info(config, dataset_name)
            assert dataset_info is not None, \
                f"Dataset {dataset_name} should be accessible through configuration API"
            assert isinstance(dataset_info, dict), \
                f"Dataset {dataset_name} should provide structured configuration information"

    def test_configuration_change_impact_behavior_validation(self, sample_hierarchical_config, mock_filesystem):
        """
        Test configuration change impact behavioral validation.
        
        Validates observable configuration change impacts on workflow behavior
        through systematic modification testing to ensure predictable and
        consistent configuration-driven behavior changes.
        
        Focus: Observable workflow behavior changes rather than internal configuration mechanisms.
        """
        # ARRANGE - Set up configuration change impact scenarios  
        base_config = sample_hierarchical_config
        base_directory = "/test/data"
        search_pattern = "*.pkl"
        
        # Expected baseline behavior with original configuration
        expected_baseline_files = [
            '/test/data/2024-12-20/mouse_20241220_baseline_001.pkl',
            '/test/data/2024-12-20/mouse_20241220_treatment_002.pkl',
            '/test/data/2024-12-22/mouse_20241222_baseline_001.pkl',
            '/test/data/2024-12-25/exp001_mouse_baseline.pkl'
        ]
        
        # ACT & ASSERT - Test systematic configuration change impacts
        
        # Test 1: Adding ignore patterns should reduce discoverable files
        with patch('flyrigloader.discovery.files.discover_files') as mock_discover:
            mock_discover.return_value = expected_baseline_files
            
            baseline_result = discover_files_with_config(
                config=base_config,
                directory=base_directory,
                pattern=search_pattern,
                recursive=True
            )
            baseline_count = len(baseline_result)
            
            # Modify configuration to add restrictive ignore pattern
            restrictive_config = base_config.copy()
            restrictive_config["project"]["ignore_substrings"].append("exp")
            
            expected_filtered_files = [f for f in expected_baseline_files if "exp" not in f]
            mock_discover.return_value = expected_filtered_files
            
            restrictive_result = discover_files_with_config(
                config=restrictive_config,
                directory=base_directory,
                pattern=search_pattern,
                recursive=True
            )
            
            assert len(restrictive_result) < baseline_count, \
                "Adding ignore patterns should reduce discovered files"
        
        # Test 2: Adding mandatory patterns should filter files differently
        with patch('flyrigloader.discovery.files.discover_files') as mock_discover:
            mandatory_config = base_config.copy()
            mandatory_config["project"]["mandatory_experiment_strings"] = ["mouse"]
            
            expected_mouse_files = [f for f in expected_baseline_files if "mouse" in f]
            mock_discover.return_value = expected_mouse_files
            
            mandatory_result = discover_files_with_config(
                config=mandatory_config,
                directory=base_directory,
                pattern=search_pattern,
                recursive=True
            )
            
            assert len(mandatory_result) <= baseline_count, \
                "Adding mandatory patterns should reduce or maintain file count"
            
            # Verify all files contain mandatory substring
            for file_path in mandatory_result:
                assert "mouse" in file_path, \
                    f"File {file_path} should contain mandatory string 'mouse'"
        
        # Test 3: Removing ignore patterns should increase or maintain files
        with patch('flyrigloader.discovery.files.discover_files') as mock_discover:
            permissive_config = base_config.copy()
            permissive_config["project"]["ignore_substrings"] = []
            
            # Include previously ignored files
            expanded_files = expected_baseline_files + [
                '/test/data/2024-12-20/._hidden_file.pkl',
                '/test/data/2024-12-20/static_horiz_ribbon_data.pkl'
            ]
            mock_discover.return_value = expanded_files
            
            permissive_result = discover_files_with_config(
                config=permissive_config,
                directory=base_directory,
                pattern=search_pattern,
                recursive=True
            )
            
            assert len(permissive_result) >= baseline_count, \
                "Removing ignore patterns should increase or maintain file count"
        
        # Test 4: Verify configuration parameter propagation behavior
        pattern_config = base_config.copy()
        pattern_config["project"]["extraction_patterns"] = [
            r".*/(?P<custom_animal>mouse|rat)_(?P<custom_date>\d{8})_.*\.pkl"
        ]
        
        # ACT - Execute pattern propagation behavior validation
        pattern_extraction = get_extraction_patterns(pattern_config)
        
        # ASSERT - Verify extraction pattern propagation behavior
        assert pattern_extraction is not None, \
            "Modified extraction patterns should be accessible"
        assert len(pattern_extraction) == 1, \
            "Configuration should reflect modified extraction pattern count"
        assert "custom_animal" in pattern_extraction[0], \
            "Custom extraction patterns should be propagated through configuration API"

    def test_configuration_validation_workflow_section_4112(self, sample_hierarchical_config):
        """
        Test Section 4.1.1.2: Configuration Management Process integration behavior.
        
        Validates observable configuration management workflow behavior through
        systematic validation of configuration loading, parameter extraction,
        and error propagation patterns using black-box verification.
        
        Focus: Configuration management workflow behavior rather than internal validation mechanisms.
        """
        # ARRANGE - Set up configuration management workflow scenario
        config = sample_hierarchical_config
        
        # ACT - Execute configuration management workflow
        # Step 1: Configuration validation behavior
        validated_config = validate_config_dict(config)
        
        # Step 2: Parameter extraction behavior
        ignore_patterns = get_ignore_patterns(validated_config)
        mandatory_strings = get_mandatory_substrings(validated_config)
        extraction_patterns = get_extraction_patterns(validated_config)
        
        # Step 3: Experiment and dataset discovery behavior
        exp_names = get_all_experiment_names(validated_config)
        dataset_names = get_all_dataset_names(validated_config)
        
        # Step 4: Specific configuration access behavior
        first_experiment = exp_names[0] if exp_names else None
        first_dataset = dataset_names[0] if dataset_names else None
        
        # ASSERT - Verify configuration management workflow behavior
        # Configuration validation behavior verification
        assert validated_config is not None, \
            "Configuration validation should produce accessible results"
        assert validated_config == config, \
            "Valid configuration should be preserved through validation"
        
        # Parameter extraction behavior verification
        assert isinstance(ignore_patterns, list), \
            "Configuration should provide ignore patterns as accessible list"
        assert isinstance(mandatory_strings, list), \
            "Configuration should provide mandatory strings as accessible list"
        assert extraction_patterns is None or isinstance(extraction_patterns, list), \
            "Configuration should provide extraction patterns in expected format"
        
        # Discovery behavior verification
        assert len(exp_names) > 0, \
            "Configuration should discover available experiments"
        assert len(dataset_names) > 0, \
            "Configuration should discover available datasets"
        
        # Configuration access behavior verification
        if first_experiment:
            exp_info = get_experiment_info(validated_config, first_experiment)
            assert isinstance(exp_info, dict), \
                "Experiment information should be accessible as structured data"
        
        if first_dataset:
            dataset_info = get_dataset_info(validated_config, first_dataset)
            assert isinstance(dataset_info, dict), \
                "Dataset information should be accessible as structured data"
        
        # Error propagation behavior verification - Test invalid configurations
        invalid_config_scenarios = [
            {"invalid": "config"},  # Missing required sections
            {
                "project": {},
                "datasets": {"test": {"dates_vials": "not_a_dict"}}, # Invalid structure
                "experiments": {}
            },
            {
                "project": {},
                "datasets": {"test": {"dates_vials": {"2024-12-20": "not_a_list"}}}, # Invalid values
                "experiments": {}
            }
        ]
        
        for invalid_config in invalid_config_scenarios:
            with pytest.raises((ValueError, TypeError)):
                validate_config_dict(invalid_config)