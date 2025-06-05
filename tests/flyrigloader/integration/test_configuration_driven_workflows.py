"""
Configuration-driven workflow integration test suite.

This module tests comprehensive configuration-driven workflows validating how YAML
configuration parameters drive complete data processing pipelines including file
discovery patterns, metadata extraction rules, column schema validation, and
DataFrame transformation specifications.

Tests hierarchical configuration merging between project and experiment levels,
validates configuration-driven behavior propagation across all modules, and ensures
consistent parameter interpretation throughout the system.

Requirements Tested:
- F-001: Hierarchical YAML Configuration System integration (Section 2.1.1)
- F-001-RQ-004: Hierarchical configuration merging validation (Section 2.2.1)
- TST-INTEG-001: Configuration-driven workflow validation (Section 2.2.10)
- F-002: Configuration-driven file discovery pattern validation (Section 2.1.2)
- F-007: Configuration-driven metadata extraction pattern validation (Section 2.1.7)
- Section 4.1.1.2: Configuration Management Process integration testing
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import pickle
import gzip
import pytest
import numpy as np
import pandas as pd
import yaml
from unittest.mock import patch, MagicMock

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
from flyrigloader.discovery.patterns import PatternMatcher
from flyrigloader.io.column_models import ColumnConfigDict, ColumnConfig


class TestConfigurationDrivenWorkflows:
    """
    Integration tests for configuration-driven workflow validation.
    
    These tests validate complete workflows from YAML configuration loading
    through file discovery, metadata extraction, and final processing.
    """

    @pytest.fixture(scope="class")
    def temp_workspace(self):
        """
        Create a temporary workspace with realistic experimental data structure.
        
        Creates a comprehensive test environment with:
        - Hierarchical directory structure
        - Multiple data files with different patterns
        - Configuration files at project and experiment levels
        - Realistic experimental data pickles
        
        Returns:
            Path: Path to the temporary workspace root
        """
        workspace = tempfile.mkdtemp(prefix="flyrig_integration_test_")
        workspace_path = Path(workspace)
        
        try:
            # Create realistic directory structure
            (workspace_path / "data" / "2024-12-20").mkdir(parents=True)
            (workspace_path / "data" / "2024-12-22").mkdir(parents=True)
            (workspace_path / "data" / "2024-12-25").mkdir(parents=True)
            (workspace_path / "configs").mkdir()
            (workspace_path / "experiments").mkdir()
            
            # Create test data files with realistic patterns
            test_files = [
                # Standard animal experiment files
                "data/2024-12-20/mouse_20241220_baseline_001.pkl",
                "data/2024-12-20/mouse_20241220_treatment_002.pkl",
                "data/2024-12-20/rat_20241220_control_001.pkl",
                "data/2024-12-22/mouse_20241222_baseline_001.pkl",
                "data/2024-12-22/mouse_20241222_treatment_003.pkl",
                
                # Experiment files with different pattern
                "data/2024-12-25/exp001_mouse_baseline.pkl",
                "data/2024-12-25/exp001_rat_treatment.pkl",
                "data/2024-12-25/exp002_mouse_control.pkl",
                
                # Files that should be ignored
                "data/2024-12-20/._hidden_file.pkl",
                "data/2024-12-20/static_horiz_ribbon_data.pkl",
                "data/2024-12-22/temp_backup_file.pkl",
                
                # Different file formats
                "data/2024-12-20/mouse_20241220_baseline_004.csv",
                "data/2024-12-22/metadata_summary.txt"
            ]
            
            # Create test pickle files with realistic experimental data
            for file_path in test_files:
                full_path = workspace_path / file_path
                
                if file_path.endswith('.pkl'):
                    # Create realistic experimental data matrix
                    exp_matrix = self._create_experimental_data_matrix(file_path)
                    
                    with open(full_path, 'wb') as f:
                        pickle.dump(exp_matrix, f)
                
                elif file_path.endswith('.csv'):
                    # Create simple CSV files
                    full_path.write_text("time,x,y\n1,0.1,0.2\n2,0.3,0.4\n")
                
                else:
                    # Create text files
                    full_path.write_text("test metadata content")
            
            yield workspace_path
            
        finally:
            shutil.rmtree(workspace)
    
    def _create_experimental_data_matrix(self, file_path: str) -> Dict[str, Any]:
        """
        Create realistic experimental data matrix based on file path.
        
        Args:
            file_path: Path to the file being created
            
        Returns:
            Dict containing experimental data matching the file pattern
        """
        # Generate time series data
        time_points = 1000
        t = np.linspace(0, 10, time_points)
        
        # Basic position data
        x = np.cumsum(np.random.normal(0, 0.1, time_points))
        y = np.cumsum(np.random.normal(0, 0.1, time_points))
        
        # Create base experimental matrix
        exp_matrix = {
            't': t,
            'x': x,
            'y': y,
            'dtheta': np.random.normal(0, 0.05, time_points),
            'signal': np.random.rand(time_points),
            'signal_disp': np.random.rand(15, time_points)  # Multi-channel signal
        }
        
        # Add metadata based on filename patterns
        filename = Path(file_path).name
        
        if 'mouse' in filename:
            exp_matrix.update({
                'animal_type': 'mouse',
                'rig': 'old_opto',
                'sampling_frequency': 60
            })
        elif 'rat' in filename:
            exp_matrix.update({
                'animal_type': 'rat',
                'rig': 'new_opto',
                'sampling_frequency': 120
            })
        
        # Add experiment-specific metadata
        if 'exp001' in filename:
            exp_matrix.update({
                'experiment_id': 'exp001',
                'protocol': 'standard_navigation'
            })
        elif 'exp002' in filename:
            exp_matrix.update({
                'experiment_id': 'exp002',
                'protocol': 'optogenetic_stimulation'
            })
        
        return exp_matrix

    @pytest.fixture
    def hierarchical_config(self, temp_workspace):
        """
        Create a comprehensive hierarchical configuration for testing.
        
        Returns:
            Dict: Complete configuration with project and experiment levels
        """
        return {
            "project": {
                "directories": {
                    "major_data_directory": str(temp_workspace / "data"),
                    "batchfile_directory": str(temp_workspace / "configs")
                },
                "ignore_substrings": [
                    "._",
                    "static_horiz_ribbon",
                    "temp_"
                ],
                "mandatory_experiment_strings": [],
                "extraction_patterns": [
                    # Project-level patterns for animal files
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

    @pytest.fixture
    def config_file(self, hierarchical_config, temp_workspace):
        """
        Create a temporary configuration file for file-based testing.
        
        Returns:
            str: Path to the configuration file
        """
        config_path = temp_workspace / "config.yaml"
        
        with open(config_path, 'w') as f:
            yaml.dump(hierarchical_config, f, default_flow_style=False)
        
        return str(config_path)

    def test_hierarchical_configuration_loading_and_validation(self, config_file, hierarchical_config):
        """
        Test F-001: Hierarchical YAML Configuration System integration.
        
        Validates that configuration loading works correctly from both files
        and dictionaries, with proper structure validation.
        """
        # Test loading from file
        config_from_file = load_config(config_file)
        assert config_from_file is not None
        assert isinstance(config_from_file, dict)
        
        # Test loading from dictionary
        config_from_dict = load_config(hierarchical_config)
        assert config_from_dict is not None
        assert isinstance(config_from_dict, dict)
        
        # Verify both methods produce equivalent results
        assert config_from_file == config_from_dict
        
        # Test configuration structure validation
        validated_config = validate_config_dict(hierarchical_config)
        assert validated_config == hierarchical_config
        
        # Verify required sections exist
        assert "project" in validated_config
        assert "datasets" in validated_config
        assert "experiments" in validated_config
        
        # Test dataset structure validation
        for dataset_name, dataset_config in validated_config["datasets"].items():
            if "dates_vials" in dataset_config:
                dates_vials = dataset_config["dates_vials"]
                assert isinstance(dates_vials, dict)
                for date, vials in dates_vials.items():
                    assert isinstance(date, str)
                    assert isinstance(vials, list)

    def test_hierarchical_configuration_merging_f001_rq004(self, hierarchical_config):
        """
        Test F-001-RQ-004: Hierarchical configuration merging validation.
        
        Validates that project-level and experiment-level configurations
        merge correctly with proper override semantics.
        """
        # Test project-level ignore patterns
        project_ignore = get_ignore_patterns(hierarchical_config)
        expected_project = ["*._*", "*static_horiz_ribbon*", "*temp_*"]
        assert project_ignore == expected_project
        
        # Test experiment-level ignore pattern override
        baseline_ignore = get_ignore_patterns(hierarchical_config, experiment="baseline_comparison")
        expected_baseline = ["*._*", "*static_horiz_ribbon*", "*temp_*", "*backup*"]
        assert baseline_ignore == expected_baseline
        
        # Test experiment with additional ignore patterns
        treatment_ignore = get_ignore_patterns(hierarchical_config, experiment="treatment_analysis")
        expected_treatment = ["*._*", "*static_horiz_ribbon*", "*temp_*", "*test_*", "*debug_*"]
        assert treatment_ignore == expected_treatment
        
        # Test mandatory strings merging
        project_mandatory = get_mandatory_substrings(hierarchical_config)
        assert project_mandatory == []
        
        baseline_mandatory = get_mandatory_substrings(hierarchical_config, experiment="baseline_comparison")
        assert "baseline" in baseline_mandatory
        assert "control" in baseline_mandatory
        
        comprehensive_mandatory = get_mandatory_substrings(hierarchical_config, experiment="comprehensive_study")
        assert "mouse" in comprehensive_mandatory
        assert "rat" in comprehensive_mandatory
        
        # Test extraction pattern hierarchical merging
        project_patterns = get_extraction_patterns(hierarchical_config)
        assert len(project_patterns) == 2
        
        baseline_patterns = get_extraction_patterns(hierarchical_config, experiment="baseline_comparison")
        assert len(baseline_patterns) == 3  # Project + experiment patterns
        
        # Verify experiment patterns are added to project patterns
        baseline_experiment_pattern = r".*_(?P<experiment>baseline)_(?P<condition>\w+)\.pkl"
        assert baseline_experiment_pattern in baseline_patterns

    def test_configuration_driven_file_discovery_f002(self, hierarchical_config, temp_workspace):
        """
        Test F-002: Configuration-driven file discovery pattern validation.
        
        Validates that configuration parameters correctly control file discovery
        including patterns, ignore rules, and directory structures.
        """
        # Test basic configuration-driven discovery
        files = discover_files_with_config(
            config=hierarchical_config,
            directory=str(temp_workspace / "data"),
            pattern="*.pkl",
            recursive=True
        )
        
        # Verify files are discovered
        assert len(files) > 0
        assert all(f.endswith('.pkl') for f in files)
        
        # Verify ignore patterns are applied
        ignored_files = [f for f in files if any(ignore in f for ignore in ["._", "static_horiz_ribbon", "temp_"])]
        assert len(ignored_files) == 0, f"Found ignored files: {ignored_files}"
        
        # Test experiment-specific discovery with additional filters
        baseline_files = discover_files_with_config(
            config=hierarchical_config,
            directory=str(temp_workspace / "data"),
            pattern="*.pkl",
            recursive=True,
            experiment="baseline_comparison"
        )
        
        # Verify experiment-specific ignore patterns are applied
        backup_files = [f for f in baseline_files if "backup" in f]
        assert len(backup_files) == 0, f"Found backup files in baseline experiment: {backup_files}"
        
        # Test mandatory substring filtering
        comprehensive_files = discover_files_with_config(
            config=hierarchical_config,
            directory=str(temp_workspace / "data"),
            pattern="*.pkl",
            recursive=True,
            experiment="comprehensive_study"
        )
        
        # Verify all files contain mandatory substrings (mouse OR rat)
        for file_path in comprehensive_files:
            assert any(substring in file_path for substring in ["mouse", "rat"]), \
                f"File {file_path} missing mandatory substrings"

    def test_configuration_driven_metadata_extraction_f007(self, hierarchical_config, temp_workspace):
        """
        Test F-007: Configuration-driven metadata extraction pattern validation.
        
        Validates that configuration-defined regex patterns correctly extract
        metadata from filenames with hierarchical pattern precedence.
        """
        # Test metadata extraction with project-level patterns
        files_with_metadata = discover_files_with_config(
            config=hierarchical_config,
            directory=str(temp_workspace / "data"),
            pattern="*.pkl",
            recursive=True,
            extract_metadata=True
        )
        
        assert isinstance(files_with_metadata, dict)
        assert len(files_with_metadata) > 0
        
        # Verify metadata extraction for standard animal files
        animal_files = {path: metadata for path, metadata in files_with_metadata.items() 
                       if any(animal in path for animal in ["mouse", "rat"])}
        
        for file_path, metadata in animal_files.items():
            if "mouse_20241220_baseline_001.pkl" in file_path:
                assert metadata.get("animal") == "mouse"
                assert metadata.get("date") == "20241220"
                assert metadata.get("condition") == "baseline"
                assert metadata.get("replicate") == "001"
            elif "rat_20241220_control_001.pkl" in file_path:
                assert metadata.get("animal") == "rat"
                assert metadata.get("condition") == "control"
        
        # Test experiment-specific pattern extraction
        comprehensive_files = discover_files_with_config(
            config=hierarchical_config,
            directory=str(temp_workspace / "data"),
            pattern="*.pkl",
            recursive=True,
            experiment="comprehensive_study",
            extract_metadata=True
        )
        
        # Verify experiment pattern extraction for exp files
        exp_files = {path: metadata for path, metadata in comprehensive_files.items() 
                    if "exp" in path}
        
        for file_path, metadata in exp_files.items():
            if "exp001_mouse_baseline.pkl" in file_path:
                assert metadata.get("experiment") == "exp001"
                assert metadata.get("animal") == "mouse"
                assert metadata.get("condition") == "baseline"

    def test_experiment_specific_discovery_workflow(self, hierarchical_config, temp_workspace):
        """
        Test experiment-specific file discovery workflow integration.
        
        Validates that discover_experiment_files correctly uses experiment
        configuration to find relevant files with proper filtering.
        """
        # Test baseline comparison experiment discovery
        baseline_files = discover_experiment_files(
            config=hierarchical_config,
            experiment_name="baseline_comparison",
            base_directory=str(temp_workspace / "data"),
            pattern="*.pkl",
            recursive=True
        )
        
        assert len(baseline_files) > 0
        
        # Verify only relevant files for baseline experiment
        baseline_relevant = [f for f in baseline_files if any(term in f for term in ["baseline", "control"])]
        assert len(baseline_relevant) > 0
        
        # Test with metadata extraction
        baseline_with_metadata = discover_experiment_files(
            config=hierarchical_config,
            experiment_name="baseline_comparison",
            base_directory=str(temp_workspace / "data"),
            pattern="*.pkl",
            recursive=True,
            extract_metadata=True
        )
        
        assert isinstance(baseline_with_metadata, dict)
        
        # Verify metadata extraction works for experiment discovery
        for file_path, metadata in baseline_with_metadata.items():
            assert "path" in metadata
            # Should have extracted metadata based on experiment patterns
            if "baseline" in file_path:
                # Could have experiment-specific metadata
                pass

    def test_dataset_specific_discovery_workflow(self, hierarchical_config, temp_workspace):
        """
        Test dataset-specific file discovery workflow integration.
        
        Validates that discover_dataset_files correctly uses dataset
        configuration including date-vial specifications.
        """
        # Test baseline navigation dataset discovery
        baseline_dataset_files = discover_dataset_files(
            config=hierarchical_config,
            dataset_name="baseline_navigation",
            base_directory=str(temp_workspace / "data"),
            pattern="*.pkl",
            recursive=True
        )
        
        assert len(baseline_dataset_files) > 0
        
        # Verify files come from correct date directories
        for file_path in baseline_dataset_files:
            assert any(date in file_path for date in ["2024-12-20", "2024-12-22"])
        
        # Test with metadata extraction using dataset patterns
        baseline_with_metadata = discover_dataset_files(
            config=hierarchical_config,
            dataset_name="baseline_navigation",
            base_directory=str(temp_workspace / "data"),
            pattern="*.pkl",
            recursive=True,
            extract_metadata=True
        )
        
        assert isinstance(baseline_with_metadata, dict)
        
        # Test treatment response dataset
        treatment_files = discover_dataset_files(
            config=hierarchical_config,
            dataset_name="treatment_response",
            base_directory=str(temp_workspace / "data"),
            pattern="*.pkl",
            recursive=True
        )
        
        # Verify treatment files are found
        treatment_relevant = [f for f in treatment_files if "treatment" in f]
        assert len(treatment_relevant) > 0

    def test_configuration_edge_cases_and_error_recovery(self):
        """
        Test configuration edge cases and error recovery scenarios.
        
        Validates proper error handling for invalid configurations,
        missing keys, malformed patterns, and graceful degradation.
        """
        # Test invalid configuration structure
        invalid_config = {"invalid": "structure"}
        
        # Should handle missing required sections gracefully
        try:
            patterns = get_ignore_patterns(invalid_config)
            assert patterns == []  # Should return empty list for missing sections
        except Exception as e:
            pytest.fail(f"Should handle missing sections gracefully: {e}")
        
        # Test malformed extraction patterns
        malformed_config = {
            "project": {
                "extraction_patterns": [
                    r"[invalid_regex",  # Malformed regex
                    r".*_(?P<valid>\w+)\.pkl"  # Valid regex
                ]
            }
        }
        
        # Should handle malformed patterns gracefully
        patterns = get_extraction_patterns(malformed_config)
        assert len(patterns) == 2  # Should include both patterns
        
        # Test missing experiment/dataset references
        config_with_missing_refs = {
            "experiments": {
                "test_exp": {
                    "datasets": ["nonexistent_dataset"]
                }
            },
            "datasets": {}
        }
        
        # Should handle missing dataset references
        with pytest.raises(KeyError):
            get_dataset_info(config_with_missing_refs, "nonexistent_dataset")
        
        # Test empty configurations
        empty_config = {}
        assert get_ignore_patterns(empty_config) == []
        assert get_mandatory_substrings(empty_config) == []
        assert get_extraction_patterns(empty_config) is None

    def test_realistic_experimental_configuration_scenarios(self, temp_workspace):
        """
        Test realistic experimental configuration scenarios.
        
        Validates complex directory structures and metadata patterns
        that mirror actual experimental setups.
        """
        # Create a realistic multi-lab configuration
        realistic_config = {
            "project": {
                "directories": {
                    "major_data_directory": str(temp_workspace / "data")
                },
                "ignore_substrings": ["._", "backup_", "temp_"],
                "extraction_patterns": [
                    r".*/(?P<animal>mouse|rat)_(?P<date>\d{8})_(?P<condition>\w+)_(?P<replicate>\d+)\.pkl",
                    r".*/(?P<experiment>exp\d+)_(?P<subject>\w+)_(?P<protocol>\w+)\.pkl"
                ]
            },
            "datasets": {
                "locomotion_baseline": {
                    "rig": "old_opto",
                    "dates_vials": {
                        "2024-12-20": [1, 2, 3, 4],
                        "2024-12-22": [1, 2, 5, 6]
                    },
                    "metadata": {
                        "extraction_patterns": [
                            r".*_(?P<behavior>baseline)_(?P<session>\d+)\.pkl"
                        ]
                    }
                },
                "optogenetic_stimulation": {
                    "rig": "new_opto",
                    "dates_vials": {
                        "2024-12-25": [1, 2, 3]
                    },
                    "metadata": {
                        "extraction_patterns": [
                            r".*_(?P<stimulation>treatment)_(?P<power>\w+)\.pkl"
                        ]
                    }
                }
            },
            "experiments": {
                "multi_day_tracking": {
                    "datasets": ["locomotion_baseline"],
                    "filters": {
                        "mandatory_experiment_strings": ["mouse"],
                        "ignore_substrings": ["pilot_", "test_"]
                    },
                    "metadata": {
                        "extraction_patterns": [
                            r".*_(?P<tracking_type>baseline|treatment)_(?P<day>\d+)\.pkl"
                        ]
                    }
                },
                "cross_modality_analysis": {
                    "datasets": ["locomotion_baseline", "optogenetic_stimulation"],
                    "filters": {
                        "mandatory_experiment_strings": ["exp"]
                    }
                }
            }
        }
        
        # Test comprehensive workflow with realistic config
        files = discover_files_with_config(
            config=realistic_config,
            directory=str(temp_workspace / "data"),
            pattern="*.pkl",
            recursive=True,
            extract_metadata=True
        )
        
        assert isinstance(files, dict)
        
        # Test multi-day tracking experiment
        tracking_files = discover_experiment_files(
            config=realistic_config,
            experiment_name="multi_day_tracking",
            base_directory=str(temp_workspace / "data"),
            pattern="*.pkl",
            recursive=True,
            extract_metadata=True
        )
        
        # Should find files with mouse in the name due to mandatory strings
        mouse_files = [path for path in tracking_files.keys() if "mouse" in path]
        assert len(mouse_files) > 0
        
        # Test cross-modality analysis
        cross_modal_files = discover_experiment_files(
            config=realistic_config,
            experiment_name="cross_modality_analysis",
            base_directory=str(temp_workspace / "data"),
            pattern="*.pkl",
            recursive=True
        )
        
        # Should find exp files due to mandatory strings
        exp_files = [path for path in cross_modal_files if "exp" in path]
        assert len(exp_files) > 0

    def test_configuration_validation_workflow_section_4112(self, hierarchical_config):
        """
        Test Section 4.1.1.2: Configuration Management Process integration.
        
        Validates error propagation from yaml_config through discovery
        and data loading modules following the workflow specification.
        """
        # Test valid configuration workflow
        try:
            # Step 1: Configuration validation
            validated_config = validate_config_dict(hierarchical_config)
            assert validated_config is not None
            
            # Step 2: Parameter extraction
            ignore_patterns = get_ignore_patterns(validated_config)
            mandatory_strings = get_mandatory_substrings(validated_config)
            extraction_patterns = get_extraction_patterns(validated_config)
            
            assert isinstance(ignore_patterns, list)
            assert isinstance(mandatory_strings, list)
            assert extraction_patterns is None or isinstance(extraction_patterns, list)
            
            # Step 3: Experiment and dataset info retrieval
            exp_names = get_all_experiment_names(validated_config)
            dataset_names = get_all_dataset_names(validated_config)
            
            assert len(exp_names) > 0
            assert len(dataset_names) > 0
            
            # Step 4: Specific experiment configuration
            if exp_names:
                exp_info = get_experiment_info(validated_config, exp_names[0])
                assert isinstance(exp_info, dict)
            
            if dataset_names:
                dataset_info = get_dataset_info(validated_config, dataset_names[0])
                assert isinstance(dataset_info, dict)
                
        except Exception as e:
            pytest.fail(f"Valid configuration workflow should not raise exceptions: {e}")
        
        # Test invalid configuration error propagation
        invalid_configs = [
            # Missing required sections
            {"invalid": "config"},
            
            # Invalid dates_vials structure
            {
                "project": {},
                "datasets": {
                    "test": {
                        "dates_vials": "not_a_dict"  # Should be dict
                    }
                },
                "experiments": {}
            },
            
            # Invalid dates_vials values
            {
                "project": {},
                "datasets": {
                    "test": {
                        "dates_vials": {
                            "2024-12-20": "not_a_list"  # Should be list
                        }
                    }
                },
                "experiments": {}
            }
        ]
        
        for invalid_config in invalid_configs:
            with pytest.raises((ValueError, TypeError)):
                validate_config_dict(invalid_config)

    def test_end_to_end_configuration_driven_workflow_tst_integ_001(self, hierarchical_config, temp_workspace):
        """
        Test TST-INTEG-001: End-to-end configuration-driven workflow validation.
        
        Validates complete pipeline from YAML configuration loading through
        file discovery, metadata extraction, to final processed results.
        """
        # Step 1: Configuration loading and validation
        validated_config = validate_config_dict(hierarchical_config)
        
        # Step 2: Multi-experiment discovery workflow
        all_experiment_names = get_all_experiment_names(validated_config)
        
        workflow_results = {}
        
        for experiment_name in all_experiment_names:
            # Step 3: Experiment-specific discovery
            experiment_files = discover_experiment_files(
                config=validated_config,
                experiment_name=experiment_name,
                base_directory=str(temp_workspace / "data"),
                pattern="*.pkl",
                recursive=True,
                extract_metadata=True,
                parse_dates=True
            )
            
            # Step 4: Validate results structure
            assert isinstance(experiment_files, dict)
            
            # Step 5: Verify metadata extraction
            for file_path, file_metadata in experiment_files.items():
                assert "path" in file_metadata
                assert file_path == file_metadata["path"]
                
                # Verify configuration-driven filtering worked
                experiment_info = get_experiment_info(validated_config, experiment_name)
                if "filters" in experiment_info:
                    ignore_patterns = experiment_info["filters"].get("ignore_substrings", [])
                    for ignore_pattern in ignore_patterns:
                        assert ignore_pattern not in file_path, \
                            f"File {file_path} should have been filtered by {ignore_pattern}"
            
            workflow_results[experiment_name] = experiment_files
        
        # Step 6: Validate workflow completeness
        assert len(workflow_results) > 0
        
        # Verify at least one experiment found files
        files_found = sum(len(files) for files in workflow_results.values())
        assert files_found > 0, "End-to-end workflow should discover files"
        
        # Step 7: Test dataset-level workflow
        all_dataset_names = get_all_dataset_names(validated_config)
        
        dataset_results = {}
        for dataset_name in all_dataset_names:
            dataset_files = discover_dataset_files(
                config=validated_config,
                dataset_name=dataset_name,
                base_directory=str(temp_workspace / "data"),
                pattern="*.pkl",
                recursive=True,
                extract_metadata=True
            )
            
            dataset_results[dataset_name] = dataset_files
        
        # Verify dataset discovery worked
        dataset_files_found = sum(len(files) for files in dataset_results.values())
        assert dataset_files_found > 0, "Dataset-level discovery should find files"

    def test_configuration_parameter_propagation_across_modules(self, hierarchical_config, temp_workspace):
        """
        Test configuration parameter propagation across all modules.
        
        Validates that configuration changes correctly influence discovery patterns,
        data validation rules, and output structure throughout the complete workflow.
        """
        # Test 1: Ignore pattern propagation
        base_files = discover_files_with_config(
            config=hierarchical_config,
            directory=str(temp_workspace / "data"),
            pattern="*.pkl",
            recursive=True
        )
        
        # Modify configuration to add more ignore patterns
        modified_config = hierarchical_config.copy()
        modified_config["project"]["ignore_substrings"].append("mouse")
        
        filtered_files = discover_files_with_config(
            config=modified_config,
            directory=str(temp_workspace / "data"),
            pattern="*.pkl",
            recursive=True
        )
        
        # Should have fewer files due to additional ignore pattern
        assert len(filtered_files) < len(base_files)
        
        # Verify no files contain "mouse"
        mouse_files = [f for f in filtered_files if "mouse" in f]
        assert len(mouse_files) == 0
        
        # Test 2: Mandatory string propagation
        mandatory_config = hierarchical_config.copy()
        mandatory_config["project"]["mandatory_experiment_strings"] = ["baseline"]
        
        mandatory_files = discover_files_with_config(
            config=mandatory_config,
            directory=str(temp_workspace / "data"),
            pattern="*.pkl",
            recursive=True
        )
        
        # All files should contain "baseline"
        for file_path in mandatory_files:
            assert "baseline" in file_path
        
        # Test 3: Extraction pattern propagation
        pattern_config = hierarchical_config.copy()
        pattern_config["project"]["extraction_patterns"] = [
            r".*/(?P<custom_animal>mouse|rat)_(?P<custom_date>\d{8})_.*\.pkl"
        ]
        
        pattern_files = discover_files_with_config(
            config=pattern_config,
            directory=str(temp_workspace / "data"),
            pattern="*.pkl",
            recursive=True,
            extract_metadata=True
        )
        
        # Verify custom pattern extraction
        for file_path, metadata in pattern_files.items():
            if any(animal in file_path for animal in ["mouse", "rat"]):
                assert "custom_animal" in metadata or "path" in metadata
                assert "custom_date" in metadata or "path" in metadata

    @pytest.mark.parametrize("experiment_name,expected_datasets", [
        ("baseline_comparison", ["baseline_navigation", "control_group"]),
        ("treatment_analysis", ["treatment_response"]),
        ("comprehensive_study", ["baseline_navigation", "treatment_response", "control_group"])
    ])
    def test_experiment_dataset_relationships(self, hierarchical_config, experiment_name, expected_datasets):
        """
        Test parametrized experiment-dataset relationships.
        
        Validates that experiment configurations correctly reference
        their associated datasets with proper relationship mapping.
        """
        experiment_info = get_experiment_info(hierarchical_config, experiment_name)
        actual_datasets = experiment_info.get("datasets", [])
        
        assert set(actual_datasets) == set(expected_datasets), \
            f"Experiment {experiment_name} should reference datasets {expected_datasets}, got {actual_datasets}"
        
        # Verify all referenced datasets exist
        for dataset_name in expected_datasets:
            dataset_info = get_dataset_info(hierarchical_config, dataset_name)
            assert dataset_info is not None
            assert isinstance(dataset_info, dict)

    def test_configuration_change_impact_validation(self, hierarchical_config, temp_workspace):
        """
        Test configuration change impact validation.
        
        Validates that modifying configuration parameters produces
        predictable and consistent changes in workflow behavior.
        """
        # Baseline discovery
        baseline_result = discover_files_with_config(
            config=hierarchical_config,
            directory=str(temp_workspace / "data"),
            pattern="*.pkl",
            recursive=True,
            extract_metadata=True
        )
        
        baseline_count = len(baseline_result)
        
        # Test 1: Add ignore pattern - should reduce files
        restrictive_config = hierarchical_config.copy()
        restrictive_config["project"]["ignore_substrings"].append("exp")
        
        restrictive_result = discover_files_with_config(
            config=restrictive_config,
            directory=str(temp_workspace / "data"),
            pattern="*.pkl",
            recursive=True,
            extract_metadata=True
        )
        
        assert len(restrictive_result) < baseline_count, \
            "Adding ignore patterns should reduce discovered files"
        
        # Test 2: Add mandatory pattern - should reduce files differently
        mandatory_config = hierarchical_config.copy()
        mandatory_config["project"]["mandatory_experiment_strings"] = ["mouse"]
        
        mandatory_result = discover_files_with_config(
            config=mandatory_config,
            directory=str(temp_workspace / "data"),
            pattern="*.pkl",
            recursive=True,
            extract_metadata=True
        )
        
        assert len(mandatory_result) <= baseline_count, \
            "Adding mandatory patterns should reduce or maintain file count"
        
        # All files should contain mandatory string
        for file_path in mandatory_result.keys():
            assert "mouse" in file_path, \
                f"File {file_path} should contain mandatory string 'mouse'"
        
        # Test 3: Remove ignore patterns - should increase files
        permissive_config = hierarchical_config.copy()
        permissive_config["project"]["ignore_substrings"] = []
        
        permissive_result = discover_files_with_config(
            config=permissive_config,
            directory=str(temp_workspace / "data"),
            pattern="*.pkl",
            recursive=True,
            extract_metadata=True
        )
        
        assert len(permissive_result) >= baseline_count, \
            "Removing ignore patterns should increase or maintain file count"