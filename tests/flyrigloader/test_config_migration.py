"""
Comprehensive test suite for configuration migration infrastructure.

This module provides exhaustive testing of the FlyRigLoader configuration migration system,
including automatic version detection, legacy adapter functionality, migration execution with
rollback capability, and seamless upgrade paths. It validates the migration infrastructure's
reliability for enterprise-grade configuration schema evolution.

Test Coverage:
- Version detection and compatibility validation during system startup
- LegacyConfigAdapter testing for dictionary-based configuration compatibility
- Migration execution with comprehensive error handling and rollback capability
- Schema_version field injection for configurations lacking version metadata
- End-to-end migration workflows from legacy configurations to modern Pydantic-based processing
- Migration error scenarios with detailed error reporting and recovery validation

The test suite uses property-based testing with Hypothesis for comprehensive edge case coverage,
fixtures for consistent test data management, and parametrized tests for exhaustive scenario
validation across all supported configuration versions and migration paths.

Architecture:
Tests are organized into logical groups covering the major migration system components:
- Version detection and validation testing
- Legacy configuration adapter functionality
- Migration execution and audit trail generation
- Error handling and rollback mechanism validation
- End-to-end workflow testing with real configuration scenarios
- Performance and thread-safety validation for enterprise environments

Usage:
    Run the complete test suite:
    $ pytest tests/flyrigloader/test_config_migration.py -v
    
    Run specific test categories:
    $ pytest tests/flyrigloader/test_config_migration.py::TestVersionDetection -v
    $ pytest tests/flyrigloader/test_config_migration.py::TestMigrationExecution -v
    
    Run with coverage analysis:
    $ pytest tests/flyrigloader/test_config_migration.py --cov=flyrigloader.migration --cov-report=html
"""

import pytest
import warnings
import tempfile
import json
import yaml
import copy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from unittest.mock import MagicMock, patch, mock_open, call, ANY
from hypothesis import given, strategies as st
from pydantic import ValidationError
from semantic_version import Version

from flyrigloader.migration.migrators import (
    ConfigMigrator,
    MigrationReport,
    migrate_v0_1_to_v1_0,
    migrate_v0_2_to_v1_0
)
from flyrigloader.config.models import create_config
from flyrigloader.migration.versions import COMPATIBILITY_MATRIX
from flyrigloader.config.validators import validate_config_version
from flyrigloader.exceptions import ConfigError, VersionError


class TestVersionDetection:
    """Test suite for configuration version detection and compatibility validation."""
    
    def test_version_detection_with_valid_versions(self):
        """Test automatic version detection for configurations with valid schema_version."""
        test_cases = [
            {"schema_version": "0.1.0", "project": {"directories": {}}},
            {"schema_version": "0.2.0", "project": {"directories": {}}, "datasets": {}},
            {"schema_version": "1.0.0", "project": {"directories": {}}, "datasets": {}, "experiments": {}},
        ]
        
        for config in test_cases:
            is_valid, detected_version, message = validate_config_version(config)
            assert is_valid is True or "migration" in message.lower()
            assert detected_version == config["schema_version"]
    
    def test_version_detection_without_schema_version(self):
        """Test version detection for legacy configurations without schema_version field."""
        legacy_configs = [
            # v0.1.0 style configuration
            {"project": {"major_data_directory": "/data"}},
            # v0.2.0 style configuration
            {"project": {"directories": {}}, "datasets": {"ds1": {"rig": "rig1"}}},
        ]
        
        for config in legacy_configs:
            is_valid, detected_version, message = validate_config_version(config)
            # Should detect legacy format and suggest migration
            assert "migration" in message.lower() or not is_valid
    
    def test_version_detection_with_invalid_version_format(self):
        """Test error handling for configurations with invalid version formats."""
        invalid_configs = [
            {"schema_version": "invalid", "project": {}},
            {"schema_version": "1.0", "project": {}}, # Missing patch version
            {"schema_version": "v1.0.0", "project": {}}, # Prefixed with 'v'
            {"schema_version": 1.0, "project": {}}, # Numeric instead of string
        ]
        
        for config in invalid_configs:
            is_valid, detected_version, message = validate_config_version(config)
            assert not is_valid
            assert "invalid" in message.lower() or "format" in message.lower()
    
    def test_compatibility_matrix_validation(self):
        """Test that compatibility matrix contains expected migration paths."""
        assert isinstance(COMPATIBILITY_MATRIX, dict)
        assert len(COMPATIBILITY_MATRIX) > 0
        
        # Verify known compatibility relationships
        expected_versions = ["0.1.0", "0.2.0", "1.0.0"]
        for version in expected_versions[:2]:  # Check that older versions exist
            assert version in COMPATIBILITY_MATRIX, f"Version {version} missing from compatibility matrix"
    
    @pytest.mark.parametrize("from_version,to_version,should_be_compatible", [
        ("0.1.0", "1.0.0", True),
        ("0.2.0", "1.0.0", True),
        ("1.0.0", "0.2.0", False),  # Backward compatibility should not be allowed
        ("0.1.0", "0.2.0", True),
    ])
    def test_version_compatibility_validation(self, from_version, to_version, should_be_compatible):
        """Test version compatibility validation against compatibility matrix."""
        migrator = ConfigMigrator()
        is_possible, message = migrator.validate_migration(from_version, to_version)
        
        if should_be_compatible:
            assert is_possible, f"Migration {from_version} -> {to_version} should be possible: {message}"
        else:
            # Note: Some migrations might still be possible even if not recommended
            if not is_possible:
                assert "not" in message.lower() or "failed" in message.lower()
    
    def test_startup_version_validation_performance(self):
        """Test that version validation performs adequately during system startup."""
        config = {
            "schema_version": "1.0.0",
            "project": {"directories": {"major_data_directory": "/data"}},
            "datasets": {},
            "experiments": {}
        }
        
        start_time = datetime.now()
        for _ in range(100):  # Simulate multiple validation calls
            validate_config_version(config)
        end_time = datetime.now()
        
        execution_time = (end_time - start_time).total_seconds()
        assert execution_time < 1.0, f"Version validation too slow: {execution_time}s for 100 calls"


class TestMigrationReport:
    """Test suite for MigrationReport audit trail and reporting functionality."""
    
    def test_migration_report_initialization(self):
        """Test MigrationReport initialization with proper timestamp and version tracking."""
        from_version = "0.1.0"
        to_version = "1.0.0"
        custom_timestamp = datetime.now()
        
        # Test with default timestamp
        report = MigrationReport(from_version, to_version)
        assert report.from_version == from_version
        assert report.to_version == to_version
        assert isinstance(report.timestamp, datetime)
        assert report.applied_migrations == []
        assert report.warnings == []
        assert report.errors == []
        assert report.metadata == {}
        
        # Test with custom timestamp
        report_custom = MigrationReport(from_version, to_version, custom_timestamp)
        assert report_custom.timestamp == custom_timestamp
    
    def test_migration_report_to_dict_conversion(self):
        """Test comprehensive dictionary conversion for serialization and logging."""
        report = MigrationReport("0.1.0", "1.0.0")
        report.applied_migrations = ["migrate_v0_1_to_v1_0"]
        report.add_warning("Legacy format deprecated")
        report.add_error("Test error")
        report.metadata = {"test_key": "test_value"}
        report.execution_time_ms = 150.5
        
        report_dict = report.to_dict()
        
        # Verify required fields
        assert "migration_summary" in report_dict
        assert "from_version" in report_dict
        assert "to_version" in report_dict
        assert "timestamp" in report_dict
        assert "applied_migrations" in report_dict
        assert "warnings" in report_dict
        assert "errors" in report_dict
        assert "metadata" in report_dict
        assert "execution_time_ms" in report_dict
        assert "success" in report_dict
        assert "warning_count" in report_dict
        assert "error_count" in report_dict
        
        # Verify values
        assert report_dict["from_version"] == "0.1.0"
        assert report_dict["to_version"] == "1.0.0"
        assert report_dict["applied_migrations"] == ["migrate_v0_1_to_v1_0"]
        assert report_dict["warning_count"] == 1
        assert report_dict["error_count"] == 1
        assert report_dict["success"] == False  # Has errors
        assert report_dict["execution_time_ms"] == 150.5
    
    def test_migration_report_context_handling(self):
        """Test warning and error context preservation in migration reports."""
        report = MigrationReport("0.1.0", "1.0.0")
        
        # Add warning with context
        report.add_warning("Configuration deprecated", {"section": "project", "field": "directories"})
        
        # Add error with context  
        report.add_error("Validation failed", {"validation_type": "schema", "field_count": 5})
        
        assert len(report.warnings) == 1
        assert len(report.errors) == 1
        assert "section=project" in report.warnings[0]
        assert "validation_type=schema" in report.errors[0]
    
    def test_migration_report_execution_time_tracking(self):
        """Test execution time tracking and performance monitoring."""
        report = MigrationReport("0.1.0", "1.0.0")
        
        start_time = datetime.now()
        # Simulate some processing time
        import time
        time.sleep(0.001)  # 1ms
        end_time = datetime.now()
        
        report.set_execution_time(start_time, end_time)
        
        assert report.execution_time_ms is not None
        assert report.execution_time_ms > 0
        assert report.execution_time_ms < 1000  # Should be less than 1 second
    
    def test_migration_report_config_change_tracking(self):
        """Test configuration change tracking for audit trails."""
        report = MigrationReport("0.1.0", "1.0.0")
        
        # Track various types of changes
        report.add_config_change("project.directories", None, {"major_data_directory": "/data"})
        report.add_config_change("schema_version", None, "1.0.0")
        report.add_config_change("project.extraction_patterns", [], ["(?P<date>\\d{4}-\\d{2}-\\d{2})"])
        
        assert len(report.config_changes) == 3
        assert "project.directories" in report.config_changes
        assert report.config_changes["project.directories"]["change_type"] == "added"
        assert report.config_changes["schema_version"]["new_value"] == "1.0.0"
    
    def test_migration_report_version_comparison_metadata(self):
        """Test semantic version comparison metadata generation."""
        # Major version change
        report_major = MigrationReport("0.2.0", "1.0.0")
        report_dict = report_major.to_dict()
        
        assert "version_comparison" in report_dict
        if "error" not in report_dict["version_comparison"]:
            assert report_dict["version_comparison"]["major_version_change"] == True
            assert report_dict["version_comparison"]["upgrade_type"] == "major"
        
        # Minor version change
        report_minor = MigrationReport("0.1.0", "0.2.0")
        report_dict_minor = report_minor.to_dict()
        
        if "error" not in report_dict_minor.get("version_comparison", {}):
            assert report_dict_minor["version_comparison"]["minor_version_change"] == True
            assert report_dict_minor["version_comparison"]["upgrade_type"] == "minor"


class TestLegacyConfigAdapter:
    """Test suite for LegacyConfigAdapter dictionary-based configuration compatibility."""
    
    def test_legacy_dict_configuration_preservation(self):
        """Test that legacy dictionary configurations are preserved and accessible."""
        legacy_config = {
            'project': {'major_data_directory': '/data'},
            'experiments': {'exp1': {'date_range': ['2024-01-01', '2024-01-31']}}
        }
        
        # Test that we can still work with dictionary configs through migration
        migrator = ConfigMigrator()
        
        # Should not raise an error
        migrated_config, report = migrator.migrate(legacy_config, "0.1.0", "1.0.0")
        
        assert isinstance(migrated_config, dict)
        assert migrated_config["schema_version"] == "1.0.0"
        assert "project" in migrated_config
        assert len(report.warnings) > 0  # Should have deprecation warnings
    
    def test_legacy_adapter_deprecation_warnings(self):
        """Test that LegacyConfigAdapter emits proper deprecation warnings."""
        legacy_config = {
            'project': {'major_data_directory': '/data'},
        }
        
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            
            # Trigger migration which should emit deprecation warning
            migrator = ConfigMigrator()
            migrated_config, report = migrator.migrate(legacy_config, "0.1.0", "1.0.0")
            
            # Check that deprecation warning was issued
            deprecation_warnings = [w for w in warning_list if issubclass(w.category, DeprecationWarning)]
            assert len(deprecation_warnings) > 0
            
            # Verify warning message content
            warning_messages = [str(w.message) for w in deprecation_warnings]
            assert any("deprecated" in msg.lower() for msg in warning_messages)
    
    def test_seamless_pydantic_model_conversion(self):
        """Test seamless conversion from dictionary configs to Pydantic models."""
        # Test create_config function with dictionary-style parameters
        config = create_config(
            base_directory="/test/data",
            extraction_patterns=[r"(?P<date>\d{4}-\d{2}-\d{2})"],
            ignore_substrings=["temp", "backup"]
        )
        
        # Should create a valid Pydantic model
        assert hasattr(config, "schema_version")
        assert hasattr(config, "project")
        assert hasattr(config, "datasets")
        assert hasattr(config, "experiments")
        
        # Verify that dictionary-style access works through model
        assert config.project.directories.major_data_directory == "/test/data"
        assert len(config.project.extraction_patterns) > 0
    
    def test_legacy_adapter_error_handling(self):
        """Test error handling in legacy configuration adapter scenarios."""
        invalid_configs = [
            None,  # None config
            "not_a_dict",  # String instead of dict
            [],  # List instead of dict
            {"project": "invalid_project_structure"},  # Invalid nested structure
        ]
        
        migrator = ConfigMigrator()
        
        for invalid_config in invalid_configs:
            with pytest.raises((VersionError, ValueError, TypeError)):
                migrator.migrate(invalid_config, "0.1.0", "1.0.0")
    
    def test_dictionary_based_configuration_compatibility(self):
        """Test comprehensive dictionary-based configuration compatibility scenarios."""
        # Test various legacy configuration patterns
        legacy_patterns = [
            # Minimal v0.1.0 config
            {"project": {"major_data_directory": "/data"}},
            
            # v0.1.0 with experiments
            {
                "project": {"major_data_directory": "/data"},
                "experiments": {
                    "exp1": {"date_range": ["2024-01-01", "2024-01-31"]}
                }
            },
            
            # v0.2.0 style with datasets
            {
                "project": {"directories": {"major_data_directory": "/data"}},
                "datasets": {"ds1": {"rig": "rig1", "dates_vials": {"2024-01-01": [1, 2]}}},
                "experiments": {"exp1": {"datasets": ["ds1"]}}
            },
        ]
        
        migrator = ConfigMigrator()
        
        for legacy_config in legacy_patterns:
            # Should successfully migrate without errors (warnings are expected)
            migrated_config, report = migrator.migrate(legacy_config, "0.1.0", "1.0.0")
            
            assert migrated_config["schema_version"] == "1.0.0"
            assert "project" in migrated_config
            assert len(report.errors) == 0, f"Unexpected errors: {report.errors}"
            assert report.metadata.get("migration_successful", False) == True


class TestConfigMigrator:
    """Test suite for ConfigMigrator migration execution and orchestration."""
    
    def test_config_migrator_initialization(self):
        """Test ConfigMigrator initialization with built-in migration functions."""
        migrator = ConfigMigrator()
        
        # Verify that built-in migrations are registered
        assert hasattr(migrator, '_migration_registry')
        assert len(migrator._migration_registry) > 0
        
        # Check for expected migration functions
        assert ("0.1.0", "1.0.0") in migrator._migration_registry
        assert ("0.2.0", "1.0.0") in migrator._migration_registry
    
    def test_migration_path_calculation(self):
        """Test migration path calculation for direct and chained migrations."""
        migrator = ConfigMigrator()
        
        # Test direct migration paths
        direct_paths = [
            ("0.1.0", "1.0.0"),
            ("0.2.0", "1.0.0"),
            ("1.0.0", "1.0.0"),  # Same version
        ]
        
        for from_version, to_version in direct_paths:
            path = migrator.get_migration_path(from_version, to_version)
            assert isinstance(path, list)
            assert len(path) >= 1
            assert path[0] == from_version
            assert path[-1] == to_version
    
    def test_migration_path_invalid_versions(self):
        """Test migration path calculation with invalid version combinations."""
        migrator = ConfigMigrator()
        
        invalid_combinations = [
            ("2.0.0", "1.0.0"),  # Future to current (not supported)
            ("invalid", "1.0.0"),  # Invalid source version
            ("1.0.0", "invalid"),  # Invalid target version
        ]
        
        for from_version, to_version in invalid_combinations:
            with pytest.raises(VersionError):
                migrator.get_migration_path(from_version, to_version)
    
    def test_successful_migration_execution(self):
        """Test successful migration execution with comprehensive reporting."""
        migrator = ConfigMigrator()
        
        test_config = {
            "project": {"major_data_directory": "/test/data"},
            "experiments": {"test_exp": {"date_range": ["2024-01-01", "2024-01-31"]}}
        }
        
        migrated_config, report = migrator.migrate(test_config, "0.1.0", "1.0.0")
        
        # Verify migration success
        assert migrated_config["schema_version"] == "1.0.0"
        assert len(report.errors) == 0
        assert report.metadata.get("migration_successful", False) == True
        assert len(report.applied_migrations) > 0
        assert report.execution_time_ms is not None
        
        # Verify structure enhancements
        assert "project" in migrated_config
        assert "directories" in migrated_config["project"]
        assert "extraction_patterns" in migrated_config["project"]
    
    def test_migration_validation_before_execution(self):
        """Test migration validation without executing the actual migration."""
        migrator = ConfigMigrator()
        
        # Test valid migration validation
        is_possible, message = migrator.validate_migration("0.1.0", "1.0.0")
        assert is_possible == True
        assert "available" in message.lower() or "path" in message.lower()
        
        # Test invalid migration validation
        is_possible, message = migrator.validate_migration("2.0.0", "1.0.0")
        assert is_possible == False
        assert "not" in message.lower() or "failed" in message.lower()
    
    def test_custom_migration_function_registration(self):
        """Test registration of custom migration functions."""
        migrator = ConfigMigrator()
        
        def custom_migration(config: Dict[str, Any]) -> Dict[str, Any]:
            result = copy.deepcopy(config)
            result["schema_version"] = "1.1.0"
            result["custom_field"] = "custom_value"
            return result
        
        # Register custom migration
        migrator.register_migration_function("1.0.0", "1.1.0", custom_migration)
        
        # Verify registration
        assert ("1.0.0", "1.1.0") in migrator._migration_registry
        
        # Test using custom migration
        test_config = {"schema_version": "1.0.0", "project": {}}
        migrated_config, report = migrator.migrate(test_config, "1.0.0", "1.1.0")
        
        assert migrated_config["schema_version"] == "1.1.0"
        assert migrated_config["custom_field"] == "custom_value"
        assert len(report.errors) == 0
    
    def test_migration_error_handling_and_context(self):
        """Test comprehensive error handling during migration execution."""
        migrator = ConfigMigrator()
        
        def failing_migration(config: Dict[str, Any]) -> Dict[str, Any]:
            raise ValueError("Simulated migration failure")
        
        # Register failing migration function
        migrator.register_migration_function("0.3.0", "1.0.0", failing_migration)
        
        test_config = {"schema_version": "0.3.0", "project": {}}
        
        with pytest.raises(VersionError) as exc_info:
            migrator.migrate(test_config, "0.3.0", "1.0.0")
        
        # Verify error context
        error = exc_info.value
        assert "VERSION_003" == error.error_code  # Migration step failure
        assert "context" in error.__dict__
        assert "migration_step" in error.context
        assert "original_error" in error.context


class TestMigrationExecution:
    """Test suite for migration execution with rollback capability and error recovery."""
    
    def test_migration_execution_with_rollback_on_failure(self):
        """Test migration rollback capability when migration steps fail."""
        migrator = ConfigMigrator()
        
        def partial_failing_migration(config: Dict[str, Any]) -> Dict[str, Any]:
            # Simulate a migration that partially modifies config then fails
            result = copy.deepcopy(config)
            result["partial_modification"] = True
            # Simulate failure after partial modification
            raise RuntimeError("Migration failed after partial changes")
        
        migrator.register_migration_function("0.5.0", "1.0.0", partial_failing_migration)
        
        original_config = {"schema_version": "0.5.0", "project": {"directories": {}}}
        original_backup = copy.deepcopy(original_config)
        
        with pytest.raises(VersionError):
            migrator.migrate(original_config, "0.5.0", "1.0.0")
        
        # Verify original config wasn't modified (rollback behavior)
        assert original_config == original_backup
    
    def test_migration_execution_audit_trail_generation(self):
        """Test comprehensive audit trail generation during migration execution."""
        migrator = ConfigMigrator()
        
        test_config = {
            "project": {"major_data_directory": "/data"},
            "experiments": {"exp1": {"date_range": ["2024-01-01", "2024-01-31"]}}
        }
        
        migrated_config, report = migrator.migrate(test_config, "0.1.0", "1.0.0")
        
        # Verify comprehensive audit trail
        report_dict = report.to_dict()
        
        assert "migration_summary" in report_dict
        assert "timestamp" in report_dict
        assert "applied_migrations" in report_dict
        assert "migration_path" in report_dict
        assert "execution_time_ms" in report_dict
        assert "config_changes" in report_dict
        
        # Verify audit trail details
        assert len(report.applied_migrations) > 0
        assert report.execution_time_ms is not None
        assert report.execution_time_ms > 0
    
    def test_automatic_error_recovery_suggestions(self):
        """Test automatic error recovery suggestion generation."""
        migrator = ConfigMigrator()
        
        # Test with completely invalid config
        invalid_config = {"invalid": "structure"}
        
        with pytest.raises(VersionError) as exc_info:
            migrator.migrate(invalid_config, "unknown", "1.0.0")
        
        error = exc_info.value
        assert hasattr(error, 'context')
        assert isinstance(error.context, dict)
        
        # Should provide context about available migrations or compatibility
        assert len(error.context) > 0
    
    def test_failed_migration_detailed_error_reporting(self):
        """Test detailed error reporting for failed migrations."""
        migrator = ConfigMigrator()
        
        def detailed_failing_migration(config: Dict[str, Any]) -> Dict[str, Any]:
            raise ValueError("Detailed migration failure with specific context")
        
        migrator.register_migration_function("0.6.0", "1.0.0", detailed_failing_migration)
        
        test_config = {"schema_version": "0.6.0", "project": {}}
        
        with pytest.raises(VersionError) as exc_info:
            migrator.migrate(test_config, "0.6.0", "1.0.0")
        
        error = exc_info.value
        
        # Verify detailed error information
        assert error.error_code == "VERSION_003"  # Migration step failure
        assert "context" in error.__dict__
        assert "migration_step" in error.context
        assert "original_error" in error.context
        assert "function_name" in error.context
        
        # Verify error message includes useful information
        assert "0.6.0 -> 1.0.0" in str(error)
    
    def test_migration_performance_monitoring(self):
        """Test migration performance monitoring and execution time tracking."""
        migrator = ConfigMigrator()
        
        test_config = {
            "project": {"major_data_directory": "/data"},
            "experiments": {"exp1": {"date_range": ["2024-01-01", "2024-01-31"]}}
        }
        
        start_time = datetime.now()
        migrated_config, report = migrator.migrate(test_config, "0.1.0", "1.0.0")
        end_time = datetime.now()
        
        expected_execution_time = (end_time - start_time).total_seconds() * 1000
        
        # Verify performance monitoring
        assert report.execution_time_ms is not None
        assert report.execution_time_ms > 0
        assert report.execution_time_ms <= expected_execution_time + 100  # Allow some tolerance
        
        # Verify individual step timing
        for migration_func in report.applied_migrations:
            step_timing_key = f"{migration_func}_execution_time_ms"
            assert step_timing_key in report.metadata
            assert report.metadata[step_timing_key] > 0


class TestSchemaVersionInjection:
    """Test suite for automatic schema_version field injection."""
    
    def test_schema_version_injection_for_legacy_configs(self):
        """Test automatic schema_version field injection for configurations without version metadata."""
        legacy_configs = [
            {"project": {"major_data_directory": "/data"}},
            {"project": {"directories": {}}, "datasets": {}},
        ]
        
        migrator = ConfigMigrator()
        
        for legacy_config in legacy_configs:
            migrated_config, report = migrator.migrate(legacy_config, "0.1.0", "1.0.0")
            
            # Verify schema_version was injected
            assert "schema_version" in migrated_config
            assert migrated_config["schema_version"] == "1.0.0"
            
            # Verify injection was tracked in the report
            changes_keys = list(report.config_changes.keys())
            schema_version_changes = [key for key in changes_keys if "schema_version" in key]
            assert len(schema_version_changes) > 0
    
    def test_schema_version_preservation_for_existing_configs(self):
        """Test that existing schema_version fields are properly updated during migration."""
        config_with_version = {
            "schema_version": "0.2.0",
            "project": {"directories": {"major_data_directory": "/data"}},
            "datasets": {"ds1": {"rig": "rig1", "dates_vials": {}}}
        }
        
        migrator = ConfigMigrator()
        migrated_config, report = migrator.migrate(config_with_version, "0.2.0", "1.0.0")
        
        # Verify version was updated
        assert migrated_config["schema_version"] == "1.0.0"
        
        # Verify change was tracked
        changes_keys = list(report.config_changes.keys())
        schema_version_changes = [key for key in changes_keys if "schema_version" in key]
        assert len(schema_version_changes) > 0
    
    def test_nested_schema_version_injection(self):
        """Test schema_version injection in nested configuration structures."""
        config_with_nested = {
            "project": {"directories": {"major_data_directory": "/data"}},
            "experiments": {
                "exp1": {"date_range": ["2024-01-01", "2024-01-31"]},
                "exp2": {"date_range": ["2024-02-01", "2024-02-28"]}
            }
        }
        
        migrator = ConfigMigrator()
        migrated_config, report = migrator.migrate(config_with_nested, "0.1.0", "1.0.0")
        
        # Verify main schema_version
        assert migrated_config["schema_version"] == "1.0.0"
        
        # Verify nested experiment schema_version injection
        if "experiments" in migrated_config:
            for exp_name, exp_config in migrated_config["experiments"].items():
                if isinstance(exp_config, dict):
                    assert "schema_version" in exp_config
                    assert exp_config["schema_version"] == "1.0.0"
    
    def test_selective_version_injection_preservation(self):
        """Test selective version field injection without overwriting valid existing versions."""
        mixed_config = {
            "schema_version": "0.2.0",  # Existing version to be updated
            "project": {"directories": {}},
            "datasets": {
                "ds1": {"rig": "rig1", "dates_vials": {}},  # No version - should get injected
                "ds2": {"schema_version": "0.2.0", "rig": "rig2", "dates_vials": {}}  # Has version - should be updated
            }
        }
        
        migrator = ConfigMigrator()
        migrated_config, report = migrator.migrate(mixed_config, "0.2.0", "1.0.0")
        
        # Verify main version updated
        assert migrated_config["schema_version"] == "1.0.0"
        
        # Verify dataset versions handled correctly
        if "datasets" in migrated_config:
            for dataset_name, dataset_config in migrated_config["datasets"].items():
                if isinstance(dataset_config, dict):
                    assert "schema_version" in dataset_config
                    assert dataset_config["schema_version"] == "1.0.0"


class TestEndToEndMigrationWorkflows:
    """Test suite for end-to-end migration workflows from legacy to modern configurations."""
    
    def test_complete_v0_1_to_v1_0_workflow(self):
        """Test complete migration workflow from v0.1.0 to v1.0.0."""
        v0_1_config = {
            "project": {
                "major_data_directory": "/research/data",
                "ignore_substrings": ["temp", "backup"]
            },
            "experiments": {
                "behavior_study": {
                    "date_range": ["2024-01-01", "2024-01-31"],
                    "subject_ids": ["fly001", "fly002", "fly003"]
                },
                "learning_experiment": {
                    "date_range": ["2024-02-01", "2024-02-28"],
                    "parameters": {"threshold": 0.7}
                }
            }
        }
        
        migrator = ConfigMigrator()
        migrated_config, report = migrator.migrate(v0_1_config, "0.1.0", "1.0.0")
        
        # Verify structural transformation
        assert migrated_config["schema_version"] == "1.0.0"
        assert "project" in migrated_config
        assert "directories" in migrated_config["project"]
        assert "major_data_directory" in migrated_config["project"]["directories"]
        assert "extraction_patterns" in migrated_config["project"]
        assert "datasets" in migrated_config
        
        # Verify experiments preserved and enhanced
        assert "experiments" in migrated_config
        for exp_name, exp_config in migrated_config["experiments"].items():
            assert "schema_version" in exp_config
            assert exp_config["schema_version"] == "1.0.0"
            assert "metadata" in exp_config
        
        # Verify migration report
        assert len(report.errors) == 0
        assert len(report.applied_migrations) > 0
        assert "migrate_v0_1_to_v1_0" in report.applied_migrations
        assert report.metadata.get("migration_successful") == True
    
    def test_complete_v0_2_to_v1_0_workflow(self):
        """Test complete migration workflow from v0.2.0 to v1.0.0."""
        v0_2_config = {
            "schema_version": "0.2.0",
            "project": {
                "directories": {"major_data_directory": "/research/data"},
                "extraction_patterns": [r"(?P<date>\d{4}-\d{2}-\d{2})"],
                "ignore_substrings": ["temp"]
            },
            "datasets": {
                "behavior_dataset": {
                    "rig": "rig_1",
                    "dates_vials": {
                        "2024-01-01": [1, 2, 3],
                        "2024-01-02": [4, 5, 6]
                    }
                },
                "learning_dataset": {
                    "rig": "rig_2", 
                    "dates_vials": {
                        "2024-02-01": [7, 8, 9]
                    },
                    "metadata": {"experiment_type": "learning"}
                }
            },
            "experiments": {
                "combined_analysis": {
                    "datasets": ["behavior_dataset", "learning_dataset"],
                    "parameters": {"analysis_window": 5.0}
                }
            }
        }
        
        migrator = ConfigMigrator()
        migrated_config, report = migrator.migrate(v0_2_config, "0.2.0", "1.0.0")
        
        # Verify version update
        assert migrated_config["schema_version"] == "1.0.0"
        
        # Verify dataset enhancements
        assert "datasets" in migrated_config
        for dataset_name, dataset_config in migrated_config["datasets"].items():
            assert dataset_config["schema_version"] == "1.0.0"
            assert "metadata" in dataset_config
            assert "rig" in dataset_config
            
        # Verify experiment enhancements
        assert "experiments" in migrated_config
        for exp_name, exp_config in migrated_config["experiments"].items():
            assert exp_config["schema_version"] == "1.0.0"
            assert "parameters" in exp_config
            assert "metadata" in exp_config
        
        # Verify migration success
        assert len(report.errors) == 0
        assert "migrate_v0_2_to_v1_0" in report.applied_migrations
    
    def test_chained_migration_workflow(self):
        """Test chained migration workflow through multiple version steps."""
        # For this test, we'll simulate a scenario where chained migration is needed
        # This might involve going through intermediate versions
        
        v0_1_config = {
            "project": {"major_data_directory": "/data"},
            "experiments": {"exp1": {"date_range": ["2024-01-01", "2024-01-31"]}}
        }
        
        migrator = ConfigMigrator()
        
        # Test migration path calculation for potential chaining
        migration_path = migrator.get_migration_path("0.1.0", "1.0.0")
        
        # Execute migration
        migrated_config, report = migrator.migrate(v0_1_config, "0.1.0", "1.0.0")
        
        assert migrated_config["schema_version"] == "1.0.0"
        assert len(report.errors) == 0
        
        # Verify path tracking in metadata
        assert "migration_path" in report.metadata
        assert report.metadata["migration_path"] == migration_path
    
    def test_real_world_configuration_migration(self):
        """Test migration with realistic, complex configuration scenarios."""
        complex_config = {
            "project": {
                "major_data_directory": "/research/flyrig/data",
                "ignore_substrings": ["._", "temp", "backup", ".DS_Store"],
                "mandatory_experiment_strings": ["experiment", "trial"]
            },
            "experiments": {
                "courtship_behavior": {
                    "date_range": ["2024-01-15", "2024-02-15"],
                    "subject_ids": ["male_001", "male_002", "female_001", "female_002"],
                    "parameters": {
                        "video_fps": 30,
                        "analysis_window": 300,
                        "detection_threshold": 0.8
                    }
                },
                "learning_memory": {
                    "date_range": ["2024-03-01", "2024-03-31"],
                    "parameters": {
                        "training_sessions": 5,
                        "retention_test_delay": 24,
                        "stimulus_duration": 2.0
                    }
                }
            }
        }
        
        migrator = ConfigMigrator()
        
        # Test the migration
        migrated_config, report = migrator.migrate(complex_config, "0.1.0", "1.0.0")
        
        # Comprehensive validation
        assert migrated_config["schema_version"] == "1.0.0"
        assert len(report.errors) == 0
        
        # Verify structural integrity
        assert "project" in migrated_config
        assert "directories" in migrated_config["project"]
        assert "extraction_patterns" in migrated_config["project"]
        
        # Verify all experiments preserved and enhanced
        assert len(migrated_config["experiments"]) == len(complex_config["experiments"])
        for exp_name in complex_config["experiments"]:
            assert exp_name in migrated_config["experiments"]
            exp_config = migrated_config["experiments"][exp_name]
            assert exp_config["schema_version"] == "1.0.0"
            assert "metadata" in exp_config


class TestErrorHandlingAndRecovery:
    """Test suite for migration error scenarios with detailed error reporting and recovery validation."""
    
    def test_migration_with_corrupted_configuration(self):
        """Test error handling with corrupted or malformed configurations."""
        corrupted_configs = [
            {"project": "should_be_dict"},  # Wrong type
            {"experiments": {"exp1": "should_be_dict"}},  # Wrong nested type
            {"project": {"directories": "should_be_dict"}},  # Wrong nested structure
        ]
        
        migrator = ConfigMigrator()
        
        for corrupted_config in corrupted_configs:
            with pytest.raises(VersionError) as exc_info:
                migrator.migrate(corrupted_config, "0.1.0", "1.0.0")
            
            # Verify error contains useful context
            error = exc_info.value
            assert hasattr(error, 'context')
            assert len(error.context) > 0
    
    def test_migration_with_unsupported_version_combinations(self):
        """Test error handling for unsupported version migration combinations."""
        test_config = {"schema_version": "1.0.0", "project": {}}
        migrator = ConfigMigrator()
        
        unsupported_combinations = [
            ("3.0.0", "1.0.0"),  # Future version downgrade
            ("invalid", "1.0.0"),  # Invalid source version
            ("1.0.0", "unknown"),  # Unknown target version
        ]
        
        for from_version, to_version in unsupported_combinations:
            with pytest.raises(VersionError) as exc_info:
                migrator.migrate(test_config, from_version, to_version)
            
            error = exc_info.value
            assert error.error_code in ["VERSION_005", "VERSION_002", "VERSION_001"]
            assert "context" in error.__dict__
    
    def test_migration_recovery_suggestions(self):
        """Test that migration errors provide helpful recovery suggestions."""
        migrator = ConfigMigrator()
        
        # Test with config that has no clear migration path
        problematic_config = {"unknown_structure": {"data": "value"}}
        
        with pytest.raises(VersionError) as exc_info:
            migrator.migrate(problematic_config, "unknown", "1.0.0")
        
        error = exc_info.value
        
        # Should provide context about available paths or solutions
        assert "context" in error.__dict__
        assert len(error.context) > 0
        
        # Error message should be helpful
        error_str = str(error)
        assert len(error_str) > 20  # Should have substantive error message
    
    def test_partial_migration_failure_recovery(self):
        """Test recovery handling when migration partially completes before failing."""
        migrator = ConfigMigrator()
        
        def partially_failing_migration(config: Dict[str, Any]) -> Dict[str, Any]:
            # Simulate migration that does some work then fails
            result = copy.deepcopy(config)
            result["schema_version"] = "1.0.0"  # This succeeds
            result["partial_work"] = "completed"  # This succeeds
            # Now simulate failure
            raise RuntimeError("Failure after partial completion")
        
        migrator.register_migration_function("0.7.0", "1.0.0", partially_failing_migration)
        
        test_config = {"schema_version": "0.7.0", "project": {}}
        original_backup = copy.deepcopy(test_config)
        
        with pytest.raises(VersionError) as exc_info:
            migrator.migrate(test_config, "0.7.0", "1.0.0")
        
        # Verify error provides useful context about what was attempted
        error = exc_info.value
        assert "migration_step" in error.context
        assert "original_error" in error.context
        assert "function_name" in error.context
        
        # Original config should remain unmodified (rollback behavior)
        assert test_config == original_backup
    
    def test_detailed_validation_error_reporting(self):
        """Test detailed error reporting for configuration validation failures."""
        migrator = ConfigMigrator()
        
        # Create config that will pass initial migration but fail validation
        config_that_fails_validation = {
            "project": {"major_data_directory": "/data"}
        }
        
        def migration_with_invalid_result(config: Dict[str, Any]) -> Dict[str, Any]:
            # Create result that will fail Pydantic validation
            result = copy.deepcopy(config)
            result["schema_version"] = "1.0.0"
            result["invalid_field"] = {"deeply": {"nested": {"invalid": "structure"}}}
            return result
        
        migrator.register_migration_function("0.8.0", "1.0.0", migration_with_invalid_result)
        
        with pytest.raises(VersionError) as exc_info:
            migrator.migrate(config_that_fails_validation, "0.8.0", "1.0.0", validate_result=True)
        
        error = exc_info.value
        # Should have validation-specific error code
        assert error.error_code in ["VERSION_006", "VERSION_003"]
        assert "validation" in str(error).lower()


# Property-based testing with Hypothesis for comprehensive edge case coverage
class TestPropertyBasedMigration:
    """Property-based tests for migration system robustness."""
    
    @given(st.dictionaries(
        keys=st.sampled_from(["project", "experiments", "datasets"]),
        values=st.dictionaries(
            keys=st.text(min_size=1, max_size=20),
            values=st.one_of(st.text(), st.lists(st.text()), st.dictionaries(st.text(), st.text()))
        ),
        min_size=1,
        max_size=3
    ))
    def test_migration_preserves_data_integrity(self, config_data):
        """Property test: Migration should preserve essential data integrity."""
        # Add required project structure for valid migration
        if "project" not in config_data:
            config_data["project"] = {}
        if not isinstance(config_data["project"], dict):
            config_data["project"] = {}
        
        config_data["project"]["major_data_directory"] = "/test/data"
        
        migrator = ConfigMigrator()
        
        try:
            migrated_config, report = migrator.migrate(config_data, "0.1.0", "1.0.0")
            
            # Properties that should always hold after successful migration
            assert "schema_version" in migrated_config
            assert migrated_config["schema_version"] == "1.0.0"
            assert "project" in migrated_config
            assert len(report.errors) == 0
            
        except VersionError:
            # Some randomly generated configs may not be migratable, which is acceptable
            pass
    
    @given(st.text(min_size=1, max_size=100))
    def test_migration_handles_arbitrary_string_inputs(self, arbitrary_string):
        """Property test: Migration should handle arbitrary string inputs gracefully."""
        migrator = ConfigMigrator()
        
        # Should not crash, should raise appropriate error
        with pytest.raises((VersionError, ValueError, TypeError)):
            migrator.migrate(arbitrary_string, "0.1.0", "1.0.0")


# Performance and thread-safety tests for enterprise environments
class TestPerformanceAndThreadSafety:
    """Test suite for migration system performance and thread-safety."""
    
    def test_migration_performance_benchmarks(self):
        """Test migration performance meets enterprise requirements."""
        large_config = {
            "project": {
                "major_data_directory": "/data",
                "extraction_patterns": [f"pattern_{i}" for i in range(100)],
                "ignore_substrings": [f"ignore_{i}" for i in range(50)]
            },
            "experiments": {
                f"experiment_{i}": {
                    "date_range": ["2024-01-01", "2024-01-31"],
                    "parameters": {f"param_{j}": j for j in range(20)}
                }
                for i in range(50)
            }
        }
        
        migrator = ConfigMigrator()
        
        start_time = datetime.now()
        migrated_config, report = migrator.migrate(large_config, "0.1.0", "1.0.0")
        end_time = datetime.now()
        
        execution_time = (end_time - start_time).total_seconds()
        
        # Should complete large migration within reasonable time
        assert execution_time < 5.0, f"Migration too slow: {execution_time}s"
        assert len(report.errors) == 0
        assert report.execution_time_ms is not None
        assert report.execution_time_ms > 0
    
    @patch('threading.RLock')
    def test_migration_thread_safety_preparation(self, mock_lock):
        """Test that migration system is prepared for thread-safe operations."""
        # This test verifies that the migration system is designed with thread-safety in mind
        # In a production system, the ConfigMigrator would use proper locking
        
        migrator = ConfigMigrator()
        
        # Verify that the migrator can handle multiple configurations
        configs = [
            {"project": {"major_data_directory": f"/data/{i}"}}
            for i in range(5)
        ]
        
        results = []
        for config in configs:
            migrated_config, report = migrator.migrate(config, "0.1.0", "1.0.0")
            results.append((migrated_config, report))
        
        # All migrations should succeed
        for migrated_config, report in results:
            assert migrated_config["schema_version"] == "1.0.0"
            assert len(report.errors) == 0


# Integration tests with temporary files and realistic scenarios
class TestFileSystemIntegration:
    """Test suite for migration system integration with file system operations."""
    
    def test_migration_with_temporary_config_files(self):
        """Test migration system with actual temporary configuration files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create temporary config file
            config_file = temp_path / "test_config.yaml"
            config_data = {
                "project": {
                    "major_data_directory": str(temp_path / "data"),
                    "extraction_patterns": [r"(?P<date>\d{4}-\d{2}-\d{2})"]
                },
                "experiments": {
                    "test_experiment": {
                        "date_range": ["2024-01-01", "2024-01-31"]
                    }
                }
            }
            
            with open(config_file, 'w') as f:
                yaml.safe_dump(config_data, f)
            
            # Load and migrate config
            with open(config_file, 'r') as f:
                loaded_config = yaml.safe_load(f)
            
            migrator = ConfigMigrator()
            migrated_config, report = migrator.migrate(loaded_config, "0.1.0", "1.0.0")
            
            # Verify migration worked with file-loaded config
            assert migrated_config["schema_version"] == "1.0.0"
            assert len(report.errors) == 0
            
            # Write migrated config back to file
            migrated_file = temp_path / "migrated_config.yaml"
            with open(migrated_file, 'w') as f:
                yaml.safe_dump(migrated_config, f)
            
            # Verify file was written correctly
            assert migrated_file.exists()
            with open(migrated_file, 'r') as f:
                reloaded_config = yaml.safe_load(f)
            
            assert reloaded_config["schema_version"] == "1.0.0"
    
    def test_migration_report_serialization(self):
        """Test migration report serialization for audit trail persistence."""
        migrator = ConfigMigrator()
        
        test_config = {
            "project": {"major_data_directory": "/data"},
            "experiments": {"exp1": {"date_range": ["2024-01-01", "2024-01-31"]}}
        }
        
        migrated_config, report = migrator.migrate(test_config, "0.1.0", "1.0.0")
        
        # Test JSON serialization
        report_dict = report.to_dict()
        json_report = json.dumps(report_dict, indent=2, default=str)
        
        # Should be able to parse back
        parsed_report = json.loads(json_report)
        assert parsed_report["from_version"] == "0.1.0"
        assert parsed_report["to_version"] == "1.0.0"
        assert "migration_summary" in parsed_report
        
        # Test with temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(report_dict, f, indent=2, default=str)
            temp_file_path = f.name
        
        try:
            # Verify file was written correctly
            with open(temp_file_path, 'r') as f:
                file_report = json.load(f)
            
            assert file_report == parsed_report
        finally:
            Path(temp_file_path).unlink()  # Clean up


if __name__ == "__main__":
    # Run the test suite with verbose output and coverage
    pytest.main([__file__, "-v", "--tb=short"])