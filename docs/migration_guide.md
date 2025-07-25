# FlyRigLoader Migration Guide

## Overview

This guide helps you migrate from the legacy dictionary-based configuration system to the new Pydantic model-based validation system in FlyRigLoader. The migration addresses six key improvements:

1. **Structured Configuration**: Replace ad-hoc dictionary handling with validated Pydantic models
2. **Clear Data Path Resolution**: Eliminate path ambiguity with explicit precedence and logging
3. **Decoupled Architecture**: Separate data discovery, loading, and transformation concerns
4. **Configuration Version Management**: Automatic version detection, migration, and compatibility handling
5. **Enhanced Registry System**: Plugin-style extensibility with automatic loader registration
6. **Comprehensive Documentation**: Complete API documentation with concrete examples and version migration workflows

**Migration Time**: ~5 minutes for most existing configurations
**Version Migration**: Automatic with comprehensive audit trails and rollback support

## Quick Start (2-Minute Migration)

### Option 1: Zero-Change Migration (Recommended)

Your existing code works unchanged. The new system maintains full backward compatibility:

```python
# Your existing code continues to work exactly as before
from flyrigloader.api import load_experiment_files

# Legacy dictionary access still works
config = load_config("config.yaml")
data = load_experiment_files(config, "my_experiment")
```

### Option 2: Gradual Migration

Enable new features incrementally while maintaining legacy compatibility:

```python
# New: Enable structured configuration validation
from flyrigloader.config.yaml_config import load_config

config = load_config("config.yaml", legacy_mode=False)  # Returns Pydantic model
# Access with both dict-style and attribute-style
datasets = config["datasets"]  # Legacy dict access
datasets = config.datasets     # New attribute access
```

### Option 3: Full Migration (Maximum Benefits)

Adopt the new decoupled architecture for maximum control:

```python
# New: Decoupled pipeline with granular control
from flyrigloader.api import discover_experiment_manifest, load_data_file, transform_to_dataframe

# Step 1: Discovery only (no data loading)
manifest = discover_experiment_manifest(config, "my_experiment")

# Step 2: Selective loading (memory efficient)
for file_info in manifest.files:
    raw_data = load_data_file(file_info.path)
    
    # Step 3: Optional transformation
    if need_dataframe:
        df = transform_to_dataframe(raw_data, config)
```

## Configuration Version Management System

FlyRigLoader now includes a comprehensive version management system that automatically handles configuration schema evolution, migration, and backward compatibility. This system ensures zero breaking changes during system upgrades while enabling continuous research workflows.

### Version Detection and Schema Management

#### Schema Version Field

All modern FlyRigLoader configurations include an embedded `schema_version` field that enables automatic version detection and migration:

```yaml
# Modern configuration with version tracking
schema_version: "1.0.0"
project:
  name: "fly_behavior_analysis"
  directories:
    major_data_directory: "/data/experiments"
  
datasets:
  plume_tracking:
    rig: "rig1"
    dates_vials:
      2023-05-01: [1, 2, 3, 4]

experiments:
  plume_navigation:
    datasets: ["plume_tracking"]
    parameters:
      threshold: 0.5
```

#### Automatic Version Detection

The system can automatically detect configuration versions even without explicit `schema_version` fields:

```python
from flyrigloader.migration.versions import ConfigVersion, detect_config_version
from semantic_version import Version

# Legacy configuration without version field
legacy_config = {
    'project': {'directories': {'major_data_directory': '/data'}},
    'experiments': {'exp1': {'date_range': ['2024-01-01', '2024-01-31']}}
}

# Automatic version detection
detected_version = detect_config_version(legacy_config)
print(f"Detected version: {detected_version.value}")  # "0.1.0"

# Access version constants
current_version = ConfigVersion.V1_0_0
oldest_supported = ConfigVersion.V0_1_0

# Semantic version comparison
version_obj = Version(current_version.value)
print(f"Current major version: {version_obj.major}")  # 1
```

#### Configuration Version Compatibility Matrix

The system uses a compatibility matrix to determine valid migration paths:

```python
from flyrigloader.migration.versions import COMPATIBILITY_MATRIX, is_migration_available, get_migration_path

# Check if migration is available
can_migrate = is_migration_available("0.1.0", "1.0.0")
print(f"Can migrate 0.1.0 -> 1.0.0: {can_migrate}")  # True

# Get the migration path
migration_path = get_migration_path("0.1.0", "1.0.0")
print(f"Migration path: {[str(v) for v in migration_path]}")  # ['0.1.0', '1.0.0']

# View full compatibility matrix
print("Supported migration paths:")
for from_version, to_versions in COMPATIBILITY_MATRIX.items():
    print(f"  {from_version.value} -> {[v.value for v in to_versions]}")
```

### ConfigMigrator Usage Patterns

#### Basic Migration Execution

The `ConfigMigrator` class orchestrates the migration process with comprehensive error handling and audit trails:

```python
from flyrigloader.migration.migrators import ConfigMigrator, MigrationReport
from flyrigloader.migration.versions import ConfigVersion

# Initialize migrator
migrator = ConfigMigrator()

# Legacy configuration to migrate
legacy_config = {
    'project': {
        'directories': {'major_data_directory': '/data/fly_experiments'},
        'ignore_substrings': ['temp_', 'backup_']
    },
    'experiments': {
        'plume_tracking': {
            'date_range': ['2024-01-01', '2024-01-31'],
            'rig_names': ['rig1', 'rig2']
        }
    }
}

# Execute migration with audit trail
try:
    migrated_config, report = migrator.migrate(legacy_config, "0.1.0", "1.0.0")
    print("Migration successful!")
    print(f"New schema version: {migrated_config['schema_version']}")
    
    # Access migration report
    print(f"Migration completed: {report.from_version} -> {report.to_version}")
    print(f"Applied migrations: {report.applied_migrations}")
    print(f"Timestamp: {report.timestamp}")
        
except Exception as e:
    print(f"Migration failed: {e}")
```

#### Advanced Migration with Custom Options

```python
from flyrigloader.migration.migrators import ConfigMigrator, migrate_v0_1_to_v1_0, migrate_v0_2_to_v1_0

# Advanced migration with detailed reporting
migrator = ConfigMigrator()

# Migration with validation
def migrate_with_validation(config_data, target_version="1.0.0"):
    """Enhanced migration with pre/post validation."""
    
    # Pre-migration validation
    from flyrigloader.migration.validators import validate_legacy_config_format
    is_valid, detected_format, messages = validate_legacy_config_format(config_data)
    
    if not is_valid:
        raise ValueError(f"Invalid legacy format: {messages}")
    
    print(f"Validated legacy format: {detected_format}")
    
    # Execute migration
    try:
        # Detect the from_version first
        from flyrigloader.migration.versions import detect_config_version
        detected_version = detect_config_version(config_data)
        
        migrated_config, migration_report = migrator.migrate(config_data, detected_version.value, target_version)
        
        # Post-migration validation using the new validate_config_with_migration
        from flyrigloader.migration.validators import validate_config_with_migration
        is_valid, final_version, validation_messages, final_config = validate_config_with_migration(
            migrated_config, 
            target_version, 
            auto_migrate=False  # Already migrated
        )
        
        if not is_valid:
            raise ValueError(f"Post-migration validation failed: {validation_messages}")
        
        print(f"Migration validation passed: {final_version}")
        return final_config
        
    except Exception as e:
        print(f"Migration failed: {e}")
        raise

# Usage with comprehensive validation
legacy_config = {
    'project': {'directories': {'major_data_directory': '/data'}},
    'datasets': {'ds1': {'rig': 'rig1', 'dates_vials': {'2024-01-01': [1, 2]}}},
    'experiments': {'exp1': {'datasets': ['ds1']}}
}

validated_config = migrate_with_validation(legacy_config)
print(f"Final configuration with version: {validated_config['schema_version']}")
```

#### Migration Report Analysis

The `MigrationReport` class provides comprehensive audit trails for migration operations:

```python
from flyrigloader.migration.migrators import MigrationReport
from datetime import datetime

# Create migration report for audit
report = MigrationReport(
    from_version="0.2.0",
    to_version="1.0.0", 
    timestamp=datetime.now()
)

# Add applied migrations and warnings after creation
report.applied_migrations.append("migrate_v0_2_to_v1_0")
report.warnings.append("Legacy date_range format converted to datasets")

# Convert to dictionary for serialization
report_dict = report.to_dict()
print("Migration Audit Trail:")
for key, value in report_dict.items():
    print(f"  {key}: {value}")

# Example output:
# Migration Audit Trail:
#   from_version: 0.2.0
#   to_version: 1.0.0
#   timestamp: 2024-01-01T10:00:00
#   applied_migrations: ['migrate_v0_2_to_v1_0']
#   warnings: ['Legacy date_range format converted to datasets']
```

### Automatic Migration and Validation

#### Using validate_config_with_migration

The system provides a comprehensive validation function that combines version detection, compatibility checking, and automatic migration:

```python
from flyrigloader.migration.validators import validate_config_with_migration

# Configuration with automatic migration
legacy_config = {
    'project': {
        'directories': {'major_data_directory': '/data/experiments'}
    },
    'experiments': {
        'baseline_study': {
            'date_range': ['2024-01-01', '2024-01-31']
        }
    }
}

# Automatic migration with comprehensive validation
is_valid, final_version, messages, migrated_config = validate_config_with_migration(
    legacy_config,
    target_version="1.0.0",
    auto_migrate=True
)

if is_valid:
    print(f"Configuration successfully migrated to {final_version}")
    print(f"Schema version: {migrated_config['schema_version']}")
    print("Migration messages:")
    for message in messages:
        print(f"  - {message}")
else:
    print("Migration failed:")
    for message in messages:
        print(f"  - {message}")
```

#### Pre-flight Validation with validate_manifest

The enhanced API includes pre-flight validation capabilities for comprehensive testing:

```python
from flyrigloader.api import validate_manifest
from flyrigloader.config.models import create_config

# Create configuration using builder pattern
config = create_config(
    project_name="behavior_analysis",
    base_directory="/data/experiments",
    experiments=["plume_navigation"],
    datasets=["plume_tracking"]
)

# Discover experiment manifest
from flyrigloader.api import discover_experiment_manifest
manifest = discover_experiment_manifest(config, "plume_navigation")

# Pre-flight validation without side effects
validation_result = validate_manifest(manifest, config, strict_validation=True)

if validation_result['is_valid']:
    print("Manifest validation passed")
    print(f"Validated {validation_result['file_count']} files")
    print(f"Total size: {validation_result['total_size_mb']} MB")
else:
    print("Manifest validation failed:")
    for error in validation_result['validation_errors']:
        print(f"  - {error}")
        
    if validation_result['missing_files']:
        print("Missing files:")
        for file_path in validation_result['missing_files']:
            print(f"  - {file_path}")
```

### LegacyConfigAdapter Integration

The `LegacyConfigAdapter` now includes comprehensive version migration support while maintaining backward compatibility:

```python
from flyrigloader.config.models import LegacyConfigAdapter

# Create adapter for legacy configuration
legacy_dict_config = {
    'project': {'directories': {'major_data_directory': '/data'}},
    'experiments': {'exp1': {'date_range': ['2024-01-01', '2024-01-31']}}
}

# Enhanced adapter with version migration support
adapter = LegacyConfigAdapter(legacy_dict_config)

# Access schema version (automatically detected and added)
print(f"Schema version: {adapter.schema_version}")  # "0.1.0"

# Dictionary-style access (preserved for backward compatibility)
data_dir = adapter['project']['directories']['major_data_directory']
print(f"Data directory: {data_dir}")

# Set operations with validation
adapter['project']['name'] = "migrated_project"
print(f"Project name: {adapter['project']['name']}")

# Validate all configuration sections
if adapter.validate_all():
    print("All configuration sections are valid")
else:
    print("Some configuration sections failed validation")
```

#### LegacyConfigAdapter with Deprecation Warnings

```python
import warnings
from flyrigloader.config.models import LegacyConfigAdapter

# Enable deprecation warnings
warnings.filterwarnings("always", category=DeprecationWarning)

# Using legacy adapter triggers appropriate warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    
    adapter = LegacyConfigAdapter({
        'project': {'directories': {'major_data_directory': '/data'}},
        'experiments': {'exp1': {}}
    })
    
    # Access triggers deprecation warning
    data_dir = adapter['project']['directories']['major_data_directory']
    
    if w:
        print(f"Deprecation warning: {w[-1].message}")
        print("Recommendation: Use Pydantic models with create_config() builder")

# Modern approach with builder pattern
from flyrigloader.config.models import create_config

modern_config = create_config(
    project_name="modern_project",
    base_directory="/data",
    experiments=["exp1"],
    datasets=["dataset1"]
)

print(f"Modern config version: {modern_config.schema_version}")
```

## Configuration Migration

### Legacy Configuration (Before)

```yaml
# example_config.yaml - Legacy format (still supported)
project:
  directories:
    major_data_directory: "/path/to/fly_data"
  ignore_substrings:
    - "._"
    - "temp_"

datasets:
  plume_tracking:
    rig: "rig1"
    dates_vials:
      2023-05-01: [1, 2, 3, 4]

experiments:
  plume_navigation_analysis:
    datasets: ["plume_tracking"]
    parameters:
      threshold: 0.5
```

### New Configuration (After)

The same YAML structure is now validated with Pydantic models:

```yaml
# example_config.yaml - Enhanced format with validation
project:
  name: "enhanced_project"  # Now validated and documented
  directories:
    major_data_directory: "/path/to/fly_data"
    # New: Multiple directory support
    alternative_data_directories:
      - "/backup/path/to/fly_data"
      - "/shared/fly_data"
  ignore_substrings:
    - "._"
    - "temp_"
  # New: Explicit validation rules
  validation_rules:
    require_rig_name: true
    minimum_vials_per_date: 1

datasets:
  plume_tracking:
    rig: "rig1"
    dates_vials:
      2023-05-01: [1, 2, 3, 4]
    # New: Dataset-specific metadata extraction
    metadata:
      extraction_patterns:
        - "(?P<condition>\\w+)_(?P<replicate>\\d+)"

experiments:
  plume_navigation_analysis:
    datasets: ["plume_tracking"]
    parameters:
      threshold: 0.5
      # New: Validated parameter schema
      analysis_type: "behavioral"
      output_format: "csv"
    # New: Experiment-specific filtering
    filters:
      ignore_substrings: ["failed_run"]
      mandatory_experiment_strings: ["plume"]
```

## Code Migration Examples

### 1. Configuration Loading

#### Before (Legacy Dictionary Access)
```python
from flyrigloader.config.yaml_config import load_config

# Legacy: Returns raw dictionary
config = load_config("config.yaml")
data_dir = config["project"]["directories"]["major_data_directory"]
datasets = config.get("datasets", {})
```

#### After (Pydantic Model Access)
```python
from flyrigloader.config.yaml_config import load_config

# New: Returns validated Pydantic model with backward compatibility
config = load_config("config.yaml", legacy_mode=False)

# Both access methods work:
data_dir = config["project"]["directories"]["major_data_directory"]  # Dict-style
data_dir = config.project.directories.major_data_directory            # Attribute-style

# Enhanced validation and IDE support
datasets = config.datasets  # Type-safe access with autocomplete
```

### 2. Data Path Resolution

#### Before (Ambiguous Path Resolution)
```python
# Legacy: Unclear which path is used
data = load_experiment_files(config, "experiment", base_directory="/override/path")
# No visibility into which path was actually used
```

#### After (Clear Path Resolution with Logging)
```python
# New: Explicit path resolution with logging
import logging
logging.basicConfig(level=logging.INFO)

data = load_experiment_files(config, "experiment", base_directory="/override/path")
# Output: INFO - Using data directory: /override/path (source: explicit_parameter)
```

### 3. Data Loading Pipeline Migration

#### Before (Monolithic Loading)
```python
# Legacy: Everything happens at once
data = process_experiment_data(config, "experiment")
# No control over individual steps
```

#### After (Decoupled Pipeline)
```python
# New: Granular control over each step
from flyrigloader.api import discover_experiment_manifest, load_data_file, transform_to_dataframe

# Step 1: Discovery only (fast, memory efficient)
manifest = discover_experiment_manifest(config, "experiment")
print(f"Found {len(manifest.files)} files")

# Step 2: Selective loading
selected_files = manifest.files[:5]  # Process only first 5 files
raw_data_list = []
for file_info in selected_files:
    raw_data = load_data_file(file_info.path)
    raw_data_list.append(raw_data)

# Step 3: Optional transformation
if need_dataframes:
    dataframes = []
    for raw_data in raw_data_list:
        df = transform_to_dataframe(raw_data, config)
        dataframes.append(df)
```

#### Migration Pattern: process_experiment_data to Decoupled Pipeline

```python
# Old monolithic approach
@deprecated("Use decoupled pipeline instead")
def old_process_experiment_data(config, experiment):
    data = process_experiment_data(config, experiment)
    return data

# New decoupled approach
def new_process_experiment_data(config, experiment):
    # Step 1: Discovery
    manifest = discover_experiment_manifest(config, experiment)
    
    # Step 2: Load files
    raw_data_list = []
    for file_info in manifest.files:
        raw_data = load_data_file(file_info.path)
        raw_data_list.append(raw_data)
    
    # Step 3: Transform to DataFrames
    dataframes = []
    for raw_data in raw_data_list:
        df = transform_to_dataframe(raw_data, config)
        dataframes.append(df)
    
    # Combine if needed
    if dataframes:
        return pd.concat(dataframes, ignore_index=True)
    return pd.DataFrame()
```

### 4. Error Handling Migration

#### Before (Generic Error Messages)
```python
try:
    config = load_config("config.yaml")
except Exception as e:
    print(f"Config error: {e}")
    # Generic error, hard to debug
```

#### After (Domain-Specific Exception Handling)
```python
from flyrigloader.exceptions import (
    ConfigError, DiscoveryError, LoadError, TransformError
)

# Configuration errors
try:
    config = load_config("config.yaml", legacy_mode=False)
except ConfigError as e:
    print(f"Configuration error: {e}")
    print(f"Error code: {e.error_code}")
    print(f"Context: {e.context}")
    # Handle specific configuration issues
    if e.error_code == "CONFIG_003":
        # Handle Pydantic validation failure
        print("Fix validation errors in config file")

# Discovery errors
try:
    manifest = discover_experiment_manifest(config, "experiment")
except DiscoveryError as e:
    print(f"Discovery error: {e}")
    if e.error_code == "DISCOVERY_002":
        # Handle directory not found
        print("Check data directory path")
    elif e.error_code == "DISCOVERY_003":
        # Handle pattern compilation error
        print("Fix regex patterns in configuration")

# Loading errors
try:
    raw_data = load_data_file("data.pkl")
except LoadError as e:
    print(f"Loading error: {e}")
    if e.error_code == "LOAD_002":
        # Handle unsupported format
        print("Check file format or register custom loader")

# Transformation errors
try:
    df = transform_to_dataframe(raw_data, config)
except TransformError as e:
    print(f"Transformation error: {e}")
    if e.error_code == "TRANSFORM_006":
        # Handle missing required columns
        print("Check data structure and column configuration")
```

#### Exception Context Preservation
```python
# Add context when re-raising exceptions
try:
    result = process_complex_operation(data)
except ValueError as e:
    raise TransformError("Data processing failed").with_context({
        "original_error": str(e),
        "data_shape": data.shape,
        "operation": "complex_analysis",
        "timestamp": datetime.now().isoformat()
    })

# Log and raise pattern
from flyrigloader.exceptions import log_and_raise
from flyrigloader import logger

try:
    validate_data(data)
except ValidationError as e:
    error = ConfigError(
        "Data validation failed",
        error_code="CONFIG_003",
        context={"validation_errors": str(e)}
    )
    log_and_raise(error, logger, "error")
```

### 5. Registry-Based Extensibility

#### New: Custom Loader Registration
```python
from flyrigloader.registries import LoaderRegistry, BaseLoader
from pathlib import Path

# Register custom loader for new file format
@LoaderRegistry.register('.custom', priority=10)
class CustomLoader(BaseLoader):
    def load(self, path: Path) -> Any:
        # Custom loading logic for .custom files
        with open(path, 'r') as f:
            return json.load(f)
    
    def supports_extension(self, extension: str) -> bool:
        return extension == '.custom'
    
    @property
    def priority(self) -> int:
        return 10

# Use custom loader
data = load_data_file("experiment.custom")  # Automatically uses CustomLoader
```

#### New: Schema Registry for Validation
```python
from flyrigloader.registries import SchemaRegistry, BaseSchema

# Register custom schema validator
@SchemaRegistry.register('experiment_schema', priority=10)
class ExperimentSchema(BaseSchema):
    def validate(self, data: Any) -> Dict[str, Any]:
        # Custom validation logic
        if 't' not in data:
            raise ValueError("Missing time column")
        return data
    
    @property
    def schema_name(self) -> str:
        return 'experiment_schema'
    
    @property
    def supported_types(self) -> List[str]:
        return ['experiment', 'behavioral']

# Use schema registry
schema_registry = SchemaRegistry()
schema = schema_registry.get_schema('experiment_schema')
validated_data = schema.validate(raw_data)
```

#### Registry Discovery and Inspection
```python
from flyrigloader.registries import LoaderRegistry, SchemaRegistry

# Inspect registered loaders
loader_registry = LoaderRegistry()
all_loaders = loader_registry.get_all_loaders()
print(f"Registered loaders: {list(all_loaders.keys())}")

# Check supported formats
from flyrigloader.io.loaders import get_supported_formats
formats = get_supported_formats()
print(f"Supported formats: {formats}")

# Inspect registered schemas
schema_registry = SchemaRegistry()
all_schemas = schema_registry.get_all_schemas()
print(f"Registered schemas: {list(all_schemas.keys())}")
```

### 6. Configuration Builder Functions

#### New: Programmatic Configuration Creation
```python
from flyrigloader.config.models import create_config, create_experiment, create_dataset

# Create project configuration programmatically
config = create_config(
    project_name="fly_behavior_analysis",
    base_directory="/data/experiments",
    ignore_substrings=["temp", "backup", "test"],
    extraction_patterns=[
        r"(?P<date>\d{4}-\d{2}-\d{2})",
        r"(?P<subject>\w+)",
        r"(?P<condition>\w+_condition)"
    ]
)

# Create experiment configuration
experiment_config = create_experiment(
    name="plume_navigation",
    datasets=["plume_tracking", "odor_response"],
    parameters={
        "analysis_window": 10.0,
        "threshold": 0.5,
        "method": "correlation"
    },
    metadata={
        "description": "Plume navigation behavior analysis",
        "researcher": "Dr. Smith"
    }
)

# Create dataset configuration
dataset_config = create_dataset(
    name="plume_tracking",
    rig="rig1",
    dates_vials={"2023-05-01": [1, 2, 3, 4]},
    metadata={
        "description": "Plume tracking behavioral data",
        "sampling_rate": 1000.0
    }
)

# Use configurations together
combined_config = {
    "project": config,
    "experiments": {"plume_navigation": experiment_config},
    "datasets": {"plume_tracking": dataset_config}
}
```

### 7. Deprecation Warnings and Migration Paths

#### Handling Deprecated Functions
```python
import warnings

# Deprecated function usage (will show warning)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    
    # This will trigger deprecation warning
    data = process_experiment_data(config, "experiment")
    
    if w:
        print(f"Deprecation warning: {w[0].message}")
        print("Migrate to: load_data_file() + transform_to_dataframe()")

# Suppress deprecation warnings during migration
warnings.filterwarnings("ignore", category=DeprecationWarning, module="flyrigloader")
```

#### Migration from Deprecated APIs
```python
# Old deprecated API
@deprecated("Use decoupled pipeline instead")
def old_process_experiment_data(config, experiment):
    return process_experiment_data(config, experiment)

# New recommended pattern
def new_process_experiment_data(config, experiment):
    # Discovery phase
    manifest = discover_experiment_manifest(config, experiment)
    
    # Loading phase
    raw_data_list = []
    for file_info in manifest.files:
        raw_data = load_data_file(file_info.path)
        raw_data_list.append(raw_data)
    
    # Transformation phase
    dataframes = []
    for raw_data in raw_data_list:
        df = transform_to_dataframe(raw_data, config)
        dataframes.append(df)
    
    return pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()
```

## Common Migration Patterns

### Pattern 1: Gradual Feature Adoption

```python
# Start with legacy mode, gradually enable new features
class ExperimentProcessor:
    def __init__(self, config_path):
        # Week 1: Legacy compatibility
        self.config = load_config(config_path, legacy_mode=True)
        
        # Week 2: Enable validation
        self.config = load_config(config_path, legacy_mode=False)
        
        # Week 3: Adopt decoupled pipeline
        self.use_new_pipeline = True
    
    def process_data(self, experiment_name):
        if self.use_new_pipeline:
            return self._process_with_new_pipeline(experiment_name)
        else:
            return self._process_legacy(experiment_name)
    
    def _process_with_new_pipeline(self, experiment_name):
        manifest = discover_experiment_manifest(self.config, experiment_name)
        # Process with granular control
        return self._selective_processing(manifest)
    
    def _process_legacy(self, experiment_name):
        # Original monolithic approach
        return process_experiment_data(self.config, experiment_name)
```

### Pattern 2: Configuration Validation

```python
# Validate configuration early in application lifecycle
def validate_setup(config_path):
    try:
        config = load_config(config_path, legacy_mode=False)
        
        # Validate required experiments exist
        required_experiments = ["control", "treatment"]
        available_experiments = list(config.experiments.keys())
        
        for exp in required_experiments:
            if exp not in available_experiments:
                raise ValueError(f"Required experiment '{exp}' not found in configuration")
        
        # Validate data directories exist
        import os
        if not os.path.exists(config.project.directories.major_data_directory):
            raise ValueError(f"Data directory not found: {config.project.directories.major_data_directory}")
        
        return config
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        raise
```

### Pattern 3: Memory-Efficient Processing with Decoupled Pipeline

```python
# Process large datasets efficiently with new decoupled approach
def process_large_experiment(config, experiment_name, batch_size=10):
    # Discovery is fast and memory-efficient
    manifest = discover_experiment_manifest(config, experiment_name)
    
    # Process in batches to manage memory
    total_files = len(manifest.files)
    results = []
    
    for i in range(0, total_files, batch_size):
        batch = manifest.files[i:i+batch_size]
        
        # Load and process batch
        batch_data = []
        for file_info in batch:
            raw_data = load_data_file(file_info.path)
            batch_data.append(raw_data)
        
        # Transform only if needed
        if need_dataframes:
            batch_dfs = [transform_to_dataframe(data, config) for data in batch_data]
            results.extend(batch_dfs)
        else:
            results.extend(batch_data)
        
        # Clear batch from memory
        del batch_data
    
    return results
```

### Pattern 4: Registry-Based Extension Development

```python
# Custom loader implementation with registry
from flyrigloader.registries import LoaderRegistry, BaseLoader
from pathlib import Path
import json

class JSONLoader(BaseLoader):
    """Custom loader for JSON files."""
    
    def load(self, path: Path) -> Any:
        with open(path, 'r') as f:
            return json.load(f)
    
    def supports_extension(self, extension: str) -> bool:
        return extension.lower() == '.json'
    
    @property
    def priority(self) -> int:
        return 5

# Register the loader
loader_registry = LoaderRegistry()
loader_registry.register_loader('.json', JSONLoader, priority=5)

# Now JSON files can be loaded automatically
data = load_data_file("experiment.json")
```

### Pattern 5: Schema Validation with Registry

```python
# Custom schema validator with registry
from flyrigloader.registries import SchemaRegistry, BaseSchema
from flyrigloader.exceptions import TransformError

class TimeSeriesSchema(BaseSchema):
    """Schema validator for time series data."""
    
    def validate(self, data: Any) -> Dict[str, Any]:
        if not isinstance(data, dict):
            raise TransformError("Data must be dictionary")
        
        if 't' not in data:
            raise TransformError("Missing time column 't'")
        
        if len(data['t']) == 0:
            raise TransformError("Empty time column")
        
        return data
    
    @property
    def schema_name(self) -> str:
        return 'time_series'
    
    @property
    def supported_types(self) -> List[str]:
        return ['time_series', 'behavioral', 'physiological']

# Register the schema
schema_registry = SchemaRegistry()
schema_registry.register_schema('time_series', TimeSeriesSchema, priority=10)

# Use schema validation
schema = schema_registry.get_schema('time_series')
validated_data = schema.validate(raw_data)
```

### Pattern 6: Complete Migration Example

```python
# Complete migration from old to new architecture
class ExperimentProcessor:
    def __init__(self, config_path: str):
        # Load configuration using new builder functions
        self.config = create_config(
            project_name="behavior_analysis",
            base_directory="/data/experiments",
            ignore_substrings=["temp", "backup"],
            extraction_patterns=[
                r"(?P<date>\d{4}-\d{2}-\d{2})",
                r"(?P<subject>\w+)",
                r"(?P<condition>\w+)"
            ]
        )
    
    def process_experiment_old(self, experiment_name: str) -> pd.DataFrame:
        """Old monolithic approach (deprecated)."""
        try:
            return process_experiment_data(self.config, experiment_name)
        except Exception as e:
            raise FlyRigLoaderError(f"Failed to process experiment: {e}")
    
    def process_experiment_new(self, experiment_name: str) -> pd.DataFrame:
        """New decoupled approach with error handling."""
        try:
            # Step 1: Discovery
            manifest = discover_experiment_manifest(self.config, experiment_name)
            logger.info(f"Discovered {len(manifest.files)} files")
            
            # Step 2: Loading with error handling
            raw_data_list = []
            for file_info in manifest.files:
                try:
                    raw_data = load_data_file(file_info.path)
                    raw_data_list.append(raw_data)
                except LoadError as e:
                    logger.warning(f"Failed to load {file_info.path}: {e}")
                    continue
            
            # Step 3: Transformation with error handling
            dataframes = []
            for raw_data in raw_data_list:
                try:
                    df = transform_to_dataframe(raw_data, self.config)
                    dataframes.append(df)
                except TransformError as e:
                    logger.warning(f"Failed to transform data: {e}")
                    continue
            
            # Combine results
            if dataframes:
                return pd.concat(dataframes, ignore_index=True)
            else:
                logger.warning("No data processed successfully")
                return pd.DataFrame()
                
        except DiscoveryError as e:
            logger.error(f"Discovery failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise FlyRigLoaderError(f"Processing failed: {e}")
    
    def migrate_processing(self, experiment_name: str) -> pd.DataFrame:
        """Gradual migration with fallback."""
        try:
            # Try new approach first
            return self.process_experiment_new(experiment_name)
        except Exception as e:
            logger.warning(f"New approach failed: {e}")
            logger.info("Falling back to legacy approach")
            
            # Fallback to old approach with deprecation warning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                return self.process_experiment_old(experiment_name)
```

## Troubleshooting Common Issues

### Issue 1: Pydantic Validation Errors

**Problem**: Configuration fails validation with cryptic error messages.

**Solution**: Enable detailed error reporting:

```python
from pydantic import ValidationError

try:
    config = load_config("config.yaml", legacy_mode=False)
except ValidationError as e:
    # Get detailed error information
    for error in e.errors():
        field_path = " -> ".join(str(loc) for loc in error["loc"])
        print(f"Field: {field_path}")
        print(f"Error: {error['msg']}")
        print(f"Input: {error['input']}")
        print("---")
```

**Common Validation Errors**:

- **Missing required fields**: Add required sections to your YAML
- **Invalid data types**: Ensure dates_vials values are lists, not strings
- **Invalid path formats**: Use absolute paths or ensure relative paths are accessible

### Issue 2: Path Resolution Confusion

**Problem**: Unclear which data directory is being used.

**Solution**: Enable INFO-level logging:

```python
import logging
logging.basicConfig(level=logging.INFO)

# The library will log exactly which path is used and why
data = load_experiment_files(config, "experiment")
# Output: INFO - Using data directory: /path/to/data (source: config_major_data_directory)
```

### Issue 3: Memory Issues with Large Datasets

**Problem**: Running out of memory when processing large experiments.

**Solution**: Use the new decoupled approach:

```python
# Instead of loading everything at once
# data = process_experiment_data(config, "large_experiment")

# Use selective loading
manifest = discover_experiment_manifest(config, "large_experiment")
print(f"Found {len(manifest.files)} files")

# Process files one at a time
for file_info in manifest.files:
    raw_data = load_data_file(file_info.path)
    # Process individual file
    result = analyze_single_file(raw_data)
    # Save result and clear memory
    save_result(result)
    del raw_data
```

### Issue 4: Registry-Based Loading Issues

**Problem**: Custom file formats not being loaded correctly.

**Solution**: Register custom loaders properly:

```python
from flyrigloader.registries import LoaderRegistry
from flyrigloader.io.loaders import get_supported_formats

# Check supported formats
formats = get_supported_formats()
print(f"Supported formats: {formats}")

# Register custom loader if needed
class CustomLoader:
    def load(self, path):
        # Custom loading logic
        pass
    
    def supports_extension(self, extension):
        return extension == '.custom'
    
    @property
    def priority(self):
        return 10

# Register the loader
registry = LoaderRegistry()
registry.register_loader('.custom', CustomLoader, priority=10)
```

### Issue 5: New Exception Types

**Problem**: Code catching generic Exception no longer works.

**Solution**: Update to catch specific exception types:

```python
# Old generic catching
try:
    data = load_experiment_files(config, "experiment")
except Exception as e:
    logger.error(f"Failed: {e}")

# New specific exception handling
from flyrigloader.exceptions import (
    ConfigError, DiscoveryError, LoadError, TransformError
)

try:
    data = load_experiment_files(config, "experiment")
except ConfigError as e:
    logger.error(f"Configuration error: {e}")
    # Handle config-specific issues
except DiscoveryError as e:
    logger.error(f"Discovery error: {e}")
    # Handle discovery-specific issues
except LoadError as e:
    logger.error(f"Loading error: {e}")
    # Handle loading-specific issues
except TransformError as e:
    logger.error(f"Transformation error: {e}")
    # Handle transformation-specific issues
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    # Handle other unexpected errors
```

### Issue 6: Deprecation Warnings

**Problem**: Getting deprecation warnings for functions that still work.

**Solution**: Understand the migration path:

```python
import warnings

# Option 1: Suppress deprecation warnings during migration
warnings.filterwarnings("ignore", category=DeprecationWarning, module="flyrigloader")

# Option 2: Handle warnings explicitly
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    
    # This will trigger deprecation warning
    data = process_experiment_data(config, "experiment")
    
    if w:
        print(f"Warning: {w[0].message}")
        print("Consider migrating to: load_data_file() + transform_to_dataframe()")

# Option 3: Migrate to new API
# Instead of: data = process_experiment_data(config, "experiment")
# Use:
manifest = discover_experiment_manifest(config, "experiment")
raw_data_list = [load_data_file(f.path) for f in manifest.files]
dataframes = [transform_to_dataframe(data, config) for data in raw_data_list]
combined_data = pd.concat(dataframes, ignore_index=True)
```

### Issue 7: Kedro Integration

**Problem**: Kedro parameters dictionary no longer works.

**Solution**: Kedro integration is preserved:

```python
# This continues to work exactly as before
def process_data(parameters: Dict[str, Any]):
    # Kedro passes parameters as dictionary
    data = load_experiment_files(parameters, "experiment")
    return data

# Or migrate to new validation:
def process_data_validated(parameters: Dict[str, Any]):
    # Validate Kedro parameters
    config = load_config(parameters, legacy_mode=False)  # Dict input still works
    manifest = discover_experiment_manifest(config, "experiment")
    return manifest

# Or use builder functions for new projects:
def process_data_with_builders(parameters: Dict[str, Any]):
    # Create validated config from Kedro parameters
    config = create_config(
        project_name=parameters.get("project_name", "kedro_project"),
        base_directory=parameters["base_directory"],
        ignore_substrings=parameters.get("ignore_substrings", []),
        extraction_patterns=parameters.get("extraction_patterns", [])
    )
    
    # Use new decoupled pipeline
    manifest = discover_experiment_manifest(config, parameters["experiment_name"])
    raw_data_list = [load_data_file(f.path) for f in manifest.files]
    dataframes = [transform_to_dataframe(data, config) for data in raw_data_list]
    
    return pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()
```

## Configuration Version Migration Workflows

### End-to-End Migration Example

Here's a comprehensive example showing the complete configuration version migration workflow:

```python
"""
Complete configuration version migration workflow example.
Demonstrates version detection, migration execution, validation, and audit trail generation.
"""

import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Import version management utilities
from flyrigloader.migration.versions import ConfigVersion, detect_config_version, is_migration_available
from flyrigloader.migration.migrators import ConfigMigrator, MigrationReport
from flyrigloader.migration.validators import (
    validate_config_with_migration, 
    validate_legacy_config_format,
    validate_schema_compatibility
)
from flyrigloader.config.models import LegacyConfigAdapter
from flyrigloader.api import validate_manifest
from semantic_version import Version

class ConfigurationMigrationWorkflow:
    """Complete configuration migration workflow with audit trails."""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.migration_reports = []
        
    def execute_migration_workflow(self, target_version: str = "1.0.0") -> Dict[str, Any]:
        """Execute complete migration workflow with comprehensive validation."""
        
        workflow_result = {
            'success': False,
            'original_version': None,
            'final_version': None,
            'migration_applied': False,
            'validation_passed': False,
            'audit_trail': [],
            'migrated_config': None,
            'errors': []
        }
        
        try:
            # Step 1: Load original configuration
            print(f"Loading configuration from: {self.config_path}")
            
            if self.config_path.suffix.lower() == '.yaml':
                import yaml
                with open(self.config_path, 'r') as f:
                    original_config = yaml.safe_load(f)
            else:
                # Assume it's a Python dict for demonstration
                original_config = self._load_demo_config()
            
            workflow_result['audit_trail'].append(f"Loaded configuration from {self.config_path}")
            
            # Step 2: Detect current version
            try:
                detected_version = detect_config_version(original_config)
                workflow_result['original_version'] = detected_version.value
                print(f"Detected configuration version: {detected_version.value}")
                workflow_result['audit_trail'].append(f"Detected version: {detected_version.value}")
            except Exception as e:
                workflow_result['errors'].append(f"Version detection failed: {e}")
                return workflow_result
            
            # Step 3: Check if migration is needed
            if detected_version.value == target_version:
                print(f"Configuration already at target version {target_version}")
                workflow_result['success'] = True
                workflow_result['final_version'] = target_version
                workflow_result['migrated_config'] = original_config
                workflow_result['validation_passed'] = True
                return workflow_result
            
            # Step 4: Validate legacy format
            is_valid_legacy, format_type, legacy_messages = validate_legacy_config_format(
                original_config, expected_version=detected_version.value
            )
            
            if not is_valid_legacy:
                workflow_result['errors'].extend(legacy_messages)
                print(f"Legacy format validation failed: {legacy_messages}")
                return workflow_result
            
            print(f"Legacy format validated: {format_type}")
            workflow_result['audit_trail'].append(f"Legacy format validated: {format_type}")
            
            # Step 5: Check migration compatibility
            is_compatible, _, compatibility_issues = validate_schema_compatibility(
                original_config, target_version
            )
            
            workflow_result['audit_trail'].extend(compatibility_issues)
            
            if not is_migration_available(detected_version.value, target_version):
                workflow_result['errors'].append(f"No migration path available: {detected_version.value} -> {target_version}")
                return workflow_result
            
            # Step 6: Execute migration with validation
            print(f"Executing migration: {detected_version.value} -> {target_version}")
            
            is_valid, final_version, migration_messages, migrated_config = validate_config_with_migration(
                original_config,
                target_version=target_version,
                auto_migrate=True
            )
            
            workflow_result['migration_applied'] = True
            workflow_result['audit_trail'].extend(migration_messages)
            
            if not is_valid:
                workflow_result['errors'].extend(migration_messages)
                print(f"Migration validation failed: {migration_messages}")
                return workflow_result
            
            # Step 7: Final validation and manifest check
            workflow_result['final_version'] = final_version
            workflow_result['migrated_config'] = migrated_config
            workflow_result['validation_passed'] = True
            
            print(f"Migration completed successfully: {detected_version.value} -> {final_version}")
            workflow_result['audit_trail'].append(f"Migration completed: {detected_version.value} -> {final_version}")
            
            # Step 8: Generate migration report
            migration_report = MigrationReport(
                from_version=detected_version.value,
                to_version=final_version,
                timestamp=datetime.now(),
                applied_migrations=[f"migrate_{detected_version.value.replace('.', '_')}_to_{final_version.replace('.', '_')}"],
                warnings=[msg for msg in migration_messages if 'warning' in msg.lower()]
            )
            
            self.migration_reports.append(migration_report)
            workflow_result['audit_trail'].append(f"Migration report generated: {migration_report.to_dict()}")
            
            workflow_result['success'] = True
            return workflow_result
            
        except Exception as e:
            workflow_result['errors'].append(f"Unexpected error: {str(e)}")
            print(f"Workflow failed with error: {e}")
            return workflow_result
    
    def _load_demo_config(self) -> Dict[str, Any]:
        """Load demonstration configuration for workflow example."""
        return {
            'project': {
                'directories': {'major_data_directory': '/data/fly_experiments'},
                'ignore_substrings': ['temp_', 'backup_']
            },
            'datasets': {
                'plume_tracking': {
                    'rig': 'rig1',
                    'dates_vials': {'2024-01-01': [1, 2, 3, 4]}
                }
            },
            'experiments': {
                'baseline_study': {
                    'datasets': ['plume_tracking'],
                    'parameters': {'threshold': 0.5}
                }
            }
        }
    
    def save_migrated_config(self, migrated_config: Dict[str, Any], output_path: str) -> None:
        """Save migrated configuration with proper formatting."""
        import yaml
        
        output_file = Path(output_path)
        
        # Ensure schema_version is at the top
        ordered_config = {'schema_version': migrated_config.get('schema_version', '1.0.0')}
        for key, value in migrated_config.items():
            if key != 'schema_version':
                ordered_config[key] = value
        
        with open(output_file, 'w') as f:
            yaml.dump(ordered_config, f, default_flow_style=False, sort_keys=False)
        
        print(f"Migrated configuration saved to: {output_file}")

# Usage example
if __name__ == "__main__":
    # Execute complete migration workflow
    workflow = ConfigurationMigrationWorkflow("legacy_config.yaml")
    result = workflow.execute_migration_workflow(target_version="1.0.0")
    
    if result['success']:
        print("\n=== Migration Workflow Completed Successfully ===")
        print(f"Original version: {result['original_version']}")
        print(f"Final version: {result['final_version']}")
        print(f"Migration applied: {result['migration_applied']}")
        print(f"Validation passed: {result['validation_passed']}")
        
        print("\nAudit Trail:")
        for i, entry in enumerate(result['audit_trail'], 1):
            print(f"  {i}. {entry}")
        
        # Save migrated configuration
        if result['migrated_config']:
            workflow.save_migrated_config(
                result['migrated_config'], 
                "migrated_config.yaml"
            )
    else:
        print("\n=== Migration Workflow Failed ===")
        print("Errors:")
        for error in result['errors']:
            print(f"  - {error}")
```

### Version-Aware Configuration Loading

Here's how to implement version-aware configuration loading in your applications:

```python
"""
Version-aware configuration loading with automatic migration and fallback handling.
"""

from flyrigloader.config.yaml_config import load_config
from flyrigloader.migration.validators import validate_config_with_migration
from flyrigloader.config.models import LegacyConfigAdapter
import warnings
import logging

# Configure logging for migration tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_version_aware_config(
    config_path: str, 
    target_version: str = "1.0.0",
    allow_auto_migration: bool = True,
    legacy_fallback: bool = True
) -> tuple:
    """
    Load configuration with comprehensive version handling.
    
    Returns:
        tuple: (config_object, was_migrated, migration_messages)
    """
    
    try:
        # Step 1: Attempt to load with modern loader
        config = load_config(config_path, legacy_mode=False)
        
        # Check if it's already the target version
        if hasattr(config, 'schema_version') and config.schema_version == target_version:
            logger.info(f"Configuration already at target version {target_version}")
            return config, False, []
        
        # Step 2: Check if migration is needed
        config_dict = config if isinstance(config, dict) else config.dict()
        
        is_valid, final_version, messages, migrated_config = validate_config_with_migration(
            config_dict,
            target_version=target_version,
            auto_migrate=allow_auto_migration
        )
        
        if is_valid and migrated_config:
            logger.info(f"Configuration migrated to {final_version}")
            
            # Load migrated configuration as Pydantic model
            from flyrigloader.config.models import create_config
            migrated_pydantic = create_config(**migrated_config)
            return migrated_pydantic, True, messages
        
        elif not allow_auto_migration:
            logger.warning("Auto-migration disabled, returning original configuration")
            return config, False, messages
        
        else:
            raise ValueError(f"Migration failed: {messages}")
    
    except Exception as e:
        if legacy_fallback:
            logger.warning(f"Modern loading failed ({e}), attempting legacy fallback")
            
            try:
                # Step 3: Legacy fallback with adapter
                legacy_config = load_config(config_path, legacy_mode=True)
                
                if isinstance(legacy_config, dict):
                    adapter = LegacyConfigAdapter(legacy_config)
                    
                    # Emit deprecation warning
                    warnings.warn(
                        f"Loaded configuration using legacy adapter. "
                        f"Consider migrating to schema version {target_version}.",
                        DeprecationWarning,
                        stacklevel=2
                    )
                    
                    return adapter, False, [f"Legacy fallback applied: {str(e)}"]
                else:
                    return legacy_config, False, []
            
            except Exception as fallback_error:
                logger.error(f"Legacy fallback also failed: {fallback_error}")
                raise ValueError(f"All loading methods failed. Original error: {e}, Fallback error: {fallback_error}")
        else:
            raise

# Example usage in application code
def initialize_application_config(config_path: str):
    """Initialize application with version-aware configuration loading."""
    
    try:
        config, was_migrated, messages = load_version_aware_config(
            config_path,
            target_version="1.0.0",
            allow_auto_migration=True,
            legacy_fallback=True
        )
        
        if was_migrated:
            print("Configuration was automatically migrated:")
            for message in messages:
                print(f"  - {message}")
            
            # Optionally save migrated configuration
            save_choice = input("Save migrated configuration? (y/n): ")
            if save_choice.lower() == 'y':
                migrated_path = config_path.replace('.yaml', '_migrated.yaml')
                
                if hasattr(config, 'dict'):
                    import yaml
                    with open(migrated_path, 'w') as f:
                        yaml.dump(config.dict(), f, default_flow_style=False)
                    print(f"Migrated configuration saved to: {migrated_path}")
        
        return config
        
    except Exception as e:
        print(f"Configuration loading failed: {e}")
        raise

# Usage example
if __name__ == "__main__":
    config = initialize_application_config("experiment_config.yaml")
    print(f"Loaded configuration with schema version: {config.schema_version}")
```

### Migration Troubleshooting Guide

#### Common Version Migration Issues

**Issue 1: Version Detection Failure**

```python
# Problem: Configuration structure doesn't match any known pattern
from flyrigloader.migration.validators import validate_legacy_config_format

problematic_config = {
    'custom_section': {'data': 'value'},
    # Missing required 'project' and 'experiments' sections
}

try:
    is_valid, format_type, messages = validate_legacy_config_format(problematic_config)
    if not is_valid:
        print("Format validation failed:")
        for message in messages:
            print(f"  - {message}")
        
        # Solution: Add missing required sections
        fixed_config = problematic_config.copy()
        fixed_config.update({
            'project': {'directories': {'major_data_directory': '/data'}},
            'experiments': {'default_exp': {'parameters': {}}}
        })
        
        print("Fixed configuration structure")
except Exception as e:
    print(f"Version detection error: {e}")
```

**Issue 2: Migration Path Not Available**

```python
# Problem: Trying to migrate from unsupported version
from flyrigloader.migration.versions import is_migration_available, COMPATIBILITY_MATRIX

from_version = "0.0.1"  # Hypothetical unsupported version
to_version = "1.0.0"

if not is_migration_available(from_version, to_version):
    print(f"No migration path available: {from_version} -> {to_version}")
    
    # Solution: Check available migration paths
    print("Available migration paths:")
    for source, targets in COMPATIBILITY_MATRIX.items():
        target_versions = [v.value for v in targets]
        print(f"  {source.value} -> {target_versions}")
    
    # Manual migration required
    print("Consider manual configuration update or contact support")
```

**Issue 3: Post-Migration Validation Failure**

```python
# Problem: Migrated configuration fails Pydantic validation
from flyrigloader.migration.validators import validate_config_with_migration
from pydantic import ValidationError

legacy_config = {
    'project': {'directories': {'major_data_directory': '/nonexistent/path'}},
    'experiments': {'exp1': {'invalid_field': 'invalid_value'}}
}

try:
    is_valid, version, messages, migrated = validate_config_with_migration(legacy_config)
    
    if not is_valid:
        print("Post-migration validation failed:")
        for message in messages:
            print(f"  - {message}")
        
        # Solution: Fix validation errors manually
        print("Fixing validation errors...")
        
        # Check for specific Pydantic validation errors
        for message in messages:
            if "path" in message.lower() and "exist" in message.lower():
                print("  Fix: Update directory paths to existing locations")
            elif "field" in message.lower():
                print("  Fix: Remove or correct invalid configuration fields")
                
except Exception as e:
    print(f"Migration validation error: {e}")
```

**Issue 4: LegacyConfigAdapter Compatibility Problems**

```python
# Problem: Dictionary operations fail with adapter
from flyrigloader.config.models import LegacyConfigAdapter

legacy_dict = {
    'project': {'directories': {'major_data_directory': '/data'}},
    'experiments': {'exp1': {}}
}

try:
    adapter = LegacyConfigAdapter(legacy_dict)
    
    # This might fail if the underlying Pydantic model is strict
    adapter['project']['new_field'] = 'new_value'
    
except Exception as e:
    print(f"Adapter operation failed: {e}")
    
    # Solution: Use proper Pydantic model fields or migrate first
    print("Solution: Migrate to modern configuration format")
    
    from flyrigloader.config.models import create_config
    modern_config = create_config(
        project_name="migrated_project",
        base_directory="/data",
        experiments=["exp1"]
    )
    
    print(f"Modern config created with version: {modern_config.schema_version}")
```

## Migration Checklist

### Week 1: Configuration Validation and Version Migration
- [ ] **New**: Test configuration version detection with `detect_config_version()`
- [ ] **New**: Validate migration paths using `is_migration_available()`
- [ ] **New**: Execute test migrations with `validate_config_with_migration()`
- [ ] **New**: Update configurations with `schema_version` field
- [ ] Test existing configuration files with `legacy_mode=False`
- [ ] Fix any validation errors (typically missing required fields)
- [ ] Add logging to verify path resolution
- [ ] Update CI/CD to use validated configurations
- [ ] **New**: Update exception handling to catch specific exception types (ConfigError, DiscoveryError, LoadError, TransformError, VersionError)
- [ ] **New**: Test configuration builder functions (create_config, create_experiment, create_dataset)
- [ ] **New**: Implement LegacyConfigAdapter integration with deprecation warnings

### Week 2: API Migration and Registry Setup
- [ ] **New**: Implement migration audit trails using `MigrationReport`
- [ ] **New**: Test pre-flight validation with `validate_manifest()`
- [ ] **New**: Setup automatic migration workflows in application code
- [ ] Identify monolithic `process_experiment_data` calls
- [ ] Replace with `discover_experiment_manifest` + `load_data_file` + `transform_to_dataframe`
- [ ] Test memory usage improvements with decoupled pipeline
- [ ] Update documentation and examples
- [ ] **New**: Register any custom loaders with LoaderRegistry
- [ ] **New**: Register any custom schemas with SchemaRegistry
- [ ] **New**: Test registry-based extensibility
- [ ] **New**: Verify version compatibility across all registry components

### Week 3: Full Migration and Advanced Features
- [ ] **New**: Deploy version-aware configuration loading in production
- [ ] **New**: Setup migration monitoring and alerting
- [ ] **New**: Create migration rollback procedures
- [ ] Adopt attribute-style configuration access
- [ ] Use new transformation utilities
- [ ] Implement custom validation rules
- [ ] Performance testing and optimization
- [ ] **New**: Implement custom transformation handlers if needed
- [ ] **New**: Test deprecation warnings and migration paths
- [ ] **New**: Update error handling to use context preservation
- [ ] **New**: Validate all migration workflows end-to-end

### Week 4: Version Management and Maintenance (Optional)
- [ ] **New**: Setup configuration version governance policies
- [ ] **New**: Implement automated migration testing for new versions
- [ ] **New**: Create migration path documentation for future versions
- [ ] **New**: Setup migration report analysis and archiving
- [ ] **New**: Develop custom loaders for additional file formats
- [ ] **New**: Implement domain-specific schema validators
- [ ] **New**: Test plugin discovery through entry points
- [ ] **New**: Validate transformation chain integrity
- [ ] **New**: Set up comprehensive error monitoring
- [ ] **New**: Document version compatibility matrix for dependencies

## New Architecture Benefits

The decoupled pipeline architecture with comprehensive version management provides several advantages:

### Memory Efficiency
- **Selective Processing**: Load only the files you need
- **Streaming Support**: Process large datasets in batches
- **Memory Monitoring**: Better control over memory usage

### Error Resilience
- **Granular Error Handling**: Specific exception types for each pipeline stage
- **Context Preservation**: Detailed error context for debugging
- **Graceful Degradation**: Continue processing even if some files fail

### Extensibility
- **Registry Pattern**: Add new file formats without modifying core code
- **Plugin Architecture**: Third-party extensions through entry points
- **Schema Validation**: Custom validation logic for different data types

### Performance
- **Parallel Processing**: Independent stages can be parallelized
- **Caching Support**: Cache discovery results for repeated processing
- **Pipeline Optimization**: Optimize each stage independently

### Developer Experience
- **Clear Separation**: Easier to understand and maintain
- **Type Safety**: Full Pydantic model support with IDE autocomplete
- **Comprehensive Testing**: Each stage can be tested independently

### Version Management & Migration
- **Zero Breaking Changes**: Automatic migration preserves research continuity
- **Audit Trails**: Complete migration reports with timestamp and change tracking
- **Rollback Support**: Safe migration with comprehensive validation and error handling
- **Forward Compatibility**: Schema versioning enables future-proof configurations

### Configuration Management
- **Automatic Detection**: Seamless version detection without manual intervention
- **Builder Pattern**: Type-safe configuration construction with comprehensive defaults
- **Legacy Support**: Transparent backward compatibility via LegacyConfigAdapter
- **Validation Integration**: Embedded validation with clear error messages and migration suggestions

## Backward Compatibility

The migration preserves complete backward compatibility:

- **All existing function signatures remain unchanged**
- **Dictionary-style configuration access continues to work**
- **Kedro integration is preserved**
- **All public APIs maintain their contracts**
- **Gradual migration path with deprecation warnings**

## Getting Help

### Documentation
- [Configuration Guide](configuration_guide.md) - Complete configuration reference
- [API Documentation](api_reference.md) - Detailed API documentation
- [Kedro Integration Guide](kedro_integration.md) - Native Kedro DataSet support
- [Examples](../examples/) - Working examples for all use cases
- **Migration Resources**: 
  - Version compatibility matrix in `flyrigloader.migration.versions`
  - Migration workflow examples in this guide
  - Automated migration test suite for validation

### Support
- Check the [FAQ](faq.md) for common questions
- Review [GitHub Issues](https://github.com/your-org/flyrigloader/issues) for known issues
- Open a new issue for migration-specific problems

### Performance
- The new system is typically 10-20% faster due to validation optimization
- Memory usage is reduced for large datasets through selective loading
- Path resolution is more efficient with explicit caching

## Complete Migration Example

Here's a comprehensive example showing the full migration from old to new architecture:

```python
# migration_example.py - Complete migration demonstration
import warnings
from typing import List, Dict, Any
import pandas as pd
from pathlib import Path

# New imports for decoupled architecture
from flyrigloader.config.models import create_config, create_experiment, create_dataset
from flyrigloader.discovery.files import discover_experiment_manifest
from flyrigloader.io.loaders import load_data_file
from flyrigloader.io.transformers import transform_to_dataframe
from flyrigloader.registries import LoaderRegistry, SchemaRegistry
from flyrigloader.exceptions import (
    ConfigError, DiscoveryError, LoadError, TransformError, log_and_raise
)
from flyrigloader import logger

class ModernExperimentProcessor:
    """Example of fully migrated experiment processing class."""
    
    def __init__(self, project_name: str, base_directory: str):
        # Use new configuration builder functions
        self.config = create_config(
            project_name=project_name,
            base_directory=base_directory,
            ignore_substrings=["temp", "backup", "test"],
            extraction_patterns=[
                r"(?P<date>\d{4}-\d{2}-\d{2})",
                r"(?P<subject>\w+)",
                r"(?P<condition>\w+_condition)",
                r"(?P<rig>rig\d+)"
            ]
        )
        
        # Register custom loaders if needed
        self._setup_custom_loaders()
        
        # Register custom schemas if needed
        self._setup_custom_schemas()
    
    def _setup_custom_loaders(self):
        """Register custom loaders for project-specific formats."""
        # Example: Register CSV loader for special format
        class CSVLoader:
            def load(self, path: Path) -> Any:
                import pandas as pd
                return pd.read_csv(path).to_dict('list')
            
            def supports_extension(self, extension: str) -> bool:
                return extension.lower() == '.csv'
            
            @property
            def priority(self) -> int:
                return 5
        
        registry = LoaderRegistry()
        registry.register_loader('.csv', CSVLoader, priority=5)
    
    def _setup_custom_schemas(self):
        """Register custom schemas for project-specific validation."""
        class TimeSeriesSchema:
            def validate(self, data: Any) -> Dict[str, Any]:
                if not isinstance(data, dict):
                    raise TransformError("Data must be dictionary")
                if 't' not in data:
                    raise TransformError("Missing time column")
                return data
            
            @property
            def schema_name(self) -> str:
                return 'time_series'
            
            @property
            def supported_types(self) -> List[str]:
                return ['time_series', 'behavioral']
        
        registry = SchemaRegistry()
        registry.register_schema('time_series', TimeSeriesSchema, priority=10)
    
    def process_experiment_with_error_handling(self, experiment_name: str) -> pd.DataFrame:
        """Process experiment using new decoupled architecture with comprehensive error handling."""
        try:
            # Step 1: Discovery with error handling
            logger.info(f"Starting discovery for experiment: {experiment_name}")
            manifest = discover_experiment_manifest(self.config, experiment_name)
            logger.info(f"Discovered {len(manifest.files)} files")
            
            # Step 2: Loading with error handling and resilience
            raw_data_list = []
            failed_files = []
            
            for file_info in manifest.files:
                try:
                    raw_data = load_data_file(file_info.path)
                    raw_data_list.append({
                        'data': raw_data,
                        'file_path': file_info.path,
                        'metadata': file_info.extracted_metadata
                    })
                except LoadError as e:
                    logger.warning(f"Failed to load {file_info.path}: {e}")
                    failed_files.append(file_info.path)
                    continue
            
            if not raw_data_list:
                raise TransformError(
                    "No files loaded successfully",
                    error_code="TRANSFORM_001",
                    context={"failed_files": failed_files}
                )
            
            # Step 3: Transformation with error handling
            dataframes = []
            failed_transforms = []
            
            for item in raw_data_list:
                try:
                    df = transform_to_dataframe(
                        item['data'], 
                        self.config,
                        metadata=item['metadata'],
                        add_file_path=True,
                        file_path=item['file_path']
                    )
                    dataframes.append(df)
                except TransformError as e:
                    logger.warning(f"Failed to transform {item['file_path']}: {e}")
                    failed_transforms.append(item['file_path'])
                    continue
            
            if not dataframes:
                raise TransformError(
                    "No data transformed successfully",
                    error_code="TRANSFORM_002",
                    context={"failed_transforms": failed_transforms}
                )
            
            # Step 4: Combine results
            combined_df = pd.concat(dataframes, ignore_index=True)
            
            # Log summary
            logger.info(f"Successfully processed {len(dataframes)} files")
            logger.info(f"Combined DataFrame shape: {combined_df.shape}")
            if failed_files:
                logger.warning(f"Failed to load {len(failed_files)} files")
            if failed_transforms:
                logger.warning(f"Failed to transform {len(failed_transforms)} files")
            
            return combined_df
            
        except DiscoveryError as e:
            log_and_raise(
                DiscoveryError(
                    f"Discovery failed for experiment '{experiment_name}'"
                ).with_context({
                    "experiment_name": experiment_name,
                    "config_path": str(self.config.directories.get("major_data_directory")),
                    "original_error": str(e)
                }),
                logger,
                "error"
            )
        except ConfigError as e:
            log_and_raise(
                ConfigError(
                    f"Configuration error for experiment '{experiment_name}'"
                ).with_context({
                    "experiment_name": experiment_name,
                    "original_error": str(e)
                }),
                logger,
                "error"
            )
        except Exception as e:
            log_and_raise(
                TransformError(
                    f"Unexpected error processing experiment '{experiment_name}'"
                ).with_context({
                    "experiment_name": experiment_name,
                    "error_type": type(e).__name__,
                    "original_error": str(e)
                }),
                logger,
                "error"
            )

# Usage example
if __name__ == "__main__":
    # Create processor with new architecture
    processor = ModernExperimentProcessor(
        project_name="fly_behavior_analysis",
        base_directory="/data/experiments"
    )
    
    # Process experiment with full error handling
    try:
        result = processor.process_experiment_with_error_handling("plume_navigation")
        print(f"Successfully processed experiment: {result.shape}")
    except Exception as e:
        print(f"Processing failed: {e}")
```

## Next Steps

After completing the migration:

1. **Review Configuration**: Use the new validation to catch configuration issues early
2. **Optimize Performance**: Use selective loading for large datasets
3. **Improve Debugging**: Enable INFO-level logging for better visibility
4. **Enhance Testing**: Use the new Protocol-based testing interfaces
5. **Consider Advanced Features**: Explore custom validation rules and transformation utilities
6. **Registry Extensions**: Develop custom loaders and schemas for your specific needs
7. **Error Monitoring**: Set up comprehensive error handling and monitoring
8. **Performance Optimization**: Profile and optimize each pipeline stage independently

The migration is designed to be incremental and safe. Start with the zero-change approach and gradually adopt new features as your team becomes comfortable with the enhanced capabilities.