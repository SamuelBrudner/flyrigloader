# FlyRigLoader Configuration Guide

## Table of Contents

1. [Overview](#overview)
2. [Configuration Schema Models](#configuration-schema-models)
3. [Hierarchical Configuration Structure](#hierarchical-configuration-structure)
4. [YAML Configuration Examples](#yaml-configuration-examples)
5. [Configuration Loading and Validation](#configuration-loading-and-validation)
6. [Path Resolution and Data Directories](#path-resolution-and-data-directories)
7. [Legacy Compatibility](#legacy-compatibility)
8. [Validation and Error Handling](#validation-and-error-handling)
9. [Advanced Configuration Patterns](#advanced-configuration-patterns)
11. [Troubleshooting](#troubleshooting)

## Overview

FlyRigLoader has been completely modernized with a new Pydantic-based configuration system that provides:

- **Type-safe configuration** with automatic validation
- **Schema-driven approach** that prevents configuration errors at load time
- **Hierarchical inheritance** from project → dataset → experiment levels
- **Backward compatibility** with existing dictionary-style access patterns
- **Enhanced error messages** for easier debugging
- **IDE autocomplete support** through Pydantic models

### Key Benefits

- **Immediate validation feedback**: Configuration errors are caught when loading, not during runtime
- **Self-documenting**: Schema models provide clear documentation of all available options
- **Flexible**: Supports both new attribute-based access and legacy dictionary patterns
- **Secure**: Built-in path traversal protection and input sanitization
- **Extensible**: Designed to accommodate future enhancements without breaking changes

## Configuration Schema Models

The new configuration system is built around three core Pydantic models that define the expected structure and validation rules for each level of the configuration hierarchy.

### ProjectConfig Model

The `ProjectConfig` model defines project-level settings that apply globally across all datasets and experiments.

**Schema Definition:**
```python
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

from flyrigloader.config.validators import PathSecurityPolicy


class ProjectConfig(BaseModel):
    path_security: Optional[PathSecurityPolicy] = Field(
        default=None,
        description="Allow/deny policy applied during directory path validation",
    )
    directories: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary of directory paths including major_data_directory"
    )
    ignore_substrings: Optional[List[str]] = Field(
        default=None,
        description="List of substring patterns to ignore during file discovery"
    )
    mandatory_experiment_strings: Optional[List[str]] = Field(
        default=None,
        description="List of strings that must be present in experiment files"
    )
    extraction_patterns: Optional[List[str]] = Field(
        default=None,
        description="List of regex patterns for extracting metadata from filenames"
    )
```

**Key Features:**
- **Directory validation**: Automatically checks path existence and security with configurable sensitive roots
- **Pattern compilation**: Validates regex patterns at load time to prevent runtime errors
- **Flexible structure**: Supports additional fields for future extensibility

### DatasetConfig Model

The `DatasetConfig` model defines dataset-specific configuration including rig identification and experimental structure.

**Schema Definition:**
```python
class DatasetConfig(BaseModel):
    rig: str = Field(
        description="Rig identifier string"
    )
    dates_vials: Dict[str, List[int]] = Field(
        default_factory=dict,
        description="Dictionary mapping date strings to lists of vial numbers"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata dictionary for dataset-specific information"
    )
```

**Validation Features:**
- **Rig name validation**: Ensures rig identifiers follow naming conventions
- **Date format checking**: Supports multiple date formats with validation
- **Vial number validation**: Ensures vial numbers are integers and properly formatted

### ExperimentConfig Model

The `ExperimentConfig` model defines experiment-specific configuration including dataset references and analysis parameters.

**Schema Definition:**
```python
class ExperimentConfig(BaseModel):
    datasets: List[str] = Field(
        default_factory=list,
        description="List of dataset names to include in this experiment"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Dictionary of experiment-specific parameters"
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Dictionary containing filter configurations"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata dictionary for experiment-specific information"
    )
```

**Advanced Features:**
- **Dataset reference validation**: Ensures referenced datasets exist
- **Parameter flexibility**: Supports any parameter structure for analysis flexibility
- **Filter inheritance**: Can override project-level filtering rules

## Hierarchical Configuration Structure

The configuration system follows a three-tier hierarchy with inheritance and override capabilities:

```
Project Level (global settings)
├── Dataset Level (inherits from project)
│   ├── Dataset A
│   ├── Dataset B
│   └── Dataset C
└── Experiment Level (inherits from project + referenced datasets)
    ├── Experiment 1 (uses Dataset A, B)
    ├── Experiment 2 (uses Dataset C)
    └── Experiment 3 (uses Dataset A, C)
```

### Inheritance Rules

1. **Project settings** are inherited by all datasets and experiments
2. **Dataset settings** can override project settings for that specific dataset
3. **Experiment settings** can override both project and dataset settings
4. **Explicit parameters** always take precedence over inherited values

### Configuration Precedence Order

For any given setting, the precedence order is:

1. **Experiment-specific setting** (highest precedence)
2. **Dataset-specific setting**
3. **Project-level setting**
4. **Default value** (lowest precedence)

## YAML Configuration Examples

### Basic Configuration Structure

```yaml
# example_config.yaml
project:
  directories:
    major_data_directory: "/path/to/fly_data"
    backup_directory: "/path/to/backup"
  ignore_substrings:
    - "._"
    - "temp"
    - "backup"
  mandatory_experiment_strings:
    - "experiment"
    - "trial"
  extraction_patterns:
    - "(?P<date>\\d{4}-\\d{2}-\\d{2})"
    - "(?P<rig>rig\\d+)"
    - "(?P<vial>vial\\d+)"

datasets:
  plume_tracking:
    rig: "rig1"
    dates_vials:
      "2023-05-01": [1, 2, 3, 4]
      "2023-05-02": [5, 6, 7, 8]
      "2023-05-03": [9, 10, 11, 12]
    metadata:
      description: "Plume tracking behavioral experiments"
      extraction_patterns:
        - "(?P<temperature>\\d+)C"
  
  odor_response:
    rig: "rig2"
    dates_vials:
      "2023-05-01": [13, 14, 15, 16]
      "2023-05-02": [17, 18, 19, 20]
    metadata:
      description: "Odor response experiments"

experiments:
  plume_navigation_analysis:
    datasets: ["plume_tracking"]
    parameters:
      analysis_window: 10.0
      threshold: 0.5
      method: "correlation"
      smoothing_kernel: "gaussian"
    filters:
      ignore_substrings: ["calibration"]  # Experiment-specific override
    metadata:
      description: "Analysis of plume navigation behavior"
      analysis_type: "behavioral"
  
  multi_rig_comparison:
    datasets: ["plume_tracking", "odor_response"]
    parameters:
      comparison_method: "statistical"
      significance_level: 0.05
    metadata:
      description: "Cross-rig comparison analysis"
```

### Advanced Configuration with Multiple Data Directories

```yaml
# advanced_config.yaml
project:
  directories:
    major_data_directory: "/primary/fly_data"
    secondary_data_directory: "/backup/fly_data"
    analysis_output_directory: "/results/analysis"
    temp_directory: "/tmp/flyrig_processing"
  ignore_substrings:
    - "._"
    - ".DS_Store"
    - "temp"
    - "backup"
    - "calibration"
  mandatory_experiment_strings:
    - "experiment"
  extraction_patterns:
    - "(?P<date>\\d{4}-\\d{2}-\\d{2})"
    - "(?P<time>\\d{2}-\\d{2}-\\d{2})"
    - "(?P<rig>rig\\d+)"
    - "(?P<vial>vial\\d+)"
    - "(?P<session>session\\d+)"

datasets:
  high_resolution_tracking:
    rig: "rig1_highres"
    dates_vials:
      "2023-06-01": [1, 2, 3, 4, 5]
      "2023-06-02": [6, 7, 8, 9, 10]
    metadata:
      description: "High-resolution tracking experiments"
      sampling_rate: 120  # Hz
      camera_resolution: "1920x1080"
      extraction_patterns:
        - "(?P<fps>\\d+fps)"
        - "(?P<resolution>\\d+x\\d+)"
  
  temperature_gradient:
    rig: "rig2_thermal"
    dates_vials:
      "2023-06-01": [11, 12, 13, 14]
      "2023-06-02": [15, 16, 17, 18]
    metadata:
      description: "Temperature gradient experiments"
      temperature_range: "20-35C"
      gradient_type: "linear"

experiments:
  comprehensive_behavior_analysis:
    datasets: ["high_resolution_tracking", "temperature_gradient"]
    parameters:
      analysis_pipeline: "comprehensive"
      include_thermal_analysis: true
      tracking_algorithm: "deeplabcut"
      smoothing_window: 5
      velocity_threshold: 2.0
      acceleration_threshold: 10.0
    filters:
      mandatory_experiment_strings: ["behavior", "tracking"]
    metadata:
      description: "Comprehensive behavioral analysis with thermal component"
      analysis_version: "2.1"
      expected_duration_hours: 48
```

## Configuration Loading and Validation

### Loading Configuration

The configuration system supports multiple loading methods:

#### Method 1: From YAML File

```python
from flyrigloader.config.yaml_config import load_config

# Load and validate configuration
config = load_config("path/to/config.yaml")

# The returned config is a dictionary with validated Pydantic models internally
print(f"Project directory: {config['project']['directories']['major_data_directory']}")
```

#### Method 2: From Dictionary (Kedro-style)

```python
# Pre-loaded configuration dictionary (e.g., from Kedro parameters)
config_dict = {
    "project": {
        "directories": {"major_data_directory": "/data/flies"}
    },
    "datasets": {
        "test_dataset": {
            "rig": "rig1",
            "dates_vials": {"2023-05-01": [1, 2, 3]}
        }
    }
}

config = load_config(config_dict)
```

#### Method 3: Using LegacyConfigAdapter for Enhanced Access

```python
from flyrigloader.config.models import LegacyConfigAdapter

# Create adapter for enhanced functionality
adapter = LegacyConfigAdapter(config_dict)

# Dictionary-style access (backward compatible)
project_dir = adapter['project']['directories']['major_data_directory']

# Get underlying Pydantic models for advanced usage
project_model = adapter.get_model('project')
dataset_model = adapter.get_model('dataset', 'test_dataset')

# Validate all sections
is_valid = adapter.validate_all()
```

### Validation Features

#### Automatic Validation During Load

```python
from flyrigloader.exceptions import ConfigError

# Invalid configuration will raise detailed ConfigError
invalid_config = {
    "project": {
        "directories": {"major_data_directory": "/invalid/../path"}  # Path traversal
    },
    "datasets": {
        "bad_dataset": {
            "rig": "",  # Empty rig name
            "dates_vials": {
                "invalid-date": ["not_a_number"]  # Invalid vial numbers
            }
        }
    }
}

try:
    config = load_config(invalid_config)
except ConfigError as e:
    print("Configuration validation failed:")
    for error in e.context.get("validation_errors", []):
        print(f"  {error}")
```

#### Runtime Validation Functions

```python
from flyrigloader.config.yaml_config import validate_config_dict

# Validate an existing configuration
is_valid = validate_config_dict(config_dict)

# Validate specific patterns
from flyrigloader.config.validators import pattern_validation

try:
    compiled_pattern = pattern_validation(r"(?P<date>\d{4}-\d{2}-\d{2})")
    print("Pattern is valid")
except ValueError as e:
    print(f"Invalid pattern: {e}")
```

## Path Resolution and Data Directories

### Path Resolution Precedence

The system follows a clear precedence order for determining data directories:

1. **Explicit `base_directory` parameter** (highest precedence)
2. **Configuration `major_data_directory`**
3. **Environment variable override** (for CI/CD scenarios)
4. **Default fallback** (current working directory)

### Path Resolution Logging

Every path resolution decision is logged clearly:

```python
import logging
logging.basicConfig(level=logging.INFO)

from flyrigloader.api import load_experiment_files

# This will log: "Using data directory: /explicit/path"
files = load_experiment_files(
    config="config.yaml",
    experiment_name="test_exp",
    base_directory="/explicit/path"  # Explicit override
)

# This will log: "Using data directory: /config/data/path (from configuration)"
files = load_experiment_files(
    config="config.yaml",
    experiment_name="test_exp"
    # Uses major_data_directory from config
)
```

### Security Features

#### Path Traversal Protection

```python
# These paths will be rejected with security errors:
dangerous_paths = [
    "/etc/passwd",           # System path
    "../../../secret",       # Path traversal
    "file:///etc/hosts",     # File URI
    "http://evil.com/data"   # Remote URL
]

for path in dangerous_paths:
    try:
        config = {"project": {"directories": {"major_data_directory": path}}}
        load_config(config)
    except (ValueError, PermissionError) as e:
        print(f"Security validation blocked: {path} - {e}")
```

#### Configuring Sensitive Root Policies

Legitimate deployments sometimes store data under system directories such as `/var`.
Use the `path_security` section of the project configuration to allow those locations
explicitly or to extend the deny list:

```yaml
project:
  path_security:
    allow_roots:
      - /var/lib/flyrigloader
    deny_roots:
      - /srv/secret_backups
    inherit_defaults: true
  directories:
    major_data_directory: /var/lib/flyrigloader
```

The validator logs each decision at DEBUG level. When a path is allowed by a configured
root you will see a message similar to `Path '/var/lib/flyrigloader' allowed by configured
allow root '/var/lib/flyrigloader'`. Setting `inherit_defaults: false` opts out of the
built-in deny list entirely.

#### Test Environment Behavior

```python
import os

# During testing, path validation is relaxed
os.environ['PYTEST_CURRENT_TEST'] = 'test_session'

# This will work in test environment even if path doesn't exist
test_config = {
    "project": {
        "directories": {"major_data_directory": "/nonexistent/test/path"}
    }
}
config = load_config(test_config)  # No FileNotFoundError in tests
```

## Legacy Compatibility

### Backward Compatibility Features

The new configuration system maintains full backward compatibility with existing code:

```python
# Old code continues to work unchanged
config = load_config("config.yaml")

# Dictionary access patterns work exactly as before
project_settings = config["project"]
dataset_info = config["datasets"]["my_dataset"]
experiment_params = config["experiments"]["my_experiment"]["parameters"]

# All existing helper functions work with new configs
from flyrigloader.config.yaml_config import (
    get_ignore_patterns,
    get_mandatory_substrings,
    get_dataset_info,
    get_experiment_info
)

patterns = get_ignore_patterns(config, experiment="my_experiment")
mandatory_strings = get_mandatory_substrings(config, experiment="my_experiment")
```

## Troubleshooting

### Common Configuration Issues

#### Issue 1: ConfigError on Load

**Problem:** Configuration fails to load with Pydantic validation errors.

**Solution:**
```python
# Debug validation errors step by step
def debug_config_validation(config_path):
    from flyrigloader.exceptions import ConfigError

    try:
        config = load_config(config_path)
        print("✅ Configuration loaded successfully")
    except ConfigError as e:
        print("❌ Validation errors found:")

        for error in e.context.get("validation_errors", []):
            print(f"\nField: {error}")

            # Provide specific guidance based on error type
            if "rig" in error and "invalid characters" in error:
                print("💡 Tip: Rig names should only contain letters, numbers, underscore, and hyphen")
            elif "dates_vials" in error:
                print("💡 Tip: Dates should be in YYYY-MM-DD format, vials should be lists of integers")
            elif "extraction_patterns" in error:
                print("💡 Tip: Check regex pattern syntax - escape backslashes in YAML strings")
```

#### Issue 2: Path Resolution Problems

**Problem:** Data directory not found or permission errors.

**Solution:**
```python
def debug_path_resolution(config_path, base_directory=None):
    """Debug path resolution issues."""
    
    config = load_config(config_path)
    
    print("Path Resolution Debug:")
    print("=====================")
    
    # Check explicit parameter
    if base_directory:
        print(f"1. Explicit base_directory: {base_directory}")
        try:
            path_existence_validator(base_directory)
            print("   ✅ Path exists and is valid")
        except Exception as e:
            print(f"   ❌ Path issue: {e}")
    
    # Check configuration directory
    config_dir = config.get("project", {}).get("directories", {}).get("major_data_directory")
    if config_dir:
        print(f"2. Configuration major_data_directory: {config_dir}")
        try:
            path_existence_validator(config_dir)
            print("   ✅ Path exists and is valid")
        except Exception as e:
            print(f"   ❌ Path issue: {e}")
    
    # Check environment variable
    env_dir = os.getenv("FLYRIG_DATA_DIR")
    if env_dir:
        print(f"3. Environment FLYRIG_DATA_DIR: {env_dir}")
        try:
            path_existence_validator(env_dir)
            print("   ✅ Path exists and is valid")
        except Exception as e:
            print(f"   ❌ Path issue: {e}")
    
    # Final resolution
    try:
        from flyrigloader.api import _resolve_base_directory
        resolved = _resolve_base_directory(config, base_directory, "debug")
        print(f"\n🎯 Final resolved path: {resolved}")
    except Exception as e:
        print(f"\n❌ Path resolution failed: {e}")
```

#### Issue 3: Regex Pattern Compilation Errors

**Problem:** Extraction patterns fail to compile.

**Solution:**
```python
def debug_regex_patterns(config_path):
    """Debug regex pattern issues."""
    
    config = load_config(config_path)
    
    # Collect all patterns from configuration
    patterns = []
    
    # Project-level patterns
    project_patterns = config.get("project", {}).get("extraction_patterns", [])
    for pattern in project_patterns:
        patterns.append(("project", pattern))
    
    # Dataset-level patterns
    for dataset_name, dataset_config in config.get("datasets", {}).items():
        dataset_patterns = dataset_config.get("metadata", {}).get("extraction_patterns", [])
        for pattern in dataset_patterns:
            patterns.append((f"dataset.{dataset_name}", pattern))
    
    # Experiment-level patterns
    for exp_name, exp_config in config.get("experiments", {}).items():
        exp_patterns = exp_config.get("metadata", {}).get("extraction_patterns", [])
        for pattern in exp_patterns:
            patterns.append((f"experiment.{exp_name}", pattern))
    
    print("Regex Pattern Validation:")
    print("========================")
    
    for source, pattern in patterns:
        print(f"\nSource: {source}")
        print(f"Pattern: {pattern}")
        
        try:
            compiled = re.compile(pattern)
            print("✅ Pattern compiles successfully")
            
            # Test with sample data
            sample_filenames = [
                "2023-05-01_rig1_vial1_experiment.pkl",
                "experiment_rig2_2023-05-02_session1.pkl",
                "data_2023-05-03_trial1.pkl"
            ]
            
            matches_found = False
            for filename in sample_filenames:
                match = compiled.search(filename)
                if match:
                    print(f"   Matches '{filename}': {match.groupdict()}")
                    matches_found = True
            
            if not matches_found:
                print("   ⚠️  Pattern doesn't match any sample filenames")
                
        except re.error as e:
            print(f"❌ Pattern compilation failed: {e}")
            
            # Provide specific guidance
            if "unterminated character set" in str(e):
                print("💡 Tip: Check for unmatched [ or ] brackets")
            elif "invalid group reference" in str(e):
                print("💡 Tip: Check named group syntax (?P<name>...)")
            elif "unbalanced parenthesis" in str(e):
                print("💡 Tip: Check for unmatched ( or ) parentheses")
```

#### Issue 4: Performance Issues with Large Configurations

**Problem:** Configuration loading is slow with many datasets/experiments.

**Solution:**
```python
def optimize_large_config(config_path):
    """Optimize configuration for better performance."""
    
    config = load_config(config_path)
    
    print("Configuration Optimization Analysis:")
    print("===================================")
    
    # Analyze configuration size
    dataset_count = len(config.get("datasets", {}))
    experiment_count = len(config.get("experiments", {}))
    
    print(f"Datasets: {dataset_count}")
    print(f"Experiments: {experiment_count}")
    
    if dataset_count > 50:
        print("\n💡 Performance tip: Consider splitting large configurations:")
        print("   - Use separate config files per project phase")
        print("   - Load only required datasets for specific analysis")
    
    # Check for redundant patterns
    all_patterns = []
    for section in ["project", "datasets", "experiments"]:
        if section == "project":
            patterns = config.get(section, {}).get("extraction_patterns", [])
            all_patterns.extend(patterns)
        else:
            for item_config in config.get(section, {}).values():
                patterns = item_config.get("metadata", {}).get("extraction_patterns", [])
                all_patterns.extend(patterns)
    
    duplicate_patterns = [p for p in set(all_patterns) if all_patterns.count(p) > 1]
    if duplicate_patterns:
        print(f"\n⚠️  Found {len(duplicate_patterns)} duplicate patterns:")
        for pattern in duplicate_patterns:
            print(f"   - {pattern}")
        print("💡 Tip: Move common patterns to project level")
    
    # Suggest caching strategies
    print("\n💡 Performance optimization suggestions:")
    print("   - Use LegacyConfigAdapter for repeated access")
    print("   - Cache compiled regex patterns")
    print("   - Consider lazy loading for unused sections")
```

### Getting Help

#### Enable Debug Logging

```python
import logging

# Enable detailed logging for troubleshooting
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Now configuration operations will provide detailed logs
config = load_config("config.yaml")
```

#### Configuration Health Check

```python
def configuration_health_check(config_path):
    """Comprehensive configuration health check."""
    
    print("FlyRigLoader Configuration Health Check")
    print("======================================")
    
    try:
        # Load configuration
        config = load_config(config_path)
        print("✅ Configuration loads successfully")
        
        # Validate with Pydantic
        adapter = LegacyConfigAdapter(config)
        if adapter.validate_all():
            print("✅ All Pydantic validations pass")
        else:
            print("⚠️  Some Pydantic validations failed")
        
        # Check required sections
        required_sections = ["project", "datasets"]
        for section in required_sections:
            if section in config:
                print(f"✅ Required section '{section}' present")
            else:
                print(f"❌ Required section '{section}' missing")
        
        # Check path accessibility
        data_dir = config.get("project", {}).get("directories", {}).get("major_data_directory")
        if data_dir:
            try:
                path_existence_validator(data_dir)
                print(f"✅ Data directory accessible: {data_dir}")
            except Exception as e:
                print(f"⚠️  Data directory issue: {e}")
        
        # Check regex patterns
        try:
            debug_regex_patterns(config_path)
            print("✅ All regex patterns compile successfully")
        except Exception as e:
            print(f"⚠️  Regex pattern issues found: {e}")
        
        print("\n✅ Configuration health check completed")
        
    except Exception as e:
        print(f"❌ Configuration health check failed: {e}")
        return False
    
    return True

# Run health check
configuration_health_check("config.yaml")
```

---

This configuration guide provides comprehensive documentation for the new Pydantic-based configuration system in FlyRigLoader. The system offers powerful validation, type safety, and enhanced developer experience while maintaining full backward compatibility with existing configurations.

For additional support or feature requests, please refer to the project documentation or open an issue in the repository.