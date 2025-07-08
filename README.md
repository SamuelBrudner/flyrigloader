# flyrigloader

A Python package for managing reading data from the opto rig.

## Project Structure

The project follows a src-layout Python package structure:

```
flyrigloader/
├── src/
│   └── flyrigloader/    # Main package code
│       ├── discovery/   # File discovery module
│       ├── config/      # Configuration handling module
│       └── io/          # Input/output utilities
├── tests/               # Test directory matching package structure
├── docs/                # Documentation
├── logs/                # Log files (auto-created)
├── config/              # Configuration files
└── pyproject.toml       # Project metadata and dependencies
```

## Features

### Logging System

`flyrigloader` includes a comprehensive logging system built with `loguru` that provides:

- **Console logging**: INFO-level logs with colored output for better readability
- **File logging**: DEBUG-level logs with automatic file rotation
- **Automatic setup**: Log directory is created automatically on import
- **Structured output**: Includes timestamps, log levels, file/function info

Example usage:

```python
from loguru import logger

# These will go to both console and log file
logger.info("Processing experiment data")

# These will only go to the log file (DEBUG level)
logger.debug("Detailed variable state: {}", variable)

# Error handling with rich context
try:
    process_data(file)
except Exception as e:
    logger.exception(f"Error processing {file}")
    # Full traceback is automatically included
```

### File Discovery Module

The `discovery` module provides utilities for finding and organizing files based on patterns, with support for:

#### Features

- **Basic file discovery**: Find files by pattern using glob matching
- **Multiple base directories**: Search across multiple directories in a single call
- **Recursive discovery**: Search through nested subdirectories
- **Extension filtering**: Filter files by one or more extensions
- **Pattern filtering**: Include or exclude files based on substring patterns
- **Metadata extraction**: Extract structured metadata from filenames using regex patterns
- **Date parsing**: Automatically parse and extract dates from filenames

#### Usage Examples

```python
from flyrigloader.discovery.files import discover_files

# Find all text files in a directory
files = discover_files("/path/to/data", "*.txt")

# Find all Python files recursively
files = discover_files("/path/to/code", "**/*.py", recursive=True)

# Find specific file types using extension filtering
files = discover_files("/path/to/data", "*", extensions=["csv", "json"])

# Search across multiple directories
dirs = ["/path/to/data1", "/path/to/data2", "/path/to/data3"]
files = discover_files(dirs, "*.csv")

# Filter files by patterns
files = discover_files(
    "/path/to/data", 
    "*.*", 
    ignore_patterns=["._", "temp_"],
    mandatory_substrings=["experiment"]
)

# Extract metadata from filenames
files_with_metadata = discover_files(
    "/path/to/data",
    "exp(?P<experiment_id>\d+)_(?P<animal>\w+).csv",
    extract_patterns=True
)
# Result: {"/path/to/data/exp001_mouse.csv": {"experiment_id": "001", "animal": "mouse"}}

# Parse dates from filenames
files_with_dates = discover_files(
    "/path/to/data",
    "experiment_*.csv",
    parse_dates=True
)
```

### Schema-Validated Configuration System

The `config` module provides a modern, type-safe configuration system built with Pydantic models for validation, type safety, and enhanced developer experience:

#### Features

- **Pydantic schema validation**: Automatic validation with clear error messages
- **Type-safe configuration**: Full IDE autocomplete and type checking support
- **Hierarchical settings**: Access settings at project, dataset, and experiment levels with inheritance
- **Backward compatibility**: Seamless migration from dictionary-based configurations
- **Path resolution**: Clear, logged precedence for data directory resolution
- **Config-aware discovery**: Find files using validated configuration filters
- **Enhanced error handling**: Detailed validation feedback with field-level error reporting

#### Schema Models

The new configuration system uses three validated Pydantic models:

- **`ProjectConfig`**: Project-level settings including directories, ignore patterns, and global extraction rules
- **`DatasetConfig`**: Dataset-specific configuration with rig identification, date/vial mappings, and metadata
- **`ExperimentConfig`**: Experiment-specific configuration including dataset references, analysis parameters, and filters

#### Basic Usage Examples

```python
from flyrigloader.config.yaml_config import load_config
from flyrigloader.config.models import ProjectConfig, DatasetConfig, ExperimentConfig

# Load and validate configuration with Pydantic models
config = load_config("/path/to/config.yaml", legacy_mode=False)

# Access configuration with type safety and autocomplete
project_config = config.get_model('project')  # Returns ProjectConfig instance
data_dir = project_config.directories['major_data_directory']

# Legacy dictionary access still works for backward compatibility
data_dir = config['project']['directories']['major_data_directory']

# Get dataset configuration with validation
dataset_config = config.get_model('dataset', 'plume_tracking')  # Returns DatasetConfig
rig_name = dataset_config.rig
vials = dataset_config.dates_vials['2023-05-01']

# Access experiment parameters with type safety
experiment_config = config.get_model('experiment', 'plume_navigation_analysis')
analysis_params = experiment_config.parameters
```

#### Advanced Configuration Discovery

```python
from flyrigloader.config.discovery import discover_experiment_files, discover_dataset_files

# Find files for a specific experiment with validated configuration
files = discover_experiment_files(
    config=config,
    experiment_name="plume_navigation_analysis",
    base_directory="/path/to/data",
    extensions=["csv"]
)

# Find files for a specific dataset
files = discover_dataset_files(
    config=config,
    dataset_name="plume_tracking",
    base_directory="/path/to/data"
)

# Get validated ignore patterns with hierarchy resolution
from flyrigloader.config.yaml_config import get_ignore_patterns
patterns = get_ignore_patterns(config, experiment="plume_navigation_analysis")
```

### Data Processing with Decoupled Architecture

The `io` module provides a modernized, decoupled architecture that separates data loading from optional transformations:

#### Core Features

- **Pure data loading**: Clean separation between file I/O and data transformation
- **Multiple pickle formats**: Support for regular, gzipped, and pandas pickle formats  
- **Automatic format detection**: Intelligently detects and loads the appropriate format
- **Optional transformations**: DataFrame conversion and metadata integration as separate utilities
- **Memory efficient**: Load only what you need, when you need it
- **Validation**: Comprehensive data dimension and structure validation
- **Pydantic Column Configuration**: Flexible configuration system with strong validation ([documentation](docs/io/column_configuration.md))

#### Decoupled Usage Patterns

```python
from flyrigloader.io.pickle import read_pickle_any_format
from flyrigloader.io.transformers import transform_to_dataframe
from flyrigloader.api import load_data_file

# Pure data loading (no transformation)
raw_data = read_pickle_any_format("/path/to/data.pkl")
raw_data = load_data_file("/path/to/data.pkl")  # Same result, preferred API

# Optional DataFrame transformation when needed
df = transform_to_dataframe(
    exp_matrix=raw_data,
    metadata={"date": "2025-04-01", "fly_id": "fly-123"},
    include_signal_disp=True,  # Include special signal_disp column
    column_list=["t", "x", "y"]  # Only include specified columns
)

# Configuration-driven transformation (recommended)
from flyrigloader.config.yaml_config import load_config

config = load_config("/path/to/config.yaml")
df = transform_to_dataframe(
    exp_matrix=raw_data,
    config=config,
    metadata={"date": "2025-04-01", "fly_id": "fly-123"}
)
```

#### Legacy Compatibility

```python
# Legacy combined approach still supported for backward compatibility
from flyrigloader.io.pickle import make_dataframe_from_matrix, make_dataframe_from_config

# Combined load and transform (legacy pattern)
df = make_dataframe_from_matrix(
    exp_matrix=raw_data,
    metadata={"date": "2025-04-01", "fly_id": "fly-123"},
    include_signal_disp=True,
    column_list=["t", "x", "y"]
)

# Configuration-based transformation (legacy pattern)
df = make_dataframe_from_config(
    exp_matrix=raw_data,
    config_source="path/to/column_config.yaml",
    metadata={"date": "2025-04-01", "fly_id": "fly-123"}
)
```

#### Memory-Efficient Processing

The decoupled architecture enables memory-efficient processing of large datasets:

```python
from flyrigloader.api import discover_experiment_manifest, load_data_file, transform_to_dataframe

# Discover files without loading data
manifest = discover_experiment_manifest(config, "large_experiment")

# Process files one at a time to manage memory
for file_info in manifest.files:
    # Load single file
    raw_data = load_data_file(file_info.path)
    
    # Transform only if needed for this analysis step
    if analysis_step_requires_dataframe:
        df = transform_to_dataframe(raw_data, config=manifest.config)
        process_dataframe(df)
        del df  # Explicit memory management
    else:
        # Work directly with raw data
        process_raw_data(raw_data)
    
    del raw_data  # Clean up
```

## Using as a Library in External Projects

`flyrigloader` is designed to be easily integrated into external data analysis projects. The modernized API provides both high-level entry points for simple use cases and a new decoupled pipeline for advanced control and better memory management.

### High-Level API (Backward Compatible)

The familiar high-level API continues to work with enhanced validation and clearer path resolution:

```python
from flyrigloader.api import load_experiment_files, get_experiment_parameters

# Load all CSV files for a specific experiment with validated configuration
files = load_experiment_files(
    config_path="/path/to/your/config.yaml",
    experiment_name="plume_navigation_analysis", 
    extensions=["csv"]
)
# Now with clear logging: "Using data directory: /path/to/fly_data (from config)"

# Get experiment-specific parameters with type-safe access
parameters = get_experiment_parameters(
    config_path="/path/to/your/config.yaml",
    experiment_name="plume_navigation_analysis"
)

# Process files with validated parameters
for file in files:
    analyze_data(file, parameters)
```

### New Decoupled Pipeline (Recommended)

The new manifest-based workflow provides granular control and better memory management:

```python
from flyrigloader.api import (
    discover_experiment_manifest, 
    load_data_file, 
    transform_to_dataframe
)

# Step 1: Discovery Phase - Get file manifest without loading data
manifest = discover_experiment_manifest(
    config_path="/path/to/your/config.yaml",
    experiment_name="plume_navigation_analysis"
)

print(f"Found {len(manifest.files)} files:")
for file_info in manifest.files:
    print(f"  {file_info.path} - {file_info.size_mb:.1f}MB")

# Step 2: Selective Loading - Load only specific files
for file_info in manifest.files:
    if file_info.metadata.get('condition') == 'treatment':
        
        # Load raw data from single file
        raw_data = load_data_file(file_info.path)
        
        # Step 3: Optional Transformation - Convert to DataFrame only if needed
        if analysis_requires_dataframe:
            df = transform_to_dataframe(
                exp_matrix=raw_data,
                config=manifest.config,
                metadata=file_info.metadata
            )
            process_dataframe(df)
        else:
            # Work directly with raw data for memory efficiency
            process_raw_data(raw_data)

# Memory-efficient batch processing
batch_size = 5
for i in range(0, len(manifest.files), batch_size):
    batch = manifest.files[i:i+batch_size]
    
    # Load batch of files
    batch_data = []
    for file_info in batch:
        raw_data = load_data_file(file_info.path)
        batch_data.append(raw_data)
    
    # Process batch
    process_batch(batch_data)
    
    # Clear memory
    del batch_data
```

### Enhanced Configuration Support

The new system provides flexible configuration loading with validation:

```python
from flyrigloader.config.yaml_config import load_config

# Load with Pydantic validation
config = load_config("/path/to/your/config.yaml", legacy_mode=False)

# Access with type safety
experiment_config = config.get_model('experiment', 'plume_navigation_analysis')
analysis_params = experiment_config.parameters

# Legacy dictionary access still works
files = load_experiment_files(
    config=config,  # Using validated config object
    experiment_name="plume_navigation_analysis",
    extract_metadata=True
)

# Kedro-style parameters dictionaries are validated automatically
kedro_params = {
    "project": {
        "directories": {"major_data_directory": "/path/to/data"}
    },
    "datasets": {
        "my_dataset": {
            "rig": "rig_name",
            "dates_vials": {"2023-01-01": [1, 2, 3]}
        }
    },
    "experiments": {
        "my_experiment": {
            "datasets": ["my_dataset"],
            "parameters": {"threshold": 0.5}  # Now formally supported
        }
    }
}

# Use pre-loaded Kedro parameters with validation
files = load_dataset_files(
    config=kedro_params,
    dataset_name="my_dataset",
    extract_metadata=True
)
```

### Metadata Extraction with Enhanced Validation

```python
# Extract metadata with validated patterns
files_with_metadata = load_experiment_files(
    config_path="/path/to/your/config.yaml",
    experiment_name="plume_navigation_analysis",
    extensions=["csv"],
    extract_metadata=True
)

# Result: {"/path/to/file.csv": {"date": "20230101", "condition": "control", ...}}

# Parse dates with multiple format support
files_with_dates = load_experiment_files(
    config_path="/path/to/your/config.yaml", 
    experiment_name="plume_navigation_analysis",
    extensions=["csv"],
    extract_metadata=True,
    parse_dates=True
)

# Result: Enhanced metadata with parsed dates
# {"/path/to/file.csv": {
#     "date": "20230101", 
#     "condition": "control", 
#     "parsed_date": datetime(2023, 1, 1),
#     "trial_id": "001"
# }}
```

### Path and File Utilities

The API also provides utility functions for common file and path operations:

```python
from flyrigloader.api import (
    get_file_statistics,        # Get comprehensive file stats
    ensure_dir_exists,          # Create directory if needed
    check_if_file_exists,       # Check if file exists
    get_path_relative_to,       # Get relative path with error handling
    get_path_absolute,          # Convert to absolute path
    get_common_base_dir         # Find common base directory for multiple paths
)

# Get comprehensive file statistics
stats = get_file_statistics("/path/to/file.txt")
# Result: {"size": 1024, "mtime": datetime(...), "is_readable": True, ...}

# Ensure directory exists before writing
output_dir = ensure_dir_exists("/path/to/output")

# For standard path operations, use Python's pathlib directly:
from pathlib import Path

filename = Path("/path/to/file.txt").name     # Get filename
extension = Path("/path/to/file.txt").suffix  # Get extension with dot
parent = Path("/path/to/file.txt").parent     # Get parent directory
resolved = Path("../file.txt").resolve()      # Normalize path
```

### Migration and Documentation

For users upgrading from the legacy system or needing detailed configuration guidance:

- **[Migration Guide](docs/migration_guide.md)** - Step-by-step instructions for transitioning from legacy configurations (5-minute migration)
- **[Configuration Guide](docs/configuration_guide.md)** - Comprehensive documentation of all configuration options and Pydantic schema models
- **[Migration Example](examples/external_project/migration_example.py)** - Practical demonstration script showing side-by-side legacy vs. new patterns

### Quick Migration Options

1. **Zero-Change Migration** (Recommended): Your existing code works unchanged with enhanced validation and clearer error messages
2. **Gradual Migration**: Enable new features incrementally while maintaining backward compatibility
3. **Full Migration**: Adopt the decoupled architecture for maximum control and memory efficiency

See the [examples directory](examples/external_project) for complete demonstrations of integrating `flyrigloader` into external analysis projects with both legacy and modern patterns.

## Configuration Structure with Pydantic Schema Validation

The configuration follows a validated three-tier hierarchical structure (project/datasets/experiments) using Pydantic models for type safety and automatic validation:

```yaml
# Enhanced configuration with full Pydantic validation
project:
  directories:
    major_data_directory: "/path/to/fly_data"
    backup_directory: "/path/to/backup"           # Optional additional directories
  ignore_substrings:                              # Validated string patterns
    - "._"
    - "temp_"
    - "backup"
  mandatory_experiment_strings:                   # Required patterns for experiment files
    - "experiment"
    - "trial"
  extraction_patterns:                            # Compiled regex patterns with validation
    - ".*_(?P<date>\\d{4}-\\d{2}-\\d{2})_(?P<condition>\\w+)_(?P<replicate>\\d+)\\.csv"

datasets:
  plume_tracking:                                 # DatasetConfig validation
    rig: "rig1"                                   # Required, validated rig identifier
    dates_vials:                                  # Date format validation
      "2023-05-01": [1, 2, 3, 4]
      "2023-05-02": [5, 6, 7, 8]
    metadata:                                     # Optional dataset-specific metadata
      extraction_patterns:
        - ".*_(?P<temperature>\\d+)C"
      description: "Temperature gradient experiments"
  
  odor_response:
    rig: "rig2"
    dates_vials:
      "2023-05-03": [1, 2, 3]
      "2023-05-04": [4, 5, 6]

experiments:
  plume_navigation_analysis:                      # ExperimentConfig validation
    datasets:                                     # Required dataset references
      - "plume_tracking"
    parameters:                                   # Previously undocumented, now formally supported
      analysis_window: 10.0                      # Analysis-specific parameters
      threshold: 0.5
      method: "correlation"
    filters:                                      # Experiment-specific filters
      ignore_substrings:
        - "test"
        - "calibration"
      mandatory_experiment_strings:
        - "trial"
        - "navigation"
    metadata:                                     # Experiment-specific metadata
      description: "Plume navigation behavioral analysis"
      analysis_type: "behavioral"
      extraction_patterns:
        - ".*_(?P<trial_id>\\d+)_navigation\\.csv"
```

### Schema Validation Benefits

The new Pydantic-based configuration system provides:

- **Immediate validation**: Configuration errors are caught at load time with specific field-level feedback
- **Type safety**: Full IDE autocomplete and type checking support
- **Clear documentation**: Each field is documented with examples and validation rules
- **Backward compatibility**: Existing YAML files work unchanged with the legacy adapter
- **Security**: Built-in path traversal protection and input sanitization

### Metadata Extraction Configuration

The configuration supports defining regex patterns for extracting metadata from filenames at three levels:

1. **Project level**: Applies to all files
2. **Experiment level**: Applies to files for that experiment
3. **Dataset level**: Applies to files for that dataset

Patterns are specified using Python regex named capture groups (`(?P<name>pattern)`). When multiple patterns match a filename, all extracted fields are combined. When conflicts occur, experiment-level patterns take precedence over dataset-level patterns, which take precedence over project-level patterns.

Example pattern for a file like `data_20230415_control_01.csv`:
```yaml
extraction_patterns:
  - .*_(?P<date>\d{8})_(?P<condition>\w+)_(?P<replicate>\d+)\.csv
```

This would extract:
- `date`: "20230415"
- `condition`: "control"
- `replicate`: "01"

When `parse_dates=True` is specified, the API will attempt to convert any field named `date` (or containing the word "date") into a Python datetime object, which will be added as a new field with the prefix `parsed_`.

## Data Path Resolution

The new configuration system provides clear, predictable data path resolution with explicit logging to eliminate ambiguity:

### Path Resolution Precedence

The system follows a strict precedence order when determining the data directory:

1. **Explicit parameter** (highest priority): `base_directory` parameter passed to API functions
2. **Configuration file**: `project.directories.major_data_directory` from validated config
3. **Environment variable** (CI/CD override): `FLYRIGLOADER_DATA_DIR` environment variable

### Path Resolution Logging

Every data path resolution decision is clearly logged for transparency:

```python
from flyrigloader.api import load_experiment_files

# Clear logging output shows exactly which path is used
files = load_experiment_files(
    config_path="/path/to/config.yaml",
    experiment_name="my_experiment"
)
# Log output: "Using data directory: /path/to/fly_data (from config: major_data_directory)"

# Explicit override takes precedence  
files = load_experiment_files(
    config_path="/path/to/config.yaml", 
    experiment_name="my_experiment",
    base_directory="/custom/path"
)
# Log output: "Using data directory: /custom/path (from explicit parameter)"
```

### Multiple Data Directory Support

The enhanced configuration schema supports multiple data directories for future flexibility:

```yaml
project:
  directories:
    major_data_directory: "/primary/data/path"
    backup_directory: "/backup/data/path"
    archive_directory: "/archive/data/path"
```

### Path Validation and Security

All data paths are validated with built-in security checks:

- **Existence validation**: Paths are checked for existence (test-environment aware)
- **Path traversal protection**: Built-in security against directory traversal attacks
- **Absolute path resolution**: Relative paths are automatically resolved to absolute paths
- **Permission checking**: Read/write permissions are validated where applicable

## Development

### Installation for Development

```bash
# Clone the repository
git clone <repository-url>
cd flyrigloader

# Create the development environment
./setup_env.sh --dev
```

### Testing

The project uses pytest for testing. Tests can be run with:

```bash
pytest
```

Or to run specific tests:

```bash
pytest tests/flyrigloader/discovery/test_files.py
```

### Development Approach

The project is developed using:

1. **Test-Driven Development (TDD)**:
   - Write failing tests first
   - Implement code to make tests pass
   - Refactor as needed while keeping tests passing

2. **Code Quality Best Practices**:
   - Modular and maintainable class-based architecture
   - Use of modern Python features (walrus operator, dictionary union)
   - Clear separation of concerns between components
   - Comprehensive test coverage for all functionality
   - Linting and code quality checks

3. **Refactoring Strategy**:
   - Break complex functions into smaller, focused methods
   - Improve error handling and logging
   - Extract reusable components (like PatternMatcher)
   - Ensure backward compatibility when enhancing functionality
   - Use descriptive naming and appropriate documentation

## License

Released under the MIT License.
