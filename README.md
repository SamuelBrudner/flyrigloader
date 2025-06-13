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

### YAML Configuration Module

The `config` module provides utilities for loading, parsing, and working with hierarchical YAML configuration files:

#### Features

- **Configuration loading**: Load and parse YAML config files
- **Hierarchical settings**: Access settings at project, dataset, and experiment levels
- **Config-aware discovery**: Find files using configuration-defined filters
- **Dataset and experiment discovery**: Find files specific to datasets or experiments

#### Usage Examples

```python
from flyrigloader.config.yaml_config import load_config, get_ignore_patterns
from flyrigloader.config.discovery import discover_experiment_files

# Load configuration
config = load_config("/path/to/config.yaml")

# Get patterns to ignore (respects hierarchy)
patterns = get_ignore_patterns(config, experiment="my_experiment")

# Find files for a specific experiment
files = discover_experiment_files(
    config=config,
    experiment_name="plume_movie_navigation",
    base_directory="/path/to/data",
    extensions=["csv"]
)

# Find files for a specific dataset
from flyrigloader.config.discovery import discover_dataset_files
files = discover_dataset_files(
    config=config,
    dataset_name="no_green_light",
    base_directory="/path/to/data"
)
```

### Data Processing

The `io` module provides utilities for loading and processing experimental data:

#### Features

- **Multiple pickle formats**: Support for regular, gzipped, and pandas pickle formats
- **Automatic format detection**: Automatically detects and loads the appropriate format
- **Multi-dimensional array handling**: Properly processes arrays of various dimensions
- **Metadata integration**: Combines experimental data with metadata
- **Column filtering**: Select specific columns to extract
- **Validation**: Validates data dimensions and structure
- **Pydantic Column Configuration**: Flexible configuration system with strong validation ([documentation](docs/io/column_configuration.md))

#### Usage Examples

```python
from flyrigloader.io.pickle import read_pickle_any_format, make_dataframe_from_matrix, make_dataframe_from_config

# Load a pickle file (auto-detects format)
data = read_pickle_any_format("/path/to/data.pkl")

# Convert experimental matrix to DataFrame
df = make_dataframe_from_matrix(
    exp_matrix=data,
    metadata={"date": "2025-04-01", "fly_id": "fly-123"},
    include_signal_disp=True,  # Include special signal_disp column
    column_list=["t", "x", "y"]  # Only include specified columns
)

# Convert experimental matrix using column configurations (recommended approach)
df = make_dataframe_from_config(
    exp_matrix=data,
    config_source="path/to/column_config.yaml",  # Can also accept dictionary or Pydantic model
    metadata={"date": "2025-04-01", "fly_id": "fly-123"}
)
```

## Using as a Library in External Projects

`flyrigloader` is designed to be easily integrated into external data analysis projects. The high-level API provides simple entry points for loading experiment data based on configuration files:

```python
from flyrigloader.api import load_experiment_files, get_experiment_parameters

# Load all CSV files for a specific experiment defined in your config
files = load_experiment_files(
    config_path="/path/to/your/config.yaml",
    experiment_name="plume_navigation_analysis",
    extensions=["csv"]
)

# Get experiment-specific parameters for analysis
parameters = get_experiment_parameters(
    config_path="/path/to/your/config.yaml",
    experiment_name="plume_navigation_analysis"
)

# Process the files using the parameters
for file in files:
    # Your analysis code here
    analyze_data(file, parameters)

# Extract metadata from filenames while loading files
files_with_metadata = load_experiment_files(
    config_path="/path/to/your/config.yaml",
    experiment_name="plume_navigation_analysis",
    extensions=["csv"],
    extract_metadata=True
)

# Result: {"/path/to/file.csv": {"date": "20230101", "condition": "control", ...}}

# Extract metadata and parse dates from filenames
files_with_dates = load_experiment_files(
    config_path="/path/to/your/config.yaml",
    experiment_name="plume_navigation_analysis",
    extensions=["csv"],
    extract_metadata=True,
    parse_dates=True
)

# Result: {"/path/to/file.csv": {"date": "20230101", "condition": "control", "parsed_date": datetime(2023, 1, 1), ...}}

# You can also use pre-loaded config instead of a config path
from flyrigloader.config.yaml_config import load_config

config = load_config("/path/to/your/config.yaml")
files = load_dataset_files(
    config=config,  # Using pre-loaded config
    dataset_name="my_dataset",
    extract_metadata=True
)

# Kedro-style parameters dictionaries are also supported
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
            "datasets": ["my_dataset"]
        }
    }
}

# Use pre-loaded Kedro parameters without modification
files = load_dataset_files(
    config=kedro_params,
    dataset_name="my_dataset",
    extract_metadata=True
)
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

See the [examples directory](examples/external_project) for a complete demonstration of integrating `flyrigloader` into an external analysis project.

## Configuration Structure

The configuration supports a hierarchical structure with the following key sections:

```yaml
project:
  directories:
    major_data_directory: /path/to/data
  ignore_substrings:
    - 'pattern_to_ignore'
    - '._'
  extraction_patterns:  # Project-level extraction patterns
    - .*_(?P<date>\d{8})_(?P<condition>\w+)_(?P<replicate>\d+)\.csv

datasets:
  my_dataset:
    rig: rig_name
    dates_vials:
      2023-01-01: [1, 2, 3]
      2023-01-02: [4, 5]
    metadata:  # Dataset-specific extraction patterns
      extraction_patterns:
        - .*_(?P<dataset>\w+)_(?P<date>\d{8})\.csv

experiments:
  my_experiment:
    datasets:
      - my_dataset
    filters:
      ignore_substrings:
        - 'experiment_specific_ignore'
      mandatory_experiment_strings:
        - 'required_pattern'
    metadata:  # Experiment-specific extraction patterns
      extraction_patterns:
        - .*_(?P<experiment>\w+)_(?P<date>\d{8})\.csv
```

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
