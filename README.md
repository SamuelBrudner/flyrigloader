# flyrigloader

A Python package for managing and controlling fly rigs for neuroscience experiments.

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

#### Usage Examples

```python
from flyrigloader.io.pickle import read_pickle_any_format, make_dataframe_from_matrix

# Load a pickle file (auto-detects format)
data = read_pickle_any_format("/path/to/data.pkl")

# Convert experimental matrix to DataFrame
df = make_dataframe_from_matrix(
    exp_matrix=data,
    metadata={"date": "2025-04-01", "fly_id": "fly-123"},
    include_signal_disp=True,  # Include special signal_disp column
    column_list=["t", "x", "y"]  # Only include specified columns
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

datasets:
  my_dataset:
    rig: rig_name
    dates_vials:
      2023-01-01: [1, 2, 3]
      2023-01-02: [4, 5]

experiments:
  my_experiment:
    datasets:
      - my_dataset
    filters:
      ignore_substrings:
        - 'experiment_specific_ignore'
      mandatory_experiment_strings:
        - 'required_pattern'
```

## Development

### Installation for Development

```bash
# Clone the repository
git clone <repository-url>
cd flyrigloader

# Install the package in development mode
pip install -e .
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

MIT