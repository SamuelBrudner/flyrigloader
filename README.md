# flyrigloader

A Python package for managing and controlling fly rigs for neuroscience experiments.

## Project Structure

The project follows a src-layout Python package structure:

```
flyrigloader/
├── src/
│   └── flyrigloader/    # Main package code
│       └── discovery/   # File discovery module
├── tests/               # Test directory matching package structure
├── docs/                # Documentation
└── pyproject.toml       # Project metadata and dependencies
```

## File Discovery Module

The `discovery` module provides utilities for finding and organizing files based on patterns, with support for:

### Features

- **Basic file discovery**: Find files by pattern using glob matching
- **Multiple base directories**: Search across multiple directories in a single call
- **Recursive discovery**: Search through nested subdirectories
- **Extension filtering**: Filter files by one or more extensions

### Usage Examples

#### Basic File Discovery

```python
from flyrigloader.discovery.files import discover_files

# Find all text files in a directory
files = discover_files("/path/to/data", "*.txt")

# Find all Python files recursively
files = discover_files("/path/to/code", "**/*.py", recursive=True)

# Find specific file types using extension filtering
files = discover_files("/path/to/data", "*", extensions=["csv", "json"])
```

#### Multiple Base Directories

```python
from flyrigloader.discovery.files import discover_files

# Search across multiple directories
dirs = ["/path/to/data1", "/path/to/data2", "/path/to/data3"]
files = discover_files(dirs, "*.csv")
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

The project is developed using Test-Driven Development (TDD):

1. Write failing tests first
2. Implement code to make tests pass
3. Refactor as needed while keeping tests passing

## License

MIT