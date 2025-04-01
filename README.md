# FlyRigLoader

A tool for managing and loading fly rigs for neuroscience experimental setups.

## Overview

FlyRigLoader is designed to streamline the process of setting up and loading fly rigs for neuroscience experiments. It provides utilities for configuration, monitoring, and control of various fly rig components, including data loading, transformation, validation, and experiment tracking.

## Installation

```bash
# Development installation
pip install -e .
```

## Project Structure

```
flyrigloader/               # Repository root
├── src/
│   └── flyrigloader/       # The actual module code
│       ├── __init__.py     # Package initialization with version info
│       ├── config_utils/   # Configuration utilities
│       │   ├── __init__.py
│       │   ├── config_loader.py
│       │   └── filter.py
│       ├── discovery/      # File and pattern discovery
│       │   ├── __init__.py
│       │   ├── files.py
│       │   └── patterns.py
│       ├── lineage/        # Experiment tracking
│       │   ├── __init__.py
│       │   ├── minimal.py
│       │   ├── tracker.py
│       │   └── data_lineage_integration.py
│       ├── pipeline/       # Data processing pipelines
│       │   ├── __init__.py
│       │   ├── data_assembly.py
│       │   └── data_pipeline.py
│       ├── readers/        # Data format readers
│       │   ├── __init__.py
│       │   ├── formats.py
│       │   └── pickle.py
│       ├── schema/         # Data schemas and validation
│       │   ├── __init__.py
│       │   ├── operations.py
│       │   └── validator.py
│       └── utils/          # Utility functions
│           ├── __init__.py
│           ├── files.py
│           └── imports.py
├── tests/                  # Tests directory
│   ├── __init__.py
│   └── test_config.py
├── pyproject.toml          # Project metadata and build configuration
├── README.md
└── .gitignore
```

## Key Components

### Configuration Management
The `config_utils` module provides functionality for loading, filtering, and managing configuration settings for rig components and experiments.

### Data Models
The project includes data models for various rig components such as:
- Cameras
- Motors
- Sensors
- Other experimental apparatus

### Schema Validation
The `schema` module implements a type system for data validation that maps string type names to pandas dtypes:

| Type Names | Pandas Dtype |
|------------|-------------|
| `int`, `int32`, `int64`, `integer` | `pd.Int64Dtype()` |
| `float`, `float32`, `float64`, `numeric` | `pd.Float64Dtype()` |
| `str`, `string`, `text` | `pd.StringDtype()` |
| `bool`, `boolean` | `pd.BooleanDtype()` |
| `list`, `array` | `pd.ArrowDtype(storage="list")` |
| `datetime`, `date` | `pd.DatetimeTZDtype(tz=None)` |
| `category` | `pd.CategoricalDtype()` |

The system uses nullable dtypes to handle missing values properly in experiment data.

### Data Pipeline
The `pipeline` module provides tools for:
- Data assembly from various sources
- Transformation and validation of experimental data
- Creating processing pipelines for automated data handling

### Experiment Tracking
The `lineage` module helps track experiment configurations and runs, maintaining provenance information.

#### Automated Lineage Tracking
The `lineage` module now includes a decorator-based approach for automatically tracking data transformations:

```python
from flyrigloader.lineage import track_lineage_step

# Add automatic lineage tracking to any function that transforms a DataFrame
@track_lineage_step("clean_data", "Remove outliers and missing values", {"phase": "preprocessing"})
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Your data transformation code
    df = df.dropna()
    df = df[df['value'] > 0]  # Remove negative values
    return df

# Simply use your function as normal - lineage is tracked automatically
df = clean_data(df)
```

For interactive or notebook environments where decorators aren't ideal:

```python
from flyrigloader.lineage import with_lineage_tracking, complete_lineage_step

# Start a lineage step
df = with_lineage_tracking(df, "manual_cleaning", "Removing outliers")

# Do your transformations
df = df[df['temperature'] < 100]  # Filter out unrealistic temperatures

# Mark the step as complete with additional metadata
df = complete_lineage_step(df, "success", {"rows_removed": len(original_df) - len(df)})
```

## Experiment Lineage Tracking

FlyRigLoader includes a data lineage system that tracks how data is processed and transformed throughout your experimental pipeline. This system attaches metadata to pandas DataFrames using the `attrs` attribute.

### Namespaced DataFrame Attributes

To avoid collisions with other libraries that might use DataFrame attributes, FlyRigLoader uses namespaced attribute keys:

| Attribute Key | Description |
|---------------|-------------|
| `__flyrig_lineage` | The primary LineageTracker object |
| `__flyrig_lineage_ids` | List of all lineage IDs associated with the DataFrame |
| `__flyrig_lineages` | Dictionary mapping lineage IDs to serialized lineage data |

#### Using Lineage Helpers (Recommended)

Instead of accessing these attributes directly, use the helper functions:

```python
from flyrigloader.lineage.tracker import get_lineage_from_dataframe, attach_lineage_to_dataframe

# Get lineage from a DataFrame
lineage = get_lineage_from_dataframe(df)

# Attach lineage to a DataFrame
df_with_lineage = attach_lineage_to_dataframe(df, lineage)
```

#### Direct Access (Advanced)

If you need to directly access the lineage attributes:

```python
# Check if a DataFrame has lineage information
has_lineage = '__flyrig_lineage' in df.attrs

# Get all lineage IDs
lineage_ids = df.attrs.get('__flyrig_lineage_ids', [])
```

## Usage

### Basic Configuration Loading

```python
from flyrigloader.config_utils import config_loader

# Load configuration from a file
config = config_loader.load_config("path/to/config.yaml")

# Access configuration settings
camera_settings = config.get("camera")
motor_settings = config.get("motor")
```

### Schema Validation

```python
from flyrigloader.schema import validator

# Define a schema
schema = {
    "temperature": "float",
    "duration": "int",
    "experiment_name": "str",
    "is_active": "bool"
}

# Validate data against schema
valid_data = validator.validate_data(data, schema)
```

### Data Processing

```python
from flyrigloader.pipeline import data_pipeline

# Create a data processing pipeline
pipeline = data_pipeline.create_pipeline([
    "load_raw_data",
    "preprocess",
    "analyze"
])

# Run the pipeline
results = pipeline.run(input_data)
```

## License

MIT