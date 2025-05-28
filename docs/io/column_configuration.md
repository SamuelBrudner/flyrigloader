# Column Configuration System

The FlyRigLoader library includes a robust column configuration system using Pydantic for validation. This README demonstrates how to use it effectively in your projects.

## Using Column Configurations

### 1. From YAML Files (Recommended for Most Cases)

The YAML configuration approach is recommended for most use cases as it provides a clear, structured way to define column requirements and is easy to maintain.

```python
from flyrigloader.io.pickle import make_dataframe_from_config

# Load experimental data
exp_matrix = {...}  # Your experimental data dictionary

# Create DataFrame using a YAML configuration file
df = make_dataframe_from_config(
    exp_matrix=exp_matrix,
    config_source="path/to/your_config.yaml",
    metadata={"date": "2025-04-01"}
)
```

See the template configuration file in `src/flyrigloader/io/column_config.yaml` for a comprehensive example with all available options. You can copy this file and modify it for your specific needs.

### 2. Using a Dictionary Configuration

If you need to generate configurations programmatically, you can pass a dictionary:

```python
from flyrigloader.io.pickle import make_dataframe_from_config

# Define column configuration as a dictionary
config_dict = {
    "columns": {
        "t": {
            "type": "numpy.ndarray",
            "dimension": 1,
            "required": True,
            "description": "Time values"
        },
        "x": {
            "type": "numpy.ndarray",
            "dimension": 1,
            "required": True,
            "description": "X position"
        }
    },
    "special_handlers": {
        "transform_to_match_time_dimension": "_handle_signal_disp"
    }
}

# Create DataFrame using the dictionary configuration
df = make_dataframe_from_config(
    exp_matrix=exp_matrix,
    config_source=config_dict,
    metadata={"date": "2025-04-01"}
)
```

### 3. Using Pydantic Models Directly

For advanced use cases with maximum type safety:

```python
from flyrigloader.io.pickle import make_dataframe_from_config
from flyrigloader.io.column_models import ColumnConfig, ColumnConfigDict, ColumnDimension

# Create a Pydantic model configuration
config_model = ColumnConfigDict(
    columns={
        "t": ColumnConfig(
            type="numpy.ndarray",
            dimension=ColumnDimension.ONE_D,
            required=True,
            description="Time values"
        ),
        "x": ColumnConfig(
            type="numpy.ndarray",
            dimension=ColumnDimension.ONE_D,
            required=True,
            description="X position"
        )
    },
    special_handlers={
        "transform_to_match_time_dimension": "_handle_signal_disp"
    }
)

# Create DataFrame using the Pydantic model
df = make_dataframe_from_config(
    exp_matrix=exp_matrix,
    config_source=config_model,
    metadata={"date": "2025-04-01"}
)
```

### 4. Using the Default Configuration

If no configuration is specified, the system will automatically use the built-in default configuration:

```python
from flyrigloader.io.pickle import make_dataframe_from_config

# Load experimental data
exp_matrix = {...}  # Your experimental data dictionary with required columns

# Create DataFrame using the default configuration (column_config.yaml)
df = make_dataframe_from_config(
    exp_matrix=exp_matrix,
    metadata={"date": "2025-04-01"}
)
```

The default configuration is located at `src/flyrigloader/io/column_config.yaml` and can be accessed programmatically:

```python
from flyrigloader.io.column_models import get_default_config_path

# Get the path to the default configuration
default_config_path = get_default_config_path()
print(f"Default configuration is at: {default_config_path}")
```

## Configuration Options

Each column configuration can include the following properties:

| Property | Type | Description |
|----------|------|-------------|
| `type` | string | Python type name (e.g., "numpy.ndarray", "str", "int") |
| `dimension` | int | For array types, the dimensionality (1, 2, or 3) |
| `required` | bool | Whether the column must be present (default: false) |
| `description` | string | Human-readable description of the column |
| `alias` | string | Alternative name to look for in the source data |
| `is_metadata` | bool | Whether this is a metadata field (default: false) |
| `default_value` | any | Value to use if column is missing |
| `special_handling` | string | Special processing to apply to this column |

## Special Handlers

The following special handlers are available:

| Handler Name | Description |
|--------------|-------------|
| `extract_first_column_if_2d` | If the array is 2D, extract only the first column |
| `transform_to_match_time_dimension` | Transform 2D arrays to match the time dimension |

You can define your own special handlers by adding them to the `special_handlers` section of your configuration and implementing the corresponding function.

When `make_dataframe_from_config` loads your YAML configuration, each entry in `special_handlers` maps a handler key to a Python function name. The function must be importable by `flyrigloader.io.pickle` and will be called whenever a column references that handler type.

For example, your configuration might include:

```yaml
special_handlers:
  my_custom_handler: _my_function
```

To use this handler you must define a function named `_my_function` that is accessible when `pickle.py` is imported. The function can either accept the entire `exp_matrix` and column name or just the current value, and it should return the processed value.

```python
def _my_function(exp_matrix, col_name):
    # access exp_matrix[col_name] and return the new value
    ...

# or for a simple value-based handler
def _my_function(value):
    # transform value and return it
    ...
```

If you place the implementation in another module, make sure it is imported in `pickle.py` so that the function can be resolved by name.

## Template Configuration

A complete template configuration is provided in `src/flyrigloader/io/column_config.yaml` which includes:

- All available configuration options with explanatory comments
- Examples of required columns, optional columns, and metadata
- Examples of special handlers and aliases
- Documentation of each field's purpose

You can use this template as a starting point for your own configuration files.

## Examples

See the `tests/flyrigloader/io/test_column_models.py` and `tests/flyrigloader/io/test_column_config.py` files for practical examples of using the column configuration system in different scenarios.
