# Column Configuration API

## Overview

The column configuration system provides Pydantic-based validation for experimental data schemas. It defines what columns are expected, their types, dimensions, and transformations.

## Quick Start

```python
from flyrigloader.io.column_models import load_column_config, get_default_config
from flyrigloader.io.transformers import make_dataframe_from_config

# Option 1: Use default configuration
config = get_default_config()

# Option 2: Load custom configuration
config = load_column_config("my_config.yaml")

# Transform experimental data
df = make_dataframe_from_config(exp_matrix, config)
```

## Primary API

### `load_column_config(path: str) -> ColumnConfigDict`

Load and validate a column configuration from a YAML file.

**Parameters:**
- `path` (str): Path to YAML configuration file

**Returns:**
- `ColumnConfigDict`: Validated configuration model

**Raises:**
- `FileNotFoundError`: If file doesn't exist
- `ValidationError`: If configuration is invalid

**Example:**
```python
config = load_column_config("experiments/config.yaml")
print(f"Loaded {len(config.columns)} column definitions")
```

### `get_default_config() -> ColumnConfigDict`

Load the built-in default configuration.

**Returns:**
- `ColumnConfigDict`: Default configuration with standard columns

**Example:**
```python
config = get_default_config()
# Use for standard experimental setups
```

### `ColumnConfigDict.model_validate(data: dict) -> ColumnConfigDict`

Validate a dictionary as a column configuration.

**Parameters:**
- `data` (dict): Dictionary with column definitions

**Returns:**
- `ColumnConfigDict`: Validated configuration model

**Raises:**
- `ValidationError`: If structure is invalid

**Example:**
```python
from flyrigloader.io.column_models import ColumnConfigDict

config_dict = {
    'columns': {
        't': {
            'type': 'numpy.ndarray',
            'dimension': 1,
            'required': True,
            'description': 'Time values'
        },
        'x': {
            'type': 'numpy.ndarray',
            'dimension': 1,
            'required': True,
            'description': 'X position'
        }
    },
    'special_handlers': {}
}

config = ColumnConfigDict.model_validate(config_dict)
```

## Configuration File Format

### Basic Structure

```yaml
columns:
  column_name:
    type: numpy.ndarray           # Data type
    dimension: 1                  # 1D, 2D, or 3D
    required: true                # Whether column must exist
    description: "Description"    # Human-readable description
    
special_handlers: {}              # Special data transformations
```

### Complete Example

```yaml
columns:
  # Required time column
  t:
    type: numpy.ndarray
    dimension: 1
    required: true
    description: "Time values in seconds"
  
  # Optional position with default
  x:
    type: numpy.ndarray
    dimension: 1
    required: false
    default_value: null
    description: "X position in mm"
  
  # Column with alias (try 'theta_smooth' first, fall back to 'theta_raw')
  theta_smooth:
    type: numpy.ndarray
    dimension: 1
    required: false
    alias: theta_raw
    description: "Smoothed heading angle in radians"
  
  # 2D array with special handling
  signal_disp:
    type: numpy.ndarray
    dimension: 2
    required: false
    special_handling: transform_to_match_time_dimension
    description: "Signal display data (channels × time)"
  
  # Metadata (not part of DataFrame)
  experiment_id:
    type: string
    required: false
    is_metadata: true
    description: "Unique experiment identifier"

special_handlers:
  transform_to_match_time_dimension: _handle_signal_disp
```

## Column Configuration Options

### Required Fields

- `type`: Data type (e.g., `numpy.ndarray`, `string`, `float`, `int`)
- `description`: Human-readable description

### Optional Fields

- `dimension`: Array dimensionality (1, 2, or 3) - only for numpy arrays
- `required`: Whether column must exist (default: `false`)
- `alias`: Alternative column name to try if primary not found
- `is_metadata`: Whether this is metadata (not included in DataFrame)
- `default_value`: Value to use if column is missing
- `special_handling`: Name of special transformation function

## Data Models

### `ColumnConfig`

Individual column configuration.

**Fields:**
- `type: str` - Data type
- `dimension: ColumnDimension | None` - Array dimensionality
- `required: bool` - Whether required (default: False)
- `description: str` - Description
- `alias: str | None` - Alternative name
- `is_metadata: bool` - Whether metadata (default: False)
- `default_value: Any` - Default if missing
- `special_handling: SpecialHandlerType | None` - Special transform

### `ColumnConfigDict`

Complete configuration for all columns.

**Fields:**
- `columns: Dict[str, ColumnConfig]` - Column definitions
- `special_handlers: Dict[str, str]` - Special handler mappings

### `ColumnDimension` (Enum)

Array dimensionality.

- `ColumnDimension.ONE_D` = 1
- `ColumnDimension.TWO_D` = 2
- `ColumnDimension.THREE_D` = 3

### `SpecialHandlerType` (Enum)

Special data handling types.

- `EXTRACT_FIRST_COLUMN` = "extract_first_column_if_2d"
- `TRANSFORM_TIME_DIMENSION` = "transform_to_match_time_dimension"

## Advanced Usage

### Schema Registry (Extensibility)

Register custom schema providers for specialized validation:

```python
from flyrigloader.io.column_models import register_schema, BaseSchema

class CustomSchemaProvider(BaseSchema):
    @property
    def schema_name(self) -> str:
        return "custom"
    
    @property
    def supported_types(self) -> List[str]:
        return ["custom_type"]
    
    def validate(self, data: Any) -> Dict[str, Any]:
        # Custom validation logic
        return validated_data

# Register
register_schema(CustomSchemaProvider(), priority=200)
```

### Validation Hooks (Testing)

For testing, you can customize validation behavior:

```python
from flyrigloader.io.column_models import (
    create_test_dependency_container,
    set_dependency_container
)

# Create test container with custom behavior
deps = create_test_dependency_container(
    yaml_loader=custom_loader,
    logger=custom_logger
)

set_dependency_container(deps)
# ... run tests ...
reset_dependency_container()
```

## Migration from Legacy API

### Old (Polymorphic)

```python
# Confusing - what does None do?
config = get_config_from_source(None)

# Too many options
config = get_config_from_source(path_or_dict_or_model_or_none)
```

### New (Explicit)

```python
# Clear intent
config = get_default_config()
config = load_column_config(path)
config = ColumnConfigDict.model_validate(dict_data)
```

**Note:** `get_config_from_source()` still works for backward compatibility but is considered legacy.

## Common Patterns

### Load and Transform

```python
from flyrigloader.io.column_models import load_column_config
from flyrigloader.io.transformers import make_dataframe_from_config

# Load configuration
config = load_column_config("config.yaml")

# Transform data
df = make_dataframe_from_config(
    exp_matrix=data,
    config=config,
    metadata={'experiment_id': 'exp_001'}
)
```

### Validate Configuration Programmatically

```python
from flyrigloader.io.column_models import ColumnConfigDict, ColumnConfig

# Build configuration in code
config = ColumnConfigDict(
    columns={
        't': ColumnConfig(
            type='numpy.ndarray',
            dimension=1,
            required=True,
            description='Time values'
        )
    },
    special_handlers={}
)

# Use it
df = make_dataframe_from_config(exp_matrix, config)
```

### Check Configuration Contents

```python
config = load_column_config("config.yaml")

# Inspect columns
for name, col_config in config.columns.items():
    print(f"{name}: {col_config.type}")
    if col_config.required:
        print(f"  → Required")
    if col_config.alias:
        print(f"  → Alias: {col_config.alias}")
```

## Error Handling

```python
from pydantic import ValidationError

try:
    config = load_column_config("config.yaml")
except FileNotFoundError:
    print("Config file not found")
    config = get_default_config()
except ValidationError as e:
    print(f"Invalid configuration: {e}")
    # Handle validation errors
```

## See Also

- [DataFrame Transformation API](API_TRANSFORMERS.md)
- [Pickle Loading API](API_PICKLE.md)
- [Migration Guide](MIGRATION.md)
