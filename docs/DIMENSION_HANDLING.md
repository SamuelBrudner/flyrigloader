# Multi-Dimensional Array Handling

**Version**: 1.0.0  
**Last Updated**: 2025-09-30

## Overview

FlyRigLoader handles arrays of varying dimensions (1D, 2D, 3D) when transforming experimental data to pandas DataFrames. This document defines the **explicit semantics** for dimension handling and special transformations.

---

## Dimension Semantics

### 1D Arrays (Most Common)
- **Shape**: `(N,)` where N is the number of time points
- **Storage**: Directly as DataFrame column (standard pandas behavior)
- **Example**: `t`, `x`, `y`, `velocity`

```python
exp_matrix = {
    't': np.array([0, 1, 2, 3, 4]),  # Shape: (5,)
    'x': np.array([10, 11, 12, 13, 14])  # Shape: (5,)
}
# DataFrame:
#    t   x
# 0  0  10
# 1  1  11
# 2  2  12
# 3  3  13
# 4  4  14
```

---

### 2D Arrays (Requires Special Handling)

**Default Behavior (No Special Handler)**: 
- **Not automatically included** in DataFrame
- **Warning logged**: "Column '{name}' is 2D but has no special_handling defined, skipping"
- **Rationale**: Ambiguous how to align with time dimension

**With `special_handling="transform_to_match_time_dimension"`**:
- **Transformation**: Convert to Series of 1D arrays
- **Requirements**: One dimension must match length of `t` (time array)
- **Orientation Detection**: Automatic
  - If `shape[0] == len(t)`: Use rows (each row is one time point's data)
  - If `shape[1] == len(t)`: Use columns (transpose first)
- **Result**: Series where each element is a 1D numpy array

**Example (signal_disp)**:
```python
exp_matrix = {
    't': np.array([0, 1, 2, 3, 4]),           # Shape: (5,)
    'signal_disp': np.random.rand(5, 10)      # Shape: (5, 10) - 5 time points, 10 channels
}

# Column config:
'signal_disp': {
    'type': 'numpy.ndarray',
    'dimension': 2,
    'special_handling': 'transform_to_match_time_dimension'
}

# Result DataFrame:
#    t  signal_disp
# 0  0  [array of 10 values]
# 1  1  [array of 10 values]
# 2  2  [array of 10 values]
# 3  3  [array of 10 values]
# 4  4  [array of 10 values]

# Access: df['signal_disp'].iloc[0] → np.array with shape (10,)
```

---

### 3D Arrays and Higher

**Default Behavior**: 
- **Not automatically included** in DataFrame
- **Error logged**: "Column '{name}' has dimension {n} (n>2), which is not supported for DataFrame conversion"
- **Rationale**: No standard way to represent in tabular format

**Workaround Options**:
1. **Pre-process to 2D**: Flatten or select slices before creating DataFrame
2. **Store separately**: Keep 3D+ arrays outside DataFrame, reference by metadata
3. **Custom handler**: Define special handler in future releases

**Example (NOT included)**:
```python
exp_matrix = {
    't': np.array([0, 1, 2, 3, 4]),              # Shape: (5,)
    'volume_data': np.random.rand(5, 10, 10)     # Shape: (5, 10, 10) - 3D volume per time point
}

# Result DataFrame:
#    t
# 0  0
# 1  1
# 2  2
# 3  3
# 4  4
# (volume_data is NOT included, warning logged)
```

---

## Special Handlers

### `transform_to_match_time_dimension`

**Purpose**: Convert 2D arrays to Series of 1D arrays, aligned with time dimension

**Configuration**:
```yaml
columns:
  signal_disp:
    type: numpy.ndarray
    dimension: 2
    special_handling: transform_to_match_time_dimension
    description: "Multi-channel signal data"

special_handlers:
  transform_to_match_time_dimension: "_handle_signal_disp"
```

**Validation Rules**:
1. **Dimension Check**: Array must be 2D (error if 1D or 3D)
2. **Time Alignment Check**: One dimension must match `len(t)`
   - Valid: `(T, X)` or `(X, T)` where T = len(t)
   - Error: `(A, B)` where neither A nor B equals len(t)
3. **Orientation Auto-Detection**:
   ```python
   if arr.shape[0] == len(t):
       # Time-first: (T, X) → use rows
       series = [arr[i, :] for i in range(len(t))]
   elif arr.shape[1] == len(t):
       # Signal-first: (X, T) → use columns
       series = [arr[:, i] for i in range(len(t))]
   else:
       raise ValueError("No dimension matches time dimension")
   ```

**Error Messages**:
```python
# Missing key
ValueError: "signal_disp transformation requires 't' (time) array in exp_matrix"

# Wrong dimension
ValueError: "signal_disp must be 2D for transform_to_match_time_dimension, got shape (100,)"

# Mismatched dimensions
ValueError: "No dimension of signal_disp (10, 20) matches time dimension (100)"
```

---

### `extract_first_column_if_2d`

**Purpose**: Extract first column from 2D array, use as 1D

**Configuration**:
```yaml
columns:
  position:
    type: numpy.ndarray
    dimension: 2
    special_handling: extract_first_column_if_2d
    description: "Position data (use first column only)"
```

**Behavior**:
```python
exp_matrix = {
    't': np.array([0, 1, 2, 3, 4]),
    'position': np.array([[10, 20], [11, 21], [12, 22], [13, 23], [14, 24]])  # Shape: (5, 2)
}

# Result: df['position'] = np.array([10, 11, 12, 13, 14])  # First column only
```

---

## Dimension Validation

### Column Configuration Validation

```python
# Valid configurations:
{
    'type': 'numpy.ndarray',
    'dimension': 1  # Or 2, 3
}

# Warnings logged:
{
    'type': 'string',
    'dimension': 1  # Warning: "dimension specified for non-array type 'string'"
}

{
    'type': 'numpy.ndarray',
    'dimension': 1,
    'special_handling': 'transform_to_match_time_dimension'  
    # Warning: "special_handling for 1D array, should be 2D"
}
```

### Runtime Validation

**Validation Order**:
1. **Type Check**: Is the data actually a numpy array?
2. **Dimension Check**: Does `ndim` match config `dimension`?
3. **Special Handler Check**: If 2D, is special handler defined?
4. **Time Alignment Check**: For special handlers, does dimension align with time?

**Example Validation Flow**:
```python
def validate_array_dimension(data, col_name, col_config, t_length):
    # 1. Type check
    if not isinstance(data, np.ndarray):
        if col_config.type == 'numpy.ndarray':
            logger.warning(f"{col_name}: expected array, got {type(data)}")
        return False
    
    # 2. Dimension check
    if col_config.dimension and data.ndim != col_config.dimension:
        logger.warning(
            f"{col_name}: config specifies {col_config.dimension}D, "
            f"but data is {data.ndim}D"
        )
    
    # 3. Special handler check (for 2D)
    if data.ndim == 2:
        if not col_config.special_handling:
            logger.warning(
                f"{col_name}: 2D array without special_handling, skipping"
            )
            return False
    
    # 4. Time alignment check (for special handlers)
    if col_config.special_handling == 'transform_to_match_time_dimension':
        if data.shape[0] != t_length and data.shape[1] != t_length:
            raise ValueError(
                f"{col_name}: neither dimension {data.shape} matches "
                f"time length {t_length}"
            )
    
    return True
```

---

## DataFrame Storage Patterns

### Pattern 1: Homogeneous 1D Columns (Standard)
```python
df = pd.DataFrame({
    't': [0, 1, 2, 3, 4],
    'x': [10, 11, 12, 13, 14],
    'y': [20, 21, 22, 23, 24]
})
```
- **Memory Efficient**: Stored as contiguous arrays
- **Operations**: Vectorized pandas operations work seamlessly

---

### Pattern 2: Mixed 1D + Series of Arrays
```python
df = pd.DataFrame({
    't': pd.Series([0, 1, 2, 3, 4], dtype=float),
    'signal_disp': pd.Series([
        np.array([...]),  # Shape: (10,)
        np.array([...]),  # Shape: (10,)
        np.array([...]),  # Shape: (10,)
        np.array([...]),  # Shape: (10,)
        np.array([...])   # Shape: (10,)
    ])
})
```
- **Heterogeneous**: Each row contains a different object (array)
- **Access**: `df['signal_disp'].iloc[i]` returns `np.ndarray`
- **Operations**: Element-wise operations require `.apply()` or list comprehension

---

## Best Practices

### ✅ **DO: Use 1D Arrays for Simple Columns**
```python
exp_matrix = {
    't': np.arange(100),
    'velocity': np.random.rand(100)  # 1D: Simple and efficient
}
```

### ✅ **DO: Define Special Handlers for 2D Arrays**
```yaml
columns:
  neural_activity:
    dimension: 2
    special_handling: transform_to_match_time_dimension
```

### ✅ **DO: Pre-process 3D+ Arrays Before DataFrame Conversion**
```python
# Extract relevant 2D slice from 3D volume
volume_slice = exp_matrix['volume_data'][:, :, slice_idx]  # (T, X)
exp_matrix['volume_slice'] = volume_slice
```

---

### ❌ **DON'T: Store 2D Without Special Handler**
```python
# BAD: No special_handling defined
exp_matrix = {
    't': np.arange(100),
    'matrix_data': np.random.rand(100, 50)  # Will be skipped with warning
}
```

### ❌ **DON'T: Expect 3D+ Arrays in DataFrame**
```python
# BAD: 3D arrays are not supported
exp_matrix = {
    't': np.arange(100),
    'volume': np.random.rand(100, 10, 10)  # Will be skipped with error
}
```

---

## Testing Requirements

All implementations must pass these test scenarios:

1. **1D array test**: 1D arrays stored directly in DataFrame
2. **2D with handler test**: 2D arrays with special_handling converted to Series of arrays
3. **2D without handler test**: 2D arrays without special_handling skipped with warning
4. **3D array test**: 3D+ arrays skipped with error logged
5. **Dimension mismatch test**: 2D array with no time-matching dimension raises error
6. **Orientation test**: Both (T,X) and (X,T) orientations handled correctly

---

## Related Documentation

- [Column Configuration Schema](COLUMN_CONFIG.md)
- [Special Handlers Reference](SPECIAL_HANDLERS.md)
- [Data Transformation Pipeline](TRANSFORMATION_PIPELINE.md)
