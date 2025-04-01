# Error Handling Standards for flyrigloader

This document outlines the standardized error handling patterns for the flyrigloader package.

## Two Complementary Approaches

The package uses two complementary error handling patterns, each appropriate for different contexts:

### 1. Tuple Returns with Metadata (for Data Processing)

**When to use:**
- In data processing pipeline functions
- For operations where detailed processing metadata is valuable
- When the caller needs structured information about the operation

**Implementation:**
- Return a tuple of `(result, metadata)`
- If successful, return `(result_object, metadata_dict)`
- If failed, return `(None, metadata_dict)`
- Always include a `"success"` key in metadata

**Standard metadata format:**
```python
{
    "success": bool,              # Required - indicates if the operation succeeded
    "error": str,                 # Present only if success=False - error message
    "error_type": str,            # Present only if success=False - exception type name
    # Additional function-specific metadata...
    "timestamp": str,             # ISO-format timestamp of when processing occurred
    "rows_processed": int,        # For DataFrame operations - number of rows
    # ... other context-specific metadata
}
```

**Example:**
```python
def load_file_into_dataframe(path) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    metadata = {"success": False, "timestamp": datetime.now().isoformat()}
    try:
        # Processing logic...
        metadata["success"] = True
        metadata["rows_processed"] = len(df)
        return df, metadata
    except Exception as e:
        metadata["error"] = str(e)
        metadata["error_type"] = type(e).__name__
        return None, metadata
```

### 2. Exception Raising (for Utilities and Low-level Functions)

**When to use:**
- In utility functions and low-level operations
- When errors are exceptional conditions that should interrupt flow
- For operations where the caller would immediately propagate errors anyway

**Implementation:**
- Raise appropriate exceptions with descriptive messages
- Log errors before raising
- Use specific exception types when possible

**Example:**
```python
def read_file(path) -> pd.DataFrame:
    try:
        # Logic...
    except (FileNotFoundError, PermissionError) as e:
        logger.error(f"Access error: {str(e)}")
        raise  # Re-raise same exception
    except ValueError as e:
        logger.error(f"Value error: {str(e)}")
        raise ValueError(f"Invalid file format: {str(e)}") from e
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise RuntimeError(f"Failed to read file: {str(e)}") from e
```

## Type Annotations

Always use type annotations to make error handling patterns explicit:

```python
# For tuple return pattern
def process_data(...) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    ...

# For exception raising pattern
def utility_function(...) -> pd.DataFrame:  # No Optional needed
    ...
```

## Handling in Client Code

When calling functions, respect their error handling pattern:

```python
# Handling tuple returns
result, metadata = load_file_into_dataframe(path)
if metadata["success"]:
    # Use result
else:
    # Handle error using metadata["error"]

# Handling exceptions
try:
    result = read_file(path)
    # Use result
except Exception as e:
    # Handle error
```
