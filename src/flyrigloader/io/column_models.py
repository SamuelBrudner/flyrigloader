"""
Column configuration models using Pydantic for validation.

This module provides Pydantic models for defining and validating
experimental data column configurations.
"""

from enum import Enum
from typing import Dict, List, Optional, Union, Any, Literal

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field, validator, root_validator


class ColumnDimension(int, Enum):
    """Enumeration of supported array dimensions."""
    ONE_D = 1
    TWO_D = 2
    THREE_D = 3


class SpecialHandlerType(str, Enum):
    """Types of special handlers for column data processing."""
    EXTRACT_FIRST_COLUMN = "extract_first_column_if_2d"
    TRANSFORM_TIME_DIMENSION = "transform_to_match_time_dimension"


class ColumnConfig(BaseModel):
    """
    Configuration for a single column in experimental data.
    
    Attributes:
        type: The expected data type (e.g., "numpy.ndarray", "str", etc.)
        dimension: For arrays, the expected dimensionality (1D, 2D, etc.)
        required: Whether the column is required in the dataset
        description: Human-readable description of the column
        alias: Optional alternative name for the column in the source data
        is_metadata: Whether the column is metadata (not part of raw data)
        default_value: Optional default value for missing columns
        special_handling: Optional special processing needed for this column
    """
    type: str
    dimension: Optional[ColumnDimension] = None
    required: bool = False
    description: str
    alias: Optional[str] = None
    is_metadata: bool = False
    default_value: Optional[Any] = None
    special_handling: Optional[SpecialHandlerType] = None
    
    @validator('dimension', pre=True)
    def validate_dimension(cls, v):
        """Convert integer dimension to enum."""
        if v is None:
            return None
        if isinstance(v, int):
            try:
                return ColumnDimension(v)
            except ValueError as e:
                raise ValueError(f"Dimension must be 1, 2, or 3, got {v}") from e
        return v
    
    @validator('special_handling', pre=True)
    def validate_special_handling(cls, v):
        """Convert string handler types to enum."""
        if v is None:
            return None
        if isinstance(v, str):
            try:
                return SpecialHandlerType(v)
            except ValueError as e:
                valid_handlers = [h.value for h in SpecialHandlerType]
                raise ValueError(f"Special handler must be one of {valid_handlers}, got {v}") from e
        return v
    
    @root_validator
    def validate_configuration(cls, values):
        """Validate the overall configuration for consistency."""
        # If we have a dimension, ensure type is compatible
        if values.get('dimension') and not values.get('type', '').startswith('numpy'):
            logger.warning(f"Dimension specified for non-numpy type {values.get('type')}")
        
        # If special handling is for transforming time dimension, ensure we have a dimension
        if (values.get('special_handling') == SpecialHandlerType.TRANSFORM_TIME_DIMENSION and 
                values.get('dimension') != ColumnDimension.TWO_D):
            logger.warning("transform_to_match_time_dimension should be used with 2D arrays")
            
        return values
        

class ColumnConfigDict(BaseModel):
    """
    Configuration for all columns in experimental data.
    
    This model allows validation of the entire column configuration
    dictionary to ensure it's properly structured.
    """
    columns: Dict[str, ColumnConfig]
    special_handlers: Dict[str, str] = Field(default_factory=dict)
    
    @validator('special_handlers')
    def validate_special_handlers(cls, v, values):
        """Ensure all referenced handlers have an implementation."""
        required_handlers = {
            col.special_handling.value
            for col in values.get('columns', {}).values()
            if col.special_handling
        }
        # Check that all required handlers are defined
        for handler in required_handlers:
            if handler not in v:
                logger.warning(f"Special handler '{handler}' is used but not defined in special_handlers")

        return v


def load_column_config(config_path: str) -> ColumnConfigDict:
    """
    Load and validate column configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        ColumnConfigDict: Validated column configuration model.
    """
    import yaml
    from loguru import logger
    
    logger.debug(f"Loading column configuration from {config_path}")
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    return ColumnConfigDict.model_validate(config_data)
