"""
models.py - Pydantic models for type-safe configuration management.

This module defines the Pydantic models that represent the configuration
structure of the flyrigloader application, providing type safety, validation,
and consistent access patterns.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Set, Literal
from enum import Enum

from pydantic import BaseModel, Field, validator, root_validator, Extra, field_validator
from pydantic_settings import BaseSettings

from ..core.utils import PathLike, ensure_path


class HardwareCamera(BaseModel):
    """Camera hardware configuration."""
    model: str = "default_camera"
    exposure: int = 500  # milliseconds
    gain: float = 1.0
    fps: int = 30
    binning: int = 1
    
    @field_validator("exposure")
    @classmethod
    def validate_exposure(cls, v: int) -> int:
        """Validate that exposure is within reasonable range."""
        if v < 0:
            raise ValueError("Exposure must be non-negative")
        if v > 10000:
            raise ValueError("Exposure value seems too high (>10000ms)")
        return v


class HardwareMotor(BaseModel):
    """Motor hardware configuration."""
    model: str = "default_motor"
    steps_per_revolution: int = 200
    max_speed: float = 100.0  # steps/second
    acceleration: float = 10.0  # steps/second^2
    
    @field_validator("steps_per_revolution")
    @classmethod
    def validate_steps(cls, v: int) -> int:
        """Validate steps per revolution."""
        if v <= 0:
            raise ValueError("Steps per revolution must be positive")
        return v


class HardwareConfig(BaseModel):
    """Hardware configuration section."""
    camera: HardwareCamera = Field(default_factory=HardwareCamera)
    motor: HardwareMotor = Field(default_factory=HardwareMotor)
    co2_valve_port: Optional[str] = None
    tank_air_port: Optional[str] = None
    
    class Config:
        """Pydantic config for this model."""
        extra = "allow"  # Allow extra fields not in the model


class ExperimentConfig(BaseModel):
    """Experiment configuration section."""
    protocol: str = "default"
    duration: int = 300  # seconds
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    wind: bool = False
    odor: bool = False
    
    @field_validator("duration")
    @classmethod
    def validate_duration(cls, v: int) -> int:
        """Validate experiment duration."""
        if v <= 0:
            raise ValueError("Duration must be positive")
        return v
    
    class Config:
        """Pydantic config for this model."""
        extra = "allow"  # Allow extra fields not in the model


class DiscoveryConfig(BaseModel):
    """File discovery configuration."""
    base_directories: List[str] = Field(default_factory=list)
    patterns: List[str] = Field(default_factory=lambda: ["*.csv", "*.parquet"])
    recursive: bool = True
    
    class Config:
        """Pydantic config for this model."""
        extra = "allow"  # Allow extra fields not in the model


class AppConfig(BaseModel):
    """Top-level application configuration."""
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    
    class Config:
        """Pydantic config for this model."""
        extra = "allow"  # Allow extra fields not in the model


class ConfigSettings(BaseSettings):
    """
    Environment-aware settings for configuration paths.
    
    This class uses Pydantic's BaseSettings to load configuration paths
    from environment variables with fallbacks to default values.
    """
    config_dir: Path = Field(
        default=Path("config"),
        description="Directory containing configuration files",
        env="FLYRIG_CONFIG_DIR"
    )
    
    base_config_file: str = Field(
        default="base_config.yaml",
        description="Base configuration filename",
        env="FLYRIG_BASE_CONFIG"
    )
    
    hardware_config_file: str = Field(
        default="hardware_config.yaml",
        description="Hardware configuration filename",
        env="FLYRIG_HARDWARE_CONFIG"
    )
    
    local_config_file: str = Field(
        default="local_config.yaml",
        description="Local configuration filename",
        env="FLYRIG_LOCAL_CONFIG"
    )
    
    run_config_file: str = Field(
        default="default_run_config.yaml",
        description="Default run configuration filename",
        env="FLYRIG_RUN_CONFIG"
    )
    
    def get_base_config_path(self) -> Path:
        """Get the full path to the base configuration file."""
        return self.config_dir / self.base_config_file
    
    def get_hardware_config_path(self) -> Path:
        """Get the full path to the hardware configuration file."""
        return self.config_dir / self.hardware_config_file
    
    def get_local_config_path(self) -> Path:
        """Get the full path to the local configuration file."""
        return self.config_dir / self.local_config_file
    
    def get_run_config_path(self) -> Path:
        """Get the full path to the run configuration file."""
        return self.config_dir / self.run_config_file
    
    def get_all_config_paths(self) -> Dict[str, Path]:
        """Get a dictionary of all configuration paths."""
        return {
            "base_config": self.get_base_config_path(),
            "hardware_config": self.get_hardware_config_path(),
            "local_config": self.get_local_config_path(),
            "default_run_config": self.get_run_config_path()
        }
    
    class Config:
        """Pydantic settings configuration."""
        env_prefix = "FLYRIG_"
        case_sensitive = False
