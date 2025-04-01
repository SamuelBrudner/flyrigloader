"""
example.py - Example usage of the Pydantic-based configuration system.

This module demonstrates how to use the configuration system,
showing the benefits of type safety, validation, and immutability.
"""

import os
from pathlib import Path

from loguru import logger

# Import the configuration system
from flyrigloader.config_utils import (
    ConfigManager,
    AppConfig,
    HardwareConfig,
    ExperimentConfig,
    get_config,
    get_hardware_config,
)


def example_simple_usage():
    """Example of simple configuration usage with global instance."""
    # Get the global configuration (automatically loaded)
    config = get_config()
    
    # Access configuration values with type safety
    camera_model = config.hardware.camera.model
    exposure = config.hardware.camera.exposure
    
    print(f"Using camera model: {camera_model} with exposure: {exposure}ms")
    
    # Access experiment settings
    protocol = config.experiment.protocol
    duration = config.experiment.duration
    
    print(f"Running experiment protocol: {protocol} for {duration} seconds")


def example_custom_config_path():
    """Example of using custom configuration paths."""
    # Override configuration directory with environment variable
    os.environ["FLYRIG_CONFIG_DIR"] = "/custom/path/to/config"
    
    # Create a new config manager (will use the environment variable)
    manager = ConfigManager()
    
    # Get configuration
    config = manager.config
    
    # Access with type hints and autocompletion
    camera: HardwareConfig = config.hardware
    print(f"Camera model: {camera.model}")
    
    # Clean up
    del os.environ["FLYRIG_CONFIG_DIR"]


def example_validation():
    """Example of validation in action."""
    try:
        # Create a hardware config with invalid values
        invalid_config = HardwareConfig(
            camera={
                "model": "test_camera",
                "exposure": -100,  # Invalid: must be non-negative
                "fps": "invalid",  # Invalid: must be an integer
            }
        )
        print("This should not print if validation works correctly")
    except Exception as e:
        print(f"Validation caught the error: {e}")
        
    # Create with valid values
    valid_config = HardwareConfig(
        camera={
            "model": "test_camera",
            "exposure": 100,
            "fps": 30,
        }
    )
    print(f"Valid config created: {valid_config.camera.model}")


def example_immutability():
    """Example demonstrating immutability benefits."""
    # Get the main configuration
    config = get_config()
    
    # Convert to dictionary for one use case
    config_dict = config.model_dump()
    
    # Modify the dictionary (without affecting the original)
    config_dict["hardware"]["camera"]["exposure"] = 1000
    
    # Original config is unchanged
    original_exposure = config.hardware.camera.exposure
    modified_exposure = config_dict["hardware"]["camera"]["exposure"]
    
    print(f"Original exposure: {original_exposure}")
    print(f"Modified exposure: {modified_exposure}")
    print("Original config was not affected by dictionary modification")


def example_recommended_usage():
    """The recommended pattern for using configuration in your code."""
    # In your module, get the specific configuration section you need
    hardware = get_hardware_config()
    
    # Use it with full type safety
    def setup_camera():
        """Set up the camera using configuration."""
        print(f"Setting up {hardware.camera.model} camera")
        print(f"Exposure: {hardware.camera.exposure}ms")
        print(f"FPS: {hardware.camera.fps}")
        print(f"Gain: {hardware.camera.gain}")
        
        # All attributes are properly typed and validated
        # This allows for IDE autocompletion and type checking
    
    setup_camera()


if __name__ == "__main__":
    # These examples can be run to demonstrate the configuration system
    print("\n=== Simple Usage ===")
    example_simple_usage()
    
    print("\n=== Custom Config Path ===")
    example_custom_config_path()
    
    print("\n=== Validation ===")
    example_validation()
    
    print("\n=== Immutability ===")
    example_immutability()
    
    print("\n=== Recommended Usage ===")
    example_recommended_usage()
