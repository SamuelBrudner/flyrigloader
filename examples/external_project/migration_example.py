#!/usr/bin/env python3
"""
Comprehensive Migration Demonstration Script for flyrigloader.

This script demonstrates how to transition from legacy dictionary-based configuration
handling to the new Pydantic model-based validation system. It provides side-by-side
comparisons, error handling examples, and practical migration patterns for existing
flyrigloader integrations.

The script showcases:
1. Legacy vs new configuration access patterns
2. Pydantic validation error handling and resolution
3. LegacyConfigAdapter usage for backward compatibility
4. Migration strategies for existing codebases
5. Best practices for configuration validation

Usage:
    python migration_example.py [example_config.yaml]
    
Requirements:
    - flyrigloader with new Pydantic models
    - pydantic >= 2.6.0
    - Valid YAML configuration file (optional)
"""

import logging
import json
import sys
from pathlib import Path
from typing import Dict, Any

# External imports for comprehensive functionality
from pydantic import ValidationError

# Internal imports from the new flyrigloader architecture
from flyrigloader.config.yaml_config import load_config
from flyrigloader.config.models import DatasetConfig
from flyrigloader.api import transform_to_dataframe

# Set up logging for demonstration purposes
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demonstrate_migration_patterns() -> None:
    """
    Demonstrate comprehensive migration patterns from legacy to new configuration system.
    
    This function shows the evolution from dictionary-based configuration access
    to Pydantic model-based validation, highlighting the benefits of the new approach
    including type safety, validation, and improved developer experience.
    """
    logger.info("=== Migration Patterns Demonstration ===")
    
    # Create a sample configuration that represents typical flyrigloader usage
    sample_config = {
        "project": {
            "directories": {
                "major_data_directory": "/path/to/fly_data",
                "backup_directory": "/path/to/backup"
            },
            "ignore_substrings": ["._", "temp", "backup"],
            "mandatory_experiment_strings": ["experiment"],
            "extraction_patterns": [r"(?P<date>\d{4}-\d{2}-\d{2})", r"(?P<subject>\w+)"]
        },
        "datasets": {
            "plume_tracking": {
                "rig": "rig1",
                "dates_vials": {
                    "2023-05-01": [1, 2, 3, 4],
                    "2023-05-02": [5, 6, 7, 8]
                },
                "metadata": {
                    "extraction_patterns": [r"(?P<temperature>\d+)C"],
                    "description": "Temperature gradient experiments"
                }
            },
            "odor_response": {
                "rig": "rig2", 
                "dates_vials": {
                    "2023-05-03": [1, 2, 3],
                    "2023-05-04": [4, 5, 6]
                }
            }
        },
        "experiments": {
            "plume_navigation_analysis": {
                "datasets": ["plume_tracking"],
                "parameters": {
                    "analysis_window": 10.0,
                    "threshold": 0.5,
                    "method": "correlation"
                },
                "filters": {
                    "ignore_substrings": ["test"],
                    "mandatory_experiment_strings": ["trial"]
                }
            }
        }
    }
    
    print("\n🔄 MIGRATION PATTERN 1: Configuration Loading")
    print("=" * 60)
    
    # Legacy pattern (pre-Pydantic)
    print("📜 LEGACY APPROACH:")
    print("  # Direct dictionary access with manual validation")
    print("  config = load_config('config.yaml')")
    print("  major_data_dir = config['project']['directories']['major_data_directory']")
    print("  # Risk: KeyError if structure is unexpected")
    print("  # Risk: No validation of directory existence or format")
    
    # New pattern (with Pydantic)
    print("\n✨ NEW APPROACH:")
    print("  # Type-safe configuration with automatic validation")
    print("  config = load_config('config.yaml')")  # Returns validated models
    print("  adapter = LegacyConfigAdapter(config)  # For backward compatibility")
    print("  major_data_dir = adapter['project']['directories']['major_data_directory']")
    print("  # OR direct model access:")
    print("  project_model = adapter.get_model('project')")
    print("  major_data_dir = project_model.directories['major_data_directory']")
    
    print("\n🔄 MIGRATION PATTERN 2: Dataset Access")
    print("=" * 60)
    
    # Demonstrate dataset access patterns
    dataset_name = "plume_tracking"
    dataset_config = sample_config["datasets"][dataset_name]
    
    print("📜 LEGACY APPROACH:")
    print(f"  dataset_config = config['datasets']['{dataset_name}']")
    print(f"  rig = dataset_config['rig']")
    print(f"  dates_vials = dataset_config['dates_vials']")
    print("  # Risk: No validation of rig name format")
    print("  # Risk: No validation of date formats or vial numbers")
    
    print("\n✨ NEW APPROACH:")
    print(f"  # Create validated dataset model")
    try:
        dataset_model = DatasetConfig(**dataset_config)
        print(f"  dataset_model = DatasetConfig(**config['datasets']['{dataset_name}'])")
        print(f"  rig = dataset_model.rig  # '{dataset_model.rig}' (validated)")
        print(f"  dates_vials = dataset_model.dates_vials  # Validated structure")
        print(f"  metadata = dataset_model.metadata  # Optional, validated if present")
        
        # Demonstrate accessing specific dates and vials
        print("\n  # Type-safe access to dates and vials:")
        for date, vials in dataset_model.dates_vials.items():
            print(f"    {date}: {vials} (validated integers)")
            break  # Just show first example
            
    except ValidationError as e:
        print(f"  # Validation would catch errors: {e}")
    
    print("\n🔄 MIGRATION PATTERN 3: Error Handling")
    print("=" * 60)
    
    # Show evolution of error handling
    print("📜 LEGACY APPROACH:")
    print("  try:")
    print("      rig = config['datasets']['invalid_dataset']['rig']")
    print("  except KeyError as e:")
    print("      print(f'Missing configuration: {e}')")
    print("  # Generic error, hard to debug configuration issues")
    
    print("\n✨ NEW APPROACH:")
    print("  try:")
    print("      dataset_model = DatasetConfig(**invalid_config)")
    print("  except ValidationError as e:")
    print("      for error in e.errors():")
    print("          field = '.'.join(str(loc) for loc in error['loc'])")
    print("          print(f'Field {field}: {error[\"msg\"]}')")
    print("  # Detailed validation errors with field-level specificity")


def compare_legacy_vs_new_config() -> None:
    """
    Compare legacy dictionary-style configuration access with new Pydantic model access.
    
    This function demonstrates the differences in syntax, validation, and error handling
    between the old and new approaches, showing practical examples that developers
    can use to understand the migration benefits.
    """
    logger.info("=== Legacy vs New Configuration Comparison ===")
    
    # Sample configuration for comparison
    sample_config = {
        "datasets": {
            "experiment_1": {
                "rig": "rig1",
                "dates_vials": {
                    "2023-05-01": [1, 2, 3, 4],
                    "2023-05-02": [5, 6, 7, 8]
                },
                "metadata": {
                    "description": "Temperature gradient experiments"
                }
            }
        }
    }
    
    print("\n📊 CONFIGURATION ACCESS COMPARISON")
    print("=" * 70)
    
    # Dictionary-style access (legacy)
    print("📜 LEGACY: Dictionary-style access")
    print("-" * 35)
    try:
        legacy_rig = sample_config["datasets"]["experiment_1"]["rig"]
        legacy_vials = sample_config["datasets"]["experiment_1"]["dates_vials"]["2023-05-01"]
        legacy_metadata = sample_config["datasets"]["experiment_1"].get("metadata", {})
        
        print(f"✓ rig = config['datasets']['experiment_1']['rig']")
        print(f"  Result: '{legacy_rig}' (type: {type(legacy_rig).__name__})")
        print(f"✓ vials = config['datasets']['experiment_1']['dates_vials']['2023-05-01']")
        print(f"  Result: {legacy_vials} (type: {type(legacy_vials).__name__})")
        print(f"✓ metadata = config['datasets']['experiment_1'].get('metadata', {{}})")
        print(f"  Result: {legacy_metadata}")
        
        print("\n🔍 Legacy Issues:")
        print("  - No validation of data types or formats")
        print("  - KeyError risk if structure changes")
        print("  - No IDE autocomplete support")
        print("  - Manual type checking required")
        
    except Exception as e:
        print(f"❌ Legacy access failed: {e}")
    
    # Pydantic model access (new)
    print("\n✨ NEW: Pydantic model access")
    print("-" * 30)
    try:
        # Create validated model
        dataset_model = DatasetConfig(**sample_config["datasets"]["experiment_1"])
        
        print(f"✓ dataset_model = DatasetConfig(**config['datasets']['experiment_1'])")
        print(f"✓ rig = dataset_model.rig")
        print(f"  Result: '{dataset_model.rig}' (validated)")
        print(f"✓ vials = dataset_model.dates_vials['2023-05-01']")
        print(f"  Result: {dataset_model.dates_vials['2023-05-01']} (validated integers)")
        print(f"✓ metadata = dataset_model.metadata")
        print(f"  Result: {dataset_model.metadata} (validated dict or None)")
        
        print("\n🚀 New Benefits:")
        print("  ✅ Automatic validation of all fields")
        print("  ✅ Type safety with IDE support")
        print("  ✅ Clear error messages for invalid data")
        print("  ✅ Documentation through type hints")
        print("  ✅ JSON schema generation capability")
        
        # Demonstrate model serialization capabilities
        print("\n📄 Model Serialization:")
        model_dict = dataset_model.model_dump()
        print(f"  model.model_dump() -> {json.dumps(model_dict, indent=2)[:100]}...")
        
        # Demonstrate validation access
        print("\n🔧 Accessing Validation Info:")
        print(f"  Model fields: {list(dataset_model.model_fields.keys())}")
        
    except ValidationError as e:
        print(f"❌ New model validation failed: {e}")
        print("  This demonstrates the validation catching configuration errors!")
    
    print("\n📈 PERFORMANCE COMPARISON")
    print("=" * 30)
    print("Legacy: Fast access, no validation overhead")
    print("New:    Slightly slower initial validation, cached afterward")
    print("       Trade-off: Small performance cost for major reliability gain")
    
    print("\n🎯 MIGRATION RECOMMENDATION")
    print("=" * 30)
    print("1. Use LegacyConfigAdapter for immediate compatibility")
    print("2. Gradually migrate to direct Pydantic model access")
    print("3. Update tests to expect ValidationError instead of KeyError")
    print("4. Leverage type hints for better development experience")


def demonstrate_pydantic_error_handling() -> None:
    """
    Demonstrate comprehensive Pydantic validation error handling and resolution strategies.
    
    This function shows common configuration errors and how the new Pydantic-based
    system provides clear, actionable error messages that help developers quickly
    identify and fix configuration issues.
    """
    logger.info("=== Pydantic Error Handling Demonstration ===")
    
    print("\n🚨 PYDANTIC VALIDATION ERROR EXAMPLES")
    print("=" * 50)
    
    # Example 1: Invalid rig name
    print("❌ EXAMPLE 1: Invalid rig name (contains illegal characters)")
    print("-" * 60)
    invalid_rig_config = {
        "rig": "rig@1!",  # Invalid characters
        "dates_vials": {
            "2023-05-01": [1, 2, 3]
        }
    }
    
    try:
        DatasetConfig(**invalid_rig_config)
    except ValidationError as e:
        print("Configuration:")
        print(json.dumps(invalid_rig_config, indent=2))
        print("\nValidation Error:")
        for error in e.errors():
            field = '.'.join(str(loc) for loc in error['loc'])
            print(f"  Field '{field}': {error['msg']}")
            print(f"  Input value: {error['input']}")
        
        print("\n🔧 Resolution:")
        print("  Change rig name to use only alphanumeric, underscore, and hyphen")
        print("  Example: 'rig@1!' → 'rig_1' or 'rig-1'")
    
    # Example 2: Invalid dates_vials structure
    print("\n❌ EXAMPLE 2: Invalid dates_vials structure")
    print("-" * 45)
    invalid_dates_config = {
        "rig": "rig1",
        "dates_vials": {
            "2023-05-01": ["not", "integers"]  # Should be list of integers
        }
    }
    
    try:
        DatasetConfig(**invalid_dates_config)
    except ValidationError as e:
        print("Configuration:")
        print(json.dumps(invalid_dates_config, indent=2))
        print("\nValidation Error:")
        for error in e.errors():
            field = '.'.join(str(loc) for loc in error['loc'])
            print(f"  Field '{field}': {error['msg']}")
            print(f"  Input value: {error['input']}")
        
        print("\n🔧 Resolution:")
        print("  Ensure all vial numbers are integers")
        print("  Example: ['not', 'integers'] → [1, 2, 3, 4]")
    
    # Example 3: Missing required field
    print("\n❌ EXAMPLE 3: Missing required field")
    print("-" * 35)
    missing_field_config = {
        "dates_vials": {
            "2023-05-01": [1, 2, 3]
        }
        # Missing required 'rig' field
    }
    
    try:
        DatasetConfig(**missing_field_config)
    except ValidationError as e:
        print("Configuration:")
        print(json.dumps(missing_field_config, indent=2))
        print("\nValidation Error:")
        for error in e.errors():
            field = '.'.join(str(loc) for loc in error['loc'])
            print(f"  Field '{field}': {error['msg']}")
        
        print("\n🔧 Resolution:")
        print("  Add the required 'rig' field")
        print("  Example: {'rig': 'rig1', 'dates_vials': {...}}")
    
    # Example 4: Invalid regex pattern in metadata
    print("\n❌ EXAMPLE 4: Invalid regex pattern in metadata")
    print("-" * 45)
    invalid_regex_config = {
        "rig": "rig1",
        "dates_vials": {
            "2023-05-01": [1, 2, 3]
        },
        "metadata": {
            "extraction_patterns": ["(?P<invalid[regex"]  # Malformed regex
        }
    }
    
    try:
        DatasetConfig(**invalid_regex_config)
    except ValidationError as e:
        print("Configuration:")
        print(json.dumps(invalid_regex_config, indent=2))
        print("\nValidation Error:")
        for error in e.errors():
            field = '.'.join(str(loc) for loc in error['loc'])
            print(f"  Field '{field}': {error['msg']}")
            print(f"  Input value: {error['input']}")
        
        print("\n🔧 Resolution:")
        print("  Fix the regex pattern syntax")
        print("  Example: '(?P<invalid[regex' → '(?P<temperature>\\d+)'")
    
    print("\n✅ EXAMPLE 5: Valid configuration")
    print("-" * 35)
    valid_config = {
        "rig": "rig1",
        "dates_vials": {
            "2023-05-01": [1, 2, 3, 4],
            "2023-05-02": [5, 6, 7, 8]
        },
        "metadata": {
            "extraction_patterns": [r"(?P<temperature>\d+)C"],
            "description": "Temperature gradient experiments"
        }
    }
    
    try:
        valid_model = DatasetConfig(**valid_config)
        print("Configuration:")
        print(json.dumps(valid_config, indent=2))
        print("\n✅ Validation Success!")
        print(f"  Model created: {type(valid_model).__name__}")
        print(f"  Rig: {valid_model.rig}")
        print(f"  Number of date entries: {len(valid_model.dates_vials)}")
        print(f"  Has metadata: {valid_model.metadata is not None}")
        
    except ValidationError as e:
        print(f"Unexpected validation error: {e}")
    
    print("\n🎯 ERROR HANDLING BEST PRACTICES")
    print("=" * 40)
    print("1. Always catch ValidationError when creating Pydantic models")
    print("2. Use error.errors() to get detailed field-level information")
    print("3. Present user-friendly error messages based on field paths")
    print("4. Log detailed errors for debugging, show simplified errors to users")
    print("5. Validate configuration early in application startup")


def demonstrate_legacy_adapter_usage() -> None:
    """
    Demonstrate LegacyConfigAdapter usage for gradual migration from dict-style access.
    
    This function shows how the LegacyConfigAdapter provides backward compatibility
    while enabling gradual migration to the new Pydantic-based system. It demonstrates
    both dictionary-style access and direct model access through the same adapter.
    """
    logger.info("=== Legacy Adapter Usage Demonstration ===")
    
    # Import the LegacyConfigAdapter for demonstration
    from flyrigloader.config.models import LegacyConfigAdapter
    
    print("\n🔄 LEGACY ADAPTER: Bridging Old and New")
    print("=" * 50)
    
    # Create a comprehensive configuration for demonstration
    full_config = {
        "project": {
            "directories": {
                "major_data_directory": "/path/to/fly_data",
                "backup_directory": "/path/to/backup"
            },
            "ignore_substrings": ["._", "temp", "backup"],
            "mandatory_experiment_strings": ["experiment"],
            "extraction_patterns": [r"(?P<date>\d{4}-\d{2}-\d{2})"]
        },
        "datasets": {
            "plume_tracking": {
                "rig": "rig1",
                "dates_vials": {
                    "2023-05-01": [1, 2, 3, 4],
                    "2023-05-02": [5, 6, 7, 8]
                },
                "metadata": {
                    "description": "Temperature gradient experiments"
                }
            },
            "odor_response": {
                "rig": "rig2",
                "dates_vials": {
                    "2023-05-03": [1, 2, 3]
                }
            }
        },
        "experiments": {
            "plume_navigation": {
                "datasets": ["plume_tracking"],
                "parameters": {
                    "analysis_window": 10.0,
                    "threshold": 0.5
                }
            }
        }
    }
    
    print("📦 Creating LegacyConfigAdapter...")
    try:
        # Create the adapter
        adapter = LegacyConfigAdapter(full_config)
        
        print("✅ Adapter created successfully!")
        print(f"   Available sections: {list(adapter.keys())}")
        
        print("\n🔍 USAGE PATTERN 1: Dictionary-style access (100% backward compatible)")
        print("-" * 70)
        
        # Dictionary-style access - exactly like legacy code
        print("📜 Legacy-style access works unchanged:")
        major_data_dir = adapter["project"]["directories"]["major_data_directory"]
        print(f"  adapter['project']['directories']['major_data_directory']")
        print(f"  → '{major_data_dir}'")
        
        dataset_rig = adapter["datasets"]["plume_tracking"]["rig"]
        print(f"  adapter['datasets']['plume_tracking']['rig']")
        print(f"  → '{dataset_rig}'")
        
        experiment_params = adapter["experiments"]["plume_navigation"]["parameters"]
        print(f"  adapter['experiments']['plume_navigation']['parameters']")
        print(f"  → {experiment_params}")
        
        print("\n🚀 USAGE PATTERN 2: Direct model access (new capability)")
        print("-" * 60)
        
        # Direct model access for enhanced functionality
        print("✨ New model-based access:")
        
        # Get project model
        project_model = adapter.get_model('project')
        if project_model:
            print(f"  project_model = adapter.get_model('project')")
            print(f"  project_model.directories → {project_model.directories}")
            print(f"  project_model.ignore_substrings → {project_model.ignore_substrings}")
        
        # Get dataset model
        dataset_model = adapter.get_model('dataset', 'plume_tracking')
        if dataset_model:
            print(f"  dataset_model = adapter.get_model('dataset', 'plume_tracking')")
            print(f"  dataset_model.rig → '{dataset_model.rig}'")
            print(f"  dataset_model.dates_vials keys → {list(dataset_model.dates_vials.keys())}")
            print(f"  dataset_model.metadata → {dataset_model.metadata}")
        
        print("\n🔧 USAGE PATTERN 3: Validation capabilities")
        print("-" * 45)
        
        # Demonstrate validation features
        print("🔍 Validation features:")
        is_valid = adapter.validate_all()
        print(f"  adapter.validate_all() → {is_valid}")
        
        all_models = adapter.get_all_models()
        print(f"  adapter.get_all_models() → {len(all_models)} models")
        for model_key in sorted(all_models.keys()):
            model_type = type(all_models[model_key]).__name__
            print(f"    {model_key}: {model_type}")
        
        print("\n🎯 USAGE PATTERN 4: Gradual migration strategy")
        print("-" * 50)
        
        print("📈 Migration phases:")
        print("  Phase 1: Replace config dict with LegacyConfigAdapter")
        print("           - All existing code works unchanged")
        print("           - Gain validation capabilities")
        print("  Phase 2: Gradually replace dict access with model access")
        print("           - Better type safety and IDE support")
        print("           - More descriptive error messages")
        print("  Phase 3: Use Pydantic models directly")
        print("           - Full benefits of the new system")
        
        print("\n📝 Example migration for a function:")
        print("  # Original code:")
        print("  def get_rig_name(config, dataset_name):")
        print("      return config['datasets'][dataset_name]['rig']")
        print()
        print("  # Phase 1: Use adapter (no code change needed)")
        print("  def get_rig_name(adapter, dataset_name):")
        print("      return adapter['datasets'][dataset_name]['rig']")
        print()
        print("  # Phase 2: Use model access")
        print("  def get_rig_name(adapter, dataset_name):")
        print("      model = adapter.get_model('dataset', dataset_name)")
        print("      return model.rig if model else None")
        
    except Exception as e:
        print(f"❌ Error creating adapter: {e}")
        print("This might indicate configuration validation issues")
    
    print("\n💡 ADAPTER BENEFITS")
    print("=" * 20)
    print("✅ 100% backward compatibility with existing code")
    print("✅ Immediate validation benefits without code changes")
    print("✅ Gradual migration path to modern patterns")
    print("✅ Better error messages and debugging")
    print("✅ Type safety when using model access")
    print("✅ No breaking changes to existing integrations")


def main() -> None:
    """
    Main function to run all migration demonstration examples.
    
    This function orchestrates all the demonstration functions and provides
    a command-line interface for running the migration examples. It can optionally
    load a real configuration file for testing with actual data.
    """
    print("🚀 flyrigloader Configuration Migration Guide")
    print("=" * 50)
    print("This script demonstrates how to migrate from legacy dictionary-based")
    print("configuration handling to the new Pydantic model-based validation system.")
    print()
    
    # Check if a configuration file was provided
    config_file = None
    if len(sys.argv) > 1:
        config_file = Path(sys.argv[1])
        if config_file.exists() and config_file.is_file():
            print(f"📁 Using configuration file: {config_file}")
            
            try:
                # Load and validate the real configuration
                real_config = load_config(config_file)
                print(f"✅ Successfully loaded configuration with {len(real_config)} sections")
                
                # Quick validation with LegacyConfigAdapter
                from flyrigloader.config.models import LegacyConfigAdapter
                adapter = LegacyConfigAdapter(real_config)
                is_valid = adapter.validate_all()
                print(f"🔍 Configuration validation: {'✅ PASSED' if is_valid else '❌ FAILED'}")
                
            except Exception as e:
                print(f"❌ Error loading configuration: {e}")
                print("Proceeding with demonstration examples...")
                
        else:
            print(f"❌ Configuration file not found: {config_file}")
            print("Proceeding with demonstration examples...")
    else:
        print("💡 No configuration file provided. Using demonstration examples.")
        print("   Usage: python migration_example.py [config.yaml]")
    
    print()
    
    try:
        # Run all demonstration functions
        demonstrate_migration_patterns()
        print("\n" + "="*70 + "\n")
        
        compare_legacy_vs_new_config()
        print("\n" + "="*70 + "\n")
        
        demonstrate_pydantic_error_handling()
        print("\n" + "="*70 + "\n")
        
        demonstrate_legacy_adapter_usage()
        print("\n" + "="*70 + "\n")
        
        print("🎉 MIGRATION DEMONSTRATION COMPLETE!")
        print("=" * 40)
        print("Key takeaways:")
        print("1. Use LegacyConfigAdapter for immediate compatibility")
        print("2. Pydantic models provide superior validation and error messages")
        print("3. Migration can be gradual without breaking existing code")
        print("4. New system provides better developer experience and reliability")
        print()
        print("📚 Next steps:")
        print("- Update your configuration loading to use load_config()")
        print("- Wrap configurations with LegacyConfigAdapter")
        print("- Gradually migrate to direct Pydantic model access")
        print("- Update error handling to catch ValidationError")
        print("- Leverage type hints for better IDE support")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Ensure flyrigloader is installed with Pydantic support:")
        print("  pip install flyrigloader[validation]")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()