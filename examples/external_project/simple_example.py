#!/usr/bin/env python
"""
Enhanced example demonstrating both legacy and new API patterns for flyrigloader.

This script showcases:
1. Legacy load_experiment_files() API for backward compatibility
2. New manifest-based workflow with discover_experiment_manifest(), load_data_file(), and transform_to_dataframe()
3. Comprehensive error handling showing Pydantic validation errors and resolution strategies
4. Explicit logging configuration for path resolution and configuration validation feedback
5. Configuration object introspection showing both dict-style and attribute-style access patterns
6. Conditional code paths demonstrating when to use legacy vs new API patterns

This example serves as a compatibility guide for existing users while demonstrating
the improved architecture and error handling capabilities of the refactored flyrigloader.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

# Standard library imports for comprehensive error handling
from pydantic import ValidationError

# Import flyrigloader API components
from flyrigloader.api import (
    # Legacy API functions for backward compatibility
    load_experiment_files,
    get_experiment_parameters,
    # New decoupled architecture functions
    discover_experiment_manifest,
    load_data_file,
    transform_to_dataframe
)

# Import configuration components for enhanced validation and introspection
from flyrigloader.config.models import LegacyConfigAdapter
from flyrigloader.config.yaml_config import load_config
from flyrigloader.exceptions import ConfigError


def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging to demonstrate new path resolution and validation feedback.
    
    Args:
        verbose: If True, enable DEBUG level logging to show detailed internal operations
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Configure flyrigloader loggers to show configuration and path resolution details
    logging.getLogger('flyrigloader').setLevel(log_level)
    logging.getLogger('flyrigloader.config').setLevel(log_level)
    logging.getLogger('flyrigloader.api').setLevel(log_level)
    
    if verbose:
        print("✓ Verbose logging enabled - detailed internal operations will be shown")
    else:
        print("✓ Standard logging enabled - key operations will be logged")


def demonstrate_config_validation_errors() -> None:
    """
    Demonstrate comprehensive error handling for configuration validation failures.
    
    This function shows how the new Pydantic-based validation provides detailed
    error messages and resolution strategies for common configuration issues.
    """
    print("\n" + "="*60)
    print("CONFIGURATION VALIDATION ERROR HANDLING DEMONSTRATION")
    print("="*60)
    
    # Example 1: Invalid configuration structure
    print("\n1. Invalid configuration structure:")
    try:
        invalid_config = {
            "project": {
                "directories": "this_should_be_a_dict_not_string"  # Invalid type
            }
        }
        
        # This will trigger Pydantic validation error
        load_config(invalid_config, use_pydantic_models=True)
        
    except ConfigError as e:
        print("   ❌ Validation Error (as expected):")
        error_details = e.context.get("validation_errors", [])
        if not error_details and isinstance(e.__cause__, ValidationError):
            error_details = [
                f"Field '{' -> '.join(str(loc) for loc in error['loc'])}': {error['msg']}"
                for error in e.__cause__.errors()
            ]

        if not error_details:
            error_details = [str(e)]

        for error in error_details:
            print(f"     {error}")
        print("   💡 Resolution: directories must be a dictionary mapping names to paths")
    
    # Example 2: Invalid regex pattern
    print("\n2. Invalid regex pattern:")
    try:
        invalid_pattern_config = {
            "project": {
                "directories": {"major_data_directory": "/tmp/test"},
                "extraction_patterns": ["[invalid_regex"]  # Invalid regex
            }
        }
        
        load_config(invalid_pattern_config, use_pydantic_models=True)
        
    except ConfigError as e:
        print("   ❌ Validation Error (as expected):")
        error_details = e.context.get("validation_errors", [])
        if not error_details and isinstance(e.__cause__, ValidationError):
            error_details = [
                f"Field '{' -> '.join(str(loc) for loc in error['loc'])}': {error['msg']}"
                for error in e.__cause__.errors()
            ]

        if not error_details:
            error_details = [str(e)]

        for error in error_details:
            print(f"     {error}")
        print("   💡 Resolution: fix the regex pattern syntax")
    
    # Example 3: Valid configuration (should work)
    print("\n3. Valid configuration:")
    try:
        valid_config = {
            "project": {
                "directories": {"major_data_directory": "/tmp/test"},
                "extraction_patterns": [r"(?P<date>\d{4}-\d{2}-\d{2})"]  # Valid regex
            },
            "datasets": {
                "test_dataset": {
                    "rig": "rig1",
                    "dates_vials": {"2023-05-01": [1, 2, 3]}
                }
            }
        }
        
        validated_config = load_config(valid_config, use_pydantic_models=True)
        print("   ✅ Configuration validated successfully!")
        print(f"   📊 Configuration type: {type(validated_config).__name__}")
        
    except ConfigError as e:
        print("   ❌ Unexpected validation error:")
        error_details = e.context.get("validation_errors", [])
        if not error_details and isinstance(e.__cause__, ValidationError):
            error_details = [
                f"Field '{' -> '.join(str(loc) for loc in error['loc'])}': {error['msg']}"
                for error in e.__cause__.errors()
            ]

        if not error_details:
            error_details = [str(e)]

        for error in error_details:
            print(f"     {error}")


def demonstrate_config_introspection(config_path: str) -> Optional[LegacyConfigAdapter]:
    """
    Demonstrate configuration object introspection with both dict-style and attribute-style access.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Loaded configuration adapter or None if loading failed
    """
    print("\n" + "="*60)
    print("CONFIGURATION OBJECT INTROSPECTION DEMONSTRATION")
    print("="*60)
    
    try:
        # Load configuration with Pydantic validation
        print(f"\n📂 Loading configuration from: {config_path}")
        config = load_config(config_path, use_pydantic_models=True)
        
        print(f"✅ Configuration loaded successfully!")
        print(f"   Type: {type(config).__name__}")
        print(f"   Sections: {list(config.keys())}")
        
        # Demonstrate dictionary-style access (legacy compatibility)
        print("\n🔍 Dictionary-style access (legacy compatibility):")
        if 'project' in config:
            print(f"   config['project'].keys(): {list(config['project'].keys())}")
            if 'directories' in config['project']:
                print(f"   config['project']['directories']: {config['project']['directories']}")
        
        # Demonstrate LegacyConfigAdapter methods
        print("\n🔍 LegacyConfigAdapter-specific methods:")
        if hasattr(config, 'get_model'):
            project_model = config.get_model('project')
            if project_model:
                print(f"   Project model type: {type(project_model).__name__}")
                print(f"   Project model fields: {list(project_model.__class__.model_fields.keys())}")
        
        # Demonstrate iteration and access patterns
        print("\n🔍 Iteration and access patterns:")
        print(f"   config.keys(): {list(config.keys())}")
        print(f"   config.get('nonexistent', 'default'): {config.get('nonexistent', 'default')}")
        
        # Show validation status
        if hasattr(config, 'validate_all'):
            validation_status = config.validate_all()
            print(f"   Validation status: {'✅ Valid' if validation_status else '❌ Invalid'}")
        
        return config
        
    except ConfigError as e:
        print("❌ Configuration validation failed:")
        error_details = e.context.get("validation_errors", [])
        if not error_details and isinstance(e.__cause__, ValidationError):
            error_details = [
                f"Field '{' -> '.join(str(loc) for loc in error['loc'])}': {error['msg']}"
                for error in e.__cause__.errors()
            ]

        if not error_details:
            error_details = [str(e)]

        for error in error_details:
            print(f"   {error}")
        print("\n💡 Resolution strategies:")
        print("   1. Check configuration file syntax and structure")
        print("   2. Ensure all required fields are present")
        print("   3. Validate data types match expected schema")
        return None
        
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        print("\n💡 Resolution strategies:")
        print("   1. Check if configuration file exists")
        print("   2. Verify YAML syntax is valid")
        print("   3. Ensure file permissions allow reading")
        return None


def demonstrate_legacy_api(config_path: str, experiment_name: str) -> bool:
    """
    Demonstrate the legacy API pattern for backward compatibility.
    
    Args:
        config_path: Path to the configuration file
        experiment_name: Name of the experiment to load
        
    Returns:
        True if successful, False otherwise
    """
    print("\n" + "="*60)
    print("LEGACY API PATTERN DEMONSTRATION")
    print("="*60)
    
    try:
        print(f"\n📂 Loading experiment '{experiment_name}' using legacy API...")
        
        # Legacy pattern: load_experiment_files (direct, all-in-one approach)
        print("🔄 Using load_experiment_files() - loads all files directly into memory")
        files = load_experiment_files(
            config_path=config_path,
            experiment_name=experiment_name,
            extensions=["csv", "pkl"]
        )
        
        print(f"✅ Found {len(files)} files using legacy API")
        
        # Legacy pattern: get_experiment_parameters
        print("🔄 Using get_experiment_parameters() - retrieves analysis parameters")
        params = get_experiment_parameters(
            config_path=config_path,
            experiment_name=experiment_name
        )
        
        print(f"✅ Retrieved {len(params)} parameters using legacy API")
        
        # Show results
        print("\n📊 Legacy API Results:")
        print(f"   Files discovered: {len(files)}")
        if files:
            print(f"   Sample files: {list(files)[:3]}{'...' if len(files) > 3 else ''}")
        
        print(f"   Parameters retrieved: {len(params)}")
        if params:
            print(f"   Parameter keys: {list(params.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Legacy API failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        print("\n💡 Common issues and solutions:")
        print("   1. Experiment not found: check experiment name spelling")
        print("   2. No files found: verify data directory and file extensions")
        print("   3. Configuration errors: validate YAML structure")
        return False


def demonstrate_new_api(config_path: str, experiment_name: str) -> bool:
    """
    Demonstrate the new manifest-based API pattern with decoupled architecture.
    
    Args:
        config_path: Path to the configuration file
        experiment_name: Name of the experiment to process
        
    Returns:
        True if successful, False otherwise
    """
    print("\n" + "="*60)
    print("NEW MANIFEST-BASED API PATTERN DEMONSTRATION")
    print("="*60)
    
    try:
        # Step 1: Discover experiment manifest (metadata only, no data loading)
        print(f"\n🔍 Step 1: Discovering experiment manifest for '{experiment_name}'...")
        print("   This step only discovers files and metadata, no data loading yet")
        
        manifest = discover_experiment_manifest(
            config_path=config_path,
            experiment_name=experiment_name,
            extensions=["csv", "pkl"],
            extract_metadata=True,
            parse_dates=True
        )
        
        print(f"✅ Discovered {len(manifest)} files in manifest")
        
        # Show manifest details
        print("\n📋 Manifest Details:")
        for i, (file_path, metadata) in enumerate(manifest.items()):
            if i < 3:  # Show first 3 files
                print(f"   File {i+1}: {Path(file_path).name}")
                print(f"     Path: {file_path}")
                print(f"     Size: {metadata.get('size', 0):,} bytes")
                print(f"     Metadata: {metadata.get('metadata', {})}")
        
        if len(manifest) > 3:
            print(f"   ... and {len(manifest) - 3} more files")
        
        # Step 2: Selective data loading (load only first few files as example)
        print(f"\n📁 Step 2: Selective data loading (loading first 2 files as example)...")
        print("   This demonstrates memory-efficient processing")
        
        loaded_data = {}
        file_paths = list(manifest.keys())[:2]  # Load only first 2 files
        
        for file_path in file_paths:
            print(f"   Loading: {Path(file_path).name}")
            try:
                raw_data = load_data_file(file_path)
                loaded_data[file_path] = raw_data
                print(f"     ✅ Loaded {len(raw_data)} data columns")
            except Exception as e:
                print(f"     ❌ Failed to load: {e}")
        
        # Step 3: Optional DataFrame transformation (only for selected files)
        print(f"\n🔄 Step 3: Optional DataFrame transformation...")
        print("   This step is optional and can be applied selectively")
        
        dataframes = []
        for file_path, raw_data in loaded_data.items():
            try:
                print(f"   Transforming: {Path(file_path).name}")
                df = transform_to_dataframe(
                    raw_data=raw_data,
                    file_path=file_path,
                    add_file_path=True
                )
                dataframes.append(df)
                print(f"     ✅ Created DataFrame with {len(df)} rows, {len(df.columns)} columns")
            except Exception as e:
                print(f"     ❌ Transformation failed: {e}")
        
        # Summary
        print("\n📊 New API Results:")
        print(f"   Files in manifest: {len(manifest)}")
        print(f"   Files loaded: {len(loaded_data)}")
        print(f"   DataFrames created: {len(dataframes)}")
        
        if dataframes:
            total_rows = sum(len(df) for df in dataframes)
            print(f"   Total rows processed: {total_rows}")
        
        return True
        
    except Exception as e:
        print(f"❌ New API failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        print("\n💡 Common issues and solutions:")
        print("   1. Manifest discovery failed: check experiment configuration")
        print("   2. Data loading failed: verify file paths and permissions")
        print("   3. DataFrame transformation failed: check data structure")
        return False


def choose_api_pattern(config_adapter: Optional[LegacyConfigAdapter], experiment_name: str) -> str:
    """
    Demonstrate conditional logic for choosing between legacy and new API patterns.
    
    Args:
        config_adapter: Loaded configuration adapter
        experiment_name: Name of the experiment
        
    Returns:
        Recommended API pattern ("legacy" or "new")
    """
    print("\n" + "="*60)
    print("API PATTERN SELECTION GUIDANCE")
    print("="*60)
    
    print("\n🤔 Analyzing experiment characteristics to recommend API pattern...")
    
    # Analyze experiment characteristics
    factors = {
        "large_dataset": False,
        "memory_constrained": False,
        "selective_processing": False,
        "existing_integration": True,  # Assume existing integration by default
        "compatibility_phase": True  # Assume we're validating coexistence of APIs
    }
    
    if config_adapter:
        try:
            # Check if experiment has many datasets (indicator of large dataset)
            if experiment_name in config_adapter.get('experiments', {}):
                experiment_config = config_adapter['experiments'][experiment_name]
                datasets = experiment_config.get('datasets', [])
                
                if len(datasets) > 3:
                    factors["large_dataset"] = True
                    print(f"   📊 Large dataset detected: {len(datasets)} datasets")
                
                # Check for memory-intensive parameters
                params = experiment_config.get('parameters', {})
                if 'memory_limit' in params or 'batch_size' in params:
                    factors["memory_constrained"] = True
                    print("   💾 Memory constraints detected in parameters")
                
                # Check for selective processing indicators
                if 'filters' in experiment_config:
                    factors["selective_processing"] = True
                    print("   🔍 Selective processing patterns detected")
                    
        except Exception as e:
            print(f"   ⚠️ Could not analyze experiment characteristics: {e}")
    
    # Decision logic
    print("\n📋 Decision factors:")
    for factor, value in factors.items():
        status = "✅ Yes" if value else "❌ No"
        print(f"   {factor.replace('_', ' ').title()}: {status}")
    
    # Recommendation logic
    new_api_score = 0
    legacy_api_score = 0
    
    if factors["large_dataset"]:
        new_api_score += 2
        print("   💡 Large dataset favors new API for memory efficiency")
    
    if factors["memory_constrained"]:
        new_api_score += 2
        print("   💡 Memory constraints favor new API for selective loading")
    
    if factors["selective_processing"]:
        new_api_score += 1
        print("   💡 Selective processing favors new API flexibility")
    
    if factors["existing_integration"]:
        legacy_api_score += 1
        print("   💡 Existing integration favors legacy API for compatibility")
    
    if factors["compatibility_phase"]:
        # During compatibility verification, recommend trying both
        print("   💡 Compatibility phase: recommend testing both approaches")
        return "both"
    
    # Make recommendation
    if new_api_score > legacy_api_score:
        recommendation = "new"
        print(f"\n🚀 Recommendation: NEW API (score: {new_api_score} vs {legacy_api_score})")
        print("   Benefits: Better memory usage, selective processing, modern architecture")
    else:
        recommendation = "legacy"
        print(f"\n🔄 Recommendation: LEGACY API (score: {legacy_api_score} vs {new_api_score})")
        print("   Benefits: Proven compatibility, simpler integration, existing workflows")
    
    return recommendation


def main():
    """
    Main function demonstrating comprehensive flyrigloader usage patterns.
    
    This function showcases:
    1. Explicit logging configuration
    2. Configuration validation and error handling
    3. Both legacy and new API patterns
    4. Conditional API selection based on requirements
    5. Best practices for compatibility and integration
    """
    print("🚀 Enhanced flyrigloader Example - Legacy and New API Patterns")
    print("="*70)
    
    # Parse command line arguments
    config_path = sys.argv[1] if len(sys.argv) > 1 else "example_config.yaml"
    experiment_name = sys.argv[2] if len(sys.argv) > 2 else "plume_navigation_analysis"
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    api_pattern = None
    
    # Check for API pattern preference
    if "--legacy" in sys.argv:
        api_pattern = "legacy"
    elif "--new" in sys.argv:
        api_pattern = "new"
    
    print(f"📋 Configuration: {config_path}")
    print(f"🧪 Experiment: {experiment_name}")
    print(f"📢 Verbose logging: {'enabled' if verbose else 'disabled'}")
    print(f"🔧 API pattern: {api_pattern or 'auto-detect'}")
    
    # Step 1: Setup logging
    setup_logging(verbose)
    
    # Step 2: Demonstrate configuration validation error handling
    demonstrate_config_validation_errors()
    
    # Step 3: Load and introspect configuration
    config_adapter = demonstrate_config_introspection(config_path)
    
    # Step 4: Choose API pattern (if not specified)
    if not api_pattern:
        api_pattern = choose_api_pattern(config_adapter, experiment_name)
    
    # Step 5: Demonstrate chosen API pattern(s)
    success = False
    
    if api_pattern in ["legacy", "both"]:
        success = demonstrate_legacy_api(config_path, experiment_name)
    
    if api_pattern in ["new", "both"]:
        new_success = demonstrate_new_api(config_path, experiment_name)
        success = success or new_success
    
    # Step 6: Summary and recommendations
    print("\n" + "="*70)
    print("SUMMARY AND RECOMMENDATIONS")
    print("="*70)
    
    if success:
        print("✅ Demonstration completed successfully!")
        
        print("\n📚 Key learnings:")
        print("   1. Pydantic validation provides detailed error messages")
        print("   2. LegacyConfigAdapter maintains backward compatibility")
        print("   3. New API enables memory-efficient selective processing")
        print("   4. Path resolution logging provides clear audit trails")
        print("   5. Both APIs can coexist during compatibility testing")

        print("\n🔧 Best practices for compatibility:")
        print("   1. Start with legacy API for existing integrations")
        print("   2. Use new API for memory-intensive or selective workflows")
        print("   3. Enable verbose logging during development and testing")
        print("   4. Validate configurations early with Pydantic models")
        print("   5. Test both patterns with your actual data and workflows")
        
        print("\n💡 Next steps:")
        print("   1. Try this example with your own configuration files")
        print("   2. Measure memory usage differences between APIs")
        print("   3. Integrate appropriate API pattern into your workflows")
        print("   4. Report any issues or feedback to the development team")
        
        return 0
    else:
        print("❌ Demonstration encountered errors")
        print("\n🔧 Troubleshooting:")
        print("   1. Check configuration file exists and is valid YAML")
        print("   2. Verify experiment name exists in configuration")
        print("   3. Ensure data directory paths are accessible")
        print("   4. Run with --verbose flag for detailed logging")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())