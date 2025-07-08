#!/usr/bin/env python
"""
Example script demonstrating how to use flyrigloader from an external project.

This shows how to use the new decoupled pipeline workflow with manifest-based
discovery, selective data loading, and optional DataFrame transformation for
improved memory management and processing control.

Demonstrates:
- New discover_experiment_manifest() for file discovery without immediate loading
- Individual load_data_file() calls for granular control over memory usage  
- Optional transform_to_dataframe() usage for selective DataFrame processing
- Enhanced logging for data directory resolution and pipeline stage progression
- Both legacy dictionary and new Pydantic model-based configuration access patterns
- Comprehensive error handling for new validation system with Pydantic errors
"""
import os
import argparse
import pandas as pd
from pathlib import Path
import sys

from flyrigloader import (
    logger,
    log_format_console,
    log_format_file,
)

# Create log file path since it's not exported from flyrigloader
import tempfile
_log_dir = Path(tempfile.gettempdir()) / "flyrigloader_logs"
_log_dir.mkdir(exist_ok=True)
log_file_path = _log_dir / "flyrigloader_analyze.log"

# Import flyrigloader utilities - new decoupled architecture functions
from flyrigloader.api import (
    discover_experiment_manifest,
    load_data_file,
    transform_to_dataframe
)
from flyrigloader.config.yaml_config import load_config, get_experiment_info
from flyrigloader.config.models import LegacyConfigAdapter

# Import for comprehensive Pydantic validation error handling
from pydantic import ValidationError


def main():
    """Run an analysis on experiment data discovered via flyrigloader."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Analyze fly experiment data")
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--experiment', type=str, required=True, help='Name of experiment to analyze')
    parser.add_argument('--data-dir', type=str, help='Override base data directory')
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        help='Console log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)'
    )
    args = parser.parse_args()

    # Reconfigure logger with user-specified console level
    logger.remove()
    logger.add(
        sys.stderr,
        level=args.log_level.upper(),
        format=log_format_console,
        colorize=True,
    )
    logger.add(
        log_file_path,
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        level="DEBUG",
        format=log_format_file,
        encoding="utf-8",
    )
    logger.debug("Debug logging enabled")

    # Load the configuration file with enhanced error handling for Pydantic validation
    try:
        config = load_config(args.config)
        logger.info("Loaded configuration from: {}", args.config)
        
        # Demonstrate new Pydantic model-based configuration access patterns
        if isinstance(config, LegacyConfigAdapter):
            logger.info("Configuration loaded as LegacyConfigAdapter (new Pydantic-based system)")
            logger.debug("Available config sections: {}", list(config.keys()))
            
            # Demonstrate validation capabilities
            if config.validate_all():
                logger.info("✓ All configuration sections passed Pydantic validation")
            else:
                logger.warning("⚠ Some configuration sections failed validation")
        else:
            logger.info("Configuration loaded as legacy dictionary (backward compatibility mode)")
            
    except ValidationError as e:
        logger.error("Pydantic validation error in configuration: {}", e)
        logger.error("Please check your configuration file for syntax and structure errors")
        return 1
    except Exception as e:
        logger.error("Error loading configuration: {}", e)
        return 1

    # Data directory resolution with enhanced logging (demonstrates new path resolution system)
    logger.info("🗂️ Starting data directory resolution with clear precedence logging...")
    base_directory = args.data_dir
    if base_directory:
        logger.info("Using explicit base_directory from command line: {}", base_directory)
    else:
        logger.info("No command line data directory specified, checking configuration...")
        # Demonstrate both legacy and new configuration access patterns
        if "project" in config and "directories" in config["project"]:
            base_directory = config["project"]["directories"].get("major_data_directory")
            if base_directory:
                logger.info("Using major_data_directory from config: {}", base_directory)
            else:
                logger.info("major_data_directory not found in config, checking environment...")
                # The API functions will handle FLYRIGLOADER_DATA_DIR environment variable
        
        if not base_directory:
            logger.error("Error: No data directory specified in config or command line")
            logger.error("Resolution methods tried: 1) command line, 2) config major_data_directory, 3) environment variable")
            return 1

    # Check if the experiment exists in the configuration with enhanced error handling
    try:
        experiment_info = get_experiment_info(config, args.experiment)
        logger.info("Found experiment '{}' in configuration", args.experiment)
        logger.debug("Experiment datasets: {}", experiment_info.get('datasets', []))
        
        # Show experiment configuration details for transparency
        if isinstance(config, LegacyConfigAdapter):
            exp_model = config.get_model('experiment', args.experiment)
            if exp_model:
                logger.debug("Experiment loaded as validated Pydantic model")
        
    except KeyError:
        available_experiments = list(config.get("experiments", {}).keys()) if "experiments" in config else []
        logger.error("Error: Experiment '{}' not found in configuration", args.experiment)
        logger.error("Available experiments: {}", available_experiments)
        return 1
    except ValidationError as e:
        logger.error("Pydantic validation error for experiment '{}': {}", args.experiment, e)
        return 1

    # NEW DECOUPLED PIPELINE WORKFLOW: Step 1 - Discover experiment manifest
    logger.info("🔍 Step 1: Discovering experiment manifest (new decoupled architecture)...")
    try:
        # Use new discover_experiment_manifest for file discovery without immediate loading
        manifest = discover_experiment_manifest(
            config=config,
            experiment_name=args.experiment,
            base_directory=base_directory,
            pattern="*.pkl",  # Look for pickle files (common in flyrigloader)
            extensions=["pkl", "csv"],  # Support both pickle and CSV for flexibility
            extract_metadata=True,
            parse_dates=True
        )
        
        file_count = len(manifest)
        total_size = sum(item.get('size', 0) for item in manifest.values())
        logger.info("✓ Manifest discovery complete: {} files found", file_count)
        logger.info("  Total data size: {:,} bytes ({:.1f} MB)", total_size, total_size / (1024**2))
        
        if file_count == 0:
            logger.info("No data files found for analysis")
            return 0
            
    except Exception as e:
        logger.error("Error discovering experiment manifest: {}", e)
        return 1

    # Display the discovered manifest
    print(f"\n📋 Experiment Manifest for '{args.experiment}':")
    print(f"Found {file_count} data files:")
    for i, (file_path, metadata) in enumerate(manifest.items(), 1):
        file_size = metadata.get('size', 0)
        print(f"  {i}. {Path(file_path).name} ({file_size:,} bytes)")
        if metadata.get('metadata'):
            print(f"     Metadata: {metadata['metadata']}")

    # NEW DECOUPLED PIPELINE WORKFLOW: Step 2 - Selective data loading
    logger.info("\n📁 Step 2: Selective data loading (granular control over memory usage)...")
    
    # Demonstrate selective processing: load only first few files for memory efficiency
    max_files_to_process = min(3, file_count)  # Process maximum 3 files for demo
    logger.info("Processing first {} files for demonstration (memory-efficient approach)", max_files_to_process)
    
    processed_data = []
    file_paths = list(manifest.keys())
    
    for i, file_path in enumerate(file_paths[:max_files_to_process]):
        try:
            logger.info("Loading file {}/{}: {}", i+1, max_files_to_process, Path(file_path).name)
            
            # Use new load_data_file for individual file loading
            raw_data = load_data_file(
                file_path=file_path,
                validate_format=True
            )
            
            # Log details about the loaded data
            if isinstance(raw_data, dict):
                data_columns = list(raw_data.keys())
                logger.info("  Raw data loaded: {} columns", len(data_columns))
                logger.debug("  Data columns: {}", data_columns[:5])  # Show first 5 columns
            
            processed_data.append((file_path, raw_data))
            
        except Exception as e:
            logger.error("Error loading file {}: {}", file_path, e)
            continue

    if not processed_data:
        logger.error("No files could be loaded successfully")
        return 1

    # NEW DECOUPLED PIPELINE WORKFLOW: Step 3 - Optional DataFrame transformation
    logger.info("\n🔄 Step 3: Optional DataFrame transformation (separation of concerns)...")
    
    # Demonstrate selective DataFrame transformation (only for analysis, not all files)
    dataframes = []
    
    for file_path, raw_data in processed_data:
        try:
            logger.info("Transforming to DataFrame: {}", Path(file_path).name)
            
            # Use new transform_to_dataframe with enhanced options
            df = transform_to_dataframe(
                raw_data=raw_data,
                column_config_path=None,  # Use default column configuration
                metadata={'source_file': str(file_path)},
                add_file_path=True,
                file_path=file_path,
                strict_schema=False
            )
            
            logger.info("  DataFrame shape: {}", df.shape)
            logger.debug("  DataFrame columns: {}", list(df.columns))
            
            dataframes.append(df)
            
        except Exception as e:
            logger.error("Error transforming file {} to DataFrame: {}", file_path, e)
            continue

    # Enhanced analysis with combined data
    if dataframes:
        print("\n📊 Analysis Results (New Decoupled Pipeline):")
        
        # Combine all DataFrames for comprehensive analysis
        combined_df = pd.concat(dataframes, ignore_index=True)
        logger.info("Combined DataFrame shape: {}", combined_df.shape)
        logger.info("Data sources: {} files", len(dataframes))
        
        print(f"\nCombined analysis across {len(dataframes)} files:")
        print(f"Total rows: {len(combined_df):,}")
        print(f"Total columns: {len(combined_df.columns)}")
        
        # Show file path distribution
        if 'file_path' in combined_df.columns:
            file_counts = combined_df['file_path'].value_counts()
            print("\nData distribution by file:")
            for file_path, count in file_counts.items():
                print(f"  {Path(file_path).name}: {count:,} rows")
        
        # Basic statistics for numeric columns
        numeric_columns = combined_df.select_dtypes(include=['number']).columns
        if len(numeric_columns) > 0:
            print(f"\nBasic statistics for {len(numeric_columns)} numeric columns:")
            print(combined_df[numeric_columns].describe())
        
        # Get analysis parameters from experiment config (enhanced with error handling)
        try:
            if "parameters" in experiment_info:
                logger.info("\n⚙️ Using experiment-specific analysis parameters:")
                for param, value in experiment_info["parameters"].items():
                    logger.info("  {}: {}", param, value)
                    print(f"  Parameter: {param} = {value}")
        except Exception as e:
            logger.warning("Could not load experiment parameters: {}", e)
    
    else:
        logger.warning("No DataFrames were successfully created")
        return 1
        
    # Summary of new decoupled pipeline benefits
    print("\n✨ New Decoupled Pipeline Benefits Demonstrated:")
    print("1. 📋 Manifest-based discovery: Files catalogued before loading (memory efficient)")
    print("2. 🎯 Selective loading: Only needed files loaded (granular control)")
    print("3. 🔄 Optional transformation: Raw data vs DataFrame separation (flexible processing)")
    print("4. 📝 Enhanced logging: Clear audit trail of data directory resolution")
    print("5. ✅ Pydantic validation: Type-safe configuration with detailed error messages")
    print("6. 🔧 Backward compatibility: Works with existing configurations")
    
    logger.info("✓ Example analysis completed successfully using new decoupled architecture")
    return 0


if __name__ == "__main__":
    exit(main())
