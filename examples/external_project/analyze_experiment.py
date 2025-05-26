#!/usr/bin/env python
"""
Example script demonstrating how to use flyrigloader from an external project.

This shows how to use the flyrigloader configuration and discovery utilities
to load experiment data defined in an external project's configuration.
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
    log_file_path,
)

# Import flyrigloader utilities
from flyrigloader.config.yaml_config import load_config, get_experiment_info
from flyrigloader.config.discovery import discover_experiment_files


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

    # Load the configuration file
    try:
        config = load_config(args.config)
        logger.info("Loaded configuration from: {}", args.config)
    except Exception as e:
        logger.error("Error loading configuration: {}", e)
        return 1

    # Override the data directory if specified
    base_directory = args.data_dir
    if not base_directory:
        # Use the directory from config if not overridden
        if "project" in config and "directories" in config["project"]:
            base_directory = config["project"]["directories"].get("major_data_directory")
        
        if not base_directory:
            logger.error("Error: No data directory specified in config or command line")
            return 1

    # Check if the experiment exists in the configuration
    try:
        experiment_info = get_experiment_info(config, args.experiment)
        logger.info("Found experiment '{}' in configuration", args.experiment)
    except KeyError:
        logger.error("Error: Experiment '{}' not found in configuration", args.experiment)
        return 1

    # Discover experiment data files
    try:
        logger.info("Searching for experiment data in: {}", base_directory)
        csv_files = discover_experiment_files(
            config=config,
            experiment_name=args.experiment,
            base_directory=base_directory,
            extensions=["csv"]
        )
        logger.info("Found {} CSV files for experiment '{}'", len(csv_files), args.experiment)
    except Exception as e:
        logger.error("Error discovering experiment files: {}", e)
        return 1

    # Extract and analyze data
    if not csv_files:
        logger.info("No data files found for analysis")
        return 0

    # Print the files found
    print("\nData files found:")
    for i, file_path in enumerate(csv_files, 1):
        print(f"  {i}. {file_path}")

    # Example analysis: Load the first CSV file and show basic statistics
    print("\nPerforming example analysis on first file:")
    try:
        first_file = csv_files[0]
        logger.info("Loading: {}", first_file)
        
        # In a real analysis, you'd process all files and perform more complex operations
        df = pd.read_csv(first_file)
        logger.info("Data shape: {}", df.shape)
        logger.debug("\nData preview:\n{}", df.head())

        logger.info("\nBasic statistics:")
        logger.info(df.describe())
        
        # Get analysis parameters from experiment config
        if "analysis_params" in experiment_info:
            logger.info("\nUsing experiment-specific analysis parameters:")
            for param, value in experiment_info["analysis_params"].items():
                logger.info("  {}: {}", param, value)
    except Exception as e:
        logger.error("Error in analysis: {}", e)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
