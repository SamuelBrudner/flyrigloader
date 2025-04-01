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
    args = parser.parse_args()

    # Load the configuration file
    try:
        config = load_config(args.config)
        print(f"Loaded configuration from: {args.config}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1

    # Override the data directory if specified
    base_directory = args.data_dir
    if not base_directory:
        # Use the directory from config if not overridden
        if "project" in config and "directories" in config["project"]:
            base_directory = config["project"]["directories"].get("major_data_directory")
        
        if not base_directory:
            print("Error: No data directory specified in config or command line")
            return 1

    # Check if the experiment exists in the configuration
    try:
        experiment_info = get_experiment_info(config, args.experiment)
        print(f"Found experiment '{args.experiment}' in configuration")
    except KeyError:
        print(f"Error: Experiment '{args.experiment}' not found in configuration")
        return 1

    # Discover experiment data files
    try:
        print(f"Searching for experiment data in: {base_directory}")
        csv_files = discover_experiment_files(
            config=config,
            experiment_name=args.experiment,
            base_directory=base_directory,
            extensions=["csv"]
        )
        print(f"Found {len(csv_files)} CSV files for experiment '{args.experiment}'")
    except Exception as e:
        print(f"Error discovering experiment files: {e}")
        return 1

    # Extract and analyze data
    if not csv_files:
        print("No data files found for analysis")
        return 0

    # Print the files found
    print("\nData files found:")
    for i, file_path in enumerate(csv_files, 1):
        print(f"  {i}. {file_path}")

    # Example analysis: Load the first CSV file and show basic statistics
    print("\nPerforming example analysis on first file:")
    try:
        first_file = csv_files[0]
        print(f"Loading: {first_file}")
        
        # In a real analysis, you'd process all files and perform more complex operations
        df = pd.read_csv(first_file)
        print(f"Data shape: {df.shape}")
        print("\nData preview:")
        print(df.head())
        
        print("\nBasic statistics:")
        print(df.describe())
        
        # Get analysis parameters from experiment config
        if "analysis_params" in experiment_info:
            print("\nUsing experiment-specific analysis parameters:")
            for param, value in experiment_info["analysis_params"].items():
                print(f"  {param}: {value}")
    except Exception as e:
        print(f"Error in analysis: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
