#!/usr/bin/env python
"""
Simple example demonstrating the high-level API for external projects.

This script shows the absolute minimum code needed to get experiment data
using the flyrigloader library from an external project.
"""
import sys
from pathlib import Path

# Import the high-level API from flyrigloader
from flyrigloader.api import load_experiment_files, get_experiment_parameters


def main():
    """Simple example of using flyrigloader from an external project."""
    # Get command line arguments or use defaults
    config_path = sys.argv[1] if len(sys.argv) > 1 else "example_config.yaml"
    experiment = sys.argv[2] if len(sys.argv) > 2 else "plume_navigation_analysis"
    
    # Print what we're doing
    print(f"Loading experiment '{experiment}' from config: {config_path}")
    
    try:
        # Load all CSV files for the experiment using a single high-level function call
        files = load_experiment_files(
            config_path=config_path,
            experiment_name=experiment,
            extensions=["csv"]
        )
        
        print(f"Found {len(files)} files for experiment '{experiment}'")
        
        # Get experiment parameters for analysis
        params = get_experiment_parameters(
            config_path=config_path,
            experiment_name=experiment
        )
        
        print(f"Loaded {len(params)} parameters for experiment '{experiment}'")
        
        # Example of processing files using parameters
        print("\nFiles to process:")
        for i, file_path in enumerate(files, 1):
            print(f"  {i}. {file_path}")
        
        print("\nAnalysis parameters:")
        for param, value in params.items():
            print(f"  {param}: {value}")
            
        print("\nIn a real application, you would process these files with these parameters.")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
