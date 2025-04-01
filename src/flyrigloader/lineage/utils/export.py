"""
export.py - Serialization and export utilities for lineage tracking.

This module provides functions for exporting lineage information to various
formats, as well as importing it back. These utilities are useful for
sharing lineage information between systems or for persistence.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, cast

import pandas as pd
import yaml
from loguru import logger

from ...core.utils import PathLike, ensure_path
from ..core import ensure_dataframe
from .dataframe import extract_lineage_dict, has_lineage


def export_lineage_to_json(
    df: pd.DataFrame, 
    path: Optional[PathLike] = None,
    indent: int = 2
) -> Optional[str]:
    """
    Export lineage information from a DataFrame to JSON.
    
    Args:
        df: DataFrame to export lineage from
        path: Optional path to save the JSON file
        indent: Indentation level for the JSON
        
    Returns:
        JSON string if path is None, None otherwise
        
    Raises:
        ValueError: If the DataFrame does not have lineage or export fails
        
    Example:
        >>> # Export to string
        >>> json_str = export_lineage_to_json(df)
        >>> 
        >>> # Export to file
        >>> export_lineage_to_json(df, "lineage.json")
    """
    try:
        df = ensure_dataframe(df)
        
        # Check if DataFrame has lineage
        if not has_lineage(df):
            logger.warning("DataFrame does not have lineage information")
            return None
        
        # Extract lineage dictionary
        lineage_dict = extract_lineage_dict(df)
        
        if lineage_dict is None:
            logger.warning("Could not extract lineage information")
            return None
        
        # Convert to JSON
        json_str = json.dumps(lineage_dict, indent=indent, default=str)
        
        # Save to file if path specified
        if path is not None:
            path_obj = ensure_path(path)
            with open(path_obj, 'w') as f:
                f.write(json_str)
            logger.info(f"Lineage exported to JSON file: {path_obj}")
            return None
        
        return json_str
    except Exception as e:
        logger.error(f"Failed to export lineage to JSON: {str(e)}")
        raise ValueError(f"Failed to export lineage to JSON: {str(e)}") from e


def export_lineage_to_yaml(
    df: pd.DataFrame, 
    path: Optional[PathLike] = None
) -> Optional[str]:
    """
    Export lineage information from a DataFrame to YAML.
    
    Args:
        df: DataFrame to export lineage from
        path: Optional path to save the YAML file
        
    Returns:
        YAML string if path is None, None otherwise
        
    Raises:
        ValueError: If the DataFrame does not have lineage or export fails
        
    Example:
        >>> # Export to string
        >>> yaml_str = export_lineage_to_yaml(df)
        >>> 
        >>> # Export to file
        >>> export_lineage_to_yaml(df, "lineage.yaml")
    """
    try:
        df = ensure_dataframe(df)
        
        # Check if DataFrame has lineage
        if not has_lineage(df):
            logger.warning("DataFrame does not have lineage information")
            return None
        
        # Extract lineage dictionary
        lineage_dict = extract_lineage_dict(df)
        
        if lineage_dict is None:
            logger.warning("Could not extract lineage information")
            return None
        
        # Convert to YAML
        yaml_str = yaml.dump(lineage_dict, sort_keys=False, default_flow_style=False)
        
        # Save to file if path specified
        if path is not None:
            path_obj = ensure_path(path)
            with open(path_obj, 'w') as f:
                f.write(yaml_str)
            logger.info(f"Lineage exported to YAML file: {path_obj}")
            return None
        
        return yaml_str
    except Exception as e:
        logger.error(f"Failed to export lineage to YAML: {str(e)}")
        raise ValueError(f"Failed to export lineage to YAML: {str(e)}") from e


def export_lineage_to_html(
    df: pd.DataFrame,
    path: PathLike,
    include_graph: bool = True,
    include_metadata: bool = True
) -> bool:
    """
    Export lineage information from a DataFrame to an HTML report.
    
    Args:
        df: DataFrame to export lineage from
        path: Path to save the HTML file
        include_graph: Whether to include a visualization graph
        include_metadata: Whether to include detailed metadata
        
    Returns:
        True if export was successful, False otherwise
        
    Raises:
        ValueError: If export fails
        
    Example:
        >>> export_lineage_to_html(df, "lineage_report.html")
    """
    try:
        df = ensure_dataframe(df)
        
        # Check if DataFrame has lineage
        if not has_lineage(df):
            logger.warning("DataFrame does not have lineage information")
            return False
        
        # Extract lineage dictionary
        lineage_dict = extract_lineage_dict(df)
        
        if lineage_dict is None:
            logger.warning("Could not extract lineage information")
            return False
        
        # Ensure path exists
        path_obj = ensure_path(path)
        
        # Generate HTML content
        html_content = _generate_lineage_html(lineage_dict, include_graph, include_metadata)
        
        # Write to file
        with open(path_obj, 'w') as f:
            f.write(html_content)
            
        logger.info(f"Lineage exported to HTML file: {path_obj}")
        return True
    except Exception as e:
        logger.error(f"Failed to export lineage to HTML: {str(e)}")
        raise ValueError(f"Failed to export lineage to HTML: {str(e)}") from e


def _generate_lineage_html(
    lineage_dict: Dict[str, Any],
    include_graph: bool = True,
    include_metadata: bool = True
) -> str:
    """
    Generate HTML content for lineage visualization.
    
    Args:
        lineage_dict: Lineage dictionary to visualize
        include_graph: Whether to include a visualization graph
        include_metadata: Whether to include detailed metadata
        
    Returns:
        HTML string
    """
    # Basic template
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Data Lineage Report - {lineage_dict.get('name', 'Unnamed')}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #3498db; }}
            .section {{ margin-bottom: 30px; }}
            .source, .step {{ 
                background-color: #f8f9fa; 
                padding: 10px; 
                margin: 10px 0; 
                border-radius: 5px;
                border-left: 5px solid #3498db;
            }}
            .metadata {{ 
                font-size: 0.9em;
                color: #7f8c8d;
                margin-left: 20px;
            }}
            pre {{ 
                background-color: #f1f1f1; 
                padding: 10px; 
                border-radius: 5px; 
                overflow-x: auto;
            }}
            .timestamp {{ color: #95a5a6; font-size: 0.8em; }}
        </style>
    </head>
    <body>
        <h1>Data Lineage Report</h1>
        <div class="timestamp">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        
        <div class="section">
            <h2>Lineage Overview</h2>
            <p><strong>Name:</strong> {lineage_dict.get('name', 'Unnamed')}</p>
            <p><strong>ID:</strong> {lineage_dict.get('id', 'Unknown')}</p>
            <p><strong>Created:</strong> {lineage_dict.get('created', 'Unknown')}</p>
        </div>
    """
    
    # Sources section
    sources = lineage_dict.get('sources', [])
    if sources:
        html += """
        <div class="section">
            <h2>Data Sources</h2>
        """
        
        for i, source in enumerate(sources):
            html += f"""
            <div class="source">
                <p><strong>Source {i+1}:</strong> {source.get('path', 'Unknown')}</p>
            """
            
            if include_metadata and 'metadata' in source:
                html += """
                <div class="metadata">
                    <p><strong>Metadata:</strong></p>
                    <pre>"""
                for key, value in source.get('metadata', {}).items():
                    html += f"{key}: {value}\n"
                html += """</pre>
                </div>
                """
                
            html += "</div>"
        
        html += "</div>"
    
    # Steps section
    steps = lineage_dict.get('steps', [])
    if steps:
        html += """
        <div class="section">
            <h2>Processing Steps</h2>
        """
        
        for i, step in enumerate(steps):
            html += f"""
            <div class="step">
                <p><strong>Step {i+1}:</strong> {step.get('name', 'Unknown')}</p>
                <p>{step.get('description', '')}</p>
            """
            
            if include_metadata and 'metadata' in step:
                html += """
                <div class="metadata">
                    <p><strong>Parameters:</strong></p>
                    <pre>"""
                for key, value in step.get('metadata', {}).items():
                    html += f"{key}: {value}\n"
                html += """</pre>
                </div>
                """
                
            html += "</div>"
        
        html += "</div>"
    
    # Include visualization graph if requested
    if include_graph:
        html += """
        <div class="section">
            <h2>Lineage Graph</h2>
            <p><em>This is a simplified visualization of the data lineage.</em></p>
            <div id="graph" style="height: 500px; border: 1px solid #ddd;"></div>
            
            <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
            <script>
            // Simple D3 graph visualization would go here
            // This is a placeholder that would need actual D3 code
            document.getElementById('graph').innerHTML = 'Graph visualization requires D3.js implementation';
            </script>
        </div>
        """
    
    # Close HTML
    html += """
    </body>
    </html>
    """
    
    return html


def export_lineage_summary(
    df: pd.DataFrame, 
    path: Optional[PathLike] = None,
    max_sources: int = 5,
    max_steps: int = 10
) -> Optional[str]:
    """
    Export a concise summary of lineage information from a DataFrame.
    
    This function creates a plain text summary of lineage information,
    useful for quick inspection or logging.
    
    Args:
        df: DataFrame to export lineage summary from
        path: Optional path to save the summary file
        max_sources: Maximum number of sources to include
        max_steps: Maximum number of steps to include
        
    Returns:
        Summary string if path is None, None otherwise
        
    Raises:
        ValueError: If the DataFrame does not have lineage or export fails
        
    Example:
        >>> # Export to string
        >>> summary = export_lineage_summary(df)
        >>> print(summary)
        >>> 
        >>> # Export to file
        >>> export_lineage_summary(df, "lineage_summary.txt")
    """
    try:
        df = ensure_dataframe(df)
        
        # Check if DataFrame has lineage
        if not has_lineage(df):
            logger.warning("DataFrame has no lineage information")
            return "No lineage information available"
        
        # Extract lineage dictionary
        lineage_dict = extract_lineage_dict(df)
        if not lineage_dict:
            logger.warning("Failed to extract lineage information")
            return "Failed to extract lineage information"
        
        # Get sources and steps
        sources = lineage_dict.get('sources', [])
        steps = lineage_dict.get('steps', [])
        
        # Generate summary
        lines = [
            "LINEAGE SUMMARY",
            "===============",
            "",
            "Name: " + str(lineage_dict.get('name', 'Unknown')),
            "ID: " + str(lineage_dict.get('lineage_id', 'Unknown')),
            "Created: " + str(lineage_dict.get('creation_time', 'Unknown')),
            "",
            "SOURCES (" + str(len(lineage_dict.get('sources', []))) + " total):",
            "-------------------------"
        ]
        
        # Add sources
        sources = lineage_dict.get('sources', [])
        for i, source in enumerate(sources[:max_sources]):
            path = source.get('path', 'Unknown')
            added_at = source.get('added_at', 'Unknown')
            lines.append(str(i+1) + ". " + path + " (added: " + added_at + ")")
        
        if len(sources) > max_sources:
            lines.append("... and " + str(len(sources) - max_sources) + " more")
        
        lines.extend([
            "",
            "PROCESSING STEPS (" + str(len(lineage_dict.get('steps', []))) + " total):",
            "-------------------------"
        ])
        
        # Add processing steps
        steps = lineage_dict.get('steps', [])
        for i, step in enumerate(steps[:max_steps]):
            name = step.get('name', 'Unknown')
            description = step.get('description', '')
            added_at = step.get('added_at', 'Unknown')
            lines.append(str(i+1) + ". " + name + ": " + description + " (added: " + added_at + ")")
        
        if len(steps) > max_steps:
            lines.append("... and " + str(len(steps) - max_steps) + " more")
        
        lines.extend([
            "",
            "Generated on " + datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ])
        
        summary = "\n".join(lines)
        
        # Write to file if path is provided
        if path:
            path_obj = ensure_path(path)
            with open(path_obj, 'w') as f:
                f.write(summary)
            logger.debug("Exported lineage summary to " + str(path_obj))
            return None
        
        return summary
    except Exception as e:
        logger.error(f"Failed to export lineage summary: {str(e)}")
        raise ValueError(f"Failed to export lineage summary: {str(e)}") from e
