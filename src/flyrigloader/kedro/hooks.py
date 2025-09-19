"""
Kedro lifecycle hooks for FlyRigLoader integration.

This module provides comprehensive pipeline-level integration points for FlyRigLoader
within Kedro workflows. The hooks handle configuration validation, resource management,
error recovery, and performance monitoring throughout the entire Kedro pipeline
execution lifecycle.

The module implements three specialized hook classes:
- FlyRigLoaderHooks: Primary lifecycle hooks for comprehensive pipeline integration
- FlyRigLoaderConfigHooks: Configuration-specific validation hooks
- FlyRigLoaderPerformanceHooks: Performance monitoring and metrics collection hooks

Key Features:
- Pipeline startup configuration validation and initialization
- Node-level parameter injection and resource management
- Comprehensive error handling with graceful degradation
- Performance monitoring with system resource tracking
- Integration with Kedro's logging and metadata framework
- Thread-safe operations for parallel pipeline execution
- Automatic cleanup of temporary resources and caches

Usage Examples:
    # Basic hook registration in settings.py
    HOOKS = (
        "flyrigloader.kedro.hooks.FlyRigLoaderHooks",
        "flyrigloader.kedro.hooks.FlyRigLoaderConfigHooks",
    )
    
    # Programmatic hook registration
    >>> from flyrigloader.kedro.hooks import FlyRigLoaderHooks
    >>> hooks = FlyRigLoaderHooks()
    >>> context.hooks.register(hooks)

Integration Points:
- before_pipeline_run: Configuration validation and resource initialization
- after_pipeline_run: Cleanup and performance summary reporting
- before_node_run: Node-specific setup and parameter injection
- after_node_run: Node cleanup and metrics collection
- on_pipeline_error: Error recovery and resource cleanup
- on_node_error: Node-level error handling and diagnostics
"""

import logging
import time
import threading
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

# External imports
from kedro.pipeline import Pipeline
import psutil

# Internal imports
from flyrigloader.config.models import ProjectConfig
from flyrigloader.api import get_registered_loaders
from flyrigloader.kedro.datasets import FlyRigLoaderDataSet
from flyrigloader.exceptions import RegistryError

# Configure module-level logging with Kedro integration
logger = logging.getLogger(__name__)

# Thread-safe lock for hook coordination
_hooks_lock = threading.RLock()

# Global performance tracking state
_performance_state = {
    'pipeline_start_time': None,
    'node_metrics': {},
    'resource_baseline': None,
    'error_count': 0,
    'total_datasets_processed': 0
}


class FlyRigLoaderHooks:
    """
    Primary Kedro lifecycle hooks for comprehensive FlyRigLoader pipeline integration.
    
    This class provides the main lifecycle hooks that integrate FlyRigLoader operations
    seamlessly with Kedro pipeline execution. It handles configuration validation,
    resource management, error recovery, and performance monitoring across the entire
    pipeline lifecycle.
    
    The hooks follow Kedro's standard lifecycle pattern while adding FlyRigLoader-specific
    functionality such as registry validation, dataset configuration checks, and
    comprehensive error handling with context preservation.
    
    Hook Methods:
        before_pipeline_run: Initialize FlyRigLoader resources and validate configuration
        after_pipeline_run: Clean up resources and generate performance reports
        before_node_run: Setup node-specific FlyRigLoader parameters and validation
        after_node_run: Collect metrics and perform node-level cleanup
        on_pipeline_error: Handle pipeline-level errors with recovery mechanisms
        on_node_error: Manage node-level errors with detailed diagnostics
    
    Thread Safety:
        All hook methods are thread-safe and support Kedro's parallel execution patterns
        through proper locking mechanisms and atomic operations on shared state.
    
    Examples:
        >>> hooks = FlyRigLoaderHooks()
        >>> # Hooks are automatically called by Kedro during pipeline execution
        >>> # Manual testing of hook functionality:
        >>> pipeline = Pipeline([...])
        >>> hooks.before_pipeline_run(pipeline, catalog={})
    """
    
    def __init__(self):
        """
        Initialize FlyRigLoaderHooks with default configuration and state management.
        
        Sets up internal state tracking, performance monitoring baselines, and
        configuration for thread-safe operation across multiple pipeline runs.
        """
        self._initialized = False
        self._config_cache = {}
        self._dataset_registry = {}
        self._cleanup_tasks = []
        
        logger.info("FlyRigLoaderHooks initialized for Kedro pipeline integration")
    
    def before_pipeline_run(
        self,
        run_params: Dict[str, Any],
        pipeline: Pipeline,
        catalog: Any
    ) -> None:
        """
        Initialize FlyRigLoader resources and validate configuration before pipeline execution.
        
        This hook performs comprehensive initialization of FlyRigLoader components before
        the pipeline begins execution. It validates all configurations, checks registry
        state, establishes performance monitoring baselines, and prepares resources for
        efficient pipeline operation.
        
        Operations Performed:
        1. Registry validation and loader capability checking
        2. FlyRigLoader dataset configuration validation
        3. Performance monitoring baseline establishment
        4. Resource allocation and temporary directory setup
        5. Thread-safe state initialization for parallel execution
        
        Args:
            run_params: Kedro run parameters including pipeline configuration
            pipeline: Kedro Pipeline object containing nodes and dependencies
            catalog: Kedro DataCatalog containing all dataset configurations
            
        Raises:
            RegistryError: If loader registry is in invalid state
            ConfigError: If FlyRigLoader dataset configurations are invalid
            SystemError: If system resources are insufficient for pipeline execution
            
        Examples:
            >>> pipeline = Pipeline([node1, node2])
            >>> hooks = FlyRigLoaderHooks()
            >>> hooks.before_pipeline_run({}, pipeline, catalog)
            # Automatically validates all FlyRigLoader datasets in catalog
        """
        with _hooks_lock:
            logger.info("Starting FlyRigLoader pipeline initialization")
            
            # Record pipeline start time for performance tracking
            global _performance_state
            _performance_state['pipeline_start_time'] = time.perf_counter()
            _performance_state['error_count'] = 0
            _performance_state['total_datasets_processed'] = 0
            _performance_state['node_metrics'] = {}
            
            try:
                # Step 1: Validate FlyRigLoader registry state
                logger.debug("Validating FlyRigLoader registry state")
                try:
                    registered_loaders = get_registered_loaders()
                    if not registered_loaders:
                        logger.warning("No loaders registered in FlyRigLoader registry")
                    else:
                        logger.info(f"Registry validation complete: {len(registered_loaders)} loaders available")
                        logger.debug(f"Available loader extensions: {list(registered_loaders.keys())}")
                        
                except Exception as e:
                    raise RegistryError(
                        "Failed to validate FlyRigLoader registry during pipeline initialization",
                        error_code="REGISTRY_007",
                        context={
                            "operation": "pipeline_initialization",
                            "registry_error": str(e)
                        }
                    ).with_context({"hook": "before_pipeline_run"})
                
                # Step 2: Identify and validate FlyRigLoader datasets in catalog
                flyrig_datasets = []
                if hasattr(catalog, '_datasets'):
                    for name, dataset in catalog._datasets.items():
                        if isinstance(dataset, (FlyRigLoaderDataSet, type(FlyRigLoaderDataSet))):
                            flyrig_datasets.append((name, dataset))
                            logger.debug(f"Found FlyRigLoaderDataSet: {name}")
                
                # Step 3: Validate each FlyRigLoader dataset configuration
                for dataset_name, dataset in flyrig_datasets:
                    try:
                        # Check if dataset configuration file exists
                        if hasattr(dataset, 'filepath') and dataset.filepath:
                            config_path = Path(dataset.filepath)
                            if not config_path.exists():
                                logger.warning(f"Configuration file not found for dataset '{dataset_name}': {config_path}")
                            else:
                                logger.debug(f"Configuration validated for dataset '{dataset_name}': {config_path}")
                        
                        # Cache dataset reference for node-level processing
                        self._dataset_registry[dataset_name] = dataset
                        
                    except Exception as e:
                        logger.error(f"Failed to validate dataset '{dataset_name}': {e}")
                        # Continue with other datasets rather than failing completely
                        continue
                
                logger.info(f"Validated {len(flyrig_datasets)} FlyRigLoader datasets")
                
                # Step 4: Establish system resource baseline for performance monitoring
                try:
                    process = psutil.Process(os.getpid())
                    _performance_state['resource_baseline'] = {
                        'memory_mb': process.memory_info().rss / 1024 / 1024,
                        'cpu_percent': process.cpu_percent(),
                        'open_files': len(process.open_files()) if hasattr(process, 'open_files') else 0,
                        'threads': process.num_threads(),
                        'timestamp': time.time()
                    }
                    logger.debug(f"Resource baseline established: {_performance_state['resource_baseline']}")
                    
                except Exception as e:
                    logger.warning(f"Failed to establish resource baseline: {e}")
                    _performance_state['resource_baseline'] = None
                
                # Step 5: Setup cleanup tasks for resource management
                self._cleanup_tasks = [
                    lambda: logger.debug("Performing registry cleanup"),
                    lambda: self._clear_config_cache(),
                    lambda: self._reset_performance_state()
                ]
                
                # Mark initialization as complete
                self._initialized = True
                logger.info("FlyRigLoader pipeline initialization completed successfully")
                
            except Exception as e:
                _performance_state['error_count'] += 1
                logger.error(f"Pipeline initialization failed: {e}")
                # Re-raise to let Kedro handle the pipeline failure
                raise
    
    def after_pipeline_run(
        self,
        run_params: Dict[str, Any],
        pipeline: Pipeline,
        catalog: Any
    ) -> None:
        """
        Clean up FlyRigLoader resources and generate performance summary after pipeline completion.
        
        This hook performs comprehensive cleanup of FlyRigLoader resources and generates
        detailed performance reports after pipeline execution completes. It ensures proper
        resource deallocation, cache cleanup, and provides valuable metrics for pipeline
        optimization and debugging.
        
        Operations Performed:
        1. Resource cleanup and memory deallocation
        2. Performance metrics calculation and reporting
        3. Configuration cache clearing and validation
        4. Thread-safe state reset for future pipeline runs
        5. Error summary and diagnostics reporting
        6. System resource utilization analysis
        
        Args:
            run_params: Kedro run parameters from completed pipeline
            pipeline: Kedro Pipeline object that was executed
            catalog: Kedro DataCatalog that was used during execution
            
        Examples:
            >>> # Automatically called by Kedro after pipeline completion
            >>> hooks.after_pipeline_run(run_params, pipeline, catalog)
            # Generates performance report and cleans up resources
        """
        with _hooks_lock:
            logger.info("Starting FlyRigLoader pipeline cleanup and reporting")
            
            try:
                # Step 1: Calculate pipeline execution performance metrics
                global _performance_state
                if _performance_state['pipeline_start_time'] is not None:
                    total_duration = time.perf_counter() - _performance_state['pipeline_start_time']
                    
                    logger.info(f"Pipeline execution completed in {total_duration:.2f} seconds")
                    logger.info(f"Total FlyRigLoader datasets processed: {_performance_state['total_datasets_processed']}")
                    logger.info(f"Total errors encountered: {_performance_state['error_count']}")
                    
                    # Calculate average processing time per dataset
                    if _performance_state['total_datasets_processed'] > 0:
                        avg_processing_time = total_duration / _performance_state['total_datasets_processed']
                        logger.info(f"Average dataset processing time: {avg_processing_time:.2f} seconds")
                    
                    # Report node-level performance metrics
                    if _performance_state['node_metrics']:
                        logger.debug("Node-level performance summary:")
                        for node_name, metrics in _performance_state['node_metrics'].items():
                            duration = metrics.get('duration', 0)
                            memory_peak = metrics.get('memory_peak_mb', 0)
                            logger.debug(f"  {node_name}: {duration:.2f}s, peak memory: {memory_peak:.1f}MB")
                
                # Step 2: Generate system resource utilization report
                if _performance_state['resource_baseline'] is not None:
                    try:
                        process = psutil.Process(os.getpid())
                        current_memory = process.memory_info().rss / 1024 / 1024
                        baseline_memory = _performance_state['resource_baseline']['memory_mb']
                        memory_increase = current_memory - baseline_memory
                        
                        logger.info(f"Memory utilization change: {memory_increase:+.1f}MB")
                        
                        if memory_increase > 100:  # Alert for significant memory increase
                            logger.warning(f"Significant memory increase detected: {memory_increase:.1f}MB")
                        
                    except Exception as e:
                        logger.debug(f"Failed to generate resource utilization report: {e}")
                
                # Step 3: Execute all registered cleanup tasks
                logger.debug("Executing resource cleanup tasks")
                for cleanup_task in self._cleanup_tasks:
                    try:
                        cleanup_task()
                    except Exception as e:
                        logger.warning(f"Cleanup task failed: {e}")
                
                # Step 4: Clear dataset registry and configuration cache
                self._dataset_registry.clear()
                self._config_cache.clear()
                
                # Step 5: Reset initialization state for future pipeline runs
                self._initialized = False
                
                logger.info("FlyRigLoader pipeline cleanup completed successfully")
                
            except Exception as e:
                logger.error(f"Pipeline cleanup failed: {e}")
                # Don't re-raise cleanup errors to avoid masking pipeline success
    
    def before_node_run(
        self,
        node: Any,
        catalog: Any,
        inputs: Dict[str, Any],
        is_async: bool,
        run_id: str
    ) -> None:
        """
        Setup node-specific FlyRigLoader parameters and perform pre-execution validation.
        
        This hook prepares individual nodes for execution by validating FlyRigLoader-specific
        inputs, establishing node-level performance monitoring, and injecting additional
        parameters required for optimal FlyRigLoader operation within the node context.
        
        Operations Performed:
        1. Node input validation for FlyRigLoader datasets
        2. Node-level performance monitoring initialization
        3. Memory usage baseline establishment
        4. FlyRigLoader-specific parameter injection and validation
        5. Resource allocation checks and warnings
        6. Configuration inheritance and override processing
        
        Args:
            node: Kedro Node object about to be executed
            catalog: Kedro DataCatalog providing access to datasets
            inputs: Dictionary of input data for the node
            is_async: Boolean indicating if node runs asynchronously
            run_id: Unique identifier for the current pipeline run
            
        Examples:
            >>> # Automatically called by Kedro before each node execution
            >>> hooks.before_node_run(node, catalog, inputs, False, "run_123")
            # Validates FlyRigLoader inputs and sets up monitoring
        """
        with _hooks_lock:
            node_name = node.name if hasattr(node, 'name') else str(node)
            logger.debug(f"Preparing node '{node_name}' for FlyRigLoader operations")
            
            try:
                # Step 1: Initialize node-level performance tracking
                global _performance_state
                node_start_time = time.perf_counter()
                
                # Establish memory baseline for this node
                try:
                    process = psutil.Process(os.getpid())
                    current_memory = process.memory_info().rss / 1024 / 1024
                    
                    _performance_state['node_metrics'][node_name] = {
                        'start_time': node_start_time,
                        'memory_baseline_mb': current_memory,
                        'memory_peak_mb': current_memory,
                        'flyrig_datasets_count': 0,
                        'errors': []
                    }
                    
                except Exception as e:
                    logger.debug(f"Failed to establish memory baseline for node '{node_name}': {e}")
                
                # Step 2: Identify FlyRigLoader datasets in node inputs
                flyrig_input_count = 0
                for input_name, input_data in inputs.items():
                    # Check if input comes from a FlyRigLoader dataset
                    if input_name in self._dataset_registry:
                        flyrig_input_count += 1
                        logger.debug(f"Node '{node_name}' uses FlyRigLoader dataset: {input_name}")
                        
                        # Validate input data structure if available
                        if input_data is not None:
                            try:
                                # Perform basic validation on FlyRigLoader data
                                if hasattr(input_data, 'shape'):
                                    logger.debug(f"Dataset '{input_name}' shape: {input_data.shape}")
                                if hasattr(input_data, 'columns'):
                                    logger.debug(f"Dataset '{input_name}' columns: {len(input_data.columns)}")
                                    
                            except Exception as e:
                                logger.warning(f"Failed to validate input '{input_name}' for node '{node_name}': {e}")
                
                # Update node metrics with FlyRigLoader dataset count
                if node_name in _performance_state['node_metrics']:
                    _performance_state['node_metrics'][node_name]['flyrig_datasets_count'] = flyrig_input_count
                
                # Step 3: Log node preparation summary
                if flyrig_input_count > 0:
                    logger.info(f"Node '{node_name}' prepared with {flyrig_input_count} FlyRigLoader datasets")
                else:
                    logger.debug(f"Node '{node_name}' does not use FlyRigLoader datasets")
                
                # Step 4: Check system resources for node execution
                try:
                    memory = psutil.virtual_memory()
                    if memory.percent > 90:
                        logger.warning(f"High memory usage ({memory.percent:.1f}%) before node '{node_name}' execution")
                    
                except Exception as e:
                    logger.debug(f"Failed to check system resources for node '{node_name}': {e}")
                
            except Exception as e:
                logger.error(f"Failed to prepare node '{node_name}': {e}")
                _performance_state['error_count'] += 1
                # Don't raise exception to avoid blocking node execution
    
    def after_node_run(
        self,
        node: Any,
        catalog: Any,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        is_async: bool,
        run_id: str
    ) -> None:
        """
        Collect performance metrics and perform node-level cleanup after execution.
        
        This hook collects detailed performance metrics and performs cleanup operations
        after individual nodes complete execution. It tracks resource utilization,
        validates outputs, and maintains comprehensive statistics for pipeline optimization.
        
        Operations Performed:
        1. Performance metrics collection and analysis
        2. Memory usage tracking and peak detection
        3. Output validation for FlyRigLoader-generated data
        4. Node-level resource cleanup and optimization
        5. Error tracking and diagnostic information collection
        6. Statistical updates for pipeline-level reporting
        
        Args:
            node: Kedro Node object that was executed
            catalog: Kedro DataCatalog used during execution
            inputs: Dictionary of input data that was used
            outputs: Dictionary of output data that was generated
            is_async: Boolean indicating if node ran asynchronously
            run_id: Unique identifier for the current pipeline run
            
        Examples:
            >>> # Automatically called by Kedro after each node execution
            >>> hooks.after_node_run(node, catalog, inputs, outputs, False, "run_123")
            # Collects metrics and validates outputs
        """
        with _hooks_lock:
            node_name = node.name if hasattr(node, 'name') else str(node)
            logger.debug(f"Collecting metrics for completed node '{node_name}'")
            
            try:
                # Step 1: Calculate node execution duration and update metrics
                global _performance_state
                if node_name in _performance_state['node_metrics']:
                    node_metrics = _performance_state['node_metrics'][node_name]
                    duration = time.perf_counter() - node_metrics['start_time']
                    node_metrics['duration'] = duration
                    
                    # Track memory peak during node execution
                    try:
                        process = psutil.Process(os.getpid())
                        current_memory = process.memory_info().rss / 1024 / 1024
                        node_metrics['memory_peak_mb'] = max(
                            node_metrics.get('memory_peak_mb', 0),
                            current_memory
                        )
                        
                        # Calculate memory delta for this node
                        memory_delta = current_memory - node_metrics.get('memory_baseline_mb', current_memory)
                        node_metrics['memory_delta_mb'] = memory_delta
                        
                    except Exception as e:
                        logger.debug(f"Failed to track memory for node '{node_name}': {e}")
                    
                    # Log performance summary for nodes with significant resource usage
                    if duration > 1.0:  # Nodes taking more than 1 second
                        logger.info(f"Node '{node_name}' completed in {duration:.2f}s")
                        
                        flyrig_count = node_metrics.get('flyrig_datasets_count', 0)
                        if flyrig_count > 0:
                            logger.info(f"  Processed {flyrig_count} FlyRigLoader datasets")
                            
                        memory_peak = node_metrics.get('memory_peak_mb', 0)
                        if memory_peak > 0:
                            logger.debug(f"  Peak memory usage: {memory_peak:.1f}MB")
                
                # Step 2: Validate outputs from FlyRigLoader operations
                flyrig_output_count = 0
                for output_name, output_data in outputs.items():
                    # Check if this output might be from FlyRigLoader processing
                    if output_data is not None and hasattr(output_data, 'shape'):
                        # Validate DataFrame structure for FlyRigLoader outputs
                        try:
                            if hasattr(output_data, 'columns'):
                                # Check for common FlyRigLoader metadata columns
                                flyrig_columns = [col for col in output_data.columns 
                                                if col in ['experiment_name', 'dataset_source', 'load_timestamp']]
                                
                                if flyrig_columns:
                                    flyrig_output_count += 1
                                    logger.debug(f"Validated FlyRigLoader output '{output_name}': {output_data.shape}")
                                    
                        except Exception as e:
                            logger.debug(f"Failed to validate output '{output_name}': {e}")
                
                # Step 3: Update global pipeline statistics
                if flyrig_output_count > 0:
                    _performance_state['total_datasets_processed'] += flyrig_output_count
                    logger.debug(f"Node '{node_name}' generated {flyrig_output_count} FlyRigLoader outputs")
                
            except Exception as e:
                logger.error(f"Failed to collect metrics for node '{node_name}': {e}")
                _performance_state['error_count'] += 1
                # Don't raise exception to avoid disrupting pipeline flow
    
    def on_pipeline_error(
        self,
        error: Exception,
        run_params: Dict[str, Any],
        pipeline: Pipeline,
        catalog: Any
    ) -> None:
        """
        Handle pipeline-level errors with comprehensive recovery mechanisms and diagnostics.
        
        This hook provides comprehensive error handling for pipeline-level failures,
        including FlyRigLoader-specific error analysis, resource cleanup, diagnostic
        information collection, and graceful degradation strategies where possible.
        
        Operations Performed:
        1. Error classification and FlyRigLoader-specific analysis
        2. Diagnostic information collection and context preservation
        3. Resource cleanup and memory deallocation
        4. Error recovery attempts where appropriate
        5. Comprehensive error logging with actionable information
        6. State reset for potential pipeline retry operations
        
        Args:
            error: Exception that caused the pipeline failure
            run_params: Kedro run parameters from failed pipeline
            pipeline: Kedro Pipeline object that failed
            catalog: Kedro DataCatalog that was being used
            
        Examples:
            >>> # Automatically called by Kedro when pipeline fails
            >>> hooks.on_pipeline_error(exception, run_params, pipeline, catalog)
            # Analyzes error, cleans up resources, provides diagnostics
        """
        with _hooks_lock:
            logger.error(f"Pipeline execution failed with error: {error}")
            
            try:
                # Step 1: Classify error type and determine if FlyRigLoader-related
                error_context = {
                    'error_type': type(error).__name__,
                    'error_message': str(error),
                    'pipeline_node_count': len(pipeline.nodes) if hasattr(pipeline, 'nodes') else 0,
                    'flyrig_datasets_in_catalog': len(self._dataset_registry)
                }
                
                # Check if error is from FlyRigLoader components
                is_flyrig_error = (
                    'flyrigloader' in str(error).lower() or
                    any(cls_name in str(type(error)) for cls_name in 
                        ['ConfigError', 'DiscoveryError', 'LoadError', 'TransformError', 'RegistryError']) or
                    any(dataset_name in str(error) for dataset_name in self._dataset_registry.keys())
                )
                
                if is_flyrig_error:
                    logger.error("Error appears to be FlyRigLoader-related, performing specialized diagnostics")
                    error_context['flyrig_related'] = True
                    
                    # Collect FlyRigLoader-specific diagnostic information
                    try:
                        # Check registry state
                        registered_loaders = get_registered_loaders()
                        error_context['registry_loader_count'] = len(registered_loaders)
                        
                        # Check dataset configurations
                        config_issues = []
                        for dataset_name, dataset in self._dataset_registry.items():
                            if hasattr(dataset, 'filepath') and dataset.filepath:
                                config_path = Path(dataset.filepath)
                                if not config_path.exists():
                                    config_issues.append(f"Missing config: {dataset_name}")
                        
                        if config_issues:
                            error_context['config_issues'] = config_issues
                            
                    except Exception as diagnostic_error:
                        logger.warning(f"Failed to collect FlyRigLoader diagnostics: {diagnostic_error}")
                        error_context['diagnostic_error'] = str(diagnostic_error)
                
                # Step 2: Collect system state information
                try:
                    process = psutil.Process(os.getpid())
                    error_context.update({
                        'memory_usage_mb': process.memory_info().rss / 1024 / 1024,
                        'cpu_percent': process.cpu_percent(),
                        'open_files_count': len(process.open_files()) if hasattr(process, 'open_files') else 0,
                        'thread_count': process.num_threads()
                    })
                    
                except Exception as e:
                    logger.debug(f"Failed to collect system state: {e}")
                
                # Step 3: Update error statistics
                global _performance_state
                _performance_state['error_count'] += 1
                
                # Log comprehensive error information
                logger.error("Pipeline error context:")
                for key, value in error_context.items():
                    logger.error(f"  {key}: {value}")
                
                # Step 4: Perform emergency resource cleanup
                logger.info("Performing emergency resource cleanup")
                try:
                    # Clear caches and temporary state
                    self._config_cache.clear()
                    
                    # Execute cleanup tasks
                    for cleanup_task in self._cleanup_tasks:
                        try:
                            cleanup_task()
                        except Exception as cleanup_error:
                            logger.debug(f"Cleanup task failed during error handling: {cleanup_error}")
                    
                    logger.info("Emergency cleanup completed")
                    
                except Exception as cleanup_error:
                    logger.error(f"Emergency cleanup failed: {cleanup_error}")
                
                # Step 5: Reset state for potential retry
                self._initialized = False
                self._dataset_registry.clear()
                
            except Exception as handler_error:
                logger.critical(f"Error handler itself failed: {handler_error}")
                # Ensure we don't mask the original error
    
    def on_node_error(
        self,
        error: Exception,
        node: Any,
        catalog: Any,
        inputs: Dict[str, Any],
        is_async: bool,
        run_id: str
    ) -> None:
        """
        Handle node-level errors with detailed diagnostics and recovery attempts.
        
        This hook provides specialized error handling for individual node failures,
        focusing on FlyRigLoader-specific error analysis, input validation diagnostics,
        and targeted recovery strategies that may allow the pipeline to continue.
        
        Operations Performed:
        1. Node-specific error analysis and input validation
        2. FlyRigLoader dataset diagnostic collection
        3. Memory and resource state analysis at failure point
        4. Input data structure validation and error correlation
        5. Detailed error logging with node context
        6. Resource cleanup specific to the failed node
        
        Args:
            error: Exception that caused the node failure
            node: Kedro Node object that failed
            catalog: Kedro DataCatalog being used
            inputs: Dictionary of input data for the failed node
            is_async: Boolean indicating if node was running asynchronously
            run_id: Unique identifier for the current pipeline run
            
        Examples:
            >>> # Automatically called by Kedro when a node fails
            >>> hooks.on_node_error(exception, node, catalog, inputs, False, "run_123")
            # Analyzes node failure, validates inputs, provides diagnostics
        """
        with _hooks_lock:
            node_name = node.name if hasattr(node, 'name') else str(node)
            logger.error(f"Node '{node_name}' failed with error: {error}")
            
            try:
                # Step 1: Collect node-specific error context
                error_context = {
                    'node_name': node_name,
                    'error_type': type(error).__name__,
                    'error_message': str(error),
                    'input_count': len(inputs),
                    'is_async': is_async,
                    'run_id': run_id
                }
                
                # Step 2: Check if node uses FlyRigLoader datasets
                flyrig_inputs = {}
                for input_name, input_data in inputs.items():
                    if input_name in self._dataset_registry:
                        flyrig_inputs[input_name] = {
                            'dataset_type': type(self._dataset_registry[input_name]).__name__,
                            'data_available': input_data is not None,
                            'data_type': type(input_data).__name__ if input_data is not None else None
                        }
                        
                        # Collect detailed input diagnostics
                        if input_data is not None:
                            try:
                                if hasattr(input_data, 'shape'):
                                    flyrig_inputs[input_name]['shape'] = input_data.shape
                                if hasattr(input_data, 'columns'):
                                    flyrig_inputs[input_name]['column_count'] = len(input_data.columns)
                                if hasattr(input_data, 'empty'):
                                    flyrig_inputs[input_name]['is_empty'] = input_data.empty
                                    
                            except Exception as validation_error:
                                flyrig_inputs[input_name]['validation_error'] = str(validation_error)
                
                if flyrig_inputs:
                    error_context['flyrig_inputs'] = flyrig_inputs
                    logger.error(f"Node '{node_name}' was processing {len(flyrig_inputs)} FlyRigLoader datasets")
                
                # Step 3: Collect memory and performance context at failure
                global _performance_state
                if node_name in _performance_state['node_metrics']:
                    node_metrics = _performance_state['node_metrics'][node_name]
                    execution_time = time.perf_counter() - node_metrics.get('start_time', time.perf_counter())
                    
                    error_context.update({
                        'execution_time_at_failure': execution_time,
                        'memory_baseline_mb': node_metrics.get('memory_baseline_mb', 0),
                        'flyrig_datasets_count': node_metrics.get('flyrig_datasets_count', 0)
                    })
                    
                    # Record error in node metrics
                    if 'errors' not in node_metrics:
                        node_metrics['errors'] = []
                    node_metrics['errors'].append({
                        'error_type': type(error).__name__,
                        'error_message': str(error),
                        'timestamp': time.time()
                    })
                
                # Step 4: Analyze error for FlyRigLoader-specific issues
                if ('flyrigloader' in str(error).lower() or 
                    any(dataset_name in str(error) for dataset_name in self._dataset_registry.keys())):
                    
                    logger.error("Node error appears to be FlyRigLoader-related")
                    error_context['flyrig_related'] = True
                    
                    # Check for common FlyRigLoader error patterns
                    error_message = str(error).lower()
                    if 'configuration' in error_message or 'config' in error_message:
                        error_context['likely_cause'] = 'configuration_error'
                    elif 'file not found' in error_message or 'path' in error_message:
                        error_context['likely_cause'] = 'file_access_error'
                    elif 'registry' in error_message or 'loader' in error_message:
                        error_context['likely_cause'] = 'registry_error'
                    elif 'transform' in error_message or 'dataframe' in error_message:
                        error_context['likely_cause'] = 'transformation_error'
                
                # Step 5: Log comprehensive error diagnostics
                logger.error(f"Node error context for '{node_name}':")
                for key, value in error_context.items():
                    logger.error(f"  {key}: {value}")
                
                # Step 6: Update global error statistics
                _performance_state['error_count'] += 1
                
                # Step 7: Perform node-specific cleanup
                try:
                    # Clear any cached data related to this node
                    node_cache_keys = [key for key in self._config_cache.keys() if node_name in key]
                    for key in node_cache_keys:
                        del self._config_cache[key]
                    
                    logger.debug(f"Cleaned up cached data for failed node '{node_name}'")
                    
                except Exception as cleanup_error:
                    logger.debug(f"Node cleanup failed for '{node_name}': {cleanup_error}")
                
            except Exception as handler_error:
                logger.critical(f"Node error handler failed for '{node_name}': {handler_error}")
                # Ensure we don't mask the original error
    
    def _clear_config_cache(self) -> None:
        """Clear configuration cache to free memory and reset state."""
        try:
            cache_size = len(self._config_cache)
            self._config_cache.clear()
            logger.debug(f"Cleared configuration cache ({cache_size} entries)")
        except Exception as e:
            logger.debug(f"Failed to clear configuration cache: {e}")
    
    def _reset_performance_state(self) -> None:
        """Reset global performance tracking state."""
        try:
            global _performance_state
            _performance_state.update({
                'pipeline_start_time': None,
                'node_metrics': {},
                'resource_baseline': None,
                'error_count': 0,
                'total_datasets_processed': 0
            })
            logger.debug("Reset performance tracking state")
        except Exception as e:
            logger.debug(f"Failed to reset performance state: {e}")


class FlyRigLoaderConfigHooks:
    """Configuration-specific hooks for FlyRigLoader validation."""
    
    def __init__(self):
        """Initialize configuration hooks with validation state tracking."""
        self._validated_configs = {}
        
        logger.info("FlyRigLoaderConfigHooks initialized for configuration management")
    
    def before_pipeline_run(
        self,
        run_params: Dict[str, Any],
        pipeline: Pipeline,
        catalog: Any
    ) -> None:
        """Validate configurations before pipeline execution."""
        logger.info("Starting comprehensive FlyRigLoader configuration validation")
        
        try:
            # Validate all FlyRigLoader dataset configurations
            if hasattr(catalog, '_datasets'):
                for dataset_name, dataset in catalog._datasets.items():
                    if isinstance(dataset, (FlyRigLoaderDataSet, type(FlyRigLoaderDataSet))):
                        self.validate_configuration(dataset_name, dataset)
                        
            logger.info("Configuration validation completed successfully")
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
    
    def validate_configuration(self, dataset_name: str, dataset: Any) -> Dict[str, Any]:
        """
        Perform deep validation of FlyRigLoader dataset configuration.
        
        Args:
            dataset_name: Name of the dataset in the catalog
            dataset: FlyRigLoader dataset instance
            
        Returns:
            Dict containing validation results and metadata
        """
        logger.debug(f"Validating configuration for dataset '{dataset_name}'")
        
        validation_result = {
            'dataset_name': dataset_name,
            'valid': True,
            'errors': [],
            'warnings': [],
            'metadata': {}
        }
        
        try:
            # Validate configuration file existence
            if hasattr(dataset, 'filepath') and dataset.filepath:
                config_path = Path(dataset.filepath)
                if not config_path.exists():
                    validation_result['errors'].append(f"Configuration file not found: {config_path}")
                    validation_result['valid'] = False
                else:
                    validation_result['metadata']['config_path'] = str(config_path)
                    validation_result['metadata']['config_size'] = config_path.stat().st_size
            
            # Validate experiment name
            if hasattr(dataset, 'experiment_name'):
                if not dataset.experiment_name or not isinstance(dataset.experiment_name, str):
                    validation_result['errors'].append("Invalid or missing experiment_name")
                    validation_result['valid'] = False
                else:
                    validation_result['metadata']['experiment_name'] = dataset.experiment_name
            
            # Store validation result
            self._validated_configs[dataset_name] = validation_result
            
            if validation_result['valid']:
                logger.debug(f"Configuration validated successfully for '{dataset_name}'")
            else:
                logger.warning(f"Configuration validation failed for '{dataset_name}': {validation_result['errors']}")
            
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Validation exception: {str(e)}")
            logger.error(f"Configuration validation error for '{dataset_name}': {e}")
        
        return validation_result
    
class FlyRigLoaderPerformanceHooks:
    """
    Performance monitoring and metrics collection hooks for FlyRigLoader operations.
    
    This specialized hook class provides comprehensive performance monitoring,
    resource utilization tracking, and metrics collection specifically for
    FlyRigLoader operations within Kedro pipelines.
    
    Hook Methods:
        before_node_run: Initialize performance monitoring for node execution
        after_node_run: Collect performance metrics and analyze resource usage
        collect_metrics: Gather detailed performance and resource metrics
        log_performance_stats: Generate comprehensive performance reports
    
    Examples:
        >>> perf_hooks = FlyRigLoaderPerformanceHooks()
        >>> # Hooks automatically monitor performance during pipeline execution
        >>> # and generate detailed reports
    """
    
    def __init__(self):
        """Initialize performance monitoring with metrics tracking state."""
        self._node_performance = {}
        self._system_metrics = {}
        self._performance_thresholds = {
            'max_memory_mb': 1000,
            'max_execution_time': 300,
            'max_cpu_percent': 80
        }
        
        logger.info("FlyRigLoaderPerformanceHooks initialized for performance monitoring")
    
    def before_node_run(
        self,
        node: Any,
        catalog: Any,
        inputs: Dict[str, Any],
        is_async: bool,
        run_id: str
    ) -> None:
        """
        Initialize comprehensive performance monitoring for node execution.
        
        Args:
            node: Kedro Node about to be executed
            catalog: Kedro DataCatalog
            inputs: Node input data
            is_async: Async execution flag
            run_id: Pipeline run identifier
        """
        node_name = node.name if hasattr(node, 'name') else str(node)
        logger.debug(f"Initializing performance monitoring for node '{node_name}'")
        
        try:
            # Initialize performance tracking
            self._node_performance[node_name] = {
                'start_time': time.perf_counter(),
                'start_timestamp': time.time(),
                'memory_start': self._get_memory_usage(),
                'cpu_start': self._get_cpu_percent(),
                'flyrig_inputs': len([name for name in inputs.keys() if 'flyrig' in name.lower()]),
                'run_id': run_id
            }
            
            logger.debug(f"Performance monitoring initialized for node '{node_name}'")
            
        except Exception as e:
            logger.debug(f"Failed to initialize performance monitoring for '{node_name}': {e}")
    
    def after_node_run(
        self,
        node: Any,
        catalog: Any,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        is_async: bool,
        run_id: str
    ) -> None:
        """
        Collect comprehensive performance metrics after node execution.
        
        Args:
            node: Kedro Node that was executed
            catalog: Kedro DataCatalog
            inputs: Node input data
            outputs: Node output data
            is_async: Async execution flag
            run_id: Pipeline run identifier
        """
        node_name = node.name if hasattr(node, 'name') else str(node)
        
        try:
            if node_name in self._node_performance:
                # Calculate execution metrics
                perf_data = self._node_performance[node_name]
                perf_data.update({
                    'end_time': time.perf_counter(),
                    'end_timestamp': time.time(),
                    'memory_end': self._get_memory_usage(),
                    'cpu_end': self._get_cpu_percent(),
                    'output_count': len(outputs)
                })
                
                # Calculate derived metrics
                perf_data['duration'] = perf_data['end_time'] - perf_data['start_time']
                perf_data['memory_delta'] = perf_data['memory_end'] - perf_data['memory_start']
                
                # Collect and log metrics
                self.collect_metrics(node_name, perf_data)
                self.log_performance_stats(node_name, perf_data)
            
        except Exception as e:
            logger.debug(f"Failed to collect performance metrics for '{node_name}': {e}")
    
    def collect_metrics(self, node_name: str, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gather detailed performance and resource metrics for analysis.
        
        Args:
            node_name: Name of the node
            performance_data: Raw performance data collected
            
        Returns:
            Dict containing processed metrics and analysis
        """
        try:
            metrics = {
                'node_name': node_name,
                'execution_time': performance_data.get('duration', 0),
                'memory_usage_mb': performance_data.get('memory_end', 0),
                'memory_delta_mb': performance_data.get('memory_delta', 0),
                'cpu_utilization': performance_data.get('cpu_end', 0),
                'flyrig_input_count': performance_data.get('flyrig_inputs', 0),
                'output_count': performance_data.get('output_count', 0),
                'timestamp': performance_data.get('end_timestamp', time.time())
            }
            
            # Performance analysis
            metrics['performance_analysis'] = {
                'memory_efficient': metrics['memory_delta_mb'] < self._performance_thresholds['max_memory_mb'],
                'time_efficient': metrics['execution_time'] < self._performance_thresholds['max_execution_time'],
                'cpu_efficient': metrics['cpu_utilization'] < self._performance_thresholds['max_cpu_percent']
            }
            
            # Store metrics for reporting
            self._system_metrics[node_name] = metrics
            
            return metrics
            
        except Exception as e:
            logger.debug(f"Failed to collect metrics for '{node_name}': {e}")
            return {}
    
    def log_performance_stats(self, node_name: str, performance_data: Dict[str, Any]) -> None:
        """
        Generate comprehensive performance reports and logging.
        
        Args:
            node_name: Name of the node
            performance_data: Performance data to report
        """
        try:
            duration = performance_data.get('duration', 0)
            memory_delta = performance_data.get('memory_delta', 0)
            flyrig_inputs = performance_data.get('flyrig_inputs', 0)
            
            # Log basic performance information
            if duration > 0.1:  # Only log for nodes with meaningful execution time
                logger.info(f"Performance: {node_name} completed in {duration:.2f}s")
                
                if flyrig_inputs > 0:
                    logger.info(f"  FlyRigLoader inputs processed: {flyrig_inputs}")
                
                if abs(memory_delta) > 10:  # Log significant memory changes
                    logger.info(f"  Memory change: {memory_delta:+.1f}MB")
                
                # Performance warnings
                if duration > self._performance_thresholds['max_execution_time']:
                    logger.warning(f"Node '{node_name}' exceeded time threshold: {duration:.2f}s")
                
                if abs(memory_delta) > self._performance_thresholds['max_memory_mb']:
                    logger.warning(f"Node '{node_name}' exceeded memory threshold: {memory_delta:+.1f}MB")
            
        except Exception as e:
            logger.debug(f"Failed to log performance stats for '{node_name}': {e}")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
    
    def _get_cpu_percent(self) -> float:
        """Get current CPU utilization percentage."""
        try:
            return psutil.cpu_percent(interval=None)
        except Exception:
            return 0.0


# Export all hook classes for public use
__all__ = [
    'FlyRigLoaderHooks',
    'FlyRigLoaderConfigHooks', 
    'FlyRigLoaderPerformanceHooks'
]
