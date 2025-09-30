"""
Contract tests for DiscoveryOptions integration with API functions.

These tests define the behavior contracts for how load_experiment_files()
and load_dataset_files() work with DiscoveryOptions.

Following TDD: These tests are written BEFORE implementation.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch


class TestAPISignatureContract:
    """
    Contract tests for API function signatures.
    
    API functions MUST accept DiscoveryOptions and use it correctly.
    """
    
    def test_load_experiment_files_accepts_discovery_options(self):
        """
        CONTRACT: load_experiment_files() accepts options parameter.
        """
        from flyrigloader.api import load_experiment_files
        from flyrigloader.discovery.options import DiscoveryOptions
        from flyrigloader.config.models import ProjectConfig
        
        # Create minimal valid config
        config = ProjectConfig(
            schema_version="1.0.0",
            directories={"major_data_directory": "/tmp/test"}
        )
        
        options = DiscoveryOptions.minimal("*.pkl")
        
        # Contract: Function accepts options parameter without error
        # (Will fail due to missing files, but signature should work)
        with patch('flyrigloader.api.discover_experiment_files') as mock_discover:
            mock_discover.return_value = []
            
            result = load_experiment_files(
                config=config,
                experiment_name="test_exp",
                options=options
            )
            
            # Contract: Function executed with options
            mock_discover.assert_called_once()
    
    def test_load_dataset_files_accepts_discovery_options(self):
        """
        CONTRACT: load_dataset_files() accepts options parameter.
        """
        from flyrigloader.api import load_dataset_files
        from flyrigloader.discovery.options import DiscoveryOptions
        from flyrigloader.config.models import ProjectConfig, DatasetConfig
        
        # Create minimal valid config with dataset
        config = ProjectConfig(
            schema_version="1.0.0",
            directories={"major_data_directory": "/tmp/test"}
        )
        
        # Mock dataset in config
        mock_dataset = DatasetConfig(
            schema_version="1.0.0",
            rig="test_rig",
            dates_vials={"2024-01-01": [1, 2, 3]}
        )
        
        options = DiscoveryOptions.minimal("*.pkl")
        
        # Contract: Function accepts options parameter
        with patch('flyrigloader.api.discover_dataset_files') as mock_discover:
            with patch.object(config, 'get_dataset', return_value=mock_dataset):
                mock_discover.return_value = []
                
                result = load_dataset_files(
                    config=config,
                    dataset_name="test_dataset",
                    options=options
                )
                
                # Contract: Function executed with options
                mock_discover.assert_called_once()
    
    def test_load_experiment_files_has_default_options(self):
        """
        CONTRACT: load_experiment_files() provides default options if not specified.
        """
        from flyrigloader.api import load_experiment_files
        from flyrigloader.config.models import ProjectConfig
        
        config = ProjectConfig(
            schema_version="1.0.0",
            directories={"major_data_directory": "/tmp/test"}
        )
        
        # Contract: Can call without options parameter
        with patch('flyrigloader.api.discover_experiment_files') as mock_discover:
            mock_discover.return_value = []
            
            result = load_experiment_files(
                config=config,
                experiment_name="test_exp"
                # No options parameter - should use defaults
            )
            
            # Should work without error
            assert result is not None


class TestAPIBehaviorContract:
    """
    Contract tests for API behavior with DiscoveryOptions.
    
    API functions MUST respect all DiscoveryOptions settings.
    """
    
    def test_api_respects_pattern_option(self):
        """
        CONTRACT: API functions use the pattern from DiscoveryOptions.
        """
        from flyrigloader.api import load_experiment_files
        from flyrigloader.discovery.options import DiscoveryOptions
        from flyrigloader.config.models import ProjectConfig
        
        config = ProjectConfig(
            schema_version="1.0.0",
            directories={"major_data_directory": "/tmp/test"}
        )
        
        options = DiscoveryOptions(pattern="exp_*.pkl")
        
        # Contract: Pattern is passed to discovery
        with patch('flyrigloader.api.discover_experiment_files') as mock_discover:
            mock_discover.return_value = []
            
            load_experiment_files(
                config=config,
                experiment_name="test_exp",
                options=options
            )
            
            # Verify pattern was used (check call args)
            call_kwargs = mock_discover.call_args[1]
            assert 'pattern' in call_kwargs
            assert call_kwargs['pattern'] == "exp_*.pkl"
    
    def test_api_respects_recursive_option(self):
        """
        CONTRACT: API functions respect recursive setting.
        """
        from flyrigloader.api import load_experiment_files
        from flyrigloader.discovery.options import DiscoveryOptions
        from flyrigloader.config.models import ProjectConfig
        
        config = ProjectConfig(
            schema_version="1.0.0",
            directories={"major_data_directory": "/tmp/test"}
        )
        
        options = DiscoveryOptions(pattern="*.pkl", recursive=False)
        
        # Contract: recursive=False is respected
        with patch('flyrigloader.api.discover_experiment_files') as mock_discover:
            mock_discover.return_value = []
            
            load_experiment_files(
                config=config,
                experiment_name="test_exp",
                options=options
            )
            
            # Verify recursive was passed
            call_kwargs = mock_discover.call_args[1]
            assert 'recursive' in call_kwargs
            assert call_kwargs['recursive'] is False
    
    def test_api_respects_extensions_option(self):
        """
        CONTRACT: API functions use extensions filter when provided.
        """
        from flyrigloader.api import load_experiment_files
        from flyrigloader.discovery.options import DiscoveryOptions
        from flyrigloader.config.models import ProjectConfig
        
        config = ProjectConfig(
            schema_version="1.0.0",
            directories={"major_data_directory": "/tmp/test"}
        )
        
        options = DiscoveryOptions(
            pattern="*.*",
            extensions=['.pkl', '.csv']
        )
        
        # Contract: Extensions filter is applied
        with patch('flyrigloader.api.discover_experiment_files') as mock_discover:
            mock_discover.return_value = []
            
            load_experiment_files(
                config=config,
                experiment_name="test_exp",
                options=options
            )
            
            # Verify extensions were passed
            call_kwargs = mock_discover.call_args[1]
            assert 'extensions' in call_kwargs
            assert call_kwargs['extensions'] == ['.pkl', '.csv']
    
    def test_api_respects_extract_metadata_option(self):
        """
        CONTRACT: API functions extract metadata when extract_metadata=True.
        """
        from flyrigloader.api import load_experiment_files
        from flyrigloader.discovery.options import DiscoveryOptions
        from flyrigloader.config.models import ProjectConfig
        
        config = ProjectConfig(
            schema_version="1.0.0",
            directories={"major_data_directory": "/tmp/test"}
        )
        
        options = DiscoveryOptions(extract_metadata=True)
        
        # Contract: Metadata extraction is enabled
        with patch('flyrigloader.api.discover_experiment_files') as mock_discover:
            mock_discover.return_value = {}  # Returns dict when extract_metadata=True
            
            result = load_experiment_files(
                config=config,
                experiment_name="test_exp",
                options=options
            )
            
            # Verify extract_metadata was passed
            call_kwargs = mock_discover.call_args[1]
            assert 'extract_metadata' in call_kwargs
            assert call_kwargs['extract_metadata'] is True
    
    def test_api_respects_parse_dates_option(self):
        """
        CONTRACT: API functions parse dates when parse_dates=True.
        """
        from flyrigloader.api import load_experiment_files
        from flyrigloader.discovery.options import DiscoveryOptions
        from flyrigloader.config.models import ProjectConfig
        
        config = ProjectConfig(
            schema_version="1.0.0",
            directories={"major_data_directory": "/tmp/test"}
        )
        
        options = DiscoveryOptions(
            extract_metadata=True,
            parse_dates=True
        )
        
        # Contract: Date parsing is enabled
        with patch('flyrigloader.api.discover_experiment_files') as mock_discover:
            mock_discover.return_value = {}
            
            load_experiment_files(
                config=config,
                experiment_name="test_exp",
                options=options
            )
            
            # Verify parse_dates was passed
            call_kwargs = mock_discover.call_args[1]
            assert 'parse_dates' in call_kwargs
            assert call_kwargs['parse_dates'] is True


class TestAPIFactoryMethodIntegrationContract:
    """
    Contract tests for using factory methods with API.
    
    Factory methods MUST work seamlessly with API functions.
    """
    
    def test_can_use_defaults_factory_with_api(self):
        """
        CONTRACT: DiscoveryOptions.defaults() works with API functions.
        """
        from flyrigloader.api import load_experiment_files
        from flyrigloader.discovery.options import DiscoveryOptions
        from flyrigloader.config.models import ProjectConfig
        
        config = ProjectConfig(
            schema_version="1.0.0",
            directories={"major_data_directory": "/tmp/test"}
        )
        
        # Contract: defaults() creates usable options
        with patch('flyrigloader.api.discover_experiment_files') as mock_discover:
            mock_discover.return_value = []
            
            result = load_experiment_files(
                config=config,
                experiment_name="test_exp",
                options=DiscoveryOptions.defaults()
            )
            
            assert mock_discover.called
    
    def test_can_use_minimal_factory_with_api(self):
        """
        CONTRACT: DiscoveryOptions.minimal() works with API functions.
        """
        from flyrigloader.api import load_experiment_files
        from flyrigloader.discovery.options import DiscoveryOptions
        from flyrigloader.config.models import ProjectConfig
        
        config = ProjectConfig(
            schema_version="1.0.0",
            directories={"major_data_directory": "/tmp/test"}
        )
        
        # Contract: minimal() creates usable options
        with patch('flyrigloader.api.discover_experiment_files') as mock_discover:
            mock_discover.return_value = []
            
            result = load_experiment_files(
                config=config,
                experiment_name="test_exp",
                options=DiscoveryOptions.minimal("*.pkl")
            )
            
            # Verify pattern was used
            call_kwargs = mock_discover.call_args[1]
            assert call_kwargs['pattern'] == "*.pkl"
    
    def test_can_use_with_metadata_factory_with_api(self):
        """
        CONTRACT: DiscoveryOptions.with_metadata() works with API functions.
        """
        from flyrigloader.api import load_experiment_files
        from flyrigloader.discovery.options import DiscoveryOptions
        from flyrigloader.config.models import ProjectConfig
        
        config = ProjectConfig(
            schema_version="1.0.0",
            directories={"major_data_directory": "/tmp/test"}
        )
        
        # Contract: with_metadata() enables metadata extraction
        with patch('flyrigloader.api.discover_experiment_files') as mock_discover:
            mock_discover.return_value = {}
            
            result = load_experiment_files(
                config=config,
                experiment_name="test_exp",
                options=DiscoveryOptions.with_metadata()
            )
            
            # Verify metadata extraction enabled
            call_kwargs = mock_discover.call_args[1]
            assert call_kwargs['extract_metadata'] is True
            assert call_kwargs['parse_dates'] is True
    
    def test_can_use_with_filtering_factory_with_api(self):
        """
        CONTRACT: DiscoveryOptions.with_filtering() works with API functions.
        """
        from flyrigloader.api import load_experiment_files
        from flyrigloader.discovery.options import DiscoveryOptions
        from flyrigloader.config.models import ProjectConfig
        
        config = ProjectConfig(
            schema_version="1.0.0",
            directories={"major_data_directory": "/tmp/test"}
        )
        
        # Contract: with_filtering() sets extensions
        with patch('flyrigloader.api.discover_experiment_files') as mock_discover:
            mock_discover.return_value = []
            
            result = load_experiment_files(
                config=config,
                experiment_name="test_exp",
                options=DiscoveryOptions.with_filtering(extensions=['.pkl'])
            )
            
            # Verify extensions filter applied
            call_kwargs = mock_discover.call_args[1]
            assert call_kwargs['extensions'] == ['.pkl']


class TestAPIOptionsReusabilityContract:
    """
    Contract tests for reusing DiscoveryOptions across API calls.
    
    DiscoveryOptions MUST be reusable across multiple API calls.
    """
    
    def test_can_reuse_options_across_experiments(self):
        """
        CONTRACT: Same DiscoveryOptions can be used for multiple experiments.
        """
        from flyrigloader.api import load_experiment_files
        from flyrigloader.discovery.options import DiscoveryOptions
        from flyrigloader.config.models import ProjectConfig
        
        config = ProjectConfig(
            schema_version="1.0.0",
            directories={"major_data_directory": "/tmp/test"}
        )
        
        # Contract: Create options once, use many times
        shared_options = DiscoveryOptions.with_metadata("*.pkl")
        
        with patch('flyrigloader.api.discover_experiment_files') as mock_discover:
            mock_discover.return_value = []
            
            # Use same options for multiple experiments
            result1 = load_experiment_files(config, "exp1", options=shared_options)
            result2 = load_experiment_files(config, "exp2", options=shared_options)
            result3 = load_experiment_files(config, "exp3", options=shared_options)
            
            # Contract: Works without error, options unchanged
            assert mock_discover.call_count == 3
            assert shared_options.pattern == "*.pkl"  # Still unchanged
    
    def test_can_reuse_options_across_datasets(self):
        """
        CONTRACT: Same DiscoveryOptions can be used for multiple datasets.
        """
        from flyrigloader.api import load_dataset_files
        from flyrigloader.discovery.options import DiscoveryOptions
        from flyrigloader.config.models import ProjectConfig, DatasetConfig
        
        config = ProjectConfig(
            schema_version="1.0.0",
            directories={"major_data_directory": "/tmp/test"}
        )
        
        mock_dataset = DatasetConfig(
            schema_version="1.0.0",
            rig="test_rig",
            dates_vials={"2024-01-01": [1]}
        )
        
        # Contract: Reusable across datasets
        shared_options = DiscoveryOptions.minimal("*.pkl")
        
        with patch('flyrigloader.api.discover_dataset_files') as mock_discover:
            with patch.object(config, 'get_dataset', return_value=mock_dataset):
                mock_discover.return_value = []
                
                result1 = load_dataset_files(config, "ds1", options=shared_options)
                result2 = load_dataset_files(config, "ds2", options=shared_options)
                
                # Contract: Works for both datasets
                assert mock_discover.call_count == 2


class TestAPIErrorHandlingContract:
    """
    Contract tests for error handling with DiscoveryOptions.
    
    API MUST provide clear errors when DiscoveryOptions are misused.
    """
    
    def test_api_rejects_invalid_options_type(self):
        """
        CONTRACT: API raises TypeError if options is not DiscoveryOptions.
        """
        from flyrigloader.api import load_experiment_files
        from flyrigloader.config.models import ProjectConfig
        
        config = ProjectConfig(
            schema_version="1.0.0",
            directories={"major_data_directory": "/tmp/test"}
        )
        
        # Contract: Wrong type for options raises clear error
        with pytest.raises((TypeError, ValueError)) as exc_info:
            load_experiment_files(
                config=config,
                experiment_name="test_exp",
                options="not a DiscoveryOptions object"  # Wrong type!
            )
        
        # Contract: Error message indicates the problem
        assert "options" in str(exc_info.value).lower() or "DiscoveryOptions" in str(exc_info.value)
    
    def test_api_validates_options_before_use(self):
        """
        CONTRACT: API validates DiscoveryOptions before attempting discovery.
        
        If DiscoveryOptions has invalid values, API should fail fast.
        """
        from flyrigloader.api import load_experiment_files
        from flyrigloader.discovery.options import DiscoveryOptions
        from flyrigloader.config.models import ProjectConfig
        
        config = ProjectConfig(
            schema_version="1.0.0",
            directories={"major_data_directory": "/tmp/test"}
        )
        
        # Contract: Invalid options caught at API level (or DiscoveryOptions creation)
        # This should fail at DiscoveryOptions creation
        with pytest.raises(ValueError):
            invalid_options = DiscoveryOptions(pattern="")  # Empty pattern
            # If it gets past creation, API should also catch it
            load_experiment_files(config, "test", options=invalid_options)
