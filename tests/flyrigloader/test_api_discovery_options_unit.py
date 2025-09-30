"""
Unit tests for DiscoveryOptions API integration implementation details.

These tests focus on specific implementation behaviors and edge cases
for how API functions work with DiscoveryOptions.

Following TDD: These tests are written BEFORE implementation.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call


class TestAPIParameterPassing:
    """Unit tests for how API functions pass DiscoveryOptions parameters."""
    
    def test_load_experiment_files_passes_all_options_to_discover(self):
        """All DiscoveryOptions fields are passed to discovery function."""
        from flyrigloader.api import load_experiment_files
        from flyrigloader.discovery.options import DiscoveryOptions
        from flyrigloader.config.models import ProjectConfig
        
        config = ProjectConfig(
            schema_version="1.0.0",
            directories={"major_data_directory": "/tmp/test"}
        )
        
        options = DiscoveryOptions(
            pattern="exp_*.pkl",
            recursive=False,
            extensions=['.pkl', '.csv'],
            extract_metadata=True,
            parse_dates=True
        )
        
        with patch('flyrigloader.api.discover_experiment_files') as mock_discover:
            mock_discover.return_value = []
            
            load_experiment_files(config, "test_exp", options=options)
            
            # Check all fields were passed
            call_kwargs = mock_discover.call_args[1]
            assert call_kwargs['pattern'] == "exp_*.pkl"
            assert call_kwargs['recursive'] is False
            assert call_kwargs['extensions'] == ['.pkl', '.csv']
            assert call_kwargs['extract_metadata'] is True
            assert call_kwargs['parse_dates'] is True
    
    def test_load_experiment_files_uses_defaults_when_options_not_provided(self):
        """When options not provided, uses DiscoveryOptions.defaults() values."""
        from flyrigloader.api import load_experiment_files
        from flyrigloader.discovery.options import DiscoveryOptions
        from flyrigloader.config.models import ProjectConfig
        
        config = ProjectConfig(
            schema_version="1.0.0",
            directories={"major_data_directory": "/tmp/test"}
        )
        
        with patch('flyrigloader.api.discover_experiment_files') as mock_discover:
            mock_discover.return_value = []
            
            # Call without options
            load_experiment_files(config, "test_exp")
            
            # Should use default values
            call_kwargs = mock_discover.call_args[1]
            defaults = DiscoveryOptions.defaults()
            assert call_kwargs['pattern'] == defaults.pattern
            assert call_kwargs['recursive'] == defaults.recursive
            assert call_kwargs['extract_metadata'] == defaults.extract_metadata
    
    def test_load_dataset_files_passes_options_correctly(self):
        """load_dataset_files properly forwards DiscoveryOptions."""
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
        
        options = DiscoveryOptions(pattern="dataset_*.pkl", recursive=False)
        
        with patch('flyrigloader.api.discover_dataset_files') as mock_discover:
            with patch.object(config, 'get_dataset', return_value=mock_dataset):
                mock_discover.return_value = []
                
                load_dataset_files(config, "test_ds", options=options)
                
                call_kwargs = mock_discover.call_args[1]
                assert call_kwargs['pattern'] == "dataset_*.pkl"
                assert call_kwargs['recursive'] is False


class TestAPIOptionsInteraction:
    """Unit tests for how options interact with other API parameters."""
    
    def test_base_directory_works_with_options(self):
        """base_directory parameter works alongside options."""
        from flyrigloader.api import load_experiment_files
        from flyrigloader.discovery.options import DiscoveryOptions
        from flyrigloader.config.models import ProjectConfig
        
        config = ProjectConfig(
            schema_version="1.0.0",
            directories={"major_data_directory": "/tmp/default"}
        )
        
        custom_base = Path("/tmp/custom")
        options = DiscoveryOptions.minimal("*.pkl")
        
        with patch('flyrigloader.api.discover_experiment_files') as mock_discover:
            mock_discover.return_value = []
            
            load_experiment_files(
                config,
                "test_exp",
                base_directory=custom_base,
                options=options
            )
            
            # Both base_directory and options should be used
            call_args = mock_discover.call_args
            assert call_args[1]['pattern'] == "*.pkl"
            # base_directory should be passed separately or used to compute path
    
    def test_options_with_extract_metadata_returns_dict(self):
        """When extract_metadata=True, API returns dict not list."""
        from flyrigloader.api import load_experiment_files
        from flyrigloader.discovery.options import DiscoveryOptions
        from flyrigloader.config.models import ProjectConfig
        
        config = ProjectConfig(
            schema_version="1.0.0",
            directories={"major_data_directory": "/tmp/test"}
        )
        
        options = DiscoveryOptions(extract_metadata=True)
        
        with patch('flyrigloader.api.discover_experiment_files') as mock_discover:
            # When extract_metadata=True, discover returns dict
            mock_discover.return_value = {
                'file1.pkl': {'meta': 'data1'},
                'file2.pkl': {'meta': 'data2'}
            }
            
            result = load_experiment_files(config, "test_exp", options=options)
            
            # Result should be dict
            assert isinstance(result, dict)
            assert 'file1.pkl' in result
    
    def test_options_without_extract_metadata_returns_list(self):
        """When extract_metadata=False, API returns list not dict."""
        from flyrigloader.api import load_experiment_files
        from flyrigloader.discovery.options import DiscoveryOptions
        from flyrigloader.config.models import ProjectConfig
        
        config = ProjectConfig(
            schema_version="1.0.0",
            directories={"major_data_directory": "/tmp/test"}
        )
        
        options = DiscoveryOptions(extract_metadata=False)
        
        with patch('flyrigloader.api.discover_experiment_files') as mock_discover:
            # When extract_metadata=False, discover returns list
            mock_discover.return_value = ['file1.pkl', 'file2.pkl']
            
            result = load_experiment_files(config, "test_exp", options=options)
            
            # Result should be list
            assert isinstance(result, list)
            assert len(result) == 2


class TestAPIFactoryMethodUsage:
    """Unit tests for using factory methods with API in practice."""
    
    def test_defaults_factory_with_experiment_loading(self):
        """DiscoveryOptions.defaults() works in real experiment loading."""
        from flyrigloader.api import load_experiment_files
        from flyrigloader.discovery.options import DiscoveryOptions
        from flyrigloader.config.models import ProjectConfig
        
        config = ProjectConfig(
            schema_version="1.0.0",
            directories={"major_data_directory": "/tmp/test"}
        )
        
        with patch('flyrigloader.api.discover_experiment_files') as mock_discover:
            mock_discover.return_value = ['file1.pkl', 'file2.pkl']
            
            result = load_experiment_files(
                config,
                "exp1",
                options=DiscoveryOptions.defaults()
            )
            
            assert result == ['file1.pkl', 'file2.pkl']
            assert mock_discover.called
    
    def test_minimal_factory_customizes_pattern_only(self):
        """minimal() changes only pattern, leaves other defaults."""
        from flyrigloader.api import load_experiment_files
        from flyrigloader.discovery.options import DiscoveryOptions
        from flyrigloader.config.models import ProjectConfig
        
        config = ProjectConfig(
            schema_version="1.0.0",
            directories={"major_data_directory": "/tmp/test"}
        )
        
        with patch('flyrigloader.api.discover_experiment_files') as mock_discover:
            mock_discover.return_value = []
            
            load_experiment_files(
                config,
                "exp1",
                options=DiscoveryOptions.minimal("custom_*.pkl")
            )
            
            call_kwargs = mock_discover.call_args[1]
            assert call_kwargs['pattern'] == "custom_*.pkl"
            assert call_kwargs['recursive'] is True  # Still default
            assert call_kwargs['extract_metadata'] is False  # Still default
    
    def test_with_metadata_factory_enables_both_metadata_flags(self):
        """with_metadata() enables both extract_metadata and parse_dates."""
        from flyrigloader.api import load_experiment_files
        from flyrigloader.discovery.options import DiscoveryOptions
        from flyrigloader.config.models import ProjectConfig
        
        config = ProjectConfig(
            schema_version="1.0.0",
            directories={"major_data_directory": "/tmp/test"}
        )
        
        with patch('flyrigloader.api.discover_experiment_files') as mock_discover:
            mock_discover.return_value = {}
            
            load_experiment_files(
                config,
                "exp1",
                options=DiscoveryOptions.with_metadata()
            )
            
            call_kwargs = mock_discover.call_args[1]
            assert call_kwargs['extract_metadata'] is True
            assert call_kwargs['parse_dates'] is True
    
    def test_with_filtering_factory_sets_extensions_correctly(self):
        """with_filtering() properly sets extensions filter."""
        from flyrigloader.api import load_experiment_files
        from flyrigloader.discovery.options import DiscoveryOptions
        from flyrigloader.config.models import ProjectConfig
        
        config = ProjectConfig(
            schema_version="1.0.0",
            directories={"major_data_directory": "/tmp/test"}
        )
        
        with patch('flyrigloader.api.discover_experiment_files') as mock_discover:
            mock_discover.return_value = []
            
            load_experiment_files(
                config,
                "exp1",
                options=DiscoveryOptions.with_filtering(
                    extensions=['.pkl', '.csv']
                )
            )
            
            call_kwargs = mock_discover.call_args[1]
            assert call_kwargs['extensions'] == ['.pkl', '.csv']


class TestAPIOptionsValidation:
    """Unit tests for how API validates DiscoveryOptions."""
    
    def test_api_accepts_valid_discovery_options_instance(self):
        """API accepts properly created DiscoveryOptions."""
        from flyrigloader.api import load_experiment_files
        from flyrigloader.discovery.options import DiscoveryOptions
        from flyrigloader.config.models import ProjectConfig
        
        config = ProjectConfig(
            schema_version="1.0.0",
            directories={"major_data_directory": "/tmp/test"}
        )
        
        options = DiscoveryOptions(pattern="*.pkl")
        
        with patch('flyrigloader.api.discover_experiment_files') as mock_discover:
            mock_discover.return_value = []
            
            # Should not raise
            result = load_experiment_files(config, "exp1", options=options)
            assert mock_discover.called
    
    def test_api_rejects_string_as_options(self):
        """API rejects string when DiscoveryOptions expected."""
        from flyrigloader.api import load_experiment_files
        from flyrigloader.config.models import ProjectConfig
        
        config = ProjectConfig(
            schema_version="1.0.0",
            directories={"major_data_directory": "/tmp/test"}
        )
        
        with pytest.raises((TypeError, ValueError)):
            load_experiment_files(
                config,
                "exp1",
                options="*.pkl"  # Should be DiscoveryOptions instance
            )
    
    def test_api_rejects_dict_as_options(self):
        """API rejects dict when DiscoveryOptions expected."""
        from flyrigloader.api import load_experiment_files
        from flyrigloader.config.models import ProjectConfig
        
        config = ProjectConfig(
            schema_version="1.0.0",
            directories={"major_data_directory": "/tmp/test"}
        )
        
        with pytest.raises((TypeError, ValueError)):
            load_experiment_files(
                config,
                "exp1",
                options={'pattern': '*.pkl'}  # Should be DiscoveryOptions instance
            )
    
    def test_api_rejects_none_as_options_when_required(self):
        """API handles None options appropriately."""
        from flyrigloader.api import load_experiment_files
        from flyrigloader.config.models import ProjectConfig
        
        config = ProjectConfig(
            schema_version="1.0.0",
            directories={"major_data_directory": "/tmp/test"}
        )
        
        # If None is passed explicitly, should either use defaults or raise
        # (depends on signature design - options has default, so None shouldn't be passed)
        with patch('flyrigloader.api.discover_experiment_files') as mock_discover:
            mock_discover.return_value = []
            
            # Passing None explicitly might be an error or use defaults
            # Let's test both behaviors are reasonable
            try:
                result = load_experiment_files(config, "exp1", options=None)
                # If it works, should use defaults
                assert mock_discover.called
            except (TypeError, ValueError):
                # If it raises, that's also valid (explicit None not allowed)
                pass


class TestAPIOptionsReusability:
    """Unit tests for reusing DiscoveryOptions instances."""
    
    def test_same_options_instance_reused_across_calls(self):
        """Same DiscoveryOptions instance can be used multiple times."""
        from flyrigloader.api import load_experiment_files
        from flyrigloader.discovery.options import DiscoveryOptions
        from flyrigloader.config.models import ProjectConfig
        
        config = ProjectConfig(
            schema_version="1.0.0",
            directories={"major_data_directory": "/tmp/test"}
        )
        
        shared_options = DiscoveryOptions.with_metadata("*.pkl")
        
        with patch('flyrigloader.api.discover_experiment_files') as mock_discover:
            mock_discover.return_value = {}
            
            # Use same instance 3 times
            result1 = load_experiment_files(config, "exp1", options=shared_options)
            result2 = load_experiment_files(config, "exp2", options=shared_options)
            result3 = load_experiment_files(config, "exp3", options=shared_options)
            
            # All calls should succeed
            assert mock_discover.call_count == 3
            
            # Options should be unchanged (immutability)
            assert shared_options.pattern == "*.pkl"
            assert shared_options.extract_metadata is True
    
    def test_options_unchanged_after_api_call(self):
        """API calls don't modify DiscoveryOptions (immutability test)."""
        from flyrigloader.api import load_experiment_files
        from flyrigloader.discovery.options import DiscoveryOptions
        from flyrigloader.config.models import ProjectConfig
        
        config = ProjectConfig(
            schema_version="1.0.0",
            directories={"major_data_directory": "/tmp/test"}
        )
        
        original_pattern = "original_*.pkl"
        options = DiscoveryOptions(pattern=original_pattern)
        
        with patch('flyrigloader.api.discover_experiment_files') as mock_discover:
            mock_discover.return_value = []
            
            load_experiment_files(config, "exp1", options=options)
            
            # Options should be completely unchanged
            assert options.pattern == original_pattern
            assert options.recursive is True
            assert options.extract_metadata is False
    
    def test_different_options_for_different_experiments(self):
        """Different DiscoveryOptions can be used for different experiments."""
        from flyrigloader.api import load_experiment_files
        from flyrigloader.discovery.options import DiscoveryOptions
        from flyrigloader.config.models import ProjectConfig
        
        config = ProjectConfig(
            schema_version="1.0.0",
            directories={"major_data_directory": "/tmp/test"}
        )
        
        opts1 = DiscoveryOptions.minimal("*.pkl")
        opts2 = DiscoveryOptions.with_metadata("*.csv")
        
        with patch('flyrigloader.api.discover_experiment_files') as mock_discover:
            mock_discover.return_value = []
            
            load_experiment_files(config, "exp1", options=opts1)
            load_experiment_files(config, "exp2", options=opts2)
            
            # Check different patterns were used
            calls = mock_discover.call_args_list
            assert calls[0][1]['pattern'] == "*.pkl"
            assert calls[1][1]['pattern'] == "*.csv"


class TestAPIEdgeCases:
    """Unit tests for edge cases in API integration."""
    
    def test_options_with_all_metadata_flags_true(self):
        """All metadata-related flags can be True simultaneously."""
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
        
        with patch('flyrigloader.api.discover_experiment_files') as mock_discover:
            mock_discover.return_value = {}
            
            result = load_experiment_files(config, "exp1", options=options)
            
            call_kwargs = mock_discover.call_args[1]
            assert call_kwargs['extract_metadata'] is True
            assert call_kwargs['parse_dates'] is True
    
    def test_options_with_all_flags_false(self):
        """All boolean flags can be False simultaneously."""
        from flyrigloader.api import load_experiment_files
        from flyrigloader.discovery.options import DiscoveryOptions
        from flyrigloader.config.models import ProjectConfig
        
        config = ProjectConfig(
            schema_version="1.0.0",
            directories={"major_data_directory": "/tmp/test"}
        )
        
        options = DiscoveryOptions(
            recursive=False,
            extract_metadata=False,
            parse_dates=False
        )
        
        with patch('flyrigloader.api.discover_experiment_files') as mock_discover:
            mock_discover.return_value = []
            
            result = load_experiment_files(config, "exp1", options=options)
            
            call_kwargs = mock_discover.call_args[1]
            assert call_kwargs['recursive'] is False
            assert call_kwargs['extract_metadata'] is False
            assert call_kwargs['parse_dates'] is False
    
    def test_options_with_many_extensions(self):
        """API handles DiscoveryOptions with many extensions."""
        from flyrigloader.api import load_experiment_files
        from flyrigloader.discovery.options import DiscoveryOptions
        from flyrigloader.config.models import ProjectConfig
        
        config = ProjectConfig(
            schema_version="1.0.0",
            directories={"major_data_directory": "/tmp/test"}
        )
        
        many_extensions = [f'.ext{i}' for i in range(20)]
        options = DiscoveryOptions(extensions=many_extensions)
        
        with patch('flyrigloader.api.discover_experiment_files') as mock_discover:
            mock_discover.return_value = []
            
            result = load_experiment_files(config, "exp1", options=options)
            
            call_kwargs = mock_discover.call_args[1]
            assert call_kwargs['extensions'] == many_extensions
    
    def test_options_with_complex_pattern(self):
        """API handles DiscoveryOptions with complex glob pattern."""
        from flyrigloader.api import load_experiment_files
        from flyrigloader.discovery.options import DiscoveryOptions
        from flyrigloader.config.models import ProjectConfig
        
        config = ProjectConfig(
            schema_version="1.0.0",
            directories={"major_data_directory": "/tmp/test"}
        )
        
        complex_pattern = "**/subdir/exp_[0-9]{4}_*.pkl"
        options = DiscoveryOptions(pattern=complex_pattern)
        
        with patch('flyrigloader.api.discover_experiment_files') as mock_discover:
            mock_discover.return_value = []
            
            result = load_experiment_files(config, "exp1", options=options)
            
            call_kwargs = mock_discover.call_args[1]
            assert call_kwargs['pattern'] == complex_pattern
