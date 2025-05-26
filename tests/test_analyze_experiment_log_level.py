import sys
import logging
from unittest.mock import patch

import pytest

from examples.external_project import analyze_experiment


@pytest.fixture
def mock_dependencies(mocker):
    mocker.patch('flyrigloader.config.yaml_config.load_config', return_value={'project': {'directories': {'major_data_directory': '/tmp'}}})
    mocker.patch('flyrigloader.config.yaml_config.get_experiment_info', return_value={})
    mocker.patch('flyrigloader.config.discovery.discover_experiment_files', return_value=[])


def run_main(args):
    with patch.object(sys, 'argv', ['analyze_experiment.py'] + args):
        return analyze_experiment.main()


def test_debug_log_level(mock_dependencies, caplog):
    caplog.set_level(logging.DEBUG)
    run_main(['--config', 'cfg.yaml', '--experiment', 'exp', '--log-level', 'DEBUG'])
    assert any('Debug logging enabled' in r.message for r in caplog.records)


def test_warning_log_level(mock_dependencies, caplog):
    caplog.set_level(logging.DEBUG)
    run_main(['--config', 'cfg.yaml', '--experiment', 'exp', '--log-level', 'WARNING'])
    assert not any('Debug logging enabled' in r.message for r in caplog.records)


def test_default_log_level_info(mock_dependencies, caplog):
    caplog.set_level(logging.DEBUG)
    run_main(['--config', 'cfg.yaml', '--experiment', 'exp'])
    # Should default to INFO so debug message should not appear
    assert not any('Debug logging enabled' in r.message for r in caplog.records)
