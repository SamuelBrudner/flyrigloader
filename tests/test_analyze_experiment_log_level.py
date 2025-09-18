import sys
import logging
from unittest.mock import patch

import pytest

from examples.external_project import analyze_experiment


@pytest.fixture
def mock_dependencies(monkeypatch):
    monkeypatch.setattr(
        'flyrigloader.config.yaml_config.load_config',
        lambda *args, **kwargs: {'project': {'directories': {'major_data_directory': '/tmp'}}},
    )
    monkeypatch.setattr(
        'flyrigloader.config.yaml_config.get_experiment_info',
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        'flyrigloader.config.discovery.discover_experiment_files',
        lambda *args, **kwargs: [],
    )


def run_main(args):
    with patch.object(sys, 'argv', ['analyze_experiment.py'] + args):
        return analyze_experiment.main()


def test_debug_log_level(mock_dependencies, caplog, capsys):
    caplog.set_level(logging.DEBUG)
    run_main(['--config', 'cfg.yaml', '--experiment', 'exp', '--log-level', 'DEBUG'])
    stderr_output = capsys.readouterr().err
    assert 'Debug logging enabled' in stderr_output


def test_warning_log_level(mock_dependencies, caplog, capsys):
    caplog.set_level(logging.DEBUG)
    run_main(['--config', 'cfg.yaml', '--experiment', 'exp', '--log-level', 'WARNING'])
    stderr_output = capsys.readouterr().err
    assert 'Debug logging enabled' not in stderr_output


def test_default_log_level_info(mock_dependencies, caplog, capsys):
    caplog.set_level(logging.DEBUG)
    run_main(['--config', 'cfg.yaml', '--experiment', 'exp'])
    # Should default to INFO so debug message should not appear
    stderr_output = capsys.readouterr().err
    assert 'Debug logging enabled' not in stderr_output
