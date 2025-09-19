import flyrigloader.api as api
from flyrigloader.api import discover_experiment_manifest
from flyrigloader.discovery.files import FileManifest, FileInfo


def test_manifest_metadata_defaults_to_empty_dict(monkeypatch):
    """Ensure manifest conversion normalizes missing metadata to empty dict."""
    captured_calls = {}

    def fake_discover(**kwargs):
        captured_calls.update(kwargs)
        return FileManifest(
            files=[
                FileInfo(path='/data/file.pkl', size=None, extracted_metadata=None, parsed_date=None)
            ]
        )

    monkeypatch.setattr(api, '_discover_experiment_manifest', fake_discover)

    config = {
        'project': {'directories': {'major_data_directory': '/data'}},
        'experiments': {'exp': {'datasets': ['dataset']}}
    }

    manifest = discover_experiment_manifest(config=config, experiment_name='exp')

    assert manifest['/data/file.pkl']['metadata'] == {}
    assert manifest['/data/file.pkl']['size'] == 0
    assert manifest['/data/file.pkl']['parsed_dates'] == {}
    assert captured_calls['parse_dates'] is True
    assert captured_calls['include_stats'] is True
