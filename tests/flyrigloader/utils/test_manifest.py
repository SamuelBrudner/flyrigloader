"""Tests for ``flyrigloader.utils.manifest`` utilities."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

import pandas as pd
import pytest

from flyrigloader.utils import manifest as m


@pytest.fixture()
def tmp_exp_files(tmp_path: Path):
    """Create a temporary experiment directory with several files."""
    paths: List[Path] = []
    # Valid file that should be discovered
    p_exp = tmp_path / "exp_matrix.pklz"
    p_exp.write_bytes(b"dummy")
    paths.append(p_exp)

    # Another pklz file with different name → should still be discovered by our
    # mocked loader but filtered later if we choose so.
    p_other_pklz = tmp_path / "other.pklz"
    p_other_pklz.write_bytes(b"dummy2")
    paths.append(p_other_pklz)

    # A non-pklz file (should be ignored for manifest construction)
    (tmp_path / "ignore.txt").write_text("text")

    return paths, tmp_path


def test_attach_file_stats(tmp_exp_files):
    paths, _ = tmp_exp_files
    records = [{"path": str(p)} for p in paths]
    enriched = m.attach_file_stats(records, add_timestamps=True, add_file_size=True)

    for rec in enriched:
        assert "file_size" in rec
        assert rec["file_size"] > 0
        assert "mtime" in rec and "ctime" in rec


def test_build_manifest_df_creates_dataframe(tmp_exp_files):
    paths, base_dir = tmp_exp_files
    df = m.build_manifest_df(paths)

    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) >= {"path"}
    # All paths should be absolute in the DataFrame
    for p in df["path"]:
        assert Path(p).is_absolute()


def test_build_file_manifest_orchestrator(monkeypatch, tmp_exp_files):
    paths, base_dir = tmp_exp_files

    # Mock the internal loader so that we don't depend on configuration parsing
    def _mock_loader(**kwargs):  # noqa: D401 – simple mock helper
        # Always return only exp_matrix.pklz files
        return [str(p) for p in paths if p.name == "exp_matrix.pklz"]

    monkeypatch.setattr(m, "_load_experiment_files", _mock_loader)

    df = m.build_file_manifest(
        config={},
        experiment_name="dummy_exp",
        base_directory=base_dir,
        store_relative_paths=True,
    )

    # We should get exactly one row corresponding to exp_matrix.pklz
    assert len(df) == 1
    assert df.iloc[0]["path"] == "exp_matrix.pklz"  # relative path stored

    # Stat columns should be present by default
    assert "file_size" in df.columns
