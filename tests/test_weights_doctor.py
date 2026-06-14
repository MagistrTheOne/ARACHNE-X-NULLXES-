"""Weights layout doctor CLI tests (CPU-safe)."""

from __future__ import annotations

from pathlib import Path

from arachne_x.weights_resolve import doctor_checkpoint_layout


def test_doctor_missing_dir():
    errors = doctor_checkpoint_layout("/nonexistent/checkpoint/path")
    assert any("not found" in e for e in errors)


def test_doctor_empty_dir(tmp_path: Path):
    errors = doctor_checkpoint_layout(str(tmp_path))
    assert len(errors) >= 4
