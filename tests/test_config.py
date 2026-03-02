"""Tests for app.config — typed settings loading and validation."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest
import yaml

from app.config import (
    AlertConfig,
    Settings,
    UIConfig,
    YOLOConfig,
    load_settings,
)


class TestDefaults:
    """Settings should have sensible defaults even without config.yaml."""

    def test_default_yolo_model(self) -> None:
        s = Settings()
        assert s.yolo.model_path == "yolo11n.pt"

    def test_default_confidence(self) -> None:
        s = Settings()
        assert s.yolo.confidence == 0.5

    def test_default_zone_is_list(self) -> None:
        s = Settings()
        assert isinstance(s.ui.default_zone, list)
        assert len(s.ui.default_zone) == 4


class TestLoadFromFile:
    """Loading from a YAML file overrides defaults."""

    def test_override_confidence(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yaml"
        cfg.write_text(yaml.dump({"yolo": {"confidence": 0.8}}))
        s = load_settings(str(cfg))
        assert s.yolo.confidence == 0.8

    def test_missing_file_uses_defaults(self, tmp_path: Path) -> None:
        s = load_settings(str(tmp_path / "nonexistent.yaml"))
        assert s.yolo.confidence == 0.5

    def test_unknown_keys_ignored(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.yaml"
        cfg.write_text(yaml.dump({"yolo": {"unknown_key": 42, "confidence": 0.9}}))
        s = load_settings(str(cfg))
        assert s.yolo.confidence == 0.9


class TestEnvOverrides:
    """Environment variables prefixed APP_ override file values."""

    def test_env_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = tmp_path / "config.yaml"
        cfg.write_text(yaml.dump({"yolo": {"confidence": 0.5}}))
        monkeypatch.setenv("APP_YOLO__CONFIDENCE", "0.75")
        s = load_settings(str(cfg))
        assert s.yolo.confidence == 0.75
