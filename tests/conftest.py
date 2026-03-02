"""Shared pytest fixtures for the video-analytics test suite."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.config import (
    AlertConfig,
    Settings,
    TrackingConfig,
    UIConfig,
    YOLOConfig,
    load_settings,
)


@pytest.fixture()
def sample_frame() -> np.ndarray:
    """A 720p black BGR frame for testing."""
    return np.zeros((720, 1280, 3), dtype=np.uint8)


@pytest.fixture()
def default_settings() -> Settings:
    """Application settings with safe defaults (no disk I/O)."""
    return Settings()


@pytest.fixture()
def test_zone() -> list[int]:
    """A small tracking zone for tests."""
    return [100, 100, 400, 400]


@pytest.fixture()
def alert_config() -> AlertConfig:
    return AlertConfig(
        enabled=True,
        cooldown_seconds=2,
        loiter_threshold_seconds=1.0,
        crowd_threshold=3,
    )
