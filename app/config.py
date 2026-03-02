"""Typed configuration system for the Video Analytics application.

Loads settings from config.yaml and validates them at startup.
Supports environment variable overrides via APP_ prefix.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class YOLOConfig:
    """YOLO model configuration."""

    model_path: str = "yolo11n.pt"
    seg_model_path: str = "yolo11n-seg.pt"
    confidence: float = 0.5
    iou: float = 0.5
    classes: list[int] = field(default_factory=lambda: [0])


@dataclass(frozen=True)
class TrackingConfig:
    """Object tracker configuration."""

    tracker: str = "botsort.yaml"
    max_age: int = 30
    min_hits: int = 3


@dataclass(frozen=True)
class AlertConfig:
    """Alert system configuration."""

    enabled: bool = True
    cooldown_seconds: int = 10
    loiter_threshold_seconds: float = 5.0
    crowd_threshold: int = 5
    default_zone: list[int] = field(default_factory=lambda: [100, 100, 400, 400])


@dataclass(frozen=True)
class UIConfig:
    """Streamlit UI configuration."""

    page_title: str = "AI-Powered Video Analytics"
    default_zone: list[int] = field(default_factory=lambda: [150, 150, 1130, 570])
    display_skip_frames: int = 3
    max_upload_mb: int = 200
    output_format: str = "mp4"


@dataclass(frozen=True)
class Settings:
    """Root application settings — immutable after construction."""

    video_input: str = "data/input_video.mp4"
    video_output: str = "output/processed_video.mp4"
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    ui: UIConfig = field(default_factory=UIConfig)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _build_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """Instantiate a dataclass from *data*, ignoring unknown keys."""
    import dataclasses

    valid_keys = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in data.items() if k in valid_keys}
    return cls(**filtered)


def load_settings(config_path: str = "config.yaml") -> Settings:
    """Load, validate, and return application :class:`Settings`.

    Resolution order (last wins):
    1. Dataclass defaults
    2. Values from *config_path* (if the file exists)
    3. Environment variables prefixed ``APP_`` (e.g. ``APP_YOLO__CONFIDENCE=0.7``)
    """

    raw: dict[str, Any] = {}
    path = Path(config_path)
    if path.exists():
        with path.open() as fh:
            raw = yaml.safe_load(fh) or {}

    # --- env-var overrides (flat: APP_YOLO__CONFIDENCE → yolo.confidence) ---
    for env_key, env_val in os.environ.items():
        if not env_key.startswith("APP_"):
            continue
        parts = env_key[4:].lower().split("__")  # APP_YOLO__CONFIDENCE → ['yolo','confidence']
        node = raw
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _coerce(env_val)

    yolo = _build_dataclass(YOLOConfig, raw.get("yolo", {}))
    tracking = _build_dataclass(TrackingConfig, raw.get("tracking", {}))
    alert = _build_dataclass(AlertConfig, raw.get("alert", {}))
    ui = _build_dataclass(UIConfig, raw.get("ui", {}))

    top_keys = {k: v for k, v in raw.items() if k not in ("yolo", "tracking", "alert", "ui", "deep_sort")}
    return Settings(
        video_input=top_keys.get("video_input", Settings.video_input),
        video_output=top_keys.get("video_output", Settings.video_output),
        yolo=yolo,
        tracking=tracking,
        alert=alert,
        ui=ui,
    )


def _coerce(value: str) -> int | float | bool | str:
    """Best-effort cast of an env-var string to a Python primitive."""
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value
