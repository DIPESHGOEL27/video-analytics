"""Tests for FrameProcessor — geometry helpers and extraction logic."""

from __future__ import annotations

import numpy as np
import pytest

from app.analytics.frame_processor import DetectedObject, FrameProcessor, FrameStats
from app.config import Settings


class TestRectToPolygon:
    def test_converts_correctly(self) -> None:
        poly = FrameProcessor._rect_to_polygon([10, 20, 110, 220])
        assert poly == [(10, 20), (110, 20), (110, 220), (10, 220)]

    def test_zero_area(self) -> None:
        poly = FrameProcessor._rect_to_polygon([0, 0, 0, 0])
        assert len(poly) == 4


class TestPointInZone:
    def test_inside(self, test_zone: list[int]) -> None:
        p = FrameProcessor(mode="TrackZone", region=test_zone, settings=Settings())
        # Don't need working model for geometry tests
        assert p._is_point_in_zone((200, 200)) is True

    def test_outside(self, test_zone: list[int]) -> None:
        p = FrameProcessor(mode="TrackZone", region=test_zone, settings=Settings())
        assert p._is_point_in_zone((0, 0)) is False

    def test_on_boundary(self, test_zone: list[int]) -> None:
        p = FrameProcessor(mode="TrackZone", region=test_zone, settings=Settings())
        assert p._is_point_in_zone((100, 100)) is True  # inclusive boundary

    def test_just_outside(self, test_zone: list[int]) -> None:
        p = FrameProcessor(mode="TrackZone", region=test_zone, settings=Settings())
        assert p._is_point_in_zone((99, 200)) is False


class TestBuildTrackMap:
    def test_builds_from_detections(self, test_zone: list[int]) -> None:
        p = FrameProcessor(mode="TrackZone", region=test_zone, settings=Settings())
        dets = [
            DetectedObject(track_id=1, bbox=(150, 150, 250, 250), class_id=0, confidence=0.9),
            DetectedObject(track_id=2, bbox=(500, 500, 600, 600), class_id=0, confidence=0.8),
        ]
        track_map = p._build_track_map(dets)
        assert track_map[1]["in_zone"] is True
        assert track_map[2]["in_zone"] is False

    def test_no_track_id_skipped(self, test_zone: list[int]) -> None:
        p = FrameProcessor(mode="TrackZone", region=test_zone, settings=Settings())
        dets = [DetectedObject(track_id=None, bbox=(150, 150, 250, 250), class_id=0, confidence=0.9)]
        track_map = p._build_track_map(dets)
        assert len(track_map) == 0


class TestFrameStats:
    def test_to_dict_keys(self) -> None:
        s = FrameStats(total_objects=5, zone_entries=2, objects_in_zone=1, fps=30.0)
        d = s.to_dict()
        assert set(d.keys()) == {"total_objects", "zone_entries", "objects_in_zone", "fps", "alerts"}

    def test_empty_alerts(self) -> None:
        s = FrameStats()
        assert s.to_dict()["alerts"] == []


class TestUnsupportedMode:
    def test_raises_on_invalid_mode(self) -> None:
        with pytest.raises(ValueError, match="Unsupported mode"):
            FrameProcessor(mode="InvalidMode", settings=Settings())
