"""Tests for AlertService — zone intrusion, loitering, crowd, cooldown."""

from __future__ import annotations

import time

import pytest

from app.alerts.alert_service import AlertService
from app.alerts.models import AlertType
from app.config import AlertConfig


@pytest.fixture()
def service(alert_config: AlertConfig) -> AlertService:
    return AlertService(alert_config)


class TestZoneIntrusion:
    def test_fires_on_entry(self, service: AlertService) -> None:
        prev: dict = {}
        curr = {1: {"centroid": (200, 200), "in_zone": True, "bbox": (150, 150, 250, 250)}}
        alerts = service.evaluate(curr, prev)
        assert len(alerts) == 1
        assert alerts[0].alert_type == AlertType.ZONE_INTRUSION

    def test_no_alert_if_already_inside(self, service: AlertService) -> None:
        state = {1: {"centroid": (200, 200), "in_zone": True, "bbox": (150, 150, 250, 250)}}
        service.evaluate(state, {})  # first entry → fires
        # Simulate cooldown expiry by pushing time
        time.sleep(0.01)
        alerts = service.evaluate(state, state)  # same state → no new entry
        zone_alerts = [a for a in alerts if a.alert_type == AlertType.ZONE_INTRUSION]
        assert len(zone_alerts) == 0

    def test_no_alert_when_outside(self, service: AlertService) -> None:
        curr = {1: {"centroid": (500, 500), "in_zone": False, "bbox": (450, 450, 550, 550)}}
        alerts = service.evaluate(curr, {})
        assert len(alerts) == 0


class TestCooldown:
    def test_respects_cooldown(self, service: AlertService) -> None:
        prev: dict = {}
        curr = {1: {"centroid": (200, 200), "in_zone": True, "bbox": (150, 150, 250, 250)}}
        alerts1 = service.evaluate(curr, prev)
        assert len(alerts1) >= 1

        # Immediately re-enter (simulate track leaving and entering within cooldown)
        prev_outside = {1: {"centroid": (200, 200), "in_zone": False, "bbox": (150, 150, 250, 250)}}
        alerts2 = service.evaluate(curr, prev_outside)
        intrusion_alerts = [a for a in alerts2 if a.alert_type == AlertType.ZONE_INTRUSION]
        assert len(intrusion_alerts) == 0  # still in cooldown


class TestCrowd:
    def test_fires_when_threshold_met(self, service: AlertService) -> None:
        curr = {
            i: {"centroid": (200, 200), "in_zone": True, "bbox": (150, 150, 250, 250)}
            for i in range(5)  # crowd_threshold = 3 in fixture
        }
        alerts = service.evaluate(curr, {})
        crowd_alerts = [a for a in alerts if a.alert_type == AlertType.CROWD]
        assert len(crowd_alerts) == 1

    def test_no_crowd_below_threshold(self, service: AlertService) -> None:
        curr = {
            1: {"centroid": (200, 200), "in_zone": True, "bbox": (150, 150, 250, 250)},
            2: {"centroid": (250, 250), "in_zone": True, "bbox": (200, 200, 300, 300)},
        }
        alerts = service.evaluate(curr, {})
        crowd_alerts = [a for a in alerts if a.alert_type == AlertType.CROWD]
        assert len(crowd_alerts) == 0


class TestDisabled:
    def test_no_alerts_when_disabled(self) -> None:
        config = AlertConfig(enabled=False)
        svc = AlertService(config)
        curr = {1: {"centroid": (200, 200), "in_zone": True, "bbox": (150, 150, 250, 250)}}
        alerts = svc.evaluate(curr, {})
        assert alerts == []


class TestReset:
    def test_clears_state(self, service: AlertService) -> None:
        curr = {1: {"centroid": (200, 200), "in_zone": True, "bbox": (150, 150, 250, 250)}}
        service.evaluate(curr, {})
        service.reset()
        # After reset, the same entry should fire again
        alerts = service.evaluate(curr, {})
        assert len(alerts) >= 1
