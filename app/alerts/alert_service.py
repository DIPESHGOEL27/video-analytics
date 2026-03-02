"""Unified alert service — replaces the legacy AlertEngine + AlertManager.

Operates on track data produced by :class:`FrameProcessor` (track_id, centroid,
bbox, in_zone flag).  Returns :class:`Alert` instances without drawing on frames
so the UI layer decides visualisation.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from app.alerts.models import Alert, AlertType, Severity
from app.config import AlertConfig

logger = logging.getLogger(__name__)


class AlertService:
    """Configurable, stateful alert evaluator.

    Supported rules
    ---------------
    * **Zone intrusion** — fires when a tracked object first enters the zone.
    * **Loitering** — fires when a tracked object remains in the zone longer
      than ``config.loiter_threshold_seconds``.
    * **Crowd** — fires when the number of simultaneous objects in the zone
      exceeds ``config.crowd_threshold``.
    """

    def __init__(self, config: AlertConfig) -> None:
        self.config = config
        self._cooldowns: dict[str, float] = {}
        self._zone_entry_times: dict[int, float] = {}  # track_id → first-seen-in-zone epoch

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        current_tracks: dict[int, dict[str, Any]],
        previous_tracks: dict[int, dict[str, Any]],
    ) -> list[Alert]:
        """Run all alert rules against the current frame's track data.

        Parameters
        ----------
        current_tracks:
            ``{track_id: {"centroid": (x,y), "in_zone": bool, "bbox": (x1,y1,x2,y2)}}``
        previous_tracks:
            Same structure, from the previous frame.

        Returns
        -------
        list[Alert]
            Zero or more alerts triggered this frame.
        """
        if not self.config.enabled:
            return []

        alerts: list[Alert] = []
        now = time.time()

        in_zone_count = 0

        for track_id, info in current_tracks.items():
            if not info["in_zone"]:
                # Left zone → clear loiter timer
                self._zone_entry_times.pop(track_id, None)
                continue

            in_zone_count += 1

            # ── Zone intrusion ──
            was_in_zone = previous_tracks.get(track_id, {}).get("in_zone", False)
            if not was_in_zone:
                alert = self._try_fire(
                    AlertType.ZONE_INTRUSION,
                    track_id,
                    info["bbox"],
                    Severity.MEDIUM,
                    f"Track {track_id} entered the zone",
                    now,
                )
                if alert:
                    alerts.append(alert)
                # Start loiter clock
                self._zone_entry_times[track_id] = now

            # ── Loitering ──
            entry_time = self._zone_entry_times.get(track_id)
            if entry_time and (now - entry_time) >= self.config.loiter_threshold_seconds:
                alert = self._try_fire(
                    AlertType.LOITERING,
                    track_id,
                    info["bbox"],
                    Severity.HIGH,
                    f"Track {track_id} loitering for {now - entry_time:.1f}s",
                    now,
                )
                if alert:
                    alerts.append(alert)

        # ── Crowd ──
        if in_zone_count >= self.config.crowd_threshold:
            alert = self._try_fire(
                AlertType.CROWD,
                None,
                None,
                Severity.CRITICAL,
                f"{in_zone_count} objects in zone (threshold: {self.config.crowd_threshold})",
                now,
            )
            if alert:
                alerts.append(alert)

        # Prune stale loiter entries for tracks that disappeared
        active_ids = set(current_tracks.keys())
        stale = [tid for tid in self._zone_entry_times if tid not in active_ids]
        for tid in stale:
            del self._zone_entry_times[tid]

        return alerts

    def reset(self) -> None:
        """Clear all internal state (e.g. when zone changes)."""
        self._cooldowns.clear()
        self._zone_entry_times.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _try_fire(
        self,
        alert_type: AlertType,
        track_id: int | None,
        bbox: tuple[int, int, int, int] | None,
        severity: Severity,
        message: str,
        now: float,
    ) -> Alert | None:
        """Fire *alert_type* if cooldown has expired, otherwise return None."""
        key = f"{alert_type.value}_{track_id}"
        last = self._cooldowns.get(key, 0.0)
        if now - last < self.config.cooldown_seconds:
            return None

        self._cooldowns[key] = now
        alert = Alert(
            alert_type=alert_type,
            severity=severity,
            track_id=track_id,
            bbox=bbox,
            message=message,
        )
        logger.warning("ALERT: %s", message)
        return alert
