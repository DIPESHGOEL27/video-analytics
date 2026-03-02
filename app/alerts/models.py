"""Data models for the alert system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class AlertType(str, Enum):
    """Supported alert categories."""

    ZONE_INTRUSION = "Zone Intrusion"
    LOITERING = "Loitering"
    CROWD = "Crowd Detected"


class Severity(str, Enum):
    """Alert severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Immutable record of a single alert event."""

    alert_type: AlertType
    severity: Severity
    track_id: int | None
    bbox: tuple[int, int, int, int] | None
    timestamp: datetime = field(default_factory=datetime.now)
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "type": self.alert_type.value,
            "severity": self.severity.value,
            "track_id": self.track_id,
            "bbox": self.bbox,
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
        }
