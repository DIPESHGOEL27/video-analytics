"""Central frame-processing orchestrator.

Wraps Ultralytics Solutions (TrackZone / InstanceSegmentation) with zone-based
analytics, stats tracking, and alert evaluation.  Returns annotated frames +
structured stats to the UI layer — never calls ``cv2.imshow()``.
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import cv2
import numpy as np
from ultralytics import solutions

from app.alerts.alert_service import AlertService
from app.alerts.models import Alert
from app.config import Settings, load_settings

# Suppress OpenCV GUI warnings (headless environment)
warnings.filterwarnings("ignore", message=".*cv2.imshow.*")
warnings.filterwarnings("ignore", message=".*The function is not implemented.*")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper types
# ---------------------------------------------------------------------------

class DetectedObject(NamedTuple):
    """Lightweight record extracted from Ultralytics results."""

    track_id: int | None
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    class_id: int
    confidence: float


@dataclass
class FrameStats:
    """Per-frame analytics payload returned alongside the annotated frame."""

    total_objects: int = 0
    zone_entries: int = 0
    objects_in_zone: int = 0
    fps: float = 0.0
    alerts: list[Alert] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_objects": self.total_objects,
            "zone_entries": self.zone_entries,
            "objects_in_zone": self.objects_in_zone,
            "fps": self.fps,
            "alerts": [a.to_dict() for a in self.alerts],
        }


# ---------------------------------------------------------------------------
# Main processor
# ---------------------------------------------------------------------------

class FrameProcessor:
    """Orchestrates detection, tracking, zone analytics, and alerting.

    Parameters
    ----------
    mode : str
        ``"TrackZone"`` or ``"Instance Segmentation"``.
    region : list[int] | None
        Tracking zone as ``[x1, y1, x2, y2]``. Falls back to config default.
    settings : Settings | None
        Application settings.  Loaded from ``config.yaml`` when *None*.
    """

    SUPPORTED_MODES = ("TrackZone", "Instance Segmentation")

    def __init__(
        self,
        mode: str = "TrackZone",
        region: list[int] | None = None,
        settings: Settings | None = None,
    ) -> None:
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"Unsupported mode {mode!r}. Choose from {self.SUPPORTED_MODES}")

        self.settings = settings or load_settings()
        self.mode = mode
        self.region: list[int] = region or list(self.settings.ui.default_zone)
        self.processor: Any | None = None

        # Cumulative stats
        self._total_objects: int = 0
        self._zone_entries: int = 0
        self._previous_tracks: dict[int, dict[str, Any]] = {}

        # FPS tracking
        self._frame_times: list[float] = []

        # Alert service
        self._alert_service = AlertService(self.settings.alert)

        # Initialise Ultralytics solution
        self._init_processor()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_processor(self) -> None:
        """Create the Ultralytics Solutions processor for the current mode."""
        cfg = self.settings.yolo
        trk = self.settings.tracking

        try:
            if self.mode == "TrackZone":
                self.processor = solutions.TrackZone(
                    show=False,
                    region=self._rect_to_polygon(self.region),
                    model=cfg.model_path,
                    conf=cfg.confidence,
                    iou=cfg.iou,
                    tracker=trk.tracker,
                )
            elif self.mode == "Instance Segmentation":
                self.processor = solutions.InstanceSegmentation(
                    show=False,
                    model=cfg.seg_model_path,
                    conf=cfg.confidence,
                    tracker=trk.tracker,
                    verbose=False,
                )
        except Exception:
            logger.exception("Failed to initialise %s processor", self.mode)
            self.processor = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Run detection + tracking on *frame* and return ``(annotated_frame, stats_dict)``.

        The returned *stats_dict* always contains keys: ``total_objects``,
        ``zone_entries``, ``objects_in_zone``, ``fps``, ``alerts``.
        """
        t0 = time.perf_counter()

        if self.processor is None:
            logger.error("Processor not initialised — returning original frame")
            return self._error_frame(frame, "Processor not initialised"), FrameStats().to_dict()

        # --- inference (never pass show=) ---
        try:
            results = self.processor(frame)
        except Exception:
            logger.exception("Inference error")
            return self._error_frame(frame, "Inference error"), FrameStats().to_dict()

        # --- visualisation ---
        frame_out = self._visualise(frame, results)

        # --- extract detections ---
        detections = self._extract_detections(results)

        # --- zone analytics ---
        current_tracks = self._build_track_map(detections)
        objects_in_zone = sum(1 for t in current_tracks.values() if t["in_zone"])

        # Count new unique objects
        new_ids = set(current_tracks) - set(self._previous_tracks)
        self._total_objects += len(new_ids)

        # Count zone entries (was outside → now inside)
        for tid, info in current_tracks.items():
            if info["in_zone"] and not self._previous_tracks.get(tid, {}).get("in_zone", False):
                self._zone_entries += 1

        # --- alerts ---
        alerts = self._alert_service.evaluate(current_tracks, self._previous_tracks)

        # --- update state ---
        self._previous_tracks = current_tracks

        # --- FPS ---
        elapsed = time.perf_counter() - t0
        self._frame_times.append(elapsed)
        if len(self._frame_times) > 30:
            self._frame_times = self._frame_times[-30:]
        avg_time = sum(self._frame_times) / len(self._frame_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0.0

        # --- draw overlay ---
        frame_out = self._draw_overlay(frame_out, objects_in_zone, fps)

        stats = FrameStats(
            total_objects=self._total_objects,
            zone_entries=self._zone_entries,
            objects_in_zone=objects_in_zone,
            fps=round(fps, 1),
            alerts=alerts,
        )
        return frame_out, stats.to_dict()

    def update_region(self, new_region: list[int]) -> None:
        """Update the tracking zone and reinitialise the processor."""
        if not new_region or len(new_region) < 4:
            return
        self.region = list(new_region)
        self._init_processor()
        self._zone_entries = 0
        self._previous_tracks.clear()
        self._alert_service.reset()
        logger.info("Tracking region updated to %s", self.region)

    def get_stats(self) -> dict[str, Any]:
        """Return current cumulative stats snapshot."""
        return FrameStats(
            total_objects=self._total_objects,
            zone_entries=self._zone_entries,
            objects_in_zone=sum(
                1 for t in self._previous_tracks.values() if t["in_zone"]
            ),
        ).to_dict()

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rect_to_polygon(rect: list[int]) -> list[tuple[int, int]]:
        """``[x1, y1, x2, y2]`` → ``[(x1,y1), (x2,y1), (x2,y2), (x1,y2)]``."""
        x1, y1, x2, y2 = rect
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    def _is_point_in_zone(self, point: tuple[int, int]) -> bool:
        """Check whether *point* falls inside ``self.region`` rectangle."""
        x, y = point
        x1, y1, x2, y2 = self.region
        return x1 <= x <= x2 and y1 <= y <= y2

    # ------------------------------------------------------------------
    # Detection extraction (single source of truth)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_detections(results: Any) -> list[DetectedObject]:
        """Pull tracked objects from an Ultralytics results object.

        Handles the various attribute layouts across Solutions (``results.boxes``,
        ``results.obb``, nested masks, or a plain ndarray frame).
        """
        boxes = None
        if hasattr(results, "boxes") and results.boxes is not None:
            boxes = results.boxes
        elif hasattr(results, "obb") and results.obb is not None:
            boxes = results.obb

        if boxes is None or len(boxes) == 0:
            return []

        detections: list[DetectedObject] = []
        for box in boxes:
            # --- bbox ---
            try:
                x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
            except (AttributeError, IndexError):
                try:
                    cxywh = box.xywh.cpu().numpy()[0]
                    x1 = int(cxywh[0] - cxywh[2] / 2)
                    y1 = int(cxywh[1] - cxywh[3] / 2)
                    x2 = int(cxywh[0] + cxywh[2] / 2)
                    y2 = int(cxywh[1] + cxywh[3] / 2)
                except (AttributeError, IndexError):
                    continue

            # --- metadata ---
            try:
                class_id = int(box.cls.item())
            except (AttributeError, ValueError):
                class_id = -1

            try:
                conf = float(box.conf.item())
            except (AttributeError, ValueError):
                conf = 0.0

            track_id: int | None = None
            if hasattr(box, "id") and box.id is not None:
                try:
                    track_id = int(box.id.item())
                except (ValueError, AttributeError):
                    pass

            detections.append(DetectedObject(track_id, (x1, y1, x2, y2), class_id, conf))

        return detections

    # ------------------------------------------------------------------
    # Track map
    # ------------------------------------------------------------------

    def _build_track_map(
        self, detections: list[DetectedObject]
    ) -> dict[int, dict[str, Any]]:
        """Convert detection list to the track map consumed by analytics + alerts."""
        tracks: dict[int, dict[str, Any]] = {}
        for det in detections:
            if det.track_id is None:
                continue
            cx = (det.bbox[0] + det.bbox[2]) // 2
            cy = (det.bbox[1] + det.bbox[3]) // 2
            tracks[det.track_id] = {
                "centroid": (cx, cy),
                "in_zone": self._is_point_in_zone((cx, cy)),
                "bbox": det.bbox,
            }
        return tracks

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def _visualise(self, frame: np.ndarray, results: Any) -> np.ndarray:
        """Render detection results onto *frame*."""
        try:
            if isinstance(results, np.ndarray):
                return results.copy()
            if hasattr(results, "plot"):
                return results.plot()
        except Exception:
            logger.debug("Built-in plot() failed, falling back to manual draw")

        frame_out = frame.copy()
        for det in self._extract_detections(results):
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(frame_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"cls:{det.class_id} {det.confidence:.2f}"
            if det.track_id is not None:
                label += f" ID:{det.track_id}"
            cv2.putText(
                frame_out, label, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
            )
        return frame_out

    def _draw_overlay(
        self, frame: np.ndarray, objects_in_zone: int, fps: float
    ) -> np.ndarray:
        """Draw zone rectangle and live stats on *frame*."""
        x1, y1, x2, y2 = self.region

        # Zone rectangle with semi-transparent fill
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
        frame_out = cv2.addWeighted(overlay, 0.08, frame, 0.92, 0)
        cv2.rectangle(frame_out, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Zone label
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.putText(
            frame_out, "Tracking Zone", (cx - 60, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )

        # Stats panel (dark background)
        panel_h = 120
        panel = frame_out.copy()
        cv2.rectangle(panel, (10, 10), (320, 10 + panel_h), (0, 0, 0), -1)
        frame_out = cv2.addWeighted(panel, 0.65, frame_out, 0.35, 0)

        y_off = 35
        stats_lines = [
            (f"Total Objects: {self._total_objects}", (255, 255, 255)),
            (f"Zone Entries: {self._zone_entries}", (255, 255, 255)),
            (f"In Zone: {objects_in_zone}", (0, 200, 255)),
            (f"FPS: {fps:.1f}", (0, 255, 100)),
        ]
        for text, colour in stats_lines:
            cv2.putText(
                frame_out, text, (20, y_off),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2,
            )
            y_off += 25

        return frame_out

    @staticmethod
    def _error_frame(frame: np.ndarray, message: str) -> np.ndarray:
        """Return *frame* with a red error banner."""
        out = frame.copy()
        cv2.rectangle(out, (0, 0), (out.shape[1], 40), (0, 0, 180), -1)
        cv2.putText(
            out, message, (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )
        return out
