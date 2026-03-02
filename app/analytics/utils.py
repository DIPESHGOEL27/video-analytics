"""Zone geometry utilities.

All zone coordinates follow the ``[x1, y1, x2, y2]`` rectangle convention
used throughout the application.
"""

from __future__ import annotations

import cv2
import numpy as np


def draw_zones(frame: np.ndarray, zones: list[list[int]]) -> np.ndarray:
    """Draw rectangular zones on *frame* (mutates in place for performance)."""
    for x1, y1, x2, y2 in zones:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return frame


def is_in_zone(center: tuple[int, int], zone: list[int]) -> bool:
    """Return ``True`` if *center* ``(x, y)`` is inside ``[x1, y1, x2, y2]``."""
    x, y = center
    x1, y1, x2, y2 = zone
    return x1 <= x <= x2 and y1 <= y <= y2
