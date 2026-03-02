"""Live Analytics — real-time video processing page.

Supports video upload and live camera input with zone-based tracking,
real-time stats, and alert evaluation.
"""

from __future__ import annotations

import atexit
import logging
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app.analytics.frame_processor import FrameProcessor
from app.config import load_settings

logger = logging.getLogger(__name__)

# ── Page config ──
st.set_page_config(page_title="Live Analytics", page_icon="📹", layout="wide")

settings = load_settings()


# ──────────────────────────────────────────────────────────────────────
# Session state defaults
# ──────────────────────────────────────────────────────────────────────

_DEFAULTS: dict = {
    "processor": None,
    "is_processing": False,
    "tracking_zone": None,
    "stats_history": [],
    "alert_history": [],
    "frame_count": 0,
    "temp_files": [],
}

for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


def _cleanup_temp_files() -> None:
    for path in st.session_state.get("temp_files", []):
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            pass


atexit.register(_cleanup_temp_files)


# ──────────────────────────────────────────────────────────────────────
# Sidebar controls
# ──────────────────────────────────────────────────────────────────────

st.sidebar.header("⚙️ Controls")
input_source = st.sidebar.radio("Input Source", ("Upload Video", "Live Camera"))
tracking_mode = st.sidebar.radio(
    "Processing Mode",
    ("TrackZone", "Instance Segmentation"),
    help="TrackZone: zone-based counting.  Instance Seg: pixel-level masks.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Model Parameters")
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, settings.yolo.confidence, 0.05)
display_skip = st.sidebar.slider(
    "Display Skip Frames",
    1, 10,
    settings.ui.display_skip_frames,
    help="Update the display every N frames (higher = faster processing)",
)


# ──────────────────────────────────────────────────────────────────────
# Zone definition helper
# ──────────────────────────────────────────────────────────────────────

def _define_zone(frame_rgb: np.ndarray, w: int, h: int) -> list[int]:
    """Render zone-definition sliders and return ``[x1, y1, x2, y2]``."""
    dz = settings.ui.default_zone
    # Clamp defaults to the actual frame size
    default_x = (min(dz[0], w - 2), min(dz[2], w))
    default_y = (min(dz[1], h - 2), min(dz[3], h))

    col_l, col_r = st.columns(2)
    with col_l:
        x_min, x_max = st.slider(
            "Horizontal (Left–Right)", 0, w, default_x, key="zone_x"
        )
    with col_r:
        y_min, y_max = st.slider(
            "Vertical (Top–Bottom)", 0, h, default_y, key="zone_y"
        )

    preview = frame_rgb.copy()
    cv2.rectangle(preview, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    st.image(preview, caption="Preview — green rectangle is the tracking zone", use_container_width=True)

    return [x_min, y_min, x_max, y_max]


# ──────────────────────────────────────────────────────────────────────
# Stats chart
# ──────────────────────────────────────────────────────────────────────

def _build_chart(history: list[dict]) -> go.Figure:
    """Build a Plotly line chart from stats history."""
    frames = list(range(len(history)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frames, y=[h["objects_in_zone"] for h in history],
        mode="lines", name="In Zone", line=dict(color="#00d4ff"),
    ))
    fig.add_trace(go.Scatter(
        x=frames, y=[h["zone_entries"] for h in history],
        mode="lines", name="Zone Entries (cum.)", line=dict(color="#ff6b6b"),
    ))
    fig.update_layout(
        title="Real-Time Analytics",
        xaxis_title="Frame",
        yaxis_title="Count",
        height=280,
        margin=dict(l=40, r=20, t=40, b=30),
        legend=dict(orientation="h", y=1.12),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────
# Main layout
# ──────────────────────────────────────────────────────────────────────

st.title("📹 Live Analytics")

# ── Metrics row ──
m1, m2, m3, m4 = st.columns(4)
metric_total = m1.empty()
metric_entries = m2.empty()
metric_in_zone = m3.empty()
metric_fps = m4.empty()

metric_total.metric("Total Objects", 0)
metric_entries.metric("Zone Entries", 0)
metric_in_zone.metric("In Zone", 0)
metric_fps.metric("FPS", "—")

# ── Video + Chart columns ──
video_col, chart_col = st.columns([3, 2])
frame_placeholder = video_col.empty()
chart_placeholder = chart_col.empty()

# ── Alert feed ──
alert_placeholder = st.empty()

# ── Status ──
status = st.empty()


# ──────────────────────────────────────────────────────────────────────
# Video source acquisition
# ──────────────────────────────────────────────────────────────────────

cap: cv2.VideoCapture | None = None
temp_path: str | None = None

if input_source == "Upload Video":
    uploaded = st.sidebar.file_uploader(
        "Upload video",
        type=["mp4", "avi", "mov", "mkv"],
    )
    if uploaded:
        size_mb = uploaded.size / (1024 * 1024)
        if size_mb > settings.ui.max_upload_mb:
            st.error(f"File too large ({size_mb:.0f} MB). Max is {settings.ui.max_upload_mb} MB.")
            st.stop()

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(uploaded.read())
        tmp.close()
        temp_path = tmp.name
        st.session_state["temp_files"].append(temp_path)

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            st.error("Could not open uploaded video.  Try a different format.")
            st.stop()

        # Read a preview frame for zone definition
        ret, preview_frame = cap.read()
        cap.release()
        if not ret:
            st.error("Could not read frames from the uploaded video.")
            st.stop()

        h, w = preview_frame.shape[:2]
        preview_rgb = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)

        st.subheader("Define Tracking Zone")
        st.session_state["tracking_zone"] = _define_zone(preview_rgb, w, h)

else:  # Live Camera
    camera_idx = st.sidebar.number_input("Camera Index", 0, 10, 0)
    test_cap = cv2.VideoCapture(int(camera_idx))
    if not test_cap.isOpened():
        st.sidebar.error(f"Camera {camera_idx} not available.")
        st.stop()

    ret, preview_frame = test_cap.read()
    test_cap.release()
    if not ret:
        st.sidebar.error("Could not capture preview frame from camera.")
        st.stop()

    h, w = preview_frame.shape[:2]
    preview_rgb = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)

    st.subheader("Define Tracking Zone")
    st.session_state["tracking_zone"] = _define_zone(preview_rgb, w, h)


# ──────────────────────────────────────────────────────────────────────
# Processing loop
# ──────────────────────────────────────────────────────────────────────

zone = st.session_state.get("tracking_zone")
can_start = zone is not None and (temp_path is not None or input_source == "Live Camera")

col_start, col_stop = st.columns(2)
start_btn = col_start.button("▶ Start Processing", disabled=not can_start, use_container_width=True)
stop_btn = col_stop.button("⏹ Stop", use_container_width=True)

if stop_btn:
    st.session_state["is_processing"] = False

if start_btn and can_start:
    st.session_state["is_processing"] = True
    st.session_state["stats_history"] = []
    st.session_state["alert_history"] = []
    st.session_state["frame_count"] = 0

    # Build a settings override with the slider confidence
    from app.config import YOLOConfig
    custom_settings = settings
    if confidence != settings.yolo.confidence:
        custom_yolo = YOLOConfig(
            model_path=settings.yolo.model_path,
            seg_model_path=settings.yolo.seg_model_path,
            confidence=confidence,
            iou=settings.yolo.iou,
            classes=list(settings.yolo.classes),
        )
        from dataclasses import replace
        custom_settings = replace(settings, yolo=custom_yolo)

    processor = FrameProcessor(
        mode=tracking_mode,
        region=zone,
        settings=custom_settings,
    )

    if processor.processor is None:
        st.error(
            "Failed to initialise the model. Check that model files "
            f"(`{custom_settings.yolo.model_path}`) exist at the project root."
        )
        st.stop()

    # Open video source
    if input_source == "Upload Video":
        cap = cv2.VideoCapture(temp_path)
    else:
        cap = cv2.VideoCapture(int(camera_idx))

    if not cap or not cap.isOpened():
        st.error("Failed to open video source.")
        st.stop()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    progress_bar = st.progress(0) if total_frames > 0 else None

    # Output video writer
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"processed_{int(time.time())}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(
        str(out_path),
        fourcc,
        source_fps,
        (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
    )

    status.info("Processing started…")
    frame_idx = 0

    while cap.isOpened() and st.session_state.get("is_processing", False):
        ret, frame = cap.read()
        if not ret:
            break

        frame_out, stats = processor.process(frame)
        frame_idx += 1
        out_writer.write(frame_out)

        # Collect history
        st.session_state["stats_history"].append(stats)
        if stats.get("alerts"):
            st.session_state["alert_history"].extend(stats["alerts"])

        # Update display every N frames to keep UI responsive
        if frame_idx % display_skip == 0:
            frame_rgb = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, use_container_width=True)

            metric_total.metric("Total Objects", stats["total_objects"])
            metric_entries.metric("Zone Entries", stats["zone_entries"])
            metric_in_zone.metric("In Zone", stats["objects_in_zone"])
            metric_fps.metric("FPS", f"{stats['fps']}")

            # Chart
            history = st.session_state["stats_history"]
            if len(history) > 1:
                chart_placeholder.plotly_chart(
                    _build_chart(history), use_container_width=True
                )

            # Recent alerts
            recent = st.session_state["alert_history"][-5:]
            if recent:
                alert_placeholder.dataframe(
                    [
                        {
                            "Time": a["timestamp"],
                            "Type": a["type"],
                            "Severity": a["severity"],
                            "Track": a.get("track_id", "—"),
                            "Message": a.get("message", ""),
                        }
                        for a in reversed(recent)
                    ],
                    use_container_width=True,
                )

            if progress_bar and total_frames > 0:
                progress_bar.progress(min(frame_idx / total_frames, 1.0))

    # ── Teardown ──
    cap.release()
    out_writer.release()
    st.session_state["is_processing"] = False

    status.success(f"Processing complete — {frame_idx} frames.")

    # Final chart
    history = st.session_state["stats_history"]
    if history:
        chart_placeholder.plotly_chart(_build_chart(history), use_container_width=True)

    # Download button
    if out_path.exists() and out_path.stat().st_size > 0:
        with open(out_path, "rb") as f:
            st.download_button(
                "⬇ Download Processed Video",
                f.read(),
                file_name=out_path.name,
                mime="video/mp4",
            )

    # Summary
    if history:
        last = history[-1]
        st.subheader("📊 Session Summary")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total Objects", last["total_objects"])
        s2.metric("Zone Entries", last["zone_entries"])
        s3.metric("Peak In Zone", max(h["objects_in_zone"] for h in history))
        s4.metric("Frames Processed", frame_idx)
