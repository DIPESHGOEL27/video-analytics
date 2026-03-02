"""Configuration — tune model parameters and alert rules.

Changes here are applied on the next processing session.
"""

from __future__ import annotations

import streamlit as st
import yaml
from pathlib import Path

from app.config import load_settings

st.set_page_config(page_title="Configuration", page_icon="⚙️", layout="wide")

CONFIG_PATH = Path("config.yaml")
settings = load_settings()


def main() -> None:
    st.title("⚙️ Configuration")
    st.markdown("Edit settings below and click **Save** to persist to `config.yaml`.")

    col_model, col_alert = st.columns(2)

    # ── Model settings ──
    with col_model:
        st.subheader("🔍 YOLO Model")
        model_path = st.text_input("Detection Model", settings.yolo.model_path)
        seg_model_path = st.text_input("Segmentation Model", settings.yolo.seg_model_path)
        confidence = st.slider("Confidence Threshold", 0.1, 1.0, settings.yolo.confidence, 0.05)
        iou = st.slider("IoU Threshold", 0.1, 1.0, settings.yolo.iou, 0.05)

        st.subheader("🏃 Tracking")
        tracker = st.selectbox(
            "Tracker",
            ("botsort.yaml", "bytetrack.yaml"),
            index=0 if settings.tracking.tracker == "botsort.yaml" else 1,
        )

    # ── Alert settings ──
    with col_alert:
        st.subheader("🚨 Alert Rules")
        alert_enabled = st.toggle("Enable Alerts", settings.alert.enabled)
        cooldown = st.number_input(
            "Cooldown (seconds)", 1, 120, settings.alert.cooldown_seconds
        )
        loiter_threshold = st.number_input(
            "Loitering Threshold (seconds)", 1.0, 60.0, settings.alert.loiter_threshold_seconds, 0.5
        )
        crowd_threshold = st.number_input(
            "Crowd Threshold (objects)", 2, 50, settings.alert.crowd_threshold
        )

        st.subheader("🖥️ UI")
        display_skip = st.slider("Display Skip Frames", 1, 10, settings.ui.display_skip_frames)
        max_upload = st.number_input("Max Upload Size (MB)", 50, 2000, settings.ui.max_upload_mb)

    st.markdown("---")

    # ── Default zone ──
    st.subheader("📐 Default Tracking Zone")
    zc1, zc2, zc3, zc4 = st.columns(4)
    dz = settings.ui.default_zone
    z_x1 = zc1.number_input("x1", 0, 3840, dz[0])
    z_y1 = zc2.number_input("y1", 0, 2160, dz[1])
    z_x2 = zc3.number_input("x2", 0, 3840, dz[2])
    z_y2 = zc4.number_input("y2", 0, 2160, dz[3])

    st.markdown("---")

    if st.button("💾 Save Configuration", type="primary", use_container_width=True):
        new_config = {
            "video_input": settings.video_input,
            "video_output": settings.video_output,
            "yolo": {
                "model_path": model_path,
                "seg_model_path": seg_model_path,
                "confidence": confidence,
                "iou": iou,
                "classes": list(settings.yolo.classes),
            },
            "tracking": {
                "tracker": tracker,
                "max_age": settings.tracking.max_age,
                "min_hits": settings.tracking.min_hits,
            },
            "alert": {
                "enabled": alert_enabled,
                "cooldown_seconds": cooldown,
                "loiter_threshold_seconds": loiter_threshold,
                "crowd_threshold": crowd_threshold,
                "default_zone": list(settings.alert.default_zone),
            },
            "ui": {
                "page_title": settings.ui.page_title,
                "default_zone": [z_x1, z_y1, z_x2, z_y2],
                "display_skip_frames": display_skip,
                "max_upload_mb": max_upload,
                "output_format": settings.ui.output_format,
            },
        }

        with open(CONFIG_PATH, "w") as f:
            yaml.dump(new_config, f, default_flow_style=False, sort_keys=False)

        st.success("Configuration saved!  Changes will apply on the next processing session.")
        st.balloons()

    # ── Current config display ──
    with st.expander("📋 Current config.yaml"):
        if CONFIG_PATH.exists():
            st.code(CONFIG_PATH.read_text(), language="yaml")
        else:
            st.info("No config.yaml on disk — using defaults.")


main()
