"""AI-Powered Video Analytics — Streamlit entrypoint.

This is the landing page of the multi-page dashboard.
Processing pages live under ``pages/``.
"""

from __future__ import annotations

import streamlit as st

from app.config import load_settings

# ── Page config (must be the FIRST Streamlit call) ──
st.set_page_config(
    page_title="AI Video Analytics",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

settings = load_settings()


def main() -> None:
    st.title("🎯 AI-Powered Video Analytics")
    st.markdown("---")

    st.markdown(
        """
        ### Welcome

        This system provides **real-time object detection, multi-object tracking,
        and zone-based analytics** powered by **YOLO11** and **BotSORT**.

        Use the sidebar to navigate between pages:

        | Page | Description |
        |------|-------------|
        | **Live Analytics** | Upload a video or use a live camera for real-time processing |
        | **Configuration** | Tune model parameters, define tracking zones, and alert rules |
        | **Alert Log** | Browse the history of triggered alerts with timestamps |

        ---
        ### Architecture

        ```
        Video Source  →  YOLO11 Detection  →  BotSORT Tracking  →  Zone Analytics  →  AlertService
                                                                          ↓
                                                                   Streamlit Dashboard
        ```

        ### Tech Stack

        - **Detection**: Ultralytics YOLO11 (object detection + instance segmentation)
        - **Tracking**: BotSORT (built-in Ultralytics tracker)
        - **Alerts**: Configurable rules — zone intrusion, loitering, crowd detection
        - **UI**: Streamlit multi-page dashboard with real-time Plotly charts
        """
    )

    st.sidebar.success("Select a page above to get started.")


if __name__ == "__main__":
    main()
