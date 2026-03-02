"""Alert Log — browse triggered alerts with timestamps and severity."""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Alert Log", page_icon="🚨", layout="wide")


def main() -> None:
    st.title("🚨 Alert Log")

    alerts: list[dict] = st.session_state.get("alert_history", [])

    if not alerts:
        st.info(
            "No alerts recorded yet.  Run a processing session from "
            "**Live Analytics** to generate alerts."
        )
        return

    st.metric("Total Alerts", len(alerts))
    st.markdown("---")

    # ── Filters ──
    col_type, col_sev = st.columns(2)
    all_types = sorted({a["type"] for a in alerts})
    all_sevs = sorted({a["severity"] for a in alerts})

    with col_type:
        selected_types = st.multiselect("Filter by Type", all_types, default=all_types)
    with col_sev:
        selected_sevs = st.multiselect("Filter by Severity", all_sevs, default=all_sevs)

    filtered = [
        a for a in alerts
        if a["type"] in selected_types and a["severity"] in selected_sevs
    ]

    st.markdown(f"**Showing {len(filtered)} / {len(alerts)} alerts**")

    # ── Table ──
    rows = [
        {
            "Timestamp": a.get("timestamp", "—"),
            "Type": a["type"],
            "Severity": a["severity"].upper(),
            "Track ID": a.get("track_id", "—"),
            "Message": a.get("message", ""),
        }
        for a in reversed(filtered)  # newest first
    ]

    st.dataframe(rows, use_container_width=True, height=500)

    # ── Clear ──
    if st.button("🗑 Clear Alert History"):
        st.session_state["alert_history"] = []
        st.rerun()


main()
