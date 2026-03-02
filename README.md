# 🎯 AI-Powered Video Analytics System

A production-grade, real-time **object detection**, **multi-object tracking**, and **zone-based analytics** system built with **YOLO11** and **BotSORT** — served through an interactive multi-page **Streamlit** dashboard.

---

## Features

- **Real-time object detection** — YOLO11 (detection + instance segmentation modes)
- **Multi-object tracking** — BotSORT with persistent track IDs across frames
- **Zone-based analytics** — configurable rectangular tracking zones with entry counting
- **Three-rule alert system** — zone intrusion, loitering detection, crowd alerts with per-track cooldowns
- **Professional dashboard** — multi-page Streamlit UI with real-time Plotly charts, session state, start/stop controls
- **Typed configuration** — `config.yaml` with dataclass validation and env-var overrides (`APP_` prefix)
- **Tested** — pytest suite covering config, frame processor geometry, and alert service logic

---

## Architecture

```
main.py                         ← Streamlit entrypoint (landing page)
pages/
  1_Live_Analytics.py           ← Video upload / camera → real-time processing
  2_Configuration.py            ← Model params, alert rules, zone defaults
  3_Alert_Log.py                ← Searchable alert history

app/
  config.py                     ← Typed Settings (dataclass) + YAML loader
  analytics/
    frame_processor.py          ← Central orchestrator (detection → tracking → zones → alerts)
    utils.py                    ← Zone geometry helpers
  alerts/
    alert_service.py            ← Unified alert evaluator (intrusion / loitering / crowd)
    models.py                   ← Alert, AlertType, Severity data models

tests/                          ← pytest suite
  conftest.py                   ← Shared fixtures
  test_config.py                ← Config loading, defaults, env overrides
  test_frame_processor.py       ← Geometry helpers, track map, stats
  test_alert_service.py         ← Intrusion, cooldown, crowd, loitering, reset
```

### Data Flow

```
Video Source → YOLO11 Detection → BotSORT Tracking → Zone Analytics → AlertService
                                                            ↓
                                                    Streamlit Dashboard
                                                  (metrics, charts, alerts)
```

---

## Tech Stack

| Component | Technology             | Version |
| --------- | ---------------------- | ------- |
| Detection | Ultralytics YOLO11     | ≥ 8.3   |
| Tracking  | BotSORT (built-in)     | —       |
| Video I/O | OpenCV (headless)      | ≥ 4.11  |
| Dashboard | Streamlit (multi-page) | ≥ 1.44  |
| Charts    | Plotly                 | ≥ 5.24  |
| Config    | PyYAML + dataclasses   | —       |
| Testing   | pytest                 | ≥ 8.0   |
| Linting   | Ruff                   | ≥ 0.8   |

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/DIPESHGOEL27/video-analytics.git
cd video-analytics
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

> For GPU support, install PyTorch with CUDA first:
>
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### 2. Download Models

Place at project root:

- [`yolo11n.pt`](https://github.com/ultralytics/assets/releases) — detection model
- [`yolo11n-seg.pt`](https://github.com/ultralytics/assets/releases) — segmentation model

### 3. Run

```bash
streamlit run main.py
```

Or use the Makefile:

```bash
make run      # Launch dashboard
make test     # Run test suite
make lint     # Run ruff linter
```

---

## Configuration

All settings live in [`config.yaml`](config.yaml). Example:

```yaml
yolo:
  model_path: "yolo11n.pt"
  confidence: 0.5

alert:
  enabled: true
  cooldown_seconds: 10
  loiter_threshold_seconds: 5.0
  crowd_threshold: 5
```

**Environment variable overrides** — prefix with `APP_`, use `__` for nesting:

```bash
APP_YOLO__CONFIDENCE=0.7 streamlit run main.py
```

---

## Alert Rules

| Rule               | Trigger                                    | Severity |
| ------------------ | ------------------------------------------ | -------- |
| **Zone Intrusion** | Object centroid enters the tracking zone   | Medium   |
| **Loitering**      | Object remains in zone > threshold seconds | High     |
| **Crowd**          | Simultaneous objects in zone ≥ threshold   | Critical |

All rules respect a per-track cooldown to prevent alert spam.

---

## Testing

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -v
```

Tests cover:

- **Config system** — defaults, YAML overrides, env-var overrides, unknown key handling
- **Frame processor** — `_rect_to_polygon`, `_is_point_in_zone` boundary cases, `_build_track_map`, stats serialisation
- **Alert service** — intrusion detection, cooldown enforcement, crowd thresholds, disable/reset

---

## License

[MIT](LICENSE)
