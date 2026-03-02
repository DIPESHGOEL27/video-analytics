# Copilot Instructions — AI-Powered Video Analytics

## Architecture

Multi-page **Streamlit** app. Entrypoint: `main.py`. Pages under `pages/`.

```
main.py → pages/1_Live_Analytics.py → FrameProcessor → ultralytics.solutions.*
                                             ↓
                                       AlertService
```

Core modules:
- `app/config.py` — typed `Settings` dataclass loaded from `config.yaml` + `APP_*` env vars
- `app/analytics/frame_processor.py` — central orchestrator: detection → tracking → zone analytics → alerts
- `app/alerts/alert_service.py` — unified alert evaluator (intrusion / loitering / crowd)
- `app/alerts/models.py` — `Alert`, `AlertType`, `Severity` dataclasses

## Critical Conventions

**Region format**: Public API uses `[x1, y1, x2, y2]`. `TrackZone` needs polygon `[(x1,y1),(x2,y1),(x2,y2),(x1,y2)]`. Use `_rect_to_polygon()`.

**Headless-first**: Never call `cv2.imshow()` / `cv2.waitKey()`. Set `show=False` only in constructors, never in `.process()` / `__call__()`.

**BGR → RGB boundary**: Convert only at `st.image()` call. Internal pipeline stays BGR.

**Stats contract**: `FrameProcessor.process()` returns `(np.ndarray, dict)` with keys: `total_objects`, `zone_entries`, `objects_in_zone`, `fps`, `alerts`.

**Config injection**: All model paths, thresholds, and UI params come from `Settings`. Never hardcode. Override via `config.yaml` or `APP_YOLO__CONFIDENCE=0.7`.

## Running

```bash
streamlit run main.py    # or: make run
python -m pytest tests/  # or: make test
ruff check .             # or: make lint
```

## Key Patterns

- `st.session_state` for all mutable UI state (processor, zone, stats history, alerts)
- `atexit` handler cleans up temp files created by video uploads
- `AlertService.evaluate()` accepts track maps, returns `list[Alert]` — no frame drawing
- Display updates skip frames (`display_skip_frames` config) for UI responsiveness
