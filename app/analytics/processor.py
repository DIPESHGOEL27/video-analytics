import cv2
from app.alerts.alert_engine import AlertEngine

class FrameProcessor:
    def __init__(self, detector, tracker, config):
        self.detector = detector
        self.tracker = tracker
        self.config = config
        self.alert_engine = AlertEngine(config.get("alerts", {}))

    def process(self, frame, frame_id):
        # Step 1: Object Detection
        detections = self.detector.detect(frame)

        # Step 2: Run Deep SORT Tracking
        tracks = self.tracker.update_tracks(detections, frame)

        # Step 3: Trigger Alerts (based on custom logic)
        self.alert_engine.evaluate(tracks, frame_id)

        # Step 4: Draw boxes on the frame
        return self._annotate(frame, tracks)

    def _annotate(self, frame, tracks):
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue

            box = track.to_tlbr()  # top-left, bottom-right
            track_id = track.track_id
            class_name = track.get_class_name() if hasattr(track, 'get_class_name') else "Object"

            # Draw bounding box
            color = (0, 255, 0)
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.putText(frame, f"{class_name} ID:{track_id}", (int(box[0]), int(box[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return frame
