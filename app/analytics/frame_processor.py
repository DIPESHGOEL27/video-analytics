import cv2

class FrameProcessor:
    def __init__(self, detector, tracker, config):
        self.detector = detector
        self.tracker = tracker
        self.config = config
        self.alert_enabled = config.get('alert', {}).get('enabled', False)
        self.alert_zone = config.get('alert', {}).get('alert_zone', [0, 0, 0, 0])

    def process_frame(self, frame):
        # Run YOLO detection
        detections = self.detector.detect(frame)
        print(f"[DEBUG] Raw YOLO Detections: {detections}")

        bbox_xyxy = []
        confidences = []
        class_ids = []

        for det in detections:
            if isinstance(det, dict):
                x, y, w, h = det['bbox']
                x1, y1 = x, y
                x2, y2 = x + w, y + h
                conf = det.get('confidence', 1.0)
                cls = det.get('class_id', 0)
            elif len(det) == 6:
                x1, y1, x2, y2, conf, cls = det
            elif len(det) == 5:
                x1, y1, x2, y2, conf = det
                cls = 0
            elif len(det) == 4:
                x1, y1, x2, y2 = det
                conf, cls = 1.0, 0
            else:
                continue

            bbox_xyxy.append([x1, y1, x2, y2])
            confidences.append(conf)
            class_ids.append(cls)

        print(f"[DEBUG] Parsed BBoxes: {bbox_xyxy}")
        print(f"[DEBUG] Confidences: {confidences}")

        # Run Deep SORT tracking
        tracked_objects = self.tracker.update(frame, bbox_xyxy, confidences)

        print(f"[DEBUG] Tracked Objects: {tracked_objects}")

        # Draw results
        for obj in tracked_objects:
            x1, y1, x2, y2 = map(int, obj['bbox'])
            track_id = obj['track_id']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if self.alert_enabled and self._in_alert_zone((x1, y1, x2, y2)):
                cv2.putText(frame, 'ALERT!', (x1, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if self.alert_enabled:
            zx, zy, zw, zh = self.alert_zone
            cv2.rectangle(frame, (zx, zy), (zx + zw, zy + zh), (255, 0, 0), 2)

        return frame

    def _in_alert_zone(self, bbox):
        x1, y1, x2, y2 = bbox
        zx, zy, zw, zh = self.alert_zone
        return x1 > zx and y1 > zy and x2 < zx + zw and y2 < zy + zh
