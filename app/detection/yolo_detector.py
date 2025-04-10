from ultralytics import YOLO
import torch

class YoloDetector:
    def __init__(self, config):
        self.model = YOLO(config["yolo"]["model_path"])
        self.conf_thresh = config["yolo"]["confidence_threshold"]
        self.classes = config["yolo"]["classes"]

    def detect(self, frame):
        results = self.model(frame)[0]
        detections = []
        for result in results.boxes.data:
            x1, y1, x2, y2, conf, cls = result.tolist()
            if conf >= self.conf_thresh and (not self.classes or int(cls) in self.classes):
                detections.append({
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "confidence": conf,
                    "class_id": int(cls)
                })
        return detections
