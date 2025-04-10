from ultralytics import YOLO
import torch

class ObjectDetector:
    def __init__(self, model_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame, classes=[0], device=self.device)
        detections = []
        for r in results:
            for i, box in enumerate(r.boxes.xyxy.cpu().numpy()):
                x1, y1, x2, y2 = map(int, box[:4])
                conf = float(r.boxes.conf[i])
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))
        return detections
