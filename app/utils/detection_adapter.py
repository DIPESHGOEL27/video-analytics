import numpy as np

def format_detections(detections):
    bbox_xywh = []
    confidences = []
    for det in detections:
        x, y, w, h = det["bbox"]
        bbox_xywh.append([x + w/2, y + h/2, w, h])
        confidences.append(det["confidence"])
    return np.array(bbox_xywh), np.array(confidences)
