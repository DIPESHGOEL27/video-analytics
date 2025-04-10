import numpy as np
from app.tracking.deep_sort.tools import generate_detections as gdet
from app.tracking.deep_sort.detection import Detection
from app.tracking.deep_sort.tracker import Tracker
from app.tracking.deep_sort import preprocessing
from app.tracking.deep_sort.nn_matching import NearestNeighborDistanceMetric
from app.tracking.deep_sort.utils import xyxy_to_xywh


def xyxy_to_tlwh(bbox):
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2 - x1, y2 - y1]

class DeepSortTracker:
    def __init__(self, config):
        deep_sort_config = config['deep_sort']
        model_filename = deep_sort_config['model_path']

        self.encoder = gdet.create_box_encoder(model_filename, batch_size=1)

        metric = NearestNeighborDistanceMetric(
            metric='cosine',
            matching_threshold=deep_sort_config['max_cosine_distance'],
            budget=deep_sort_config['nn_budget']
        )

        self.tracker = Tracker(
            metric,
            max_age=deep_sort_config['max_age'],
            n_init=deep_sort_config['n_init']
        )

    def update(self, frame, bboxes, confs):

        # Convert to required formats
        bbox_xywh = [xyxy_to_xywh(bbox) for bbox in bboxes]
        bbox_tlwh = [xyxy_to_tlwh(bbox) for bbox in bboxes]
        confidences = confs

        # Encode appearance features
        features = self.encoder(frame, bbox_xywh)


        # Create Detection objects
        detections = [Detection(bbox, conf, feat) for bbox, conf, feat in zip(bbox_tlwh, confidences, features)]

        # Apply NMS
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, 0.5, scores)
        detections = [detections[i] for i in indices]

        # Update tracker
        self.tracker.predict()
        self.tracker.update(detections, frame)

        print(f"[INFO] {len(detections)} detections after NMS")
        print(f"[INFO] {len(self.tracker.tracks)} active tracks")


        # Collect results
        results = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            results.append({
                "bbox": bbox,
                "track_id": track.track_id
            })

        return results
