# app/analytics/tracker.py

from app.tracking.deep_sort.deep_sort_tracker import DeepSort

class ObjectTracker:
    def __init__(self, max_age=30, nn_budget=100, nms_max_overlap=1.0):
        self.tracker = DeepSort(
            model_path='app/tracking/deep_sort/resources/networks/mars-small128.pb',
            max_cosine_distance=0.4,
            nn_budget=nn_budget,
            nms_max_overlap=nms_max_overlap
        )

    def update(self, detections, frame):
        return self.tracker.update_tracks(detections, frame)
