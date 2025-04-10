import time
import logging

class AlertEngine:
    def __init__(self, config):
        self.config = config
        self.cooldowns = {}
        self.logger = logging.getLogger("AlertEngine")

    def evaluate(self, tracks, frame_id):
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue

            # Example Rule 1: Person in restricted zone
            if self._in_restricted_zone(track):
                self._trigger_alert("RestrictedAreaBreach", track.track_id, frame_id)

            # Example Rule 2: Unusual behavior placeholder (extend later)
            # if self._suspicious_behavior(track):
            #     self._trigger_alert("SuspiciousBehavior", track.track_id, frame_id)

    def _in_restricted_zone(self, track):
        box = track.to_tlbr()
        x_center = (box[0] + box[2]) / 2
        y_center = (box[1] + box[3]) / 2

        restricted_zone = self.config.get("restricted_zone")  # [x1, y1, x2, y2]
        if not restricted_zone:
            return False

        x1, y1, x2, y2 = restricted_zone
        return x1 <= x_center <= x2 and y1 <= y_center <= y2

    def _trigger_alert(self, alert_type, track_id, frame_id):
        key = f"{alert_type}_{track_id}"
        cooldown_time = self.config.get("cooldown_seconds", 10)
        now = time.time()

        if key not in self.cooldowns or now - self.cooldowns[key] > cooldown_time:
            self.cooldowns[key] = now
            self._log_alert(alert_type, track_id, frame_id)

    def _log_alert(self, alert_type, track_id, frame_id):
        msg = f"[ALERT] {alert_type} | Track ID: {track_id} | Frame: {frame_id}"
        self.logger.warning(msg)
        print(msg)  # You could also send this to a database, Slack, etc.
