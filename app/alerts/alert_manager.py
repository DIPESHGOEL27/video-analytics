import cv2

class AlertManager:
    def __init__(self, config):
        self.zone = config["alert_zone"]  # x, y, w, h

    def check_alert_zone(self, bbox, frame):
        x1, y1, x2, y2 = bbox
        zx, zy, zw, zh = self.zone
        zone_rect = (zx, zy, zx + zw, zy + zh)

        if self._is_inside((x1, y1, x2, y2), zone_rect):
            cv2.putText(frame, "ALERT ZONE!", (zx, zy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.rectangle(frame, (zx, zy), (zx + zw, zy + zh), (0, 0, 255), 2)

    def _is_inside(self, bbox, zone):
        bx1, by1, bx2, by2 = bbox
        zx1, zy1, zx2, zy2 = zone
        return bx1 >= zx1 and by1 >= zy1 and bx2 <= zx2 and by2 <= zy2
