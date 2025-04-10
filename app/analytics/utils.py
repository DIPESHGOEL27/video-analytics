import cv2

def draw_zones(frame, zones):
    for (x1, y1, x2, y2) in zones:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return frame

def is_in_zone(center, zones):
    for (x1, y1, x2, y2) in zones:
        if x1 < center[0] < x2 and y1 < center[1] < y2:
            return True
    return False
