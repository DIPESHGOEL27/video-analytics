def xyxy_to_xywh(bbox_xyxy):
    """Convert bounding boxes from [x1, y1, x2, y2] to [x, y, w, h]"""
    x1, y1, x2, y2 = bbox_xyxy
    w = x2 - x1
    h = y2 - y1
    return [x1, y1, w, h]
