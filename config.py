yolo = {
    "conf_thres": 0.25,  # confidence threshold
    "iou_thres": 0.45,  # NMS IOU threshold
    "agnostic_nms": False,
    "max_det": 1000, # maximum detections per image
}

matching = {
    "match_distance_threshold" : 0.6,
    "max_stored_feature" : 10,
}