yolo = {
    "conf_thres": 0.40,  # confidence threshold
    "iou_thres": 0.45,  # NMS IOU threshold
    "agnostic_nms": False,
    "max_det": 1000, # maximum detections per image
}

matching = {
    "match_distance_threshold" : 0.50,
    "reid_distance_threshold" : 0.65,
    "max_stored_features" : 25,
    "max_idle_age" : 5,
    "reid_weight" : 0.9,
    "iou_weight" : 0.1
}

camera = [
    {
        "id": 0,
        "port": "5555"
    },
    {
        "id": 1,
        "port": "5565"
    }
]