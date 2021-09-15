import zmq
import cv2
import threading
import time

from utils.multicamera import TrackDatabase, PersonDatabase
from utils.plots import colors, plot_one_box
import config as cfg

camera_db = [0, 0]


def recvCameraInfo():
  global camera_db
  # Prepare our context and publisher
  context = zmq.Context()
  subscriber = context.socket(zmq.SUB)
  subscriber.connect("tcp://localhost:5555")
  subscriber.connect("tcp://localhost:5565")
  subscriber.setsockopt_string(zmq.SUBSCRIBE, "")

  while True:
    # Read envelope with address
    contents = subscriber.recv_pyobj()
    camera_db[contents.cameraid] = contents
  subscriber.close()
  context.term()


def reidMatching():
  global camera_db

  tracks = TrackDatabase(len(cfg.camera), cfg.matching["max_idle_age"])
  persons = PersonDatabase(cfg.matching["max_stored_features"],
                           cfg.matching["match_distance_threshold"],
                           cfg.matching["iou_weight"],
                           cfg.matching["reid_weight"],
                           cfg.matching["reid_distance_threshold"])
  n_del = 0
  tot_del = 0
  while (1):
    start = time.time()
    camera_db_temp = camera_db.copy()  # Create deep copy of global variable

    for cam in cfg.camera:
      if camera_db_temp[cam["id"]]:
        curr_cam = camera_db_temp[cam["id"]]
        match_arr = persons.feature_matching(curr_cam.reid_feat, tracks.get_iou_matrix(cam["id"], curr_cam.reid_xyxy, persons.get_num_person()))
        curr_tracks = tracks.update_tracks(cam["id"], curr_cam.reid_xyxy,
                                           match_arr)

        for key, value in curr_tracks.items():
          plot_one_box(value.xyxy,
                       camera_db_temp[cam["id"]].frame,
                       label=str(value.personid),
                       color=colors(0, True),
                       line_thickness=1)

        cv2.namedWindow("Cam" + str(cam["id"]), cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Cam" + str(cam["id"]), 600, 600)
        cv2.imshow("Cam" + str(cam["id"]),
                   camera_db_temp[cam["id"]].frame)
        
        end = time.time()
        n_del += 1
        tot_del += end - start
        print(tot_del / n_del)

    if cv2.waitKey(1) == 27:
      break


p1 = threading.Thread(target=recvCameraInfo, args=())
p1.start()

p2 = threading.Thread(target=reidMatching, args=())
p2.start()
