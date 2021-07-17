from time import time, sleep
import numpy as np
import torch
import zmq
import cv2
import threading
import pandas as pd
from tabulate import tabulate

from utils.classes import Track
from utils.plots import colors, plot_one_box

CAMERA_ID = [0, 1]

camera_db = [0, 0]


def recvCameraInfo():
    global camera_db
    # Prepare our context and publisher
    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.connect("tcp://localhost:5555")
    subscriber.connect("tcp://localhost:5565")
    subscriber.setsockopt_string(zmq.SUBSCRIBE, '')

    while True:
        # Read envelope with address
        contents = subscriber.recv_pyobj()
        camera_db[contents.cameraid] = contents

    subscriber.close()
    context.term()


def querySearch(tracks_db, feat):
    '''return distance with matched person and the matched person id

    Look for the nearest cosine distance between Re-ID features 
    of person in tracks database
    '''
    max_distance = 0.0
    max_person_id = False

    for key, value in tracks_db.items():
        distance = float(torch.nn.functional.cosine_similarity(
            value.reid_feat, feat, dim=0))
        if (max_distance < distance):
            max_distance = distance
            max_person_id = value.personid

    return max_distance, max_person_id


def reidMatching():
    global camera_db

    MATCH_DISTANCE_THRESHOLD = 0.6
    tracks = {}
    person = 1 	# Person ID counter
    remove_track = []
    c = int(0)

    while (1):
        camera_db_temp = camera_db.copy()

        for i in range(len(CAMERA_ID)):
            if (camera_db_temp[i] != 0):
                for j in range(len(camera_db_temp[i].reid_feat)):
                    res, match_id = querySearch(
                        tracks, camera_db_temp[i].reid_feat[j])
                    if res and match_id and res > MATCH_DISTANCE_THRESHOLD:
                        tracks[str(i) + '_' + str(match_id)].update(
                            camera_db_temp[i].reid_feat[j], camera_db_temp[i].reid_xyxy[j])

                    else:
                        tracks[str(i) + '_' + str(person)] = Track(i, person,
                                                                   camera_db_temp[i].reid_feat[j], camera_db_temp[i].reid_xyxy[j])
                        person = person + 1

        if (camera_db_temp[0] != 0 or camera_db_temp[1] != 0):
            for key, value in tracks.items():
                plot_one_box(value.reid_xyxy, camera_db_temp[value.cameraid].frame, label=str(value.personid),
                             color=colors(c, True), line_thickness=1)
                if (value.add_no_input_age()):
                    remove_track.append(key)

            if (len(remove_track) > 0):
                for track_name in remove_track:
                    tracks.pop(track_name,None)

            cv2.namedWindow("Frames", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Frames", 600,600)
            cv2.imshow("Frames", camera_db_temp[0].frame)

        if cv2.waitKey(1) == 27:
            break


p1 = threading.Thread(target=recvCameraInfo, args=())
p1.start()

p2 = threading.Thread(target=reidMatching, args=())
p2.start()
