from torchreid.utils import FeatureExtractor

import time
import cv2
import argparse
import numpy as np
import glob
import config as cfg

import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, \
    scale_coords,  set_logging, min_max_normalize, letterbox
from utils.torch_utils import select_device
from utils.multicamera import ClientPayload, TrackDatabase, PersonDatabase
from utils.plots import colors, plot_one_box

if __name__ == "__main__":

  # Parse argument from user
  parser = argparse.ArgumentParser()
  parser.add_argument('--cameraid', type=int, default=0, help='Camera ID')
  opt = parser.parse_args()
  print(opt)

  # Initialize
  set_logging()
  device = select_device('0')
  f = open("result.txt", 'w')

  # Load Yolov5 model
  model = attempt_load('weight/best.pt', map_location=device)  # load FP32 model
  stride = int(model.stride.max())  # model stride
  imgsz = check_img_size(640, s=stride)  # check image size
  model.half()  # to FP16
  cudnn.benchmark = True

  # Prepare GPU inference
  if device.type != 'cpu':
    model(
        torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
            next(model.parameters())))  # run once

  # Instantiate ReID Network
  extractor = FeatureExtractor(model_name='osnet_ain_x1_0',
                               model_path='weight/reid.pth.tar',
                               device='cuda')

  # # Create a VideoCapture object and read from input file
  # # If the input is the camera, pass 0 instead of the video file name
  # cap = cv2.VideoCapture("test-vid/" + "0" + ".avi")

  # # Check if camera opened successfully
  # if (cap.isOpened() == False):
  #   print("Error opening video stream or file")

  # load coco labels
  categories = ["head", "person"]

  # Tracker Initiate
  tracks = TrackDatabase(len(cfg.camera), cfg.matching["max_idle_age"])
  persons = PersonDatabase(cfg.matching["max_stored_features"],
                           cfg.matching["match_distance_threshold"],
                           cfg.matching["iou_weight"],
                           cfg.matching["reid_weight"],
                           cfg.matching["reid_distance_threshold"])

  n_frame = 0
  total_time = 0
  n_det = 0

  # Start video inferencing
  print("This camera is running...")

  all_file = []
  for filepath in glob.iglob(r'/home/naufal/Documents/tugas-akhir/MOT20/train/MOT20-01/img1/*.jpg'):
    all_file.append(filepath)
    
  all_file.sort()

  for filepath in all_file:
    # Capture frame-by-frame
    frame = cv2.imread(filepath)
    n_frame += 1

    frame = cv2.resize(frame, (1920,1080), interpolation = cv2.INTER_AREA)
    start = time.time()
    height = frame.shape[0]
    width = frame.shape[1]

    # Padded resize
    img_input = letterbox(frame, 640, stride=stride)[0]

    # Convert
    # BGR to RGB, to 3x416x416
    img_input = img_input[:, :, ::-1].transpose(2, 0, 1)
    img_input = np.ascontiguousarray(img_input)

    img_input = torch.from_numpy(img_input).to(device)
    img_input = img_input.half()  # uint8 to fp16/32
    img_input /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img_input.ndimension() == 3:
      img_input = img_input.unsqueeze(0)

    # Inference
    pred = model(img_input, augment=False)[0]

    # Apply NMS
    pred = non_max_suppression(pred,
                                cfg.yolo["conf_thres"],
                                cfg.yolo["iou_thres"], [1],
                                cfg.yolo["agnostic_nms"],
                                max_det=cfg.yolo["max_det"])

    reid_img = []
    reid_xyxy = []

    # Draw rectangles and labels on the original image
    for i, det in enumerate(pred):
      # Rescale boxes from img_size to im0 size
      det[:, :4] = scale_coords(img_input.shape[2:], det[:, :4],
                                frame.shape).round()

      for *xyxy, conf, cls in reversed(det):
        x1 = int(xyxy[0])
        x2 = int(xyxy[2])
        y1 = int(xyxy[1])
        y2 = int(xyxy[3])
        im_input = frame[y1:y2, x1:x2]
        cvt_img = im_input[:, :, ::-1]  # to RGB
        reid_img.append(cvt_img)
        reid_xyxy.append([x1, y1, x2, y2])

    # ReID Inference
    reid_feat = torch.tensor([])

    if (len(reid_img) != 0):
      n_det += len(reid_img)
      reid_feat = extractor(reid_img)
      reid_feat = reid_feat.to('cpu')

    match_arr = persons.feature_matching(reid_feat, 
                  tracks.get_iou_matrix(opt.cameraid, reid_xyxy, persons.get_num_person()))
    curr_tracks = tracks.update_tracks(opt.cameraid, reid_xyxy, match_arr)

    for key, value in curr_tracks.items():
      plot_one_box(value.xyxy,
                    frame,
                    label=str(value.personid),
                    color=colors(0, True),
                    line_thickness=1)
      f.writelines(
          f'{n_frame},{float(value.personid)},{round(float(value.xyxy[0]),0)},{round(float(value.xyxy[1]),0)},{round(float(value.xyxy[2]-value.xyxy[0]), 0)},{round(float(value.xyxy[3]-value.xyxy[1]), 0)},-1,-1,-1,-1\n'
      )

    cv2.namedWindow("Cam", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Cam", 1400, 1000)
    cv2.imshow("Cam", frame)

    end = time.time()
    timeDiff = end - start
    total_time += timeDiff
    
    print(total_time / n_frame)
    print(n_det / n_frame)

    if (timeDiff < 1.0/(10)):
      time.sleep(1.0/(10) - timeDiff)

    # Press Q on keyboard to  exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break


  # Closes all the frames
  cv2.destroyAllWindows()
