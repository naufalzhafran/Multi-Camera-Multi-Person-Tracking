from torchreid.utils import FeatureExtractor

import time
import cv2
import argparse
import numpy as np
import zmq
import config as cfg

import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, \
    scale_coords,  set_logging, min_max_normalize, letterbox
from utils.torch_utils import select_device
from utils.multicamera import ClientPayload

if __name__ == "__main__":

  # Parse argument from user
  parser = argparse.ArgumentParser()
  parser.add_argument('--cameraid', type=int, default=0, help='Camera ID')
  parser.add_argument('--port', type=str, default=5555, help='Port number')
  opt = parser.parse_args()
  print(opt)

  # Prepare our context and publisher
  context = zmq.Context()
  publisher = context.socket(zmq.PUB)
  publisher.bind("tcp://*:" + opt.port)

  # Initialize
  set_logging()
  device = select_device('0')

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

  # Create a VideoCapture object and read from input file
  # If the input is the camera, pass 0 instead of the video file name
  cap = cv2.VideoCapture("test-vid/" + str(opt.cameraid) + ".avi")

  # Check if camera opened successfully
  if (cap.isOpened() == False):
    print("Error opening video stream or file")

  # load coco labels
  categories = ["head", "person"]

  # FPS Counter Initiate
  n = 0
  start = time.time()
  end = time.time()
  fps = 0

  # Start video inferencing
  print("This camera is running...")
  while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret == True:
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
          x1 = int(xyxy[0]), x2 = int(xyxy[2])
          y1 = int(xyxy[1]), y2 = int(xyxy[3])

          im_input = frame[y1:y2, x1:x2]
          cvt_img = im_input[:, :, ::-1]  # to RGB
          reid_img.append(cvt_img)
          reid_xyxy.append([x1, y1, x2, y2])

      # ReID Inference
      reid_feat = torch.tensor([])

      if (len(reid_img) != 0):
        reid_feat = extractor(reid_img)
        reid_feat = reid_feat.to('cpu')

      # Publish payload data to server
      sent_data = ClientPayload(frame, reid_feat, opt.cameraid, reid_xyxy)
      publisher.send_pyobj(sent_data)

      # FPS Counter Print
      if (n % 20 == 0):
        n = 0
        end = time.time()
        fps = 20 / (end - start)
        start = end
        # print(fps)

      # Press Q on keyboard to  exit
      if cv2.waitKey(int(1000 / 10)) & 0xFF == ord('q'):
        break

    # Break the loop
    else:
      break

  # When everything done, release the video capture object
  cap.release()

  # Closes all the frames
  cv2.destroyAllWindows()
