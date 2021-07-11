from models.yolotrt import YoLov5TRT

import time
import cv2
import numpy as np

import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

def min_max_normalize(num,min,max):
    if (num < min):
        return min
    elif (num > max):
        return max
    else:
        return num

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


if __name__ == "__main__":

  # Initialize
  set_logging()
  device = select_device('0')

  # Load model
  model = attempt_load('weight/best.pt', map_location=device)  # load FP32 model
  stride = int(model.stride.max())  # model stride
  imgsz = check_img_size(640, s=stride)  # check image size
  names = model.module.names if hasattr(model, 'module') else model.names  # get class names
  model.half()  # to FP16
  cudnn.benchmark = True
  
  # Run inference
  if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

  # Create a VideoCapture object and read from input file
  # If the input is the camera, pass 0 instead of the video file name
  cap = cv2.VideoCapture("test-vid/1.avi")

  # Check if camera opened successfully
  if (cap.isOpened()== False): 
    print("Error opening video stream or file")

  

  # load coco labels

  categories = ["head","person"]

  # FPS Counter Initiate
  n = 0
  start = time.time()
  end = time.time()
  fps = 0

  # Read until video is completed
  while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if ret == True:
      height = frame.shape[0]
      width = frame.shape[1]
      mulai = time.time()

      # Padded resize
      img_input = letterbox(frame, 640, stride=stride)[0]

      # Convert
      img_input = img_input[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
      img_input = np.ascontiguousarray(img_input)

      img_input = torch.from_numpy(img_input).to(device)
      img_input = img_input.half()  # uint8 to fp16/32
      img_input /= 255.0  # 0 - 255 to 0.0 - 1.0
      if img_input.ndimension() == 3:
        img_input = img_input.unsqueeze(0)

      # Inference
      pred = model(img_input, augment=False)[0]

      # Apply NMS
      conf_thres=0.25  # confidence threshold
      iou_thres=0.45  # NMS IOU threshold
      agnostic_nms=False
      max_det=1000  # maximum detections per image
      pred = non_max_suppression(pred, conf_thres, iou_thres, [1], agnostic_nms, max_det=max_det)

      reid_feat = []
      akhir = time.time()
      print(akhir - mulai)
      # Draw rectangles and labels on the original image

      for i, det in enumerate(pred):
        # Rescale boxes from img_size to im0 size
          det[:, :4] = scale_coords(img_input.shape[2:], det[:, :4], frame.shape).round()

          # box = res_boxes[j]

          # x1 = int(min_max_normalize(float(box[0]),0,width-1))
          # x2 = int(min_max_normalize(float(box[2]),0,width-1))
          # y1 = int(min_max_normalize(float(box[1]),0,height-1))
          # y2 = int(min_max_normalize(float(box[3]),0,height-1))

          # im_input = frame[y1:y2,x1:x2]
          # cvt_img = im_input[:, :, ::-1]

          for *xyxy, conf, cls in reversed(det):
            c = int(cls)  # integer class
            label =  f'{categories[c]} {conf:.2f}'
            plot_one_box(xyxy, frame, label=label, color=colors(c, True), line_thickness=1)

        
      frame = cv2.putText(frame, "fps : {:.2f}".format(fps) , (10,10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (0,0,0), 2, cv2.LINE_AA)
      # Display the resulting frame
      cv2.imshow('Frame',frame)
      n = n + 1
      # Press Q on keyboard to  exit
      if cv2.waitKey(25) & 0xFF == ord('q'):
        break
      
      # FPS Counter Print
      if (n % 20 == 0):
        n = 0
        end = time.time()
        fps = 20 /(end - start)
        start = end
        print(fps)


    # Break the loop
    else: 
      break
    


  # When everything done, release the video capture object
  cap.release()

  # Closes all the frames
  cv2.destroyAllWindows()