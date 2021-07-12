from torchreid.utils import FeatureExtractor

import socket
import pickle
import struct
import time
import cv2
import numpy as np

import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, \
    scale_coords,  set_logging, min_max_normalize, letterbox
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device
from utils.network import ClientPayload


if __name__ == "__main__":

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host_ip = '127.0.1.1'  # Here according to your server ip write the address

    port = 9999
    client_socket.connect((host_ip, port))

    if client_socket:

        # Initialize
        set_logging()
        device = select_device('0')

        # Load model
        model = attempt_load(
            'weight/best.pt', map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(640, s=stride)  # check image size
        names = model.module.names if hasattr(
            model, 'module') else model.names  # get class names
        model.half()  # to FP16
        cudnn.benchmark = True

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
                next(model.parameters())))  # run once

        extractor = FeatureExtractor(
            model_name='osnet_ain_x1_0',
            model_path='weight/reid.pth.tar',
            device='cuda'
        )

        # Create a VideoCapture object and read from input file
        # If the input is the camera, pass 0 instead of the video file name
        cap = cv2.VideoCapture("test-vid/1.avi")

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
                conf_thres = 0.25  # confidence threshold
                iou_thres = 0.45  # NMS IOU threshold
                agnostic_nms = False
                max_det = 1000  # maximum detections per image
                pred = non_max_suppression(pred, conf_thres, iou_thres, [
                                           1], agnostic_nms, max_det=max_det)

                reid_img = []

                # Draw rectangles and labels on the original image

                for i, det in enumerate(pred):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img_input.shape[2:], det[:, :4], frame.shape).round()

                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        label = f'{categories[c]} {conf:.2f}'
                        plot_one_box(xyxy, frame, label=label,
                                     color=colors(c, True), line_thickness=1)

                        x1 = int(min_max_normalize(float(xyxy[0]), 0, width-1))
                        x2 = int(min_max_normalize(float(xyxy[2]), 0, width-1))
                        y1 = int(min_max_normalize(
                            float(xyxy[1]), 0, height-1))
                        y2 = int(min_max_normalize(
                            float(xyxy[3]), 0, height-1))

                        im_input = frame[y1:y2, x1:x2]
                        cvt_img = im_input[:, :, ::-1]  # to RGB
                        reid_img.append(cvt_img)

                # ReID Inference
                reid_feat = False
                if (len(reid_img) != 0):
                    reid_feat = extractor(reid_img)
                    reid_feat = reid_feat.to('cpu')

                frame = cv2.putText(frame, "fps : {:.2f}".format(fps), (10, 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 0, 0), 2, cv2.LINE_AA)

                sent_data = ClientPayload(frame,reid_feat)

                # Send to server
                a = pickle.dumps(sent_data)
                message = struct.pack("Q", len(a))+a
                client_socket.sendall(message)

                # Display the resulting frame
                # cv2.imshow('Frame', frame)
                n = n + 1
                akhir = time.time()
                print(akhir - mulai)

                # FPS Counter Print
                if (n % 20 == 0):
                    n = 0
                    end = time.time()
                    fps = 20 / (end - start)
                    start = end
                    print(fps)

                # Press Q on keyboard to  exit
                if cv2.waitKey(int(1000/30)) & 0xFF == ord('q'):
                    client_socket.close()
                    break

            # Break the loop
            else:
                break

        # When everything done, release the video capture object
        cap.release()

        # Closes all the frames
        cv2.destroyAllWindows()
