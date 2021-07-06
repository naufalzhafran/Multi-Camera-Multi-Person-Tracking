from models.yolotrt import YoLov5TRT
from utils.general import plot_one_box

import ctypes
import time
import cv2

if __name__ == "__main__":

  PLUGIN_LIBRARY = "weight/libmyplugins.so"
  engine_file_path = "weight/yolov5s.engine"

  ctypes.CDLL(PLUGIN_LIBRARY)
  # a YoLov5TRT instance
  yolov5_wrapper = YoLov5TRT(engine_file_path)

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

      res_boxes, res_scores, res_classid = yolov5_wrapper.infer(frame)

      # Draw rectangles and labels on the original image
      for j in range(len(res_boxes)):
          box = res_boxes[j]
          plot_one_box(
              box,
              frame,
              color=(0,0,0),
              label="{}:{:.2f}".format(
                  categories[int(res_classid[j])], res_scores[j]
              ),
          )

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
  yolov5_wrapper.destroy()