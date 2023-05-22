#------------------------------------------------------------------------------
#
# Object Detection Demo (Adapted from TensorFlow tutorial examples,
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.)
#
# This script will detect various objects in the video stream.
#
# Execute the script from the command line prompt [$] as shown below:
#
#    ~/demos $ python module-8-demo-object-detetcion.py
#
# A window will be displayed with the video stream from a connected camera
# and each frame will be annotated with object detection results which includes
# a colored bounding box, the type of object detected, and the confidence of
# the detection.
#
# Developed by Big Vision LLC for Emerging Technologies Institute (ETI).
#------------------------------------------------------------------------------

import argparse
import sys
import time
import numpy as np
import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision

ap= argparse.ArgumentParser()
ap.add_argument('--model', help='Path of the object detection model.', required=False, default='../models/efficientdet_lite0.tflite')
ap.add_argument('--input', help='Path to input video (optional)', required=False)
ap.add_argument('--cameraId', help='Id of camera.', required=False, type=int, default=0)
ap.add_argument('--frameWidth',help='Width of frame to capture from camera.',required=False,type=int,default=600)
ap.add_argument('--frameHeight',help='Height of frame to capture from camera.',required=False,type=int,default=400)
ap.add_argument('--numThreads', help='Number of CPU threads to run the model.', required=False, type=int, default=4)
args= vars(ap.parse_args())

np.random.seed(2000)

_MARGIN = 10    # pixels
_ROW_SIZE = 10  # pixels
_FONT_SIZE = 1
_FONT_THICKNESS = 1

COLORS = np.random.uniform(0, 255, size=(200, 3))

def visualize( image: np.ndarray, detection_result: processor.DetectionResult) -> np.ndarray:
  """Draws bounding boxes on the input image and return it.

  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.

  Returns:
    Image with bounding boxes.
  """
  lw = max(round(sum(image.shape) / 2 * 0.003), 2)  # Line width.
  tf = max(lw - 1, 1) # Font thickness.
  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height

    # Draw label and score.
    category = detection.categories[0]
    color = COLORS[category.index]
    cv2.rectangle(image, start_point, end_point, color, 3)
    category_name = category.category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    w, h = cv2.getTextSize(result_text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=lw / 3, thickness=tf)[0]  # text width, height
    w = int(w - (0.20 * w))
    outside = start_point[1] - h >= 3
    p2 = start_point[0] + w, start_point[1] - h - 3 if outside else start_point[1] + h + 3
    cv2.rectangle(image, start_point, p2, color=color, thickness=-1, lineType=cv2.LINE_AA)
    # text_location = (_MARGIN + bbox.origin_x,_MARGIN + _ROW_SIZE + bbox.origin_y)
    cv2.putText(image, result_text,
                (start_point[0], start_point[1] - 5 if outside else start_point[1] + h + 2),
                cv2.FONT_HERSHEY_PLAIN, _FONT_SIZE, (255, 255, 255), _FONT_THICKNESS)
  return image

def run(model: str, camera_id: int, width: int, height: int,  num_threads: int, input_video ) -> None:
  """Continuously run inference on images acquired from the camera.

  Args:
    model: Name of the TFLite object detection model.
    camera_id: The camera id to be passed to OpenCV.
    width: The width of the frame captured from the camera.
    height: The height of the frame captured from the camera.
    num_threads: The number of CPU threads to run the model.
    enable_edgetpu: True/False whether the model is a EdgeTPU model.
  """

  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()

  # Start capturing video input from the camera.
  if input_video is not None:
    cap = cv2.VideoCapture(input_video)
  else:
    cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Visualization parameters.
  # row_size = 20  # pixels
  # left_margin = 24  # pixels
  # text_color = (0, 0, 255)  # red
  # font_size = 1.5
  # font_thickness = 2
  # fps_avg_frame_count = 10

  # Initialize the object detection model.
  base_options = core.BaseOptions(file_name=model, use_coral=False, num_threads=num_threads)
  detection_options = processor.DetectionOptions(max_results=3, score_threshold=0.3)
  options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
  detector = vision.ObjectDetector.create_from_options(options)

  # Continuously capture images from the camera and run inference
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      sys.exit('ERROR: Unable to read from webcam. Please verify your webcam settings.')

    counter += 1
    image = cv2.flip(image, 1)

    # Convert the image from BGR to RGB as required by the TFLite model.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a TensorImage object from the RGB image.
    input_tensor = vision.TensorImage.create_from_array(rgb_image)

    # Run object detection estimation using the model.
    detection_result = detector.detect(input_tensor)

    # Draw keypoints and edges on input image
    image = visualize(image, detection_result)

    # # Calculate the FPS
    # if counter % fps_avg_frame_count == 0:
    #   end_time = time.time()
    #   fps = fps_avg_frame_count / (end_time - start_time)
    #   start_time = time.time()

    # # Show the FPS
    # fps_text = 'FPS = {:.1f}'.format(fps)
    # text_location = (left_margin, row_size)
    # cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
    #             font_size, text_color, font_thickness)

    key = cv2.waitKey(1)
    # Quit program when `q` or the `esc` key is selected.
    if key == ord('Q') or key == ord('q') or key == 27:
        break
    cv2.imshow('Object Detector', image)

  cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':

    run(args['model'], int(args['cameraId']), args['frameWidth'], args['frameHeight'],
        int(args['numThreads']), args['input'])
