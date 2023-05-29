#------------------------------------------------------------------------------
#
# Artistic Filter Demo
#
# This script demonstrates the use of artistic filters in OpenCV.
# Execute the script from the command line prompt [$] as shown below:
#
#    ~/demos $ python module-7-demo-artistic-filters.py
#
# Upon execution, the script will launch a window that filters the video stream
# to render the video with an artistic effect known as a "pencil sketch."  The
# Pencil Sketch filter is the default, but you can also specify 
#
# You can use the optional input argument "filter" tp specify eith
#
# Developed by Big Vision LLC for Emerging Technologies Institute (ETI).
#------------------------------------------------------------------------------

import argparse
import cv2

# Set font scale and thickness for STEM-Kit resolution.
font_scale = .8
font_thickness = 2

parser = argparse.ArgumentParser()
parser.add_argument('--scale', '-sc', type=float, default=.75, help='Scale factor used to resize input video frames.')
parser.add_argument('--filter', '-f', type=str, default='pencil', help='Type of filter: pencil or style')
args = parser.parse_args()

# Initialize videocapture object.
deviceId = 0
cap = cv2.VideoCapture(deviceId)

# Set the detector input size based on the video frame size.
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * args.scale)
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * args.scale)

move_window_flag = True

# Enter infinite loop to process an endless video stream from the connected camera.
while(True):

    # Read one frame at a time from the video camera.
    has_frame, frame = cap.read()

    if has_frame:

        frame = cv2.resize(frame, (frameWidth, frameHeight))

        cv2.imshow('Live', frame)

        key = cv2.waitKey(1)
        # Quit program when `q` or the `esc` key is selected.
        if key == 32:
            break;

cv2.destroyAllWindows()
cap.release()

cap2 = cv2.VideoCapture(deviceId)

# Set the detector input size based on the video frame size.
frameWidth = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH) * args.scale)
frameHeight = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT) * args.scale)

while (True):

    # Read one frame at a time from the video camera.
    has_frame, frame = cap2.read()
    if has_frame:

        frame = cv2.resize(frame, (frameWidth, frameHeight))

        if args.filter == 'pencil':
            img_blur = cv2.GaussianBlur(frame, (5, 5), 0, 0)
            img_sketch_bw, img_sketch_color = cv2.pencilSketch(img_blur)
            color = True
            if color:
                frame = img_sketch_color
            else:
                frame = cv2.cvtColor(img_sketch_bw, cv2.COLOR_GRAY2BGR)
        else:
            img_blur = cv2.GaussianBlur(frame, (7, 7), 0, 0)
            frame = cv2.stylization(img_blur, sigma_s=60, sigma_r=.3)

        cv2.imshow('Live', frame)

        key = cv2.waitKey(1)
        # Quit program when `q` or the `esc` key is selected.
        if key == ord('Q') or key == ord('q') or key == 27:
            break
    
# Release video object and destroy windows.
cap2.release()
cv2.destroyAllWindows()
