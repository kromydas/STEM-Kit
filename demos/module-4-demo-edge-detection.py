#------------------------------------------------------------------------------
#
# Edge Detection Demo
#
# This script will detect prominent edges in the video stream of a connected
# camera. Execute the script from the command line prompt [$] as shown below:
#
#    ~/demos $ python module-4-demo-edge-detection.py
#
# A window should appear that shows the video feed from the camera that has been
# filtered to show the prominent edges.
#
# Developed by Big Vision LLC for Emerging Technologies Institute (ETI).
#------------------------------------------------------------------------------

# Import libraries.
import argparse
import cv2

# Set font scale and thickness for STEM-Kit resolution.
font_scale = .8
font_thickness = 2

parser = argparse.ArgumentParser()
parser.add_argument('--scale', '-sc', type=float, default=.75, help='Scale factor used to resize input video frames.')
args = parser.parse_args()

# Initialize videocapture object.
deviceId = 0
cap = cv2.VideoCapture(deviceId)

# Set the detector input size based on the video frame size.
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * args.scale)
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * args.scale)

# Initialize QR detector.
qcd = cv2.QRCodeDetector()

move_window_flag = True

lower_threshold = 120
upper_threshold = 180

# Enter infinite loop to process an endless video stream from the connected camera.
while(True):

    # Read one frame at a time from the video camera.
    has_frame, frame = cap.read()

    if has_frame:

        frame = cv2.resize(frame, (frameWidth, frameHeight))

        # Convert the current frame to grayscale and decode any found QR codes.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edge_frame = cv2.Canny(gray, lower_threshold, upper_threshold)

        # Display annotated frame in window.
        cv2.imshow('Live', edge_frame)

        if move_window_flag:
            cv2.moveWindow('Live', 170, 20)
            move_window_flag = False

        key = cv2.waitKey(1)
        # Quit program when `q` or the `esc` key is selected.
        if key == ord('Q') or key == ord('q') or key == 27:
            break
    
# Release video object and destroy windows.
cap.release()
cv2.destroyAllWindows()
