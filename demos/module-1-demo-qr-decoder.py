#------------------------------------------------------------------------------
#
# QR Code Decoder Demo
#
# This script will monitor the image frames from a connected video camera. If a
# QR code is detected in the image frame, the QR code will be outlined in green
# and the decoded message will be displayed. Execute the script from the command
# line prompt [$] as shown below:
#
#    ~/demos $ python module-1-demo-qr-decoder.py
#
# A window should appear that shows the video feed from the camera. Point the
# camera at a QR code to decode the embedded message.
#
# Developed by Big Vision LLC for Emerging Technologies Institute (ETI).
#------------------------------------------------------------------------------

# Import libraries.
import cv2
import numpy as np

# Set font scale and thickness for STEM-Kit resolution.
font_scale = 1
font_thickness = 2

def find_centroid_vertices(points):
    '''
    Finds the centroid of the input coordinates.
    '''
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
    mid_x = (min_x + max_x) / 2
    mid_y = (min_y + max_y) / 2

    # Return the coordinates of the centroid.
    return mid_x, mid_y

def draw_label_banner(frame, text, centroid, font_color=(0, 0, 0), fill_color=(255, 255, 255), font_scale=1, font_thickness=1):
    '''
    Annotate the image frame with a text banner overlayed on a filled rectangle.
    '''
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness + 2)[0]
    text_pad = 8

    # Define upper left and lower right vertices of rectangle.
    px1 = int(centroid[0]) - int(text_size[0] / 2) - text_pad
    px2 = int(centroid[0]) + int(text_size[0] / 2) + text_pad
    py1 = int(centroid[1]) - int(text_size[1] / 2) - text_pad
    py2 = int(centroid[1]) + int(text_size[1] / 2) + text_pad

    frame = cv2.rectangle(frame, (px1, py1), (px2, py2), fill_color, thickness=-1, lineType=cv2.LINE_8)

    # Define the lower left coordinate for the text.
    px = int(centroid[0]) - int(text_size[0] / 2)
    py = int(centroid[1]) + int(text_size[1] / 2)

    cv2.putText(frame, text, (px, py), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness, cv2.LINE_AA)

# Initialize videocapture object.
camera_index = 0
vid = cv2.VideoCapture(camera_index)

# Initialize QR detector.
qcd = cv2.QRCodeDetector()

# Enter infinite loop to process an endless video stream from the connected camera.
while(True):

    # Read one frame at a time from the video camera.
    has_frame, frame = vid.read()

    if has_frame:

        # Convert the current frame to grayscale and decode any found QR codes.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found_qr_code, decoded_info, points, straight_qrcode = qcd.detectAndDecodeMulti(gray)

        if found_qr_code is True:

            # Outline each detected QR code.
            frame = cv2.polylines(frame, points.astype(int), True, (0, 255, 0), thickness=2)

            # Annotate the frame with the decoded message.
            for msg, points in zip(decoded_info, points):
                centroid = find_centroid_vertices(points)
                draw_label_banner(frame, msg, centroid, font_color=(255, 255, 255), fill_color=(255, 0, 0), font_scale=font_scale,
                                  font_thickness=font_thickness)

        # Display annotated frame in window.
        cv2.imshow('Decoded Image', frame)

        key = cv2.waitKey(1)
        # Quit program when `q` or the `esc` key is selected.
        if key == ord('Q') or key == ord('q') or key == 27:
            break
    
# Release video object and destroy windows.
vid.release()
cv2.destroyAllWindows()