# ------------------------------------------------------------------------------
#
# Facial Recognition Demo
#
# This script will monitor the image frames from a connected video camera. If a
# human face is detected in the image frame, a green bounding box and facial
# landmark points for the eyes, nose and mouth will be displayed. Execute the
# script from the command line as shown below:
#
#    $ python module-2-face-recognition.py
#
# If you supply a target image, then the script will also attempt to identify faces
# in the input video stream that match the target image. This mode is specified as
# shown below:
#
#   $ python module-2-face-recognition.py --target_image target_image.png
#
# Developed by Big Vision LLC for Emerging Technologies Institute (ETI).
# ------------------------------------------------------------------------------

import argparse
import numpy as np
import cv2

font_scale = .8
font_thickness = 2

parser = argparse.ArgumentParser()

parser.add_argument('--target_image', '-trg', type=str, help='Path to target image to match.')
parser.add_argument('--video', '-v', type=str, help='Path to the input video.')
parser.add_argument('--scale', '-sc', type=float, default=.75, help='Scale factor used to resize input video frames.')
parser.add_argument('--face_detection_model', '-fd', type=str, default='../models/face_detection_yunet_2022mar_int8.onnx', help='Path to the face detection model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet')
parser.add_argument('--face_recognition_model', '-fr', type=str, default='../models/face_recognition_sface_2021dec_int8.onnx', help='Path to the face recognition model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_recognition_sface')
parser.add_argument('--score_threshold', type=float, default=0.95, help='Filtering out faces of score < score_threshold.')
parser.add_argument('--nms_threshold', type=float, default=0.3, help='Suppress bounding boxes of iou >= nms_threshold.')
parser.add_argument('--top_k', type=int, default=5000, help='Keep top_k bounding boxes before NMS.')

args = parser.parse_args()

def visualize(input, faces, fps, thickness=2):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            # print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(idx, face[0], face[1], face[2], face[3], face[-1]))

            coords = face[:-1].astype(np.int32)
            cv2.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
            cv2.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            cv2.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            cv2.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            cv2.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            cv2.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
    cv2.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def find_centroid_bbox(bbox):
    '''
    Finds the centroid of the input bounding box [x, y, w, h].
    '''
    mid_x = int(bbox[0] + bbox[2] / 2)
    mid_y = int(bbox[1] + bbox[3] / 2)

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

if __name__ == '__main__':

    # Initialize_FaceDetectorYN.
    detector = cv2.FaceDetectorYN.create(
        args.face_detection_model,
        "",
        (320, 320),
        args.score_threshold,
        args.nms_threshold,
        args.top_k
    )

    tm = cv2.TickMeter()

    # Create the video capture object.
    if args.video is not None:
        deviceId = args.video
    else:
        deviceId = 0
    cap = cv2.VideoCapture(deviceId)

    # Set the detector input size based on the video frame size.
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)*args.scale)
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*args.scale)

    detector.setInputSize([frameWidth, frameHeight])

    # If a target image is specified, read it for facial recogition.
    if args.target_image is not None:
        target_image = cv2.imread(cv2.samples.findFile(args.target_image))

    iter = 0
    while (True):
        hasFrame, frame = cap.read()
        if not hasFrame:
            print('No frames grabbed!')
            break
        skip_frame = False

        frame = cv2.resize(frame, (frameWidth, frameHeight))

        #--------------------------------------------------
        # Perform face detection on the input video frame.
        #--------------------------------------------------
        tm.start()
        stream_faces = detector.detect(frame) # faces is a tuple
        tm.stop()

        # Check for face detections in the video frame.
        if stream_faces[1] is not None:
            coords = stream_faces[1][0].astype(np.int32)
            bbox = coords[0:4]
            # Annotate each detected face with a bounding box and facial landmark points.
            visualize(frame, stream_faces, tm.getFPS())
        else:
            msg = "No Face Detected in the current frame."
            frame_bbox= [0, 0, frameWidth, frameHeight]
            centroid = find_centroid_bbox(frame_bbox)
            draw_label_banner(frame, msg, centroid, font_color=(255, 0, 255), fill_color=(0, 0, 0), font_scale=font_scale, font_thickness=font_thickness)
            skip_frame = True

        # Display the annotated video frame in a window.
        cv2.imshow('Live', frame)

        #--------------------------------------------------------------------
        # If a target image was specified, continue with facial recognition.
        # --------------------------------------------------------------------
        if args.target_image is not None and skip_frame == False:
            if iter == 0:
                target_image = cv2.resize(target_image, (frameWidth, frameHeight))
                detector.setInputSize((target_image.shape[1], target_image.shape[0]))
                target_face = detector.detect(target_image)

            # Confirm a face was detected in the target image provided.
            assert target_face[1] is not None, 'Cannot find a face in {}'.format(args.target_image)

            # Initialize FaceRecognizerSF.
            recognizer = cv2.FaceRecognizerSF.create(args.face_recognition_model,"")

            # Align detected face in video frame and extract features.
            stream_face_align = recognizer.alignCrop(frame, stream_faces[1][0])
            stream_face_feature = recognizer.feature(stream_face_align)

            # Align target face and extract features.
            if iter == 0:
                target_face_align = recognizer.alignCrop(target_image, target_face[1][0])
                target_face_feature = recognizer.feature(target_face_align)

            cosine_similarity_threshold = 0.363
            #l2_similarity_threshold = 1.128
            l2_similarity_threshold = 1.3

            # Feature match scores.
            #cosine_score = recognizer.match(stream_face_feature, target_face_feature, cv2.FaceRecognizerSF_FR_COSINE)
            l2_score = recognizer.match(stream_face_feature, target_face_feature, cv2.FaceRecognizerSF_FR_NORM_L2)

            if l2_score <= l2_similarity_threshold:
                msg = 'Keanu Reeves [Actor]'
                centroid = find_centroid_bbox(bbox)
                draw_label_banner(frame, msg, centroid, font_color=(0, 255, 0), fill_color=(0,0,0), font_scale=font_scale, font_thickness=font_thickness)
            else:
                msg = 'No Match'
                text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness + 2)[0]
                centroid = find_centroid_bbox(bbox)
                draw_label_banner(frame, msg, centroid, font_color=(0, 0, 255), fill_color=(0,0,0), font_scale=font_scale, font_thickness=font_thickness)
            tm.stop()
            cv2.imshow('Live', frame)
            iter += 1

        key = cv2.waitKey(1)
        if key == ord('Q') or key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()
