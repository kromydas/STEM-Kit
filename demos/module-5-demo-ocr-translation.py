#------------------------------------------------------------------------------
#
# OCR + Translation Demo
#
# This script uses the Google Translate API to detect text in an image and
# translate the text to English. The translated text will be annotated on a
# copy of the input image. Execute the script from the command line prompt [$]
# as shown below:
#
#    ~/demos $ python module-5-demo-ocr-translation.py
#
# A window will first appear that displays the input image (foreign-language.png).
# Select any key on the keyboard to initiate the processing and a new window will
# be displayed that shows the detected text and the associated translation to
# English.
#
# Developed by Big Vision LLC for Emerging Technologies Institute (ETI).
#------------------------------------------------------------------------------

import cv2
import numpy as np
import googletrans
import argparse

ap= argparse.ArgumentParser()
ap= argparse.ArgumentParser()
ap.add_argument('--image', '-i', default='./foreign-language.png', help='Path to input image that contains foriegn text.')
args= vars(ap.parse_args())
args= vars(ap.parse_args())

def draw_label_banner(frame, text, lower_left, font_color=(0, 0, 0), fill_color=(255, 255, 255), font_scale=1, font_thickness=1):
    '''
    Annotate the image frame with a text banner overlayed on a filled rectangle.
    '''
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness + 2)[0]
    text_pad = 8

    # Define upper left and lower right vertices of rectangle.
    px1 = lower_left[0] - text_pad
    px2 = lower_left[0] + int(text_size[0]) + text_pad
    py1 = lower_left[1] - int(text_size[1]) - text_pad
    py2 = lower_left[1] + text_pad

    frame = cv2.rectangle(frame, (px1, py1), (px2, py2), fill_color, thickness=-1, lineType=cv2.LINE_8)

    cv2.putText(frame, text, lower_left, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness, cv2.LINE_AA)

# This function performs a transformation for each bounding box detected by the text detection model.
def fourPointsTransform(frame, vertices):

    # Print vertices of each bounding box.
    vertices = np.asarray(vertices).astype(np.float32)
    outputSize = (100, 32)
    targetVertices = np.array([
        [0, outputSize[1] - 1],
        [0, 0],
        [outputSize[0] - 1, 0],
        [outputSize[0] - 1, outputSize[1] - 1]], dtype="float32")

    # Apply perspective transform.
    rotationMatrix = cv2.getPerspectiveTransform(vertices, targetVertices)
    result = cv2.warpPerspective(frame, rotationMatrix, outputSize)
    return result

# Perform Language Translation on Recognized Text.
def recognizeTranslateText(image, dest='en', src=''):

    # Use the DB text detector initialized previously to detect the presence of text in the image.
    boxes, confs = textDetector.detect(image)

    # Process each detected text block.
    for box in boxes:

        # Apply transformation on the bounding box detected by the text detection algorithm.
        croppedRoi  = fourPointsTransform(image, box)

        # Recognize the text using the crnn model.
        recognizedText = textRecognizer.recognize(croppedRoi)

        if recognizedText is not None and recognizedText.strip() != '':
            # translation = translator.translate(recognizedText, dest, src)
            if src:
                translation = translator.translate(recognizedText, dest, src)
            else:
                translation = translator.translate(recognizedText, dest)
        else:
            print("No text was recognized in the frame.")

        pad_x = 10
        shift_y = 10
        px = int(np.max(box[0:4,0])) + pad_x
        py = int(np.average(box[0:4,1])) + shift_y

        lower_left = (px, py)

        draw_label_banner(image, translation.text, lower_left, font_color=(255, 255, 255),
                          fill_color=(255, 0, 0), font_scale=0.7, font_thickness=2)

    # Draw the bounding boxes of text detected.
    cv2.polylines(image, boxes, True, (255, 0, 255), 3)

    return image

if __name__ == "__main__":
    print(googletrans.LANGUAGES)

    # Create a Translator Object.
    translator = googletrans.Translator()

    vocabulary = []
    try:
        with open("../models/alphabet_94.txt") as f:
            # Read the file line by line
            for l in f:
                # Append each line into the vocabulary list.
                vocabulary.append(l.strip())
    except FileNotFoundError:
        print("File not found!")
        # Handle error appropriately, maybe exit the program
    except PermissionError:
        print("No permission to read the file!")
        # Handle error appropriately

    try:
        # DB model for text-detection based on resnet50
        textDetector = cv2.dnn_TextDetectionModel_DB("../models/DB_TD500_resnet50.onnx")
    except FileNotFoundError:
        print("File not found!")
        # Handle error appropriately, maybe exit the program
    except PermissionError:
        print("No permission to read the file!")
        # Handle error appropriately

    try:
        # CRNN model for text-recognition.
        textRecognizer = cv2.dnn_TextRecognitionModel("../models/crnn_cs.onnx")
    except FileNotFoundError:
        print("File not found!")
        # Handle error appropriately, maybe exit the program
    except PermissionError:
        print("No permission to read the file!")
        # Handle error appropriately

    # Set threshold for Binary Map creation and polygon detection
    binThresh = 0.3
    polyThresh = 0.5

    mean = (122.67891434, 116.66876762, 104.00698793)
    inputSize = (640, 640)

    textDetector.setBinaryThreshold(binThresh).setPolygonThreshold(polyThresh)
    textDetector.setInputParams(1.0/255, inputSize, mean, True)

    # CRNN model for text-recognition.
    textRecognizer = cv2.dnn_TextRecognitionModel("../models/crnn_cs.onnx")
    textRecognizer.setDecodeType("CTC-greedy")
    textRecognizer.setVocabulary(vocabulary)
    textRecognizer.setInputParams(1/127.5, (100,32), (127.5, 127.5, 127.5), True)

    # Set the desired frame size.
    frameWidth = 512
    frameHeight = 512

    source = args['image']
    ip_image = cv2.imread(source)

    frame = ip_image.copy()
    frame = cv2.resize(frame, (frameWidth, frameHeight))

    cv2.imshow("Input", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # frame = cv2.resize(frame, (frameWidth, frameHeight))

    # ocr_result = recognizeTranslateText(frame, src='de')
    ocr_result = recognizeTranslateText(frame)

    cv2.imshow('OCR Translation Result', ocr_result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()




