#------------------------------------------------------------------------------
#
# Binary Decoder Demo
#
#
#    ~/demos $ python module-6-demo-binary-decoder.py
#
#
# Developed by Big Vision LLC for Emerging Technologies Institute (ETI).
#------------------------------------------------------------------------------
import cv2
import numpy as np
import argparse

ap= argparse.ArgumentParser()
ap.add_argument('--image', '-i', default='./binary_input.png', help='Path to input image that contains binary codes.')
args= vars(ap.parse_args())

def draw_label_banner_ocr(frame, text, lower_left, font_color=(0, 0, 0), fill_color=(255, 255, 255), font_scale=1,
                          font_thickness=1):
    """
    Annotate the image frame with a text banner overlayed on a filled rectangle.
    """
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness + 2)[0]
    text_pad = 8

    # Define upper left and lower right vertices of rectangle.
    px1 = lower_left[0] - text_pad
    px2 = lower_left[0] + int(text_size[0]) + text_pad
    py1 = lower_left[1] - int(text_size[1]) - text_pad
    py2 = lower_left[1] + text_pad

    frame = cv2.rectangle(frame, (px1, py1), (px2, py2), fill_color, thickness=-1, lineType=cv2.LINE_8)

    cv2.putText(frame, text, lower_left, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness,
                cv2.LINE_AA)

def fourPointsTransform(frame, vertices):
    """
    Function: Align text boxes - transformation over the bounding boxes
                detected by the text detection model
    """
    # Print vertices of each bounding box 
    vertices = np.asarray(vertices).astype(np.float32)
    outputSize = (100, 32)
    targetVertices = np.array([
        [0, outputSize[1] - 1],
        [0, 0],
        [outputSize[0] - 1, 0],
        [outputSize[0] - 1, outputSize[1] - 1]], dtype="float32")
    
    # Apply perspective transform
    rotationMatrix = cv2.getPerspectiveTransform(vertices, targetVertices)
    result = cv2.warpPerspective(frame, rotationMatrix, outputSize)
    
    return result

'''
Function: Conversion of Binary to Decimal variable
'''
def binaryToDecimal(binary):

    decimal, i = 0, 0
    
    while(binary != 0):
        dec = binary % 10
        decimal = decimal + dec * pow(2, i)
        binary = binary // 10
        i += 1
    return decimal

'''
Function: Recognition of text from a given input image and perform conversion
'''
def recognizeText(image, dest='en', src='', debug=False):
    # Variable declaration
    code = []

    # Image resizing for screen fit
    image = cv2.resize(image, (640, 640))

    # Check for presence of text on image
    boxes, confs = textDetector.detect(image)

    if boxes is not None:
        # Draw the bounding boxes of text detected.
        cv2.polylines(image, boxes, True, (255, 0, 255), 3)

    # Iterate through detected text regions
    for box in boxes:

        # Variable declaration
        code = []

        # Apply transformation on the detected bounding box
        croppedRoi = fourPointsTransform(image, box)

        # Recognise the text using the crnn model
        recognizedText = textRecognizer.recognize(croppedRoi)

        if recognizedText is not None and recognizedText.strip() != '':

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # pt_code = pytesseract.image_to_string(gray, config="--psm 4")

            # Type conversion operations
            for item in recognizedText:
                code.append(str(item))
            binary = "".join(code)
            binary = int(binary)
            # print("code:   ", str(code))
            # print("binary: ", binary)
            # print("intbin: ", int(binary))
            recognizedDec = str(binaryToDecimal(binary))

            pad_x = 10
            shift_y = 10
            px = int(np.max(box[0:4, 0])) + pad_x
            py = int(np.average(box[0:4, 1])) + shift_y

            lower_left = (px, py)

            draw_label_banner_ocr(image, recognizedDec, lower_left, font_color=(255, 255, 255),
                                       fill_color=(255, 0, 0), font_scale=1, font_thickness=2)

        else:
            print("No text was recognized in the frame.")

    return image
        
if __name__ == "__main__":
    
    # Variable declaration
    inputSize = (640, 640)
    binThresh = 0.3
    polyThresh = 0.5
    mean = (122.67891434, 116.66876762, 104.00698793)

    vocabulary = []
    try:
        with open("../models/alphabet_01.txt") as f:
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

    # DB model for text-detection based on resnet50
    textDetector.setBinaryThreshold(binThresh).setPolygonThreshold(polyThresh)
    textDetector.setInputParams(1.0/255, inputSize, mean, True)

    # CRNN model for text-recognition
    textRecognizer.setDecodeType("CTC-greedy")
    textRecognizer.setVocabulary(vocabulary)
    textRecognizer.setInputParams(1/127.5, (100,32), (127.5, 127.5, 127.5),True)

    # Set the detector input size based on the video frame size.
    frameWidth = 600
    frameHeight = 480

    move_window_flag = True

    source = args['image']
    ip_image = cv2.imread(source)

    frame = ip_image.copy()
    frame = cv2.resize(frame, (frameWidth, frameHeight))

    cv2.imshow("Input", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Perform inference on input image
    decoded_image = recognizeText(frame, src='en')

    cv2.imshow('Binary Decoded Result', decoded_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

