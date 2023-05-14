import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import argparse

ap= argparse.ArgumentParser()
ap= argparse.ArgumentParser()
ap.add_argument('--image', '-i', default='./blurry_license_plate.png', help='Path to input blurred image')
args= vars(ap.parse_args())
args= vars(ap.parse_args())

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

def process(ip_image, length, deblur_angle):
    noise = 0.01
    size = 200
    length= int(length)
    angle = (deblur_angle*np.pi) /180

    psf = np.ones((1, length), np.float32) #base image for psf
    costerm, sinterm = np.cos(angle), np.sin(angle)
    Ang = np.float32([[-costerm, sinterm, 0], [sinterm, costerm, 0]])
    size2 = size // 2
    Ang[:,2] = (size2, size2) - np.dot(Ang[:,:2], ((length-1)*0.5, 0))
    psf = cv2.warpAffine(psf, Ang, (size, size), flags=cv2.INTER_CUBIC) #Warp affine to get the desired psf

    gray = ip_image
    gray = np.float32(gray) / 255.0
    gray_dft = cv2.dft(gray, flags=cv2.DFT_COMPLEX_OUTPUT) #DFT of the image
    psf /= psf.sum() #Dividing by the sum
    psf_mat = np.zeros_like(gray)
    psf_mat[:size, :size] = psf
    psf_dft = cv2.dft(psf_mat, flags=cv2.DFT_COMPLEX_OUTPUT) #DFT of the psf
    PSFsq = (psf_dft**2).sum(-1)
    imgPSF = psf_dft / (PSFsq + noise)[...,np.newaxis] #H in the equation for wiener deconvolution
    gray_op = cv2.mulSpectrums(gray_dft, imgPSF, 0)
    gray_res = cv2.idft(gray_op,flags = cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT) #Inverse DFT
    gray_res = np.roll(gray_res, -size//2,0)
    gray_res = np.roll(gray_res, -size//2,1)

    return gray_res


# Function to visualize the Fast Fourier Transform of the blurred images.
def create_fft(img):
    img = np.float32(img) / 255.0
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    mag_spec = 20 * np.log(np.abs(fshift))
    mag_spec = np.asarray(mag_spec, dtype=np.uint8)

    return mag_spec

# Change this variable with the name of the trained models.
angle_model_name='../models/deblurring_angle_model.hdf5'
length_model_name= '../models/deblurring_length_model.hdf5'

print('Loading models (this will take a few seconds)...')
model1= load_model(angle_model_name)
model2= load_model(length_model_name)

# read blurred image
source = args['image']
ip_image = cv2.imread(source)

frame = ip_image.copy()

frameHeight = frame.shape[0]
frameWidth = frame.shape[1]

frame_bbox = [0, 0, frameWidth, frameHeight]
[px, py] = find_centroid_bbox(frame_bbox)

msg = "Can you read this license plate?"
font_scale = 1
font_thickness = 2
draw_label_banner(frame, msg, [px, py - 80], font_color=(255, 255, 255), fill_color=(255, 0, 0), font_scale=font_scale,
                  font_thickness=font_thickness)
skip_frame = True

cv2.imshow("Input", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

ip_image=  cv2.cvtColor(ip_image, cv2.COLOR_BGR2GRAY)
ip_image= cv2.resize(ip_image, (640, 480))

# Predicting the psf parameters of length and angle.
img= cv2.resize(create_fft(ip_image), (224,224))
img= np.expand_dims(img_to_array(img), axis=0)/ 255.0
preds= model1.predict(img)
# angle_value= np.sum(np.multiply(np.arange(0, 180), preds[0]))
angle_value = np.mean(np.argsort(preds[0])[-3:])

print("Predicted Blur Angle: ", angle_value)
length_value= model2.predict(img)[0][0]
print("Predicted Blur Length: ",length_value)

op_image = process(ip_image, length_value, angle_value)
op_image = (op_image*255).astype(np.uint8)
op_image = (255/(np.max(op_image)-np.min(op_image))) * (op_image-np.min(op_image))

cv2.imwrite("result.png", op_image)

# op_image = cv2.imread("result.png")
cv2.imshow("Output", cv2.imread("result.png"))
cv2.waitKey(0)
cv2.destroyAllWindows()


