# ------------------------------------------------------------------------------
#
# User Interface for STEM Kit computer vision applications.
#
#   Assumes supporting models are present in ./models
#
# Developed by Big Vision LLC for Emerging Technologies Institute (ETI).
# ------------------------------------------------------------------------------
import os
os.environ["KIVY_NO_ARGS"] = "1"

import sys
import numpy as np
import cv2
#import pyzbar.pyzbar as pyzbar
import onnxruntime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

import kivy
from kivy.clock import Clock
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.slider import Slider
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.graphics.texture import Texture
from kivy.core.window import Window
import threading
import queue

kivy.require("2.0.0")

def parse_command_line_args():
    mode = "SK"
    for i, arg in enumerate(sys.argv):
        if arg in ["--mode", "-m"] and i + 1 < len(sys.argv):
            mode = sys.argv[i + 1]
    return mode

# Set run mode configurations that appropriately scale application elements.
mode = parse_command_line_args()
if mode == 'SK':
    # STEM-Kit run mode (default).
    Window.left = 140  # horizontal position
    Window.top = 30  # vertical position (distance from the top of the screen)
    font_scale = 0.8
    font_thickness = 2
    layout_padding_y = 25
else:
    # Laptop run mode.
    Window.left = 450  # horizontal position
    Window.top = 200   # vertical position (distance from the top of the screen)
    font_scale = .9
    font_thickness = 2
    layout_padding_y = 40

def find_centroid_bbox(bbox):
    '''
    Finds the centroid of the input bounding box [x, y, w, h].
    '''
    mid_x = int(bbox[0] + bbox[2] / 2)
    mid_y = int(bbox[1] + bbox[3] / 2)

    # Return the coordinates of the centroid.
    return mid_x, mid_y

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

def draw_label_banner(frame, text, centroid, font_color=(0, 0, 0), fill_color=(255, 255, 255), font_scale=1,
                      font_thickness=1):
    '''
    Annotate the image frame with a text banner overlayed on a filled rectangle.
    '''
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness + 2)[0]
    text_pad = 10
    # Define upper left and lower right vertices of rectangle.
    px1 = int(centroid[0]) - int(text_size[0] / 2) - text_pad
    px2 = int(centroid[0]) + int(text_size[0] / 2) + text_pad

    py1 = int(centroid[1]) - int(text_size[1] / 2) - text_pad
    py2 = int(centroid[1]) + int(text_size[1] / 2) + text_pad

    frame = cv2.rectangle(frame, (px1, py1), (px2, py2), fill_color, thickness=-1, lineType=cv2.LINE_8)

    # Define the lower left coordinate for the text.
    px = int(centroid[0]) - int(text_size[0] / 2)
    py = int(centroid[1]) + int(text_size[1] / 2)

    cv2.putText(frame, text, (px, py), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness,
                cv2.LINE_AA)

class MainLayout(BoxLayout):
    def __init__(self, **kwargs):
        super(MainLayout, self).__init__(**kwargs)

        self.orientation = "vertical"

        self.modules_layout = GridLayout(cols=2, rows=4)
        self.add_widget(self.modules_layout)

        button_names = [
            "Edge Detection",
            "Image Deblurring",
            "Face Recognition",
            "QR Code Decoder",
            "Module 5",
            "Module 6",
            "Module 7",
            "Module 8",
        ]

        for i in range(len(button_names)):
            btn = Button(text=button_names[i])
            btn.bind(on_press=self.open_module_popup)
            self.modules_layout.add_widget(btn)

        self.capture = cv2.VideoCapture(0)
        self.frame_queue = queue.Queue(maxsize=5)
        self.capture_thread = threading.Thread(target=self.capture_frames)
        self.capture_thread.daemon = True
        self.capture_thread.start()

    def capture_frames(self):
        while True:
            ret, frame = self.capture.read()
            if not ret:
                break

            if self.frame_queue.full():
                self.frame_queue.get()

            self.frame_queue.put(frame)

    def open_module_popup(self, instance):
        module_name = instance.text

        if module_name == "QR Code Decoder":
            popup = QRCodeDecoderPopup(self)
        elif module_name == "Face Recognition":
            popup = FaceRecognitionPopup(self)
        elif module_name == "Image Deblurring":
            popup = DeblurringPopup(self)
        elif module_name == "Edge Detection":
            popup = EdgeDetectionPopup(self)
        else:
            # Add additional modules here
            popup = UnderConstructionPopup(self)

        popup.open()

    def on_stop(self):
        self.capture.release()

class BasePopup(Popup):
    def __init__(self, main_layout, **kwargs):
        super(BasePopup, self).__init__(**kwargs)
        self.main_layout = main_layout
        self.capture = main_layout.capture

    def close_popup(self, *args):
        self.dismiss()

    @staticmethod
    def convert_frame_to_texture(self, frame):
        buf = cv2.flip(frame, 0).tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        return texture

    def create_labeled_slider(self, label_text, min_value, max_value, initial_value, value_format='{}', size_hint_label=.4):
        layout = BoxLayout(orientation="horizontal", size_hint_y=None, height="30dp")

        label = Label(text=label_text, size_hint_x=size_hint_label)
        layout.add_widget(label)

        slider = Slider(min=min_value, max=max_value, value=initial_value, size_hint_x=0.5)
        layout.add_widget(slider)

        value_label = Label(text=value_format.format(initial_value), size_hint_x=0.1)
        layout.add_widget(value_label)
        slider.bind(value=lambda instance, value: setattr(value_label, 'text', value_format.format(value)))

        return layout, slider

    def create_base_widgets(self):
        self.content = BoxLayout(orientation="vertical")
        self.add_widget(self.content)

        self.image = Image(allow_stretch=True, size_hint_y=0.7)
        self.content.add_widget(self.image)

        self.slider_layout = BoxLayout(orientation="vertical", size_hint_y=0.15)
        self.content.add_widget(self.slider_layout)

        button_container = BoxLayout(size_hint_y=None, height="40dp")
        self.slider_layout.add_widget(button_container)

        self.close_button = Button(text="Close")
        self.close_button.bind(on_press=self.close_popup)
        button_container.add_widget(self.close_button)

        self.process_button = Button(text="Process Image")
        self.process_button.bind(on_press=self.process_image)
        button_container.add_widget(self.process_button)

    def get_latest_frame(self):
        if not self.main_layout.frame_queue.empty():
            return self.main_layout.frame_queue.get()
        else:
            return None

class QRCodeDecoderPopup(BasePopup):
    def __init__(self, main_layout, **kwargs):
        super(QRCodeDecoderPopup, self).__init__(main_layout, **kwargs)
        self.title = "Decode QR Code"

        self.content = BoxLayout(orientation="vertical", spacing=layout_padding_y)

        self.image = Image(allow_stretch=True, size_hint_y=0.7)
        self.content.add_widget(self.image)

        button_layout = BoxLayout(size_hint_y=0.1)
        self.content.add_widget(button_layout)

        self.close_button = Button(text="Close")
        self.close_button.bind(on_press=self.close_popup)
        button_layout.add_widget(self.close_button)

        # Clock.schedule_interval(self.process_image, 1.0 / 30.0)
        Clock.schedule_interval(self.process_image_cv2, 1.0 / 30.0)

        self.frame_count = 0

    def process_image_cv2(self, dt, *args):

        frame = self.get_latest_frame()
        if frame is None:
            return

        else:
            qcd = cv2.QRCodeDetector()

            # Convert the current frame to grayscale and decode any found QR codes.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found_qr_code, decoded_info, points, straight_qrcode = qcd.detectAndDecodeMulti(gray)

            if found_qr_code is True:

                # Outline each detected QR code.
                frame = cv2.polylines(frame, points.astype(int), True, (0, 255, 0), thickness=3)

                # Annotate the frame with the decoded message.
                for msg, points in zip(decoded_info, points):
                    centroid = find_centroid_vertices(points)
                    draw_label_banner(frame, msg, centroid, font_color=(255, 255, 255), fill_color=(255, 0, 0),
                                      font_scale=font_scale, font_thickness=font_thickness)

            texture = self.convert_frame_to_texture(self, frame)
            self.image.texture = texture


    # def process_image(self, dt, *args):
    #
    #     frame = self.get_latest_frame()
    #     if frame is None:
    #         return
    #
    #     else:
    #
    #         qcd = cv2.QRCodeDetector()
    #
    #         # Convert the current frame to grayscale and decode any found QR codes.
    #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #         decoded = pyzbar.decode(gray)
    #
    #         for code in decoded:
    #             # Extract bounding box location and size.
    #             bbox = [x, y, w, h] = code.rect
    #
    #             # Draw bounding box rectangle around the QR code
    #             cv2.polylines(frame, [np.array(code.polygon)], True, (0, 255, 0), 2)
    #
    #             # Get decoded text from the QR code
    #             msg = code.data.decode('utf-8')
    #
    #             centroid = find_centroid_bbox(bbox)
    #             draw_label_banner(frame, msg, centroid, font_color=(255, 255, 255), fill_color=(255, 0, 0),
    #                               font_scale=font_scale,
    #                               font_thickness=font_thickness)
    #
    #         texture = self.convert_frame_to_texture(self, frame)
    #         self.image.texture = texture

class DeblurringPopup(BasePopup):
    def __init__(self, main_layout, **kwargs):
        super(DeblurringPopup, self).__init__(main_layout, **kwargs)
        self.title = "Deblur Image"

        self.content = BoxLayout(orientation="vertical", spacing=layout_padding_y)
        self.input_source = './input_media/image_1.jpg'
        # self.image = Image(source='./input_media/license_plate.jpg', allow_stretch=True, size_hint_y=0.7)
        self.image = Image(source=self.input_source, allow_stretch=True, size_hint_y=0.7)
        self.content.add_widget(self.image)

        button_layout = BoxLayout(size_hint_y=0.1)
        self.content.add_widget(button_layout)

        self.close_button = Button(text="Close")
        self.close_button.bind(on_press=self.close_popup)
        button_layout.add_widget(self.close_button)

        self.process_button = Button(text="Process image")
        self.process_button.bind(on_press=self.process_image)
        button_layout.add_widget(self.process_button)

    # Function to visualize the Fast Fourier Transform of the blurred images.
    def create_fft(self, img):
        img = np.float32(img) / 255.0
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        mag_spec = 20 * np.log(np.abs(fshift))
        mag_spec = np.asarray(mag_spec, dtype=np.uint8)

        return mag_spec

    def process(self, ip_image, length, deblur_angle):
        noise = 0.01
        size = 200
        length = int(length)
        angle = (deblur_angle * np.pi) / 180

        psf = np.ones((1, length), np.float32)  # base image for psf
        costerm, sinterm = np.cos(angle), np.sin(angle)
        Ang = np.float32([[-costerm, sinterm, 0], [sinterm, costerm, 0]])
        size2 = size // 2
        Ang[:, 2] = (size2, size2) - np.dot(Ang[:, :2], ((length - 1) * 0.5, 0))
        psf = cv2.warpAffine(psf, Ang, (size, size), flags=cv2.INTER_CUBIC)  # Warp affine to get the desired psf

        gray = ip_image
        gray = np.float32(gray) / 255.0
        gray_dft = cv2.dft(gray, flags=cv2.DFT_COMPLEX_OUTPUT)  # DFT of the image
        psf /= psf.sum()  # Dividing by the sum
        psf_mat = np.zeros_like(gray)
        psf_mat[:size, :size] = psf
        psf_dft = cv2.dft(psf_mat, flags=cv2.DFT_COMPLEX_OUTPUT)  # DFT of the psf
        PSFsq = (psf_dft ** 2).sum(-1)
        imgPSF = psf_dft / (PSFsq + noise)[..., np.newaxis]  # H in the equation for wiener deconvolution
        gray_op = cv2.mulSpectrums(gray_dft, imgPSF, 0)
        gray_res = cv2.idft(gray_op, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)  # Inverse DFT
        gray_res = np.roll(gray_res, -size // 2, 0)
        gray_res = np.roll(gray_res, -size // 2, 1)

        return gray_res

    def process_image(self, dt, *args):

        frame = self.get_latest_frame()
        if frame is None:
            return

        else:

            # Change this variable with the name of the trained models.
            angle_model_name = './models/angle_model.hdf5'
            length_model_name = './models/length_model.hdf5'
            model1 = load_model(angle_model_name)
            model2 = load_model(length_model_name)

            # read blurred image
            ip_image = cv2.imread(self.input_source)
            ip_image = cv2.cvtColor(ip_image, cv2.COLOR_BGR2GRAY)
            ip_image = cv2.resize(ip_image, (640, 480))
            # FFT visualization of the blurred image
            # fft_img = self.create_fft(ip_image)

            # Predicting the psf parameters of length and angle.
            img = cv2.resize(self.create_fft(ip_image), (224, 224))
            img = np.expand_dims(img_to_array(img), axis=0) / 255.0
            preds = model1.predict(img)
            # angle_value= np.sum(np.multiply(np.arange(0, 180), preds[0]))
            angle_value = np.mean(np.argsort(preds[0])[-3:])

            print("Predicted Blur Angle: ", angle_value)
            length_value = model2.predict(img)[0][0]
            print("Predicted Blur Length: ", length_value)

            op_image = self.process(ip_image, length_value, angle_value)
            op_image = (op_image * 255).astype(np.uint8)
            op_image = (255 / (np.max(op_image) - np.min(op_image))) * (op_image - np.min(op_image))

            cv2.imwrite("result_new.png", op_image)

            op_image = cv2.imread('./result_new.png')

            texture = self.convert_frame_to_texture(self, op_image)
            self.image.texture = texture

class FaceRecognitionPopup(BasePopup):
    def __init__(self, main_layout, **kwargs):
        super(FaceRecognitionPopup, self).__init__(main_layout, **kwargs)
        self.title = "Facial Recognition"

        self.face_detetcion_model   = './models/face_detection_yunet_2022mar_int8.onnx'
        self.face_recognition_model = './models/face_recognition_sface_2021dec_int8.onnx'
        self.target_image           = './input_media/target_image.png'

        self.content = BoxLayout(orientation="vertical", spacing=layout_padding_y)

        self.image = Image(allow_stretch=True, size_hint_y=0.7)
        self.content.add_widget(self.image)

        slider_layout, self.slider = self.create_labeled_slider("Similarity Threshold: ", 1.00, 1.30, 1.10, value_format='{:.2f}', size_hint_label=0.4)

        slider_box = BoxLayout(size_hint_y=0.05)
        self.content.add_widget(slider_box)
        slider_box.add_widget(slider_layout)

        button_layout = BoxLayout(size_hint_y=0.1)
        self.content.add_widget(button_layout)

        self.close_button = Button(text="Close")
        self.close_button.bind(on_press=self.close_popup)
        button_layout.add_widget(self.close_button)

        Clock.schedule_interval(self.process_image, 1.0 / 30.0)

    def visualize(self, input, faces, thickness=2):
        if faces[1] is not None:
            for idx, face in enumerate(faces[1]):
                # print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(idx, face[0], face[1], face[2], face[3], face[-1]))

                coords = face[:-1].astype(np.int32)
                cv2.rectangle(input, (coords[0], coords[1]), (coords[0] + coords[2], coords[1] + coords[3]),
                              (0, 255, 0), thickness)
                cv2.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
                cv2.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
                cv2.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
                cv2.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
                cv2.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)

    def process_image(self, dt, *args):

        similarity_threshold = self.slider.value
        nms_threshold = 0.3
        score_threshold = 0.9
        top_k = 5000
        skip_frame = False

        detector = cv2.FaceDetectorYN.create(
            self.face_detetcion_model,
            "",
            (320, 320),
            score_threshold,
            nms_threshold,
            top_k
        )
        # scale = 1.0
        # frameWidth = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
        # frameHeight = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)
        frameWidth = 640
        frameHeight = 480
        detector.setInputSize([frameWidth, frameHeight])

        target_image = cv2.imread(cv2.samples.findFile(self.target_image))

        frame = self.get_latest_frame()
        if frame is None:
            return

        else:

            frame = cv2.resize(frame, (frameWidth, frameHeight))

            # --------------------------------------------------
            # Perform face detection on the input video frame.
            # --------------------------------------------------
            stream_faces = detector.detect(frame)  # faces is a tuple

            # Check for face detections in the video frame.
            if stream_faces[1] is not None:
                coords = stream_faces[1][0].astype(np.int32)
                bbox = coords[0:4]
                # Annotate each detected face with a bounding box and facial landmark points.
                self.visualize(frame, stream_faces)
            else:
                msg = "No Face Detected in the current frame."
                frame_bbox = [0, 0, frameWidth, frameHeight]
                centroid = find_centroid_bbox(frame_bbox)
                draw_label_banner(frame, msg, centroid, font_color=(255, 0, 255), fill_color=(0, 0, 0),
                                  font_scale=font_scale, font_thickness=font_thickness)
                skip_frame = True

                texture = self.convert_frame_to_texture(self, frame)
                self.image.texture = texture

            if skip_frame == False:
                target_image = cv2.resize(target_image, (frameWidth, frameHeight))
                detector.setInputSize((target_image.shape[1], target_image.shape[0]))
                target_face = detector.detect(target_image)

                # Confirm a face was detected in the target image provided.
                assert target_face[1] is not None, 'Cannot find a face in {}'.format(args.target_image)

                # Initialize FaceRecognizerSF.
                recognizer = cv2.FaceRecognizerSF.create(self.face_recognition_model, "")

                # Align detected face in video frame and extract features.
                stream_face_align = recognizer.alignCrop(frame, stream_faces[1][0])
                stream_face_feature = recognizer.feature(stream_face_align)

                # Align target face and extract features.
                target_face_align = recognizer.alignCrop(target_image, target_face[1][0])
                target_face_feature = recognizer.feature(target_face_align)

                cosine_similarity_threshold = 0.363
                # l2_similarity_threshold = 1.128
                l2_similarity_threshold = similarity_threshold

                cosine_score = recognizer.match(stream_face_feature, target_face_feature, cv2.FaceRecognizerSF_FR_COSINE)
                l2_score = recognizer.match(stream_face_feature, target_face_feature, cv2.FaceRecognizerSF_FR_NORM_L2)

                if l2_score <= l2_similarity_threshold:
                    msg = 'Jack Ryan [Under Cover FBI Agent]'
                    centroid = find_centroid_bbox(bbox)
                    draw_label_banner(frame, msg, centroid, font_color=(0, 255, 0), fill_color=(0, 0, 0),
                                      font_scale=font_scale, font_thickness=font_thickness)
                else:
                    msg = 'No Match'
                    centroid = find_centroid_bbox(bbox)
                    draw_label_banner(frame, msg, centroid, font_color=(0, 0, 255), fill_color=(0, 0, 0),
                                      font_scale=font_scale, font_thickness=font_thickness)

                texture = self.convert_frame_to_texture(self, frame)
                self.image.texture = texture


class EdgeDetectionPopup(BasePopup):
    def __init__(self, main_layout, **kwargs):
        super(EdgeDetectionPopup, self).__init__(main_layout, **kwargs)
        self.title = "Canny Edge Detection"

        self.content = BoxLayout(orientation="vertical", spacing=layout_padding_y)

        self.image = Image(allow_stretch=True, size_hint_y=0.7)
        self.content.add_widget(self.image)

        slider1_layout, self.lower_threshold_slider = self.create_labeled_slider("Lower Threshold: ", 0, 255, 100)
        slider2_layout, self.upper_threshold_slider = self.create_labeled_slider("Upper Threshold: ", 0, 255, 200)

        slider_box = BoxLayout(orientation="vertical", size_hint_y=0.2)
        self.content.add_widget(slider_box)
        slider_box.add_widget(slider1_layout)
        slider_box.add_widget(slider2_layout)

        button_layout = BoxLayout(size_hint_y=0.1)
        self.content.add_widget(button_layout)

        self.close_button = Button(text="Close")
        self.close_button.bind(on_press=self.close_popup)
        button_layout.add_widget(self.close_button)

        Clock.schedule_interval(self.process_image, 1.0 / 30.0)

        self.frame_count = 0

    def process_image(self, *args):

        frame = self.get_latest_frame()
        if frame is None:
            return
        else:
            lower_threshold = self.lower_threshold_slider.value
            upper_threshold = self.upper_threshold_slider.value
            edge_frame = self.apply_canny_edge_detection(frame, lower_threshold, upper_threshold)
            colored_edge_frame = cv2.cvtColor(edge_frame, cv2.COLOR_GRAY2BGR)
            texture = self.convert_frame_to_texture(self, colored_edge_frame)
            self.image.texture = texture

    @staticmethod
    def apply_canny_edge_detection(img, lower_threshold, upper_threshold):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray, lower_threshold, upper_threshold)


class UnderConstructionPopup(BasePopup):
    def __init__(self, main_layout, **kwargs):
        super(UnderConstructionPopup, self).__init__(main_layout, **kwargs)
        self.title = "Under Construction"

        self.content = BoxLayout(orientation="vertical", spacing=40)

        self.image = Image(allow_stretch=True, size_hint_y=0.7)
        self.content.add_widget(self.image)

        button_layout = BoxLayout(size_hint_y=0.1)
        self.content.add_widget(button_layout)

        self.close_button = Button(text="Close")
        self.close_button.bind(on_press=self.close_popup)
        button_layout.add_widget(self.close_button)

class STEMKitApp(App):
    def build(self):
        if (mode == 'LT'):
            Window.size = (800, 600)
        else:
            Window.size = (500, 400)
        return MainLayout()

    def on_stop(self):
        self.root.on_stop()

if __name__ == "__main__":

    STEMKitApp().run()