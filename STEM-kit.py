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
import googletrans
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

import kivy
from kivy.clock import Clock
from kivy.app import App
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.slider import Slider
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from kivy.uix.filechooser import FileChooserIconView, FileChooserListView
import threading
import queue

kivy.require("2.0.0")
#--------------------------
# Load application models.
#--------------------------
angle_model_name = './models/deblurring_angle_model.hdf5'
length_model_name = './models/deblurring_length_model.hdf5'
model_angle = load_model(angle_model_name)
model_length = load_model(length_model_name)

try:
    # DB model for text-detection based on resnet50
    textDetector = cv2.dnn_TextDetectionModel_DB("./models/DB_TD500_resnet50.onnx")
except FileNotFoundError:
    print("File not found!")
    exit()
except PermissionError:
    print("No permission to read the file!")
    exit()

try:
    # CRNN model for text-recognition.
    textRecognizer = cv2.dnn_TextRecognitionModel("./models/crnn_cs.onnx")
except FileNotFoundError:
    print("File not found!")
    exit()
except PermissionError:
    print("No permission to read the file!")
    exit()

frame_rate = 1./15.

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
    Window.left = 100  # horizontal position
    Window.top = 10  # vertical position (distance from the top of the screen)
    font_scale = 0.8
    font_thickness = 2
    layout_padding_y = 25
    font_size_slider = 12
    default_media_path = '/media/pi'
    default_media_path = '/Users/billk/dev/BigVision/ETI/STEM-Kit/demos/test-case/thumbdrive_files'
else:
    # Laptop run mode.
    Window.left = 450  # horizontal position
    Window.top = 200   # vertical position (distance from the top of the screen)
    font_scale = .9
    font_thickness = 2
    layout_padding_y = 40
    font_size_slider = 24
    default_media_path = '/Users/billk/dev/BigVision/ETI/STEM-Kit/demos/test-case/thumbdrive_files'

def check_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return False
    cap.release()
    return True

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
            "OCR + Translation",
            "QR Code Decoder",
            "Binary Decoder",
            "Module 7",
            "Module 8",
        ]

        for i in range(len(button_names)):
            btn = Button(text=button_names[i])
            btn.bind(on_press=self.open_module_popup)
            self.modules_layout.add_widget(btn)

        if not check_camera():
            Clock.schedule_once(lambda dt: self.show_error_popup('Camera not connected!'), 0)
            cv2.waitKey(0)
            return

        self.capture = cv2.VideoCapture(0)
        self.frame_queue = queue.Queue(maxsize=1)
        self.capture_thread = threading.Thread(target=self.capture_frames)
        self.capture_thread.daemon = True
        self.capture_thread.start()

    def show_error_popup(self, dt):  # dt argument is required for functions called by Clock.
        content = BoxLayout(orientation='vertical')  # A layout for the Label and Button.
        message = Label(text='No camera connected! Please connect a camera and restart the application.',
                        color=(1, 0, 0, 1) ) # Red font color.
        close_button = Button(text='Close', size_hint=(1, 0.3))  # Close button at 20% of the popup height.

        content.add_widget(message)
        content.add_widget(close_button)

        popup = Popup(title='Error', content=content, size_hint=(None, None), size=(400, 200))
        message.text_size = (
        popup.size[0]*.8, None)  # Set the Label's text_size to the 80% of the width of the popup, and unlimited height.
        message.valign = 'middle'  # Optional: align the text vertically in the middle.
        close_button.bind(on_release=self.close_app)  # Close the app when the button is pressed.

        popup.open()

    def close_app(self, instance):  # We don't use 'instance', but it's passed by Kivy so we must include it
        App.get_running_app().stop()  # Stop the currently running Kivy app


    def capture_frames(self):

        while True:
            ret, frame = self.capture.read()

            if not ret:
                break

            if self.frame_queue.full():
                self.frame_queue.get()

            self.frame_queue.put(frame)
            # Reset the frame counter
            self.frame_counter = 0


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
        elif module_name == "OCR + Translation":
            popup = OCRTranslationPopup(self)
        elif module_name == "Binary Decoder":
            popup = BinaryDecoderPopup(self)
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

    def convert_frame_to_texture(self, frame):
        buf = cv2.flip(frame, 0).tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        return texture

    def create_labeled_slider(self, label_text, min_value, max_value, initial_value, value_format='{}', rounding=None,
                              size_hint_label=.4, font_size=24):
        layout = BoxLayout(orientation="vertical", size_hint_x=None, width="30dp")

        label = Label(text=label_text, size_hint_y=size_hint_label, font_size=font_size)
        layout.add_widget(label)

        slider = Slider(min=min_value, max=max_value, value=initial_value, orientation='vertical', size_hint_y=0.5)
        layout.add_widget(slider)

        value_label = Label(text=value_format.format(initial_value), size_hint_y=0.1, font_size=font_size)
        layout.add_widget(value_label)

        slider.bind(value=lambda instance, value: setattr(value_label, 'text', value_format.format(
            round(value, rounding) if rounding is not None else int(value))))

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

        # Clock.schedule_interval(self.process_image, frame_rate)
        Clock.schedule_interval(self.process_image_cv2, frame_rate)

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

            texture = self.convert_frame_to_texture(frame)
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
    #         texture = self.convert_frame_to_texture(frame)
    #         self.image.texture = texture

class DeblurringPopup(BasePopup):
    def __init__(self, main_layout, **kwargs):
        super(DeblurringPopup, self).__init__(main_layout, **kwargs)
        self.title = "Deblur Image"

        self.content = BoxLayout(orientation="horizontal",
                                 spacing=layout_padding_y)  # Changed orientation to "horizontal"
        self.input_source = None

        # Added a new BoxLayout with "vertical" orientation to hold the image and the buttons
        self.image_and_buttons = BoxLayout(orientation="vertical", size_hint_x=0.6)
        self.image = Image(source=self.input_source, allow_stretch=True)
        self.image_and_buttons.add_widget(self.image)

        button_layout = BoxLayout(size_hint_y=0.1)
        self.image_and_buttons.add_widget(button_layout)

        self.close_button = Button(text="Close")
        self.close_button.bind(on_press=self.close_popup)
        button_layout.add_widget(self.close_button)

        self.load_button = Button(text="Load Image")
        self.load_button.bind(on_press=self.load_image)
        button_layout.add_widget(self.load_button)

        self.process_button = Button(text="Process image")
        self.process_button.bind(on_press=self.process_image)
        button_layout.add_widget(self.process_button)

        self.filechooser = FileChooserListView(
            path=default_media_path,
            size_hint_x=0.4)
        self.content.add_widget(self.filechooser)

        self.content.add_widget(self.image_and_buttons)

    def load_image(self, instance):
        if len(self.filechooser.selection) > 0:
            self.input_source = self.filechooser.selection[0]
            self.image.source = self.input_source

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
        if length > 0:
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
        else:
            gray_res = ip_image

        return gray_res

    def process_image(self, instance):
        Clock.schedule_once(self.process_image_thread, 0)

    def process_image_thread(self, dt, *args):

        global texture_result
        texture_result = None  # Reset to None before processing starts

        frame = self.get_latest_frame()
        if frame is None:
            return
        else:
            # read blurred image
            ip_image = cv2.imread(self.input_source)
            ip_image = cv2.cvtColor(ip_image, cv2.COLOR_BGR2GRAY)
            ip_image = cv2.resize(ip_image, (640, 480))

            # Predicting the psf parameters of length and angle.
            img = cv2.resize(self.create_fft(ip_image), (224, 224))
            img = np.expand_dims(img_to_array(img), axis=0) / 255.0
            preds = model_angle.predict(img)
            angle_value = np.mean(np.argsort(preds[0])[-3:])

            print("Predicted Blur Angle: ", angle_value)
            length_value = model_length.predict(img)[0][0]
            print("Predicted Blur Length: ", length_value)

            op_image = self.process(ip_image, length_value, angle_value)
            op_image = (op_image * 255).astype(np.uint8)
            op_image = (255 / (np.max(op_image) - np.min(op_image))) * (op_image - np.min(op_image))

            op_image_path = './result_new.png'
            #------------------------------------------------------------------
            # Passing the  op_image directly to convert_frame_to_texture() does
            # not work for some reason. The displayed image is garbage. This
            # hack works for now to write/read the output image and then pass
            # it to convert_frame_to_texture().
            #------------------------------------------------------------------
            cv2.imwrite(op_image_path, op_image)
            op_image = cv2.imread(op_image_path)

            # After processing, convert the output image to a texture.
            texture_result = self.convert_frame_to_texture(op_image)

            # Schedule a function to be called every frame until texture_result is not None.
            Clock.schedule_interval(self.update_image_texture, 0)

    def update_image_texture(self, dt):
        global texture_result
        if texture_result is not None:
            self.image.texture = texture_result
            return False  # This stops the function from being called again.


class FaceRecognitionPopup(BasePopup):
    def __init__(self, main_layout, **kwargs):
        super(FaceRecognitionPopup, self).__init__(main_layout, **kwargs)
        self.title = "Facial Recognition"

        # These models load fast, so ok here on start-up.
        self.face_detetcion_model = './models/face_detection_yunet_2022mar_int8.onnx'
        self.face_recognition_model = './models/face_recognition_sface_2021dec_int8.onnx'
        self.target_image = './input_media/target_image.png'

        self.content = BoxLayout(orientation="vertical", spacing=layout_padding_y)

        slider_layout, self.slider = self.create_labeled_slider("Similarity Threshold: ", 1.00, 1.30, 1.20,
                                                                value_format='{:.2f}', rounding=2, size_hint_label=0.4, font_size=font_size_slider)
        slider_box = AnchorLayout(anchor_x='center', anchor_y='center', size_hint=(0.35, 1))
        slider_box.add_widget(slider_layout)

        image_and_buttons_box = BoxLayout(orientation="horizontal")
        image_and_buttons_box.add_widget(slider_box)

        self.image = Image(allow_stretch=True, size_hint_y=0.7)
        image_box = BoxLayout(orientation="vertical")
        image_box.add_widget(self.image)
        image_and_buttons_box.add_widget(image_box)

        self.content.add_widget(image_and_buttons_box)

        self.close_button = Button(text="Close", size_hint_y=0.1)
        self.close_button.bind(on_press=self.close_popup)
        self.content.add_widget(self.close_button)
        Clock.schedule_interval(self.process_image, frame_rate)

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
        frameHeight = 420
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

                texture = self.convert_frame_to_texture(frame)
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

                # cosine_similarity_threshold = 0.363
                # l2_similarity_threshold = 1.128
                l2_similarity_threshold = similarity_threshold

                #cosine_score = recognizer.match(stream_face_feature, target_face_feature, cv2.FaceRecognizerSF_FR_COSINE)
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

                texture = self.convert_frame_to_texture(frame)
                self.image.texture = texture

class EdgeDetectionPopup(BasePopup):
    def __init__(self, main_layout, **kwargs):
        super(EdgeDetectionPopup, self).__init__(main_layout, **kwargs)
        self.title = "Canny Edge Detection"

        self.content = BoxLayout(orientation="vertical", spacing=layout_padding_y)
        video_and_controls = BoxLayout(orientation="horizontal")

        sliders_layout = BoxLayout(orientation="horizontal", size_hint_x=0.2, padding=[10, 0])

        slider1_layout, self.lower_threshold_slider = self.create_labeled_slider("Lower T: ", 0, 255, 100,
                                                                                 value_format='{}', rounding=0, font_size=font_size_slider)
        slider1_box = AnchorLayout(anchor_x='center', anchor_y='center')
        slider1_box.add_widget(slider1_layout)
        sliders_layout.add_widget(slider1_box)

        slider2_layout, self.upper_threshold_slider = self.create_labeled_slider("Upper T: ", 0, 255, 200,
                                                                                 value_format='{}', rounding=0, font_size=font_size_slider)
        slider2_box = AnchorLayout(anchor_x='center', anchor_y='center')
        slider2_box.add_widget(slider2_layout)
        sliders_layout.add_widget(slider2_box)

        video_and_controls.add_widget(sliders_layout)

        self.image = Image(allow_stretch=True, size_hint_x=0.8)
        video_and_controls.add_widget(self.image)

        self.content.add_widget(video_and_controls)

        button_layout = BoxLayout(size_hint_y=0.1)
        self.content.add_widget(button_layout)

        self.close_button = Button(text="Close")
        self.close_button.bind(on_press=self.close_popup)
        button_layout.add_widget(self.close_button)

        Clock.schedule_interval(self.process_image, frame_rate)

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
            texture = self.convert_frame_to_texture(colored_edge_frame)
            self.image.texture = texture

    @staticmethod
    def apply_canny_edge_detection(img, lower_threshold, upper_threshold):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray, lower_threshold, upper_threshold)


class OCRTranslationPopup(BasePopup):
    def __init__(self, main_layout, **kwargs):
        super(OCRTranslationPopup, self).__init__(main_layout, **kwargs)
        self.title = "OCR + Translation"

        self.content = BoxLayout(orientation="horizontal",
                                 spacing=layout_padding_y)  # Changed orientation to "horizontal"
        self.input_source = None

        # Added a new BoxLayout with "vertical" orientation to hold the image and the buttons
        self.image_and_buttons = BoxLayout(orientation="vertical", size_hint_x=0.6)
        self.image = Image(source=self.input_source, allow_stretch=True)
        self.image_and_buttons.add_widget(self.image)

        button_layout = BoxLayout(size_hint_y=0.1)
        self.image_and_buttons.add_widget(button_layout)

        self.close_button = Button(text="Close")
        self.close_button.bind(on_press=self.close_popup)
        button_layout.add_widget(self.close_button)

        self.load_button = Button(text="Load Image")
        self.load_button.bind(on_press=self.load_image)
        button_layout.add_widget(self.load_button)

        self.process_button = Button(text="Process image")
        self.process_button.bind(on_press=self.process_image)
        button_layout.add_widget(self.process_button)

        self.filechooser = FileChooserListView(
            path=default_media_path,
            size_hint_x=0.4)
        self.content.add_widget(self.filechooser)

        self.content.add_widget(self.image_and_buttons)

        # Create a Translator Object.
        self.translator = googletrans.Translator()

        vocabulary = []
        try:
            with open("./models/alphabet_94.txt") as f:
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

        # Set threshold for Binary Map creation and polygon detection.
        binThresh = 0.3
        polyThresh = 0.5

        mean = (122.67891434, 116.66876762, 104.00698793)
        self.inputSize = (480, 480)

        textDetector.setBinaryThreshold(binThresh).setPolygonThreshold(polyThresh)
        textDetector.setInputParams(1.0 / 255, self.inputSize, mean, True)

        textRecognizer.setDecodeType("CTC-greedy")
        textRecognizer.setVocabulary(vocabulary)
        textRecognizer.setInputParams(1 / 127.5, (100, 32), (127.5, 127.5, 127.5), True)

    def load_image(self, instance):
        if len(self.filechooser.selection) > 0:
            self.input_source = self.filechooser.selection[0]
            self.image.source = self.input_source

    def draw_label_banner_ocr(self, frame, text, lower_left, font_color=(0, 0, 0), fill_color=(255, 255, 255), font_scale=1,
                              font_thickness=1):
        """
        Annotate the image frame with a text banner overlaid on a filled rectangle.
        :param frame: Input image frame.
        :param text: Text string to annotate on frame.
        :param lower_left: (x,y) coordinates for the lower-left corner of the text block.
        :param font_color: Font color for the annotaed text string.
        :param fill_color: Fill color for the background rectangle.
        :param font_scale: Font scale for the annotated text.
        :param font_thickness: Font thickness for the annotated text.
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

    def fourPointsTransform(self, frame, vertices):
        """
        # This function performs a transformation for each bounding box detected by the text detection model.
        :param frame: Input image frame with detected text.
        :param vertices: Verticies of polygon for detected text block.
        :return: Annotated image frame.
        """
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

    def recognizeTranslateText(self, image, dest='en', src=None):
        """
        # Perform Language Translation on Recognized Text.
        :param image: Input image frame.
        :param dest: Destination langauge (default is English).
        :param src: Source langauge (default: unspecified)
        :return: Annotated image frame with text bounding boxes and translated text.
        """

        # Use the DB text detector initialized previously to detect the presence of text in the image.
        boxes, confs = textDetector.detect(image)

        image = cv2.resize(image, self.inputSize)

        if boxes is not None:
            # Draw the bounding boxes of text detected.
            cv2.polylines(image, boxes, True, (255, 0, 255), 3)

        # Iterate through the bounding boxes detected by the text detector model
        for box in boxes:

            # Apply transformation on the bounding box detected by the text detection algorithm.
            croppedRoi = self.fourPointsTransform(image, box)

            # Recognize the text using the crnn model.
            recognizedText = textRecognizer.recognize(croppedRoi)

            if recognizedText is not None and recognizedText.strip() != '':
                if src is not None:
                    translation = self.translator.translate(recognizedText, dest, src)
                else:
                    translation = self.translator.translate(recognizedText, dest)

                if translation is not None:
                    pad_x = 10
                    shift_y = 10
                    px = int(np.max(box[0:4, 0])) + pad_x
                    py = int(np.average(box[0:4, 1])) + shift_y
                    lower_left = (px, py)
                    self.draw_label_banner_ocr(image, translation.text, lower_left, font_color=(255, 255, 255),
                                               fill_color=(255, 0, 0), font_scale=0.7, font_thickness=2)
            else:
                print("No text was recognized in the frame.")

        return image
    def process_image(self, instance):
        Clock.schedule_once(self.process_image_thread, 0)

    def process_image_thread(self, dt, *args):

        global texture_result
        texture_result = None  # Reset to None before processing starts.

        frame = self.get_latest_frame()
        if frame is None:
            return
        else:

            ip_image = cv2.imread(self.input_source)
            ip_image = cv2.resize(ip_image, self.inputSize)

            op_image = self.recognizeTranslateText(ip_image, src='ru')

            # After processing, convert the output image to a texture.
            texture_result = self.convert_frame_to_texture(op_image)

            # Schedule a function to be called every frame until texture_result is not None.
            Clock.schedule_interval(self.update_image_texture, 0)

    def update_image_texture(self, dt):
        global texture_result
        if texture_result is not None:
            self.image.texture = texture_result
            return False  # This stops the function from being called again.

class BinaryDecoderPopup(BasePopup):
    def __init__(self, main_layout, **kwargs):
        super(BinaryDecoderPopup, self).__init__(main_layout, **kwargs)
        self.title = "Decode Binary Code"

        self.content = BoxLayout(orientation="vertical", spacing=layout_padding_y)

        self.image = Image(allow_stretch=True, size_hint_y=0.7)
        self.content.add_widget(self.image)

        button_layout = BoxLayout(size_hint_y=0.1)
        self.content.add_widget(button_layout)

        self.close_button = Button(text="Close")
        self.close_button.bind(on_press=self.close_popup)
        button_layout.add_widget(self.close_button)

        # Create a Translator Object.
        self.translator = googletrans.Translator()

        vocabulary = []
        try:
            with open("./models/alphabet_01.txt") as f:
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

        # Set threshold for Binary Map creation and polygon detection
        binThresh = 0.3
        polyThresh = 0.5

        mean = (122.67891434, 116.66876762, 104.00698793)
        self.inputSize = (480, 480)

        textDetector.setBinaryThreshold(binThresh).setPolygonThreshold(polyThresh)
        textDetector.setInputParams(1.0 / 255, self.inputSize, mean, True)

        textRecognizer.setDecodeType("CTC-greedy")
        textRecognizer.setVocabulary(vocabulary)
        textRecognizer.setInputParams(1 / 127.5, (100, 32), (127.5, 127.5, 127.5), True)

        Clock.schedule_interval(self.process_image, frame_rate)

    def binaryToDecimal(self, binary):

        decimal, i = 0, 0

        while (binary != 0):
            dec = binary % 10
            decimal = decimal + dec * pow(2, i)
            binary = binary // 10
            i += 1
        return decimal

    def draw_label_banner_ocr (self, frame, text, lower_left, font_color=(0, 0, 0), fill_color=(255, 255, 255), font_scale=1,
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

    # Align text boxes.
    # This Function does transformation over the bounding boxes detected by the text detection model
    def fourPointsTransform(self, frame, vertices):
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

    def recognizeText(self, image, dest='en', src='', debug=False):

        image = cv2.resize(image, self.inputSize)

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
            croppedRoi = self.fourPointsTransform(image, box)

            # Recognise the text using the crnn model
            recognizedText = textRecognizer.recognize(croppedRoi)

            if recognizedText is not None and recognizedText.strip() != '':

                # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # pt_code = pytesseract.image_to_string(gray, config="--psm 4")

                # Type conversion operations
                for item in recognizedText:
                    code.append(str(item))
                binary = "".join(code)
                binary = int(binary)
                # print("code:   ", str(code))
                # print("binary: ", binary)
                # print("intbin: ", int(binary))
                decoded_result = str(self.binaryToDecimal(binary))

                pad_x = 10
                shift_y = 10
                px = int(np.max(box[0:4, 0])) + pad_x
                py = int(np.average(box[0:4, 1])) + shift_y

                lower_left = (px, py)

                self.draw_label_banner_ocr(image, decoded_result, lower_left, font_color=(255, 255, 255),
                                           fill_color=(255, 0, 0), font_scale=1, font_thickness=2)

            else:
                print("No text was recognized in the frame.")

        return image

    def process_image(self, dt, *args):

        frame = self.get_latest_frame()
        if frame is None:
            return
        else:
            # Perform inference on input image.
            frame = self.recognizeText(frame, src='en')
            if frame is not None:
                texture = self.convert_frame_to_texture(frame)
                self.image.texture = texture

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

class STEMKitv2App(App):
    def build(self):
        if (mode == 'LT'):
            Window.size = (800, 600)
        else:
            Window.size = (620, 420)
        return MainLayout()

    def on_stop(self):
        self.root.on_stop()

if __name__ == "__main__":
    STEMKitv2App().run()