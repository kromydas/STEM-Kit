import os
import numpy as np
import cv2
import onnxruntime
import kivy
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

kivy.require("2.0.0")

# Set the position of the window
Window.left = 100  # horizontal position
Window.top = 500   # vertical position (distance from the top of the screen)


class MainLayout(BoxLayout):
    def __init__(self, **kwargs):
        super(MainLayout, self).__init__(**kwargs)

        self.orientation = "vertical"

        self.modules_layout = GridLayout(cols=2, rows=4)
        self.add_widget(self.modules_layout)

        button_names = [
            "Color Segmentation",
            "Edge Detection",
            "Text Detection",
            "Object Detection",
            "Super Resolution",
            "Module 6",
            "Module 7",
            "Module 8",
        ]

        for i in range(len(button_names)):
            btn = Button(text=button_names[i])
            btn.bind(on_press=self.open_module_popup)
            self.modules_layout.add_widget(btn)

        self.capture = cv2.VideoCapture(0)

    def open_module_popup(self, instance):
        module_name = instance.text

        if module_name == "Color Segmentation":
            popup = ColorSegmentationPopup(self)
        elif module_name == "Edge Detection":
            popup = CannyPopup(self)
        elif module_name == "Text Detection":
            popup = TextDetectionPopup(self)
        elif module_name == "Object Detection":
            popup = ObjectDetectionPopup(self)
        elif module_name == "Super Resolution":
            popup = SuperResolutionPopup(self)
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
    def convert_frame_to_texture(frame):
        buf = cv2.flip(frame, 0).tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        return texture

    def create_labeled_slider(self, label_text, min_value, max_value, initial_value, value_format='{}'):
        layout = BoxLayout(orientation="horizontal", size_hint_y=None, height="30dp")

        label = Label(text=label_text, size_hint_x=0.2)
        layout.add_widget(label)

        slider = Slider(min=min_value, max=max_value, value=initial_value, size_hint_x=0.85)
        layout.add_widget(slider)

        value_label = Label(text=value_format.format(initial_value), size_hint_x=0.1)
        layout.add_widget(value_label)

        slider.bind(value=lambda instance, value: setattr(value_label, 'text', value_format.format(int(value))))

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

class ColorSegmentationPopup(BasePopup):
    def __init__(self, main_layout, **kwargs):
        super(ColorSegmentationPopup, self).__init__(main_layout, **kwargs)
        self.title = "Adjust Color Thresholds"

        self.content = BoxLayout(orientation="vertical", spacing=40)

        self.image = Image(allow_stretch=True, size_hint_y=0.7)
        self.content.add_widget(self.image)

        slider1_layout, self.lower_threshold_slider = self.create_labeled_slider("Lower Threshold: ", 0, 255, 100)
        slider2_layout, self.upper_threshold_slider = self.create_labeled_slider("Upper Threshold: ", 0, 255, 200)

        slider_box = BoxLayout(orientation="vertical", size_hint_y=0.1)
        self.content.add_widget(slider_box)
        slider_box.add_widget(slider1_layout)
        slider_box.add_widget(slider2_layout)

        button_layout = BoxLayout(size_hint_y=0.1)
        self.content.add_widget(button_layout)

        self.close_button = Button(text="Close")
        self.close_button.bind(on_press=self.close_popup)
        button_layout.add_widget(self.close_button)

        self.process_button = Button(text="Process image")
        self.process_button.bind(on_press=self.process_image)
        button_layout.add_widget(self.process_button)

    def process_image(self, *args):
        ret, frame = self.capture.read()

        if ret:
            lower_threshold = self.lower_threshold_slider.value
            upper_threshold = self.upper_threshold_slider.value

            img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            g_lb = np.array([lower_threshold, 10, 10], np.uint8)
            g_ub = np.array([upper_threshold, 255, 255], np.uint8)

            g_mask = cv2.inRange(img_hsv, g_lb, g_ub)
            adjusted_frame = cv2.bitwise_and(frame, frame, mask=g_mask)
            texture = self.convert_frame_to_texture(adjusted_frame)
            self.image.texture = texture

class CannyPopup(BasePopup):
    def __init__(self, main_layout, **kwargs):
        super(CannyPopup, self).__init__(main_layout, **kwargs)
        self.title = "Canny Edge Detection"

        self.content = BoxLayout(orientation="vertical", spacing=40)

        self.image = Image(allow_stretch=True, size_hint_y=0.7)
        self.content.add_widget(self.image)

        slider1_layout, self.lower_threshold_slider = self.create_labeled_slider("Lower Threshold: ", 0, 255, 100)
        slider2_layout, self.upper_threshold_slider = self.create_labeled_slider("Upper Threshold: ", 0, 255, 200)

        slider_box = BoxLayout(orientation="vertical", size_hint_y=0.1)
        self.content.add_widget(slider_box)
        slider_box.add_widget(slider1_layout)
        slider_box.add_widget(slider2_layout)

        button_layout = BoxLayout(size_hint_y=0.1)
        self.content.add_widget(button_layout)

        self.close_button = Button(text="Close")
        self.close_button.bind(on_press=self.close_popup)
        button_layout.add_widget(self.close_button)

        self.process_button = Button(text="Process image")
        self.process_button.bind(on_press=self.process_image)
        button_layout.add_widget(self.process_button)

    def process_image(self, *args):
        ret, frame = self.capture.read()

        if ret:
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


class TextDetectionPopup(BasePopup):
    def __init__(self, main_layout, **kwargs):
        super(TextDetectionPopup, self).__init__(main_layout, **kwargs)
        self.title = "Adjust Text Detection"

        self.content = BoxLayout(orientation="vertical", spacing=40)

        self.image = Image(allow_stretch=True, size_hint_y=0.7)
        self.content.add_widget(self.image)

        slider_layout, self.slider = self.create_labeled_slider("Text Detection: ", 1, 100, 60, value_format='{}%')

        slider_box = BoxLayout(size_hint_y=0.05)
        self.content.add_widget(slider_box)
        slider_box.add_widget(slider_layout)

        button_layout = BoxLayout(size_hint_y=0.1)
        self.content.add_widget(button_layout)

        self.close_button = Button(text="Close")
        self.close_button.bind(on_press=self.close_popup)
        button_layout.add_widget(self.close_button)

        self.process_button = Button(text="Process image")
        self.process_button.bind(on_press=self.process_image)
        button_layout.add_widget(self.process_button)

    def process_image(self, *args):
        ret, frame = self.capture.read()

        if ret:
            # Set input image size.
            inputSize = (320, 320)

            # Load pre-trained models.
            # East model for text-detection
            textDetectorEAST = cv2.dnn_TextDetectionModel_EAST("./models/frozen_east_text_detection.pb")

            # Set the Detection Confidence Threshold and NMS threshold
            conf_thresh = self.slider.value/100.0
            print("conf_thresh: ", conf_thresh)
            nms_thresh = 0.4

            textDetectorEAST.setConfidenceThreshold(conf_thresh).setNMSThreshold(nms_thresh)
            textDetectorEAST.setInputParams(1.0, inputSize, (123.68, 116.78, 103.94), True)

            # # DB model for text-detection based on resnet50
            # textDetectorDB50 = cv2.dnn_TextDetectionModel_DB("./resources/DB_TD500_resnet50.onnx")
            # # DB model for text-detection based on resnet18
            # textDetectorDB18 = cv2.dnn_TextDetectionModel_DB("./resources/DB_TD500_resnet18.onnx")

            # Set threshold for Binary Map creation and polygon detection
            # bin_thresh = 0.3
            # poly_thresh = 0.5

            # mean = (122.67891434, 116.66876762, 104.00698793)

            # textDetectorDB18.setBinaryThreshold(bin_thresh).setPolygonThreshold(poly_thresh)
            # textDetectorDB18.setInputParams(1.0 / 255, inputSize, mean, True)
            #
            # textDetectorDB50.setBinaryThreshold(bin_thresh).setPolygonThreshold(poly_thresh)
            # textDetectorDB50.setInputParams(1.0 / 255, inputSize, mean, True)

            # Making copies of the original image
            imEAST = frame.copy()
            # imDB18 = image.copy()
            # imDB50 = image.copy()

            # Use the East text detector to detect the presence of text in the image
            boxesEAST, confsEAST = textDetectorEAST.detect(frame)

            # # Use the DB18 text detector to detect the presence of text in the image
            # boxesDB18, confsDB18 = textDetectorDB18.detect(image)
            #
            # # Use the DB50 text detector to detect the presence of text in the image
            # boxesDB50, confsDB50 = textDetectorDB50.detect(image)

            # Inspect the output of one of the detected text boxes
            # print(boxesEAST[0])

            # Draw the bounding boxes of text detected using EAST.
            cv2.polylines(imEAST, boxesEAST, isClosed=True, color=(255, 0, 255), thickness=4)

            # # Draw the bounding boxes of text detected using DB18.
            # cv2.polylines(imDB18, boxesDB18, True, (255, 0, 255), 4)
            # cv2.imshow('Bounding boxes for DB18', imDB18)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            #
            # # Draw the bounding boxes of text detected using DB50.
            # cv2.polylines(imDB50, boxesDB50, True, (255, 0, 255), 4)
            # cv2.imshow('Bounding boxes for DB50', imDB50)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # output = cv2.hconcat([image, imEAST, imDB18, imDB50])
            # cv2.imwrite('./visuals/english_signboard_detected.jpg', output)
            # cv2.imshow('Original | EAST | DB18 | DB50', cv2.resize(output, None, fx=0.6, fy=0.6))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # adjusted_frame = cv2.bitwise_and(frame, frame, mask=g_mask)
            adjusted_frame = imEAST
            texture = self.convert_frame_to_texture(adjusted_frame)
            self.image.texture = texture
#------------------------------------------------------------------------------
# Object Detection
#------------------------------------------------------------------------------
class ObjectDetectionPopup(BasePopup):
    def __init__(self, main_layout, **kwargs):
        super(ObjectDetectionPopup, self).__init__(main_layout, **kwargs)
        self.title = "Adjust Detection Threshold"

        self.content = BoxLayout(orientation="vertical", spacing=40)

        self.image = Image(allow_stretch=True, size_hint_y=0.7)
        self.content.add_widget(self.image)

        # slider_layout, self.slider = self.create_labeled_slider("Brightness", 0, 100, 50)
        slider_layout, self.slider = self.create_labeled_slider("Confidence: ", 1, 100, 60, value_format='{}%')

        slider_box = BoxLayout(size_hint_y=0.05)
        self.content.add_widget(slider_box)
        slider_box.add_widget(slider_layout)

        button_layout = BoxLayout(size_hint_y=0.1)
        self.content.add_widget(button_layout)

        self.close_button = Button(text="Close")
        self.close_button.bind(on_press=self.close_popup)
        button_layout.add_widget(self.close_button)

        self.process_button = Button(text="Process image")
        self.process_button.bind(on_press=self.process_image)
        button_layout.add_widget(self.process_button)

        self.INPUT_WIDTH = 640
        self.INPUT_HEIGHT = 640
        self.SCORE_THRESHOLD = 0.25
        self.NMS_THRESHOLD = 0.45
        self.CONFIDENCE_THRESHOLD = 0.45

        # Text parameters.
        self.FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
        self.FONT_SCALE = 0.7
        self.THICKNESS = 1

        # Colors
        self.BLACK = (0, 0, 0)
        self.BLUE = (255, 178, 50)
        self.YELLOW = (0, 255, 255)
        self.RED = (0, 0, 255)

        self.classes = ('classes.txt')

    # Constants.
    INPUT_WIDTH = 640


    def draw_label(self, input_image, label, left, top):
        """Draw text onto image at location."""

        # Get text size.
        text_size = cv2.getTextSize(label, self.FONT_FACE, self.FONT_SCALE, self.THICKNESS)
        dim, baseline = text_size[0], text_size[1]
        # Use text size to create a BLACK rectangle.
        cv2.rectangle(input_image, (left, top), (left + dim[0], top + dim[1] + baseline), self.BLACK, cv2.FILLED);
        # Display text inside the rectangle.
        cv2.putText(input_image, label, (left, top + dim[1]), self.FONT_FACE, self.FONT_SCALE, self.YELLOW, self.THICKNESS, cv2.LINE_AA)

    def preprocess(self, img):
        img = cv2.resize(img, (self.INPUT_WIDTH, self.INPUT_HEIGHT))
        image_data = np.array(img, dtype='float32')
        image_data /= 255.
        image_data = np.transpose(image_data, [2, 0, 1])
        image_data = np.expand_dims(image_data, 0)
        return image_data

    def post_process(self, input_image, outputs):
        # Lists to hold respective values while unwrapping.
        class_ids = []
        confidences = []
        boxes = []

        # Rows.
        rows = outputs[0].shape[1]

        image_height, image_width = input_image.shape[:2]

        # Resizing factor.
        x_factor = image_width / self.INPUT_WIDTH
        y_factor = image_height / self.INPUT_HEIGHT

        # Iterate through 25200 detections.
        for r in range(rows):
            row = outputs[0][0][r]
            confidence = row[4]

            # Discard bad detections and continue.
            if confidence >= self.CONFIDENCE_THRESHOLD:
                classes_scores = row[5:]

                # Get the index of max class score.
                class_id = np.argmax(classes_scores)

                #  Continue if the class score is above threshold.
                if (classes_scores[class_id] > self.SCORE_THRESHOLD):
                    confidences.append(confidence)
                    class_ids.append(class_id)

                    cx, cy, w, h = row[0], row[1], row[2], row[3]

                    left = int((cx - w / 2) * x_factor)
                    top = int((cy - h / 2) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)

                    # box = np.array([left, top, width, height]) \
                    box = [left, top, width, height]
                    boxes.append(box)

        # Perform non-maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.CONFIDENCE_THRESHOLD, self.NMS_THRESHOLD)
        for i in indices:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            cv2.rectangle(input_image, (left, top), (left + width, top + height), self.BLUE, 3 * self.THICKNESS)

            print("i:", i)
            print("class_ids:", class_ids)
            print("len(self.classes):", len(self.classes))

            label = "{}:{:.2f}".format(self.classes[class_ids[i]], confidences[i])

            print("label:", label)

            self.draw_label(input_image, label, left, top)
        return input_image

    def process_image(self, *args):
        ret, frame = self.capture.read()

        if ret:
            session = onnxruntime.InferenceSession('./models/yolov5n-640.onnx')

            imgarr = np.array(frame)

            img_data = self.preprocess(imgarr)

            # start = time.time()
            output = session.run(None, {session.get_inputs()[0].name: img_data})

            img = self.post_process(imgarr, output)

            texture = self.convert_frame_to_texture(img)
            self.image.texture = texture


class SuperResolutionPopup(BasePopup):
    def __init__(self, main_layout, **kwargs):
        super(SuperResolutionPopup, self).__init__(main_layout, **kwargs)
        self.title = "Super Resolution"

        self.content = BoxLayout(orientation="vertical", spacing=40)

        self.image = Image(allow_stretch=True, size_hint_y=0.7)
        self.content.add_widget(self.image)

        # slider_layout, self.slider = self.create_labeled_slider("Brightness", 0, 100, 50)
        slider_layout, self.slider = self.create_labeled_slider("Upsample Factor: ", 1, 5, 4, value_format='{}')

        slider_box = BoxLayout(size_hint_y=0.05)
        self.content.add_widget(slider_box)
        slider_box.add_widget(slider_layout)

        button_layout = BoxLayout(size_hint_y=0.1)
        self.content.add_widget(button_layout)

        self.close_button = Button(text="Close")
        self.close_button.bind(on_press=self.close_popup)
        button_layout.add_widget(self.close_button)

        self.process_button = Button(text="Process image")
        self.process_button.bind(on_press=self.process_image)
        button_layout.add_widget(self.process_button)

    def process_image(self, *args):
        ret, frame = self.capture.read()

        downsample = 32
        # Downsampling.
        frame = cv2.resize(frame, None, fx=1 / downsample, fy=1 / downsample)

        # frame = cv2.imread('images/image_1_downsampled.png')
        if ret:
            upsample_value = self.slider.value
            print("upsample_value: ", upsample_value)
            adjusted_frame = self.super_resolution(frame, upsample_value)
            texture = self.convert_frame_to_texture(adjusted_frame)
            self.image.texture = texture

    @staticmethod
    def super_resolution(img, value):
        # Super resolution instance.
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel('models/EDSR_x4.pb')
        # Set the model with the method and scale factor.
        sr.setModel('edsr', value)
        # Pass image through super resolution model.
        result_edsr = sr.upsample(img)
        return result_edsr

class BrightnessPopup(BasePopup):
    def __init__(self, main_layout, **kwargs):
        super(BrightnessPopup, self).__init__(main_layout, **kwargs)
        self.title = "Adjust Brightness"

        self.content = BoxLayout(orientation="vertical", spacing=40)

        self.image = Image(allow_stretch=True, size_hint_y=0.7)
        self.content.add_widget(self.image)

        # slider_layout, self.slider = self.create_labeled_slider("Brightness", 0, 100, 50)
        slider_layout, self.slider = self.create_labeled_slider("Confidence: ", 1, 100, 60, value_format='{}%')

        slider_box = BoxLayout(size_hint_y=0.05)
        self.content.add_widget(slider_box)
        slider_box.add_widget(slider_layout)

        button_layout = BoxLayout(size_hint_y=0.1)
        self.content.add_widget(button_layout)

        self.close_button = Button(text="Close")
        self.close_button.bind(on_press=self.close_popup)
        button_layout.add_widget(self.close_button)

        self.process_button = Button(text="Process image")
        self.process_button.bind(on_press=self.process_image)
        button_layout.add_widget(self.process_button)

    def process_image(self, *args):
        ret, frame = self.capture.read()

        if ret:
            brightness_value = self.slider.value
            print("brightness_value: ", brightness_value)
            adjusted_frame = self.adjust_brightness(frame, brightness_value)
            texture = self.convert_frame_to_texture(adjusted_frame)
            self.image.texture = texture

    @staticmethod
    def adjust_brightness(img, value):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, value)
        v[v > 255] = 255
        v[v < 0] = 0
        final_hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

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
        Window.size = (400, 300)
        return MainLayout()

    def on_stop(self):
        self.root.on_stop()

if __name__ == "__main__":
    STEMKitApp().run()