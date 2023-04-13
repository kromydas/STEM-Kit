import os
import time
import subprocess
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.slider import Slider
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.graphics import Color, Rectangle
from kivy.core.window import Window
from kivy.uix.image import Image
from kivy.uix.video import Video
from kivy.uix.textinput import TextInput

# Set the window size (width, height)
Window.size = (1000, 600)

# Set the minimum and maximum allowed dimensions for the window
Window.minimum_width, Window.minimum_height = (400, 300)
Window.maximum_width, Window.maximum_height = (1200, 900)

Builder.load_file('main.kv')

class MainView(GridLayout):

    def __init__(self, **kwargs):
        super(MainView, self).__init__(**kwargs)
        self.cols = 2
        self.rows = 4
        self.spacing = [2, 2]

        script_summary = [
            "This application will allow you to detect motion in a video stream.",
            "This application will allow you monitor video for social distancing.",
            "This application will allow you segment regions of an image based on color.",
            "This application will allow you to do Module 4.",
            "This application will allow you to do Module 5.",
            "This application will allow you to do Module 6.",
            "This application will allow you to do Module 7.",
            "This application will allow you to do Module 8.",
        ]

        script_names = [
            "Intrusion_Detection.py",
            "Social_Distancing.py",
            "Color_Segmentation.py",
            "another_script.py",
            "another_script.py",
            "another_script.py",
            "another_script.py",
            "another_script.py"
        ]

        button_names = [
            "Intrusion Detection",
            "Social Distancing",
            "Color Segmentation",
            "Module 4",
            "Module 5",
            "Module 6",
            "Module 7",
            "Module 8",
        ]

        for i in range(len(script_names)):
            btn = Button(text=button_names[i])
            btn.bind(on_release=lambda instance, script=script_names[i], title=button_names[i], summary=script_summary[i]: self.open_application(instance, script, title, summary))
            self.add_widget(btn)

    def open_application(self, instance, script_name, title, summary):
        if script_name == "Color_Segmentation.py":
            color_values = [0, 0]
            popup_content = CustomPopupContentWithTextInputs(script_name=script_name, custom_message=summary,
                                                             color_values=color_values)
        else:
            popup_content = CustomPopupContent(script_name=script_name, custom_message=summary)

        popup = Popup(title="Application: " + title, content=popup_content, size_hint=(0.8, 0.8))
        popup.open()

class CustomPopupContent(BoxLayout):
    def __init__(self, script_name, custom_message, **kwargs):
        super(CustomPopupContent, self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.process = None

        # Add a container for the script output (image or video)
        self.output_container = BoxLayout(size_hint=(1, 0.6))
        self.add_widget(self.output_container)

        slider_name = ["Threshold-1", "Threshold-2"]
        slider_defaults = [30, 60]

        with self.canvas.before:
            Color(0.6, 0.6, 0.6, 1)  # Light grey color
            self.rect = Rectangle(size=self.size, pos=self.pos)

        self.bind(size=self._update_rect, pos=self._update_rect)

        self.add_widget(Label(text=custom_message, size_hint=(1, 0.1)))

        slider1 = Slider(min=1, max=100, value=slider_defaults[0], orientation='horizontal', size_hint=(1, 0.3))
        label1 = Label(text=self.slider_text(slider1, slider_name[0]), size_hint=(1, .1))
        self.add_widget(label1)
        slider1.bind(value=lambda instance, value: self.update_label_text(label1, slider1, slider_name[0]))
        self.add_widget(slider1)

        button_row = BoxLayout(orientation='horizontal', size_hint=(1, 0.15), pos_hint={'bottom': 1})
        button_row.add_widget(Button(text="Close", on_release=lambda x: (self.kill_script(), self.parent.parent.parent.dismiss())))
        if script_name == "Color_Segmentation.py":
            button_row.add_widget(Button(text="Execute",
                                         on_release=lambda x: self.handle_execute_script(script_name, slider1.value)))
        else:
            button_row.add_widget(
                Button(text="Execute", on_release=lambda x: self.handle_execute_script(script_name, slider1.value)))

        self.add_widget(button_row)

    def slider_text(self, slider, slider_name):
        return slider_name + f": {slider.value:.0f} (min: {slider.min}, max: {slider.max})"

    def update_label_text(self, label, slider, slider_name):
            label.text = self.slider_text(slider, slider_name)

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def execute_script(self, script_name, slider1_value, text_input_values=None):
        print(f"Executing {script_name}")
        if isinstance(self, CustomPopupContentWithTextInputs):
            print("text_input_values: ", text_input_values)
            self.process = subprocess.Popen(["python", script_name] + [str(x) for x in text_input_values],
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE, text=True, bufsize=1)
        else:
            self.process = subprocess.Popen(["python", script_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                            text=True, bufsize=1)

        # Read the output and store the image path
        self.output_image_path = None
        output = self.process.communicate()
        for line in output[0].splitlines():
            print(line.strip())
            if line.startswith("Output image path:"):
                self.output_image_path = line.strip().split(":")[-1].strip()

    def handle_execute_script(self, script_name, slider1_value):
        if script_name == "Color_Segmentation.py":
            text_input_values = [t.text for t in self.children[2].children[::-1] if isinstance(t, TextInput)]
            print("Extracted text_input_values:", text_input_values)  # Add this line
            self.execute_script(script_name, slider1_value, text_input_values=text_input_values)

            # Wait for the script to finish executing
            while self.process.poll() is None:
                time.sleep(0.1)

            # Display the output image
            if self.output_image_path:
                self.display_image(self.output_image_path)
        else:
            super().handle_execute_script(script_name, slider1_value)

    def display_image(self, file_path):
        # Clear the output_container before adding new content
        self.output_container.clear_widgets()

        # Create an Image widget and load the file
        image = Image(source=file_path, nocache=True)
        image.reload()  # Force the image to reload
        self.output_container.add_widget(image)

    def display_video(self, file_path):
        # Clear the output_container before adding new content
        self.output_container.clear_widgets()

        # Create a Video widget, load the file, and set it to play and loop
        video = Video(source=file_path, play=True, loop=True)
        self.output_container.add_widget(video)

    def kill_script(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()

class CustomPopupContentWithTextInputs(CustomPopupContent):
    def __init__(self, script_name, custom_message, color_values, **kwargs):
        super(CustomPopupContentWithTextInputs, self).__init__(script_name, custom_message, **kwargs)
        self.color_values = color_values
        self.create_text_boxes()

    def create_text_boxes(self):
        # Create a new GridLayout to contain the text boxes and labels
        text_box_container = GridLayout(cols=2, rows=3, size_hint=(None, None), spacing=(50, 10), padding=(10, 0, 0, 0))
        text_box_container.size = (300, 180)  # Adjust the size of the container
        text_box_container.pos_hint = {'center_x': 0.5, 'center_y': 0.5}

        colors = ['Green Lower', 'Green Upper']

        # Create a list to store the TextInput widgets
        self.text_boxes = []

        # Create the three text boxes with labels and add them to the container
        for color in colors:
            text_box_label = Label(text=f'{color}:', size_hint_x=None, width=20, height=10)
            text_box = TextInput(input_filter='int', multiline=False, size_hint_x=None,
                                 width=100)  # Set the width property here
            self.text_boxes.append(text_box)
            text_box_container.add_widget(text_box_label)
            text_box_container.add_widget(text_box)

        # Add the text_box_container to the CustomPopupContentWithTextInputs instance
        self.add_widget(text_box_container, index=4)

    def handle_execute_script(self, script_name, slider1_value):
        if script_name == "Color_Segmentation.py":
            # Get the values directly from the TextInput widgets
            text_input_values = [t.text for t in self.text_boxes]
            print("Extracted text_input_values:", text_input_values)
            self.execute_script(script_name, slider1_value, text_input_values=text_input_values)

            # Wait for the script to finish executing
            while self.process.poll() is None:
                time.sleep(0.1)

            # Display the output image
            if self.output_image_path:
                self.display_image(self.output_image_path)
        else:
            super().handle_execute_script(script_name, slider1_value)

class RootWidget(BoxLayout):
    def __init__(self, **kwargs):
        super(RootWidget, self).__init__(**kwargs)
        self.orientation = 'vertical'

        # Set the background color (RGBA format)
        with self.canvas.before:
            Color(0.8, 0.8, 0.8, 1)  # Grey color
            self.rect = Rectangle(size=self.size, pos=self.pos)

        # Update the background rectangle when the window size changes
        self.bind(size=self._update_rect, pos=self._update_rect)

        # Add the text panel at the top
        text_panel = Label(text='Select an application to run...', size_hint=(1, 0.2), color=(0, 0, .8, 1))
        self.add_widget(text_panel)

        # Add the main view with the buttons
        main_view = MainView()
        self.add_widget(main_view)

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

class ComputerVisionUI(App):
    def build(self):
        # return MainView()
        return RootWidget()

if __name__ == '__main__':
    ComputerVisionUI().run()