import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
import cv2
import torch
from interface import Ui_MainWindow
import os

class CarDetectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.setWindowTitle("Car Detector")
        self.setGeometry(100, 100, 800, 600)

        # Load the custom model
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp18/weights/best.pt',
                                    force_reload=True)

        # Initialize variables
        self.current_image_index = 0
        self.image_files = []
        self.directory = None

        # Connect buttons to functions
        self.ui.pushButton.clicked.connect(self.open_image_folder)
        self.ui.preButton.clicked.connect(self.show_previous_image)
        self.ui.nextButton.clicked.connect(self.show_next_image)

    def open_image_folder(self):
        self.directory = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if self.directory:
            self.image_files = [f for f in os.listdir(self.directory) if f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg")]
            self.current_image_index = 0
            self.show_image(self.current_image_index)

    def show_image(self, index):
        if self.directory and index < len(self.image_files):
            file_path = os.path.join(self.directory, self.image_files[index])
            image = cv2.imread(file_path)
            results = self.model(image)
            annotated_image = cv2.cvtColor(results.render()[0], cv2.COLOR_BGR2RGB)
            h, w, c = annotated_image.shape
            q_image = QImage(annotated_image.data, w, h, 3 * w, QImage.Format_RGB888)
            self.ui.label.setPixmap(QPixmap.fromImage(q_image))

    def show_previous_image(self):
        if self.directory and self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_image(self.current_image_index)

    def show_next_image(self):
        if self.directory and self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.show_image(self.current_image_index)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    detector_app = CarDetectorApp()
    detector_app.show()
    sys.exit(app.exec_())