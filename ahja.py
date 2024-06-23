import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
import cv2
import torch
from interface import Ui_MainWindow


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
        self.images = []

        # Connect buttons to functions
        self.ui.pushButton.clicked.connect(self.open_image)
        self.ui.preButton.clicked.connect(self.show_previous_image)
        self.ui.nextButton.clicked.connect(self.show_next_image)

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_path:
            image = cv2.imread(file_path)
            self.images.append(image)
            self.current_image_index = len(self.images) - 1
            self.show_image(self.current_image_index)

    def show_image(self, index):
        image = self.images[index]
        results = self.model(image)
        annotated_image = cv2.cvtColor(results.render()[0], cv2.COLOR_BGR2RGB)
        h, w, c = annotated_image.shape
        q_image = QImage(annotated_image.data, w, h, 3 * w, QImage.Format_RGB888)
        self.ui.label.setPixmap(QPixmap.fromImage(q_image))

    def show_previous_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_image(self.current_image_index)

    def show_next_image(self):
        if self.current_image_index < len(self.images) - 1:
            self.current_image_index += 1
            self.show_image(self.current_image_index)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    detector_app = CarDetectorApp()
    detector_app.show()
    sys.exit(app.exec_())
