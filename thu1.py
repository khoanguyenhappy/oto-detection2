import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer
import cv2
import os
from video_extractor_ui import Ui_MainWindow


class VideoApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.btnOpenVideo.clicked.connect(self.openVideo)

        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self.playVideo)

        self.frame_extract_timer = QTimer(self)
        self.frame_extract_timer.timeout.connect(self.extractFrame)

        self.capture = None
        self.frame_rate = 1
        self.save_dir = 'extracted_frames'
        self.frame_count = 0  # Biến đếm số thứ tự khung hình
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def openVideo(self):
        video_path, _ = QFileDialog.getOpenFileName(self, 'Open Video File', '', 'Video Files (*.mp4 *.avi *.mov)')
        if video_path:
            self.capture = cv2.VideoCapture(video_path)
            self.frame_rate = int(self.capture.get(cv2.CAP_PROP_FPS))
            self.video_timer.start(1000 // self.frame_rate)
            self.frame_extract_timer.start(1000)

    def playVideo(self):
        if self.capture and self.capture.isOpened():
            ret, frame = self.capture.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channel = frame.shape
                step = channel * width
                qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
                self.ui.lblVideo.setPixmap(QPixmap.fromImage(qImg))
            else:
                self.video_timer.stop()
                self.frame_extract_timer.stop()

    def extractFrame(self):
        if self.capture and self.capture.isOpened():
            current_pos = self.capture.get(cv2.CAP_PROP_POS_FRAMES)
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, current_pos - 1)
            ret, frame = self.capture.read()
            if ret:
                frame_path = os.path.join(self.save_dir, f'frame_{self.frame_count}.jpg')
                cv2.imwrite(frame_path, frame)
                self.frame_count += 1  # Tăng biến đếm số thứ tự
            else:
                self.frame_extract_timer.stop()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = VideoApp()
    mainWindow.show()
    sys.exit(app.exec_())
