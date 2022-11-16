import sys
import os

from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

absolute_path = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(absolute_path, "videos/countdown.mp4")


class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Platform50 Launch")
        # self.setGeometry(350, 100, 700, 500)
        # self.showMaximized()
        self.showFullScreen()
        p = self.palette()
        p.setColor(QPalette.Window, Qt.black)
        self.setPalette(p)

        self.play_video()

    def play_video(self):
        # QMediaPlayer
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(video_path)))

        # Set widget
        self.videoWidget = QVideoWidget()
        self.setCentralWidget(self.videoWidget)
        self.mediaPlayer.setVideoOutput(self.videoWidget)

        self.mediaPlayer.mediaStatusChanged.connect(self.media_status)
        # Play
        self.mediaPlayer.play()

    def media_status(self, status):  # <---
        if status == 7:
            print("The End!")
            self.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
