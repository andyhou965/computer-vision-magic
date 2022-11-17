import sys

from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


class Window(QMainWindow):
    def __init__(self, video_path, window_title=""):
        super().__init__()
        self.video_path = video_path
        self.setWindowTitle(window_title)
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
        self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(self.video_path)))

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
