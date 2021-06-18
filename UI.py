# import
import os
import sys
import cv2 as cv
import PyQt5.QtWidgets as qwd
from PyQt5 import QtGui as gui
import datetime
from imutils.video import VideoStream
import imutils
import cv2

# fileName=""
videoPath = ""
# MACRO DEFINITIONS
WINDOW_TITLE = "Drowsy Driver"
WINDOW_SIZE_X = 1280  # 1920
WINDOW_SIZE_Y = 720  # 1080
QDIALOG_IMAGE_PREFIX_TYPE = "*.png *.jpg *.jpeg *.JPG"
QDIALOG_IAMGE_SELECTION_DIALOG_NAME = "Select image"


class PicButton(qwd.QAbstractButton):
    def __init__(self, pixmap, parent=None):
        super(PicButton, self).__init__(parent)
        self.pixmap = pixmap

    def paintEvent(self, event):
        painter = gui.QPainter(self)
        painter.drawPixmap(event.rect(), self.pixmap)

    def sizeHint(self):
        return self.pixmap.size()


class firstGui(qwd.QDialog):

    def __init__(self, parent=None):
        super(firstGui, self).__init__(parent)
        self.setWindowTitle(WINDOW_TITLE)
        self.setGeometry(0, 0, 600, 600)
        self.InitUI()

    def InitUI(self):

        # Label OpenImg
        logData("creating main UI")

        self.setStyleSheet(""" firstGui{   background-repeat: no-repeat; 
       background-position: center;}""")

        btnClose = qwd.QPushButton('Close', self)
        btnClose.setToolTip("Close Program")
        btnClose.clicked.connect(self.BtnCloseSystem)
        btnClose.setStyleSheet("QPushButton"
                               "{"
                               "background-color : red ;color : white; border-radius: 10 ; border-color : black; border-width: 2px; border-bottom: 2px solid ;padding : 8;"
                               "}"
                               "QPushButton::pressed"
                               "{"


                               "}"
                               )

        # button open video
        btnOpenVideo = qwd.QPushButton("Start Script", self)
        btnOpenVideo.clicked.connect(self.startScript)
        btnOpenVideo.setStyleSheet("QPushButton"
                                   "{"
                                   "background-color : lightblue ;color : black; border-radius: 10 ; border-color : black; border-width: 2px; border-bottom: 2px solid ;padding : 8;"
                                   "}"
                                   "QPushButton::pressed"
                                   "{"


                                   "}"
                                   )

        # layouts
        lytQHBtns = qwd.QHBoxLayout()
        lytQHBtns.addWidget(btnOpenVideo)
        lytQHBtns.addWidget(btnClose)
        # Box Layout
        # image
        lytQBoxImg = qwd.QBoxLayout(qwd.QBoxLayout.TopToBottom, parent=None)
        # lytQBoxImg.addWidget(self.lblOpenImg)
        lytQVImg = qwd.QVBoxLayout()
        lytQVImg.addLayout(lytQBoxImg)
        # Line Edit + Lbl Layout
        lytQVLnedsAndLabels = qwd.QVBoxLayout()
        # Grid Layout
        lytQGridMain = qwd.QGridLayout()
        # lytQGridMain.addLayout(lytQVImg, 1, 0, 6, 1)
        # lytQGridMain.addLayout(lytQVLnedsAndLabels, 1, 0)
        # lytQGridMain.addLayout(lytQBoxTxt, 8, 0, 6, 1)
        lytQGridMain.addLayout(lytQHBtns, 7, 0)
        self.setLayout(lytQGridMain)

    def startScript(self):
        # put here the python file that will execute all stuff
        os.system("python main.py")
        
    def BtnCloseSystem(self):
        sys.exit()

    def SetSystemSearchedFlag(self, flag):
        self.isSystemSearched = flag


def logData(data):
    print("[" + str(datetime.datetime.now())+"]  " + data)


def main():
    cv.__version__
    app = qwd.QApplication(sys.argv)
    # main gui
    ex = firstGui()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':

    main()
