# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'video_player.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1147, 871)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.buttonChooseVideo = QtWidgets.QPushButton(self.centralwidget)
        self.buttonChooseVideo.setGeometry(QtCore.QRect(50, 180, 111, 51))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.buttonChooseVideo.setFont(font)
        self.buttonChooseVideo.setObjectName("buttonChooseVideo")
        self.labelVideo = QtWidgets.QLabel(self.centralwidget)
        self.labelVideo.setGeometry(QtCore.QRect(240, 90, 781, 481))
        self.labelVideo.setFrameShape(QtWidgets.QFrame.Box)
        self.labelVideo.setLineWidth(5)
        self.labelVideo.setText("")
        self.labelVideo.setObjectName("labelVideo")
        self.buttonSelectFolder = QtWidgets.QPushButton(self.centralwidget)
        self.buttonSelectFolder.setGeometry(QtCore.QRect(50, 260, 111, 51))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.buttonSelectFolder.setFont(font)
        self.buttonSelectFolder.setObjectName("buttonSelectFolder")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1147, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.buttonChooseVideo.setText(_translate("MainWindow", "Open Video"))
        self.buttonSelectFolder.setText(_translate("MainWindow", "Select Save"))
