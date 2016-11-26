import sys
from PyQt4.QtCore import *
from PyQt4 import QtGui
from PyQt4.QtGui import *
import threading
from read_files import read_DB
from read_files import plot_waves_comparison
from read_files import all_data_file

class Window(QtGui.QMainWindow):
    subject1=""
    subject2=""
    subject3=""

def __init__(self):
    super(Window, self).__init__()
    self.setGeometry(50, 30, 430, 300)
    self.setWindowTitle("Proyecto TISC")
    button1 = QPushButton("Leer Base de Datos", self)
    button1.clicked.connect(lambda:self.start())
    button1.setGeometry(142, 30, 150, 50)
    self.progress = QtGui.QProgressBar(self)
    self.progress.setGeometry(132, 100, 190, 10)
    l1= QtGui.QLabel("Seleccione los sujetos a comparar",self)
    l1.setGeometry(125, 120,251,20)
    box1 = QtGui.QComboBox(self)
    box1.setGeometry(65,160,90,20)
    box2 = QtGui.QComboBox(self)
    box2.setGeometry(160, 160, 90, 20)
    box3 = QtGui.QComboBox(self)
    box3.setGeometry(255, 160, 90, 20)
    for index in all_data_file:
        box1.addItem(index)
        box2.addItem(index)
        box3.addItem(index)
        self.subject1 = str(box1.currentText())
        self.subject2 = str(box2.currentText())
        self.subject3 = str(box3.currentText())
        button2 = QPushButton("Generar Grafica", self)
        button2.clicked.connect(lambda:self.plot(box1,box2,box3))
        button2.setGeometry(142, 220, 130, 50)
        self.show()

def plot(self,s1,s2,s3):
    self.subject1 = str(s1.currentText())
    self.subject2 = str(s2.currentText())
    self.subject3 = str(s3.currentText())
    plot_waves_comparison(self.subject1,self.subject2,self.subject3)

def start(self):
    read_DB()
print "End Read Data"


def run():
    app = QtGui.QApplication(sys.argv)
    GUI = Window()
    sys.exit(app.exec_())

run()