import sys
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from read_files import all_data_file


class tab(QTabWidget):
    subject = ""
    subject1=""
    subject2=""
    subject3=""
    def __init__(self, parent=None):
        super(tab, self).__init__(parent)
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.setGeometry(470, 150, 430, 300)
        self.addTab(self.tab1, "Tab 1")
        self.addTab(self.tab2, "Tab 2")
        self.tab1UI()
        self.tab2UI()
        self.setWindowTitle("Clasificacion patologica con EEG en el CAP")

    def tab1UI(self):
        layout = QFormLayout()
        h1 = QHBoxLayout()
        button1 = QPushButton("Leer Base de Datos", self)
        button1.clicked.connect(lambda: self.start())
        h1.addWidget(button1)
        h2 = QHBoxLayout()
        progress = QProgressBar()
        h2.addWidget(progress)
        h3 = QHBoxLayout()
        box1 = QComboBox()
        box2 = QComboBox()
        box3 = QComboBox()
        h3.addWidget(box1)
        h3.addWidget(box2)
        h3.addWidget(box3)
        for index in all_data_file:
            box1.addItem(index)
            box2.addItem(index)
            box3.addItem(index)
        h4 = QHBoxLayout()
        button2 = QPushButton("Generar Grafica")
        button2.clicked.connect(lambda: self.plot(box1, box2, box3))
        h4.addWidget(button2)
        layout.addRow(h1)
        layout.addRow(h2)
        layout.addRow(QLabel("Seleccione los sujetos a comparar"))
        layout.addRow(h3)
        layout.addRow(h4)
        self.setTabText(0, "Graficar")
        self.tab1.setLayout(layout)



    def tab2UI(self):
        layout = QFormLayout()
        v1 = QVBoxLayout()
        b1=QRadioButton("Adaboost usando kernels")
        b2=QRadioButton("SVM usando kernels")
        b3=QRadioButton("Random forest usando kernels")
        b4=QRadioButton("Adaboost usando fourier")
        b5=QRadioButton("SVM usando fourier.")
        b6=QRadioButton("Random forest usando fourier")
        v1.addWidget(b1)
        v1.addWidget(b2)
        v1.addWidget(b3)
        v1.addWidget(b4)
        v1.addWidget(b5)
        v1.addWidget(b6)
        h2 = QHBoxLayout()
        boxy = QComboBox()
        h2.addWidget(boxy)
        for index in all_data_file:
            boxy.addItem(index)
        h3 = QHBoxLayout()
        button = QPushButton("Clasificar")
        button.clicked.connect(lambda: self.clasificar(boxy,b1,b2,b3,b4,b5,b6))
        h3.addWidget(button)
        layout.addRow(QLabel("Seleccione el Clasificador"), v1)
        layout.addRow(h2)
        layout.addRow(h3)
        self.setTabText(1, "Clasificar")
        self.tab2.setLayout(layout)

    def clasificar(self,s1,b1,b2,b3,b4,b5,b6):
        b=0
        self.subject = str(s1.currentText())
        if b1.isChecked()== True:
            b=1
        if b2.isChecked()== True:
            b=2
        if b3.isChecked()== True:
            b=3
        if b4.isChecked()== True:
            b=4
        if b5.isChecked()== True:
            b=6
        if b6.isChecked()== True:
            b=6
        #funcionqueejecutaclasificador(b,self.subject)
        print "se calsificara el sujeto"
        print self.subject
        print "con el clasificador"
        print b
        print "--------"

    def plot(self,s1,s2,s3):
        self.subject1 = str(s1.currentText())
        self.subject2 = str(s2.currentText())
        self.subject3 = str(s3.currentText())
        #plot_waves_comparison(self.subject1,self.subject2,self.subject3)
        print "graficas generadas"
        print self.subject1
        print self.subject2
        print self.subject3
        print "--------"
    def start(self):
        #read_DB()
        print "base de datos leida"




def main():
    app = QApplication(sys.argv)
    ex = tab()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()