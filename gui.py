##==============================================================
# PySide2 API
#   https://doc.qt.io/qtforpython/api.html
##==============================================================
from PySide2.QtGui import QBitmap, QIcon, QPainter, QMouseEvent
from PySide2.QtCore import QSize, Qt
from PySide2.QtWidgets import QLabel, QMainWindow, QPushButton, QSizePolicy, QWidget, QBoxLayout

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUI()

    """ USER INTERFACE SETUP """
    def setupUI(self):
        # WINDOW SETUP
        self.setWindowTitle("Keras.QuickDraw")
        self.setMinimumSize(QSize(800, 600))
        self.setWindowIcon(QIcon("res/eon-icon.png"))

        # WINDOW CENTRAL
        self.widget_central = QWidget(self)
        self.layout_central = QBoxLayout(QBoxLayout.TopToBottom, self.widget_central)
        self.setCentralWidget(self.widget_central)


        # CENTRAL HEADER
        self.widget_header = QWidget(self.widget_central)
        self.widget_header_indicater = QLabel(parent = self.widget_header)
        self.widget_header_caption   = QLabel("(TARGET)", self.widget_header)
        self.widget_header_counter   = QLabel(parent = self.widget_header)
        
        self.layout_header = QBoxLayout(QBoxLayout.LeftToRight, self.widget_header)
        self.layout_header.addWidget(self.widget_header_indicater, 0)
        self.layout_header.addWidget(self.widget_header_caption  , 1, Qt.AlignCenter)
        self.layout_header.addWidget(self.widget_header_counter  , 0)


        # CENTRAL CANVAS
        canvas = QBitmap(400,300)
        self.widget_canvas = QLabel(parent=self.widget_central)
        self.widget_canvas.setPixmap(canvas)
        

        # CENTRAL BUTTONS
        self.widget_footer = QWidget(self.widget_central)
        self.widget_footer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.widget_footer_clear = QPushButton("Clear", self.widget_footer)

        self.layout_footer = QBoxLayout(QBoxLayout.LeftToRight, self.widget_footer)
        self.layout_footer.addWidget(self.widget_footer_clear, 1)


        self.layout_central.addWidget(self.widget_header, 0)
        self.layout_central.addWidget(self.widget_canvas, 1)
        self.layout_central.addWidget(self.widget_footer, 0)


        self.show()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        ...