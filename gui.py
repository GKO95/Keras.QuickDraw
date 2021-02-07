#===============================================================
# PySide2 API
#   https://doc.qt.io/qtforpython/api.html
#===============================================================
from PySide2.QtGui import QBitmap, QIcon, QPainter, QPen, QPaintEvent, QMouseEvent
from PySide2.QtCore import QSize, Qt
from PySide2.QtWidgets import QLabel, QMainWindow, QPushButton, QSizePolicy, QWidget, QBoxLayout

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.strokes = []
        self.setupUI()

    """ USER INTERFACE SETUP """
    def setupUI(self):
        # WINDOW SETUP
        self.setWindowTitle("Keras.QuickDraw")
        self.setMinimumSize(QSize(800, 600))
        self.setFixedSize(QSize(800, 600))
        self.setWindowIcon(QIcon("res/eon-icon.png"))

        # INITIALIZE: WINDOW CENTRAL
        self.widget_central = QWidget(self)
        self.layout_central = QBoxLayout(QBoxLayout.TopToBottom, self.widget_central)
        self.setCentralWidget(self.widget_central)

        # INITIALIZE: CENTRAL HEADER
        self.widget_header = QWidget(self.widget_central)
        self.widget_header_indicater = QLabel(parent = self.widget_header)
        self.widget_header_caption   = QLabel("(TARGET)", self.widget_header)
        self.widget_header_counter   = QLabel(parent = self.widget_header)
        
        self.layout_header = QBoxLayout(QBoxLayout.LeftToRight, self.widget_header)
        self.layout_header.addWidget(self.widget_header_indicater, 0, Qt.AlignLeft)
        self.layout_header.addWidget(self.widget_header_caption  , 1, Qt.AlignCenter)
        self.layout_header.addWidget(self.widget_header_counter  , 0, Qt.AlignRight)

        # INITIALIZE: CENTRAL BUTTONS
        self.widget_footer = QWidget(self.widget_central)
        self.widget_footer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.widget_footer_clear = QPushButton("Clear", self.widget_footer)

        self.layout_footer = QBoxLayout(QBoxLayout.LeftToRight, self.widget_footer)
        self.layout_footer.addWidget(self.widget_footer_clear, 1)

        # INITIALIZE: CENTRAL CANVAS
        self.widget_canvas = QLabel(parent=self.widget_central)
        self.widget_canvas.setStyleSheet("border: 1px solid black;")
        
        # LAYOUT: HEADER + CANVAS + FOOTER -> CENTRAL WINDOW CENTRAL
        self.layout_central.addWidget(self.widget_header, 0, Qt.AlignHCenter)
        self.layout_central.addWidget(self.widget_canvas, 1, Qt.AlignCenter)
        self.layout_central.addWidget(self.widget_footer, 0, Qt.AlignHCenter)

        # IMPLEMENT: QPaintDevice -> CENTRAL CANVAS
        canvas = QBitmap(self.width(), self.height() - (self.widget_header.height() + self.widget_footer.height()))
        canvas.clear()
        self.widget_canvas.setPixmap(canvas)

        self.show()

    """ EVENT: MOUSE CLICK/PRESS """
    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.x() > self.widget_canvas.x() and event.x() < (self.widget_canvas.x() + self.widget_canvas.width()) \
            and event.y() > self.widget_canvas.y() and event.y() < (self.widget_canvas.y() + self.widget_canvas.height()):
            self.strokes.append(list())
            self.strokes[-1].append((event.x(), event.y()))
        return super().mousePressEvent(event)

    """ EVENT: MOUSE MOVE WHILE PRESSED """
    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if event.x() > self.widget_canvas.x() and event.x() < (self.widget_canvas.x() + self.widget_canvas.width()) \
            and event.y() > self.widget_canvas.y() and event.y() < (self.widget_canvas.y() + self.widget_canvas.height()):
            self.strokes[-1].append((event.x(), event.y()))
        return super().mouseMoveEvent(event)

    """ EVENT: MOUSE RELEASE """
    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        return super().mouseReleaseEvent(event)

    """ EVENT: UPDATE PAINT 
        => Beware, every time paintEvent updates removes previous drawings.
        Therefore, store the data of previous drawing to prevent being erased.
    """
    def paintEvent(self, event: QPaintEvent) -> None:
        pen = QPen()
        pen.setWidth(4)
        painter = QPainter(self)
        painter.setPen(pen)
        if len(self.strokes) != 0:
            for stroke in self.strokes:
                if len(stroke) == 1:
                    painter.drawPoint(stroke[0][0], stroke[0][1])
                else:
                    for index in range(len(stroke) - 1):
                        painter.drawLine(stroke[index][0], stroke[index][1], stroke[index+1][0], stroke[index+1][1])
        painter.end()
        self.update()
        return super().paintEvent(event)


# https://forum.qt.io/topic/64693/unable-to-paint-on-qt-widget-shows-error-paintengine-should-no-longer-be-called/2