#===============================================================
# PySide2 API
#   https://doc.qt.io/qtforpython/api.html
#===============================================================
from PySide2.QtGui import QBitmap, QIcon, QPainter, QPen, QPaintEvent, QMouseEvent, QShowEvent
from PySide2.QtCore import QSize, Qt
from PySide2.QtWidgets import QLabel, QMainWindow, QPushButton, QSizePolicy, QWidget, QBoxLayout

class QCanvas(QLabel):
    def __init__(self, parent: QWidget):
        super().__init__(parent)        
        self.strokes = []
        self.painter = QPainter()
        self.setStyleSheet("background-color: white;")
    
    """ EVENT: MOUSE CLICK/PRESS """
    def mousePressEvent(self, event: QMouseEvent) -> None:
        self.strokes.append(list())
        self.strokes[-1].append((event.x(), event.y()))
        return super().mousePressEvent(event)

    """ EVENT: MOUSE MOVE WHILE PRESSED """
    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        self.strokes[-1].append((event.x(), event.y()))
        return super().mouseMoveEvent(event)

    """ EVENT: MOUSE RELEASE """
    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        # SCREENSHOT WIDGET OF QPaintDevice, SUCH AS QBITMAP
        self.render(self.pixmap())
        self.pixmap().save("./res/buff.bmp", "BMP")
        return super().mouseReleaseEvent(event)

    """ EVENT: UPDATE PAINT 
        => Beware, every time paintEvent updates removes previous drawings.
        Therefore, store the data of previous drawing to prevent being erased.
    """
    def paintEvent(self, event: QPaintEvent) -> None:
        pen = QPen()
        pen.setWidth(4)
        pen.setColor(Qt.color1)  
        self.painter.begin(self)
        self.painter.setPen(pen)
        if len(self.strokes) != 0:
            for stroke in self.strokes:
                if len(stroke) == 1:
                    self.painter.drawPoint(stroke[0][0], stroke[0][1])
                else:
                    for index in range(len(stroke) - 1):
                        self.painter.drawLine(stroke[index][0], stroke[index][1], stroke[index+1][0], stroke[index+1][1])
        self.painter.end()
        self.update()
        return super().paintEvent(event)

    """ EVENT: UPON SHOWN """
    def showEvent(self, event: QShowEvent) -> None:
        margin = self.parentWidget().layout().margin()
        width = self.topLevelWidget().width()
        height = self.topLevelWidget().height()
        for index in range(self.parentWidget().layout().count()):
            if index != self.parentWidget().layout().indexOf(self): 
                height -= ( self.parentWidget().layout().itemAt(index).widget().height() + margin * 2)
        canvas = QBitmap(width - margin * 2, height)
        canvas.clear()
        self.setPixmap(canvas)
        return super().showEvent(event)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUI()

    """ USER INTERFACE SETUP """
    def setupUI(self):
        # WINDOW SETUP
        self.setWindowTitle("Keras.QuickDraw")
        self.setMinimumSize(QSize(800, 600))
        self.setFixedSize(QSize(800, 600))
        self.setWindowIcon(QIcon("res/eon-icon.png"))
        self.setStyleSheet("background-color: blue")

        # INITIALIZE: WINDOW CENTRAL
        self.widget_central = QWidget(self)
        self.widget_central.setObjectName("Window Central")
        self.layout_central = QBoxLayout(QBoxLayout.TopToBottom, self.widget_central)
        self.setCentralWidget(self.widget_central)

        # INITIALIZE: CENTRAL HEADER
        self.widget_header = QWidget(self.widget_central)
        self.widget_header.setObjectName("Widget Header")
        self.widget_header_indicater = QLabel(parent = self.widget_header)
        self.widget_header_caption   = QLabel("(TARGET)", self.widget_header)
        self.widget_header_counter   = QLabel(parent = self.widget_header)
        
        self.layout_header = QBoxLayout(QBoxLayout.LeftToRight, self.widget_header)
        self.layout_header.addWidget(self.widget_header_indicater, 0, Qt.AlignLeft)
        self.layout_header.addWidget(self.widget_header_caption  , 1, Qt.AlignCenter)
        self.layout_header.addWidget(self.widget_header_counter  , 0, Qt.AlignRight)

        # INITIALIZE: CENTRAL BUTTONS
        self.widget_footer = QWidget(self.widget_central)
        self.widget_footer.setObjectName("Widget Footer")
        self.widget_footer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.widget_footer_clear = QPushButton("Clear", self.widget_footer)

        self.layout_footer = QBoxLayout(QBoxLayout.LeftToRight, self.widget_footer)
        self.layout_footer.addWidget(self.widget_footer_clear, 1)

        # INITIALIZE: CENTRAL CANVAS
        self.widget_canvas = QCanvas(self.widget_central)
        self.widget_canvas.setObjectName("Widget Canvas")
        self.widget_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # LAYOUT: HEADER + CANVAS + FOOTER -> CENTRAL WINDOW CENTRAL
        self.layout_central.addWidget(self.widget_header, 0, Qt.AlignHCenter)
        self.layout_central.addWidget(self.widget_canvas, 1, Qt.AlignCenter)
        self.layout_central.addWidget(self.widget_footer, 0, Qt.AlignHCenter)

        self.show()

    def paintEvent(self, event: QPaintEvent) -> None:
        return super().paintEvent(event)
