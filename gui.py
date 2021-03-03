#===============================================================
# PySide2 API
#   https://doc.qt.io/qtforpython/api.html
#===============================================================
from PySide2.QtGui import QBitmap, QIcon, QPainter, QPen, QPaintEvent, QMouseEvent, QShowEvent, QCloseEvent
from PySide2.QtCore import QSize, Qt, QBuffer, QThread, QTime
from PySide2.QtWidgets import QLabel, QMainWindow, QPushButton, QSizePolicy, QWidget, QBoxLayout
from PIL import Image
import numpy as np
import io

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.buffer = QBuffer()
        self.buffer.open(QBuffer.ReadWrite)
        self.setupUI()

        self.model = QModel(self)
        self.model.start()

    """ USER INTERFACE SETUP """
    def setupUI(self):
        # WINDOW SETUP
        self.setWindowTitle("Keras.QuickDraw")
        self.setMinimumSize(QSize(800, 600))
        self.setFixedSize(QSize(800, 600))
        self.setWindowIcon(QIcon("favicon.ico"))

        # INITIALIZE: WINDOW CENTRAL
        self.widget_central = QWidget(self)
        self.widget_central.setObjectName("Window Central")
        self.widget_central.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.layout_central = QBoxLayout(QBoxLayout.TopToBottom, self.widget_central)
        self.setCentralWidget(self.widget_central)

        # INITIALIZE: CENTRAL HEADER
        self.widget_header = QWidget(self.widget_central)
        self.widget_header.setObjectName("Widget Header")
        self.widget_header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.widget_header_indicater = QLabel(parent = self.widget_header)
        self.widget_header_caption   = QLabel("(TARGET)", self.widget_header)
        self.widget_header_counter   = QLabel(parent = self.widget_header)
        
        self.layout_header = QBoxLayout(QBoxLayout.LeftToRight, self.widget_header)
        self.layout_header.addWidget(self.widget_header_indicater, 0, Qt.AlignLeft)
        self.layout_header.addWidget(self.widget_header_caption  , 1, Qt.AlignCenter)
        self.layout_header.addWidget(self.widget_header_counter  , 0, Qt.AlignRight)

        # INITIALIZE: CENTRAL CANVAS
        self.widget_canvas = QCanvas(self.widget_central, self)
        self.widget_canvas.setObjectName("Widget Canvas")
        self.widget_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # INITIALIZE: CENTRAL BUTTONS
        self.widget_footer = QWidget(self.widget_central)
        self.widget_footer.setObjectName("Widget Footer")
        self.widget_footer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.widget_footer_clear = QPushButton("Clear", self.widget_footer)
        self.widget_footer_clear.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.widget_footer_clear.clicked.connect(self.widget_canvas.resetCanvas)
        self.widget_footer_undo  = QPushButton("Undo", self.widget_footer)
        self.widget_footer_undo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.widget_footer_undo.clicked.connect(self.widget_canvas.undoCanvas)

        self.layout_footer = QBoxLayout(QBoxLayout.LeftToRight, self.widget_footer)
        self.layout_footer.addWidget(self.widget_footer_undo, 0)
        self.layout_footer.addWidget(self.widget_footer_clear, 0)
        self.layout_footer.setMargin(0)

        # LAYOUT: HEADER + CANVAS + FOOTER -> CENTRAL WINDOW CENTRAL
        self.layout_central.addWidget(self.widget_header, 0)
        self.layout_central.addWidget(self.widget_canvas, 1, Qt.AlignCenter)
        self.layout_central.addWidget(self.widget_footer, 0)

        self.show()

    """ EVENT: SAVING CANVAS (QThread ALTERNATIVE) """
    def paintEvent(self, event: QPaintEvent) -> None:
        # SCREENSHOT WIDGET OF QPaintDevice, SUCH AS QBITMAP
        self.widget_canvas.render(self.widget_canvas.pixmap())
        if self.isVisible:
            self.buffer.reset()
            self.widget_canvas.pixmap().save(self.buffer, "BMP")
        return super().paintEvent(event)

    """ EVENT: CLOSING MAINWINDOW """
    def closeEvent(self, event: QCloseEvent) -> None:
        self.model.terminate()
        self.model.wait()
        self.buffer.close()
        return super().closeEvent(event)


class QModel(QThread):
    def __init__(self, window: MainWindow):
        super().__init__(window)
        self.pWindow = window

    def run(self):
        while(True):
            QThread.msleep(100)
            npQDraw = np.array(Image.open(io.BytesIO(self.pWindow.buffer.data())))


class QCanvas(QLabel):
    def __init__(self, parent: QWidget, window: MainWindow):
        super().__init__(parent)
        self.pWindow = window
        self.strokeX = []
        self.strokeY = []
        self.strokeT = []
        self.timing = QTime()
        self.timing.start()
        self.paused = 0
        self.painter = QPainter()
        self.setStyleSheet("background-color: white;")
    
    """ EVENT: MOUSE CLICK/PRESS """
    def mousePressEvent(self, event: QMouseEvent) -> None:
        self.timing.restart()
        self.strokeX.append(list())
        self.strokeY.append(list())
        self.strokeT.append(list())
        self.strokeX[-1].append(event.x())
        self.strokeY[-1].append(event.y())
        self.strokeT[-1].append(self.paused)
        return super().mousePressEvent(event)

    """ EVENT: MOUSE MOVE WHILE PRESSED """
    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if event.x() > 0 and event.y() > 0 and event.x() <= self.width() and event.y() <= self.height():
            # IF DRAWING IS NOT JUST A MERE POINT...
            if len(self.strokeT[-1]) == 1:
                self.timing.restart()
                self.paused += 1
            self.strokeX[-1].append(event.x())
            self.strokeY[-1].append(event.y())
            self.strokeT[-1].append(self.timing.elapsed() + self.paused)
        else: return
        return super().mouseMoveEvent(event)

    """ EVENT: MOUSE RELEASE """
    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        # TO ABLE TO RECOGNIZE POINTS AS PART OF THE DRAWING...
        if len(self.strokeT[-1]) == 1:
            self.paused += 1
        # OTHERWISE, CHECKPOINTS THE MILLISECONDS AS FOLLOWS.
        else:
            self.paused += (self.timing.elapsed() + 1)
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
        if len(self.strokeX) != 0:
            for stroke in range(len(self.strokeX)):
                if len(self.strokeX[stroke]) == 1:
                    self.painter.drawPoint(self.strokeX[stroke][0], self.strokeY[stroke][0])
                else:
                    for index in range(len(self.strokeX[stroke]) - 1):
                        self.painter.drawLine(self.strokeX[stroke][index], self.strokeY[stroke][index], self.strokeX[stroke][index+1], self.strokeY[stroke][index+1])
        self.painter.end()
        self.update()
        return super().paintEvent(event)

    """ EVENT: UPON SHOWN """
    def showEvent(self, event: QShowEvent) -> None:
        self.blankCanvas()
        return super().showEvent(event)

    """ METHOD: CREATE CANVAS """
    def blankCanvas(self) -> None:
        margin = self.parentWidget().layout().margin()
        width = self.topLevelWidget().width()
        height = self.topLevelWidget().height()
        for index in range(self.parentWidget().layout().count()):
            if index != self.parentWidget().layout().indexOf(self): 
                height -= (self.parentWidget().layout().itemAt(index).widget().height() + margin * 2)
        canvas = QBitmap(width - margin * 2, height)
        canvas.clear()
        self.setPixmap(canvas)
        self.update()

    """ METHOD: RESET CANVAS """
    def resetCanvas(self) -> None:
        self.strokeX = []
        self.strokeY = []
        self.strokeT = []
        self.paused = 0
        self.blankCanvas()

    """ METHOD: UNDO CANVAS """
    def undoCanvas(self) -> None:
        try:
            self.paused = self.strokeT[-1][0]
            self.strokeX.pop()
            self.strokeY.pop()
            self.strokeT.pop()
        except IndexError:
            print("The canvas is completely empty!")
        self.blankCanvas()
