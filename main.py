import sys
import gui as GUI
from PySide2.QtWidgets import QApplication

if __name__ == '__main__':
    app = QApplication()
    window = GUI.MainWindow()
    sys.exit(app.exec_())
