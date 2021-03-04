import numpy as np
import matplotlib.pyplot as plt
from matplotlib import figure
from matplotlib.backend_tools import ToolBase

plt.rcParams['toolbar'] = 'toolmanager'

class ToolNext(ToolBase):
    description = 'View next data'
    image = 'forward'
    default_keymap = 'right'
    
    def trigger(self, *args, **kwargs):
        plt.clf()
        self.figure.index += 1
        if self.figure.index == self.figure.dataset.shape[0]:
            self.figure.index = 0
        self.figure.subplots(1,1).matshow(np.reshape(self.figure.dataset[self.figure.index], (28, 28)))
        plt.draw()


class ToolPrev(ToolBase):
    description = 'View previous data'
    image = 'back'
    default_keymap = 'left'
    
    def trigger(self, *args, **kwargs):
        plt.clf()
        self.figure.index -= 1
        if self.figure.index < 0:
            self.figure.index = self.figure.dataset.shape[0] - 1
        self.figure.subplots(1,1).matshow(np.reshape(self.figure.dataset[self.figure.index], (28, 28)))
        plt.draw()


class npyViewer(figure.Figure):
    def __init__(self, *args, dataset, index, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = dataset
        self.index = index
        self.subplots(1,1).matshow(np.reshape(dataset[index], (28, 28)))


#=============================================================
# NUMPY DATASET: 28 x 28 pixel (Grayscale)
#=============================================================
def npyView(subject, index = 0):
    fig = plt.figure(FigureClass=npyViewer, dataset = np.load("./dataset/full_numpy_bitmap_"+subject+".npy"), index = index)
    fig.canvas.set_window_title("Dataset Visualizer")

    fig.canvas.manager.toolmanager.remove_tool('back')
    fig.canvas.manager.toolmanager.remove_tool('forward')

    fig.canvas.manager.toolmanager.add_tool('Prev', ToolPrev)
    fig.canvas.manager.toolmanager.add_tool('Next', ToolNext)

    fig.canvas.manager.toolbar.add_tool('Prev', 'navigation', -1)
    fig.canvas.manager.toolbar.add_tool('Next', 'navigation', -1)

    plt.show()


""" VISUALIZE NUMPY TRAINING DATA """
if __name__ == "__main__":
    npyView("fish", 117)
