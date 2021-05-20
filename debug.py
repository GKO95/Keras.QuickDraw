import tensorflow as tf
import numpy as np

import random
import os

DATAPATH = "datasets"

""" DATAFILE =
['full_numpy_bitmap_ant.npy', 'full_numpy_bitmap_bee.npy', 'full_numpy_bitmap_bird.npy', 'full_numpy_bitmap_butterfly.npy', 'full_numpy_bitmap_circle.npy',
 'full_numpy_bitmap_cow.npy', 'full_numpy_bitmap_dog.npy', 'full_numpy_bitmap_dolphin.npy', 'full_numpy_bitmap_donut.npy', 'full_numpy_bitmap_dragon.npy',
 'full_numpy_bitmap_fish.npy', 'full_numpy_bitmap_grass.npy', 'full_numpy_bitmap_lightning.npy', 'full_numpy_bitmap_line.npy', 'full_numpy_bitmap_mermaid.npy',
 'full_numpy_bitmap_mosquito.npy', 'full_numpy_bitmap_mountain.npy', 'full_numpy_bitmap_rain.npy', 'full_numpy_bitmap_river.npy', 'full_numpy_bitmap_shark.npy',
 'full_numpy_bitmap_snake.npy', 'full_numpy_bitmap_spider.npy', 'full_numpy_bitmap_square.npy', 'full_numpy_bitmap_star.npy', 'full_numpy_bitmap_stitches.npy', 
 'full_numpy_bitmap_sun.npy', 'full_numpy_bitmap_triangle.npy', 'full_numpy_bitmap_watermelon.npy', 'full_numpy_bitmap_whale.npy', 'full_numpy_bitmap_wheel.npy']
"""
DATAFILE = [file for file in os.listdir(DATAPATH) if os.path.isfile(os.path.join(DATAPATH, file))]

""" DATANAME =
['ant', 'bee', 'bird', 'butterfly', 'circle', 'cow', 'dog', 'dolphin', 'donut', 'dragon',
 'fish', 'grass', 'lightning', 'line', 'mermaid', 'mosquito', 'mountain', 'rain', 'river', 'shark', 
 'snake', 'spider', 'square', 'star', 'stitches', 'sun', 'triangle', 'watermelon', 'whale', 'wheel']
"""
DATANAME = [name.split('.')[0].split('_')[-1] for name in DATAFILE]

# RANDOM DATASET SELECTOR FROM DATASETS
def Index():
    random.seed()
    return int(random.random() * 1000) % len(DATAFILE)

# RANDOM INSTANCE SELECTOR FROM A DATASET
def Instance(instance: int = 0):
    index = Index()
    return np.load(os.path.join(DATAPATH, DATAFILE[index]))[instance].reshape([1, 28, 28, 1]) / 255, index

def Sequence(instance):
    layerConv2D_1 = tf.keras.layers.Conv2D(
        filters = 32,               # The number of filter in the Conv2D layer: 32
        kernel_size = (5, 5),       # The size of the filter in the Conv2D layer.
        strides = (1, 1),           # The amount of pixels to shift the kernel.
        padding = 'valid',          # Leave the input data AS IS wihtout any padding ('valid').
        activation = 'relu',        # Activation function (default: None).
        input_shape = (28, 28, 1),  # First Conv2D must specify the shape of input: 28 x 28 Grayscale channel 
    )
    tensor = layerConv2D_1(instance)
    #print(tensor)                   # tf.Tensor(shape=(batch, 24, 24, 32), dtype=float32)

    layerMaxPooling2D_1 = tf.keras.layers.MaxPool2D(
        pool_size = (2, 2),         # The size of a pooling window in the Conv2D layer.
        strides = (2, 2),           # The amount of pixels to shift a pooling window.
        padding = 'valid',          # Leave the input data AS IS wihtout any padding ('valid').
    )
    tensor = layerMaxPooling2D_1(tensor)
    #print(tensor)                   # tf.Tensor(shape=(batch, 12, 12, 32), dtype=float32)

    layerConv2D_2 = tf.keras.layers.Conv2D(
        filters = 64,
        kernel_size = (3, 3),
    )
    tensor = layerConv2D_2(tensor)
    #print(tensor)                   # tf.Tensor(shape=(batch, 10, 10, 64), dtype=float32)

    layerMaxPooling2D_2 = tf.keras.layers.MaxPool2D(
        pool_size = (2, 2),
    )
    tensor = layerMaxPooling2D_2(tensor)
    #print(tensor)                   # tf.Tensor(shape=(batch, 5, 5, 64), dtype=float32)

    layerConv2D_3 = tf.keras.layers.Conv2D(
        filters = 64,
        kernel_size= (3, 3),
    )
    tensor = layerConv2D_3(tensor)
    #print(tensor)                   # tf.Tensor(shape=(batch, 3, 3, 64), dtype=float32)

    layerFlatten = tf.keras.layers.Flatten()
    tensor = layerFlatten(tensor)
    #print(tensor)                   # tf.Tensor(shape=(batch, 576), dtype=float32)

    layerDense_1 = tf.keras.layers.Dense(
        units = 192,
        activation = 'relu',
    )
    tensor = layerDense_1(tensor)
    #print(tensor)                   # tf.Tensor(shape=(batch, 192), dtype=float32)

    layerDense_2 = tf.keras.layers.Dense(
        units = 64,
        activation = 'relu',
    )
    tensor = layerDense_2(tensor)
    #print(tensor)                   # tf.Tensor(shape=(batch, 64), dtype=float32)

    layerSoftmax = tf.keras.layers.Dense(
        units = 30,
        activation = 'softmax',
    )
    tensor = layerSoftmax(tensor)
    #print(tensor)                   # tf.Tensor(shape=(batch, 30), dtype=float32)


if __name__ == "__main__":
    instance, _ = Instance()
    Sequence(instance)
