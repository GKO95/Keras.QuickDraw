import tensorflow as tf
import numpy as np
import os

DATAPATH = "dataset"

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

if __name__ == "__main__":
    print(DATAFILE)