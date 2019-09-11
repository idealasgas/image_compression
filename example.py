import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np

# First, load the image again
dir_path = os.path.dirname(os.path.realpath(__file__))
filename = dir_path + "/forest.jpg"
i = Image.open(filename)
iar = np.asarray(i)
print(iar)

def random_img(output, width, height):
  array = np.random.random_integers(0,255, (height,width,3))

  array = np.array(array, dtype=np.uint8)
  img = Image.fromarray(array)
  img.save(output)
  plt.imshow(img)
  plt.show()

random_img('random.png', 2, 2)

# сначала нужно сделать разбиение картинки на маленькие прямоугольники
