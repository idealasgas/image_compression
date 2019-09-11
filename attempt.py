import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np

# First, load the image again
dir_path = os.path.dirname(os.path.realpath(__file__))
filename = dir_path + "/original.jpg"
image = Image.open(filename)
image_array = np.asarray(image)
print(image_array)

def random_img(output, width, height):
  array = np.random.random_integers(0,255, (height,width,3))

  array = np.array(array, dtype=np.uint8)
  img = Image.fromarray(array)
  img.save(output)
  plt.imshow(img)
  plt.show()

M = 2
N = 2

tiles = [image_array[x:x+M,y:y+N] for x in range(0,image_array.shape[0],M) for y in range(0,image_array.shape[1],N)]
# import code; code.interact(local=dict(globals(), **locals()))
# print(tiles)

# i=0
# while i<25:
#   arr = np.array(tiles[i], dtype=np.uint8)
#   img = Image.fromarray(arr)
#   img.save(str(i) + '.jpg')
#   i+=1

# i=0
# while i<25:
image = Image.open('1.jpg')
plt.imshow(image)
plt.show()

image = Image.open('2.jpg')
plt.imshow(image)
plt.show()

image = Image.open('3.jpg')
plt.imshow(image)
plt.show()

image = Image.open('4.jpg')
plt.imshow(image)
plt.show()

image = Image.open('5.jpg')
plt.imshow(image)
plt.show()

image = Image.open('6.jpg')
plt.imshow(image)
plt.show()

image = Image.open('7.jpg')
plt.imshow(image)
plt.show()

image = Image.open('8.jpg')
plt.imshow(image)
plt.show()

image = Image.open('9.jpg')
plt.imshow(image)
plt.show()


image = Image.open('10.jpg')
plt.imshow(image)
plt.show()

image = Image.open(filename)
plt.imshow(image)
plt.show()



plt.show()


# сначала нужно сделать разбиение картинки на маленькие прямоугольники
# делим на 256 квадратов 16x16 для этого нужно
