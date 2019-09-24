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
# print(tiles)

# i=0
# while i<25:
#   arr = np.array(tiles[i], dtype=np.uint8)
#   img = Image.fromarray(arr)
#   img.save(str(i) + '.jpg')
#   i+=1

array_of_x = []

for i, tile in enumerate(tiles):
  array_of_x.append([])

for i, tile in enumerate(tiles):
  for j, row in enumerate(tile):
    for k, pixel in enumerate(row):
      for p, color in enumerate(pixel):
        array_of_x[i].append(((2 * color) / 255) - 1)

print(array_of_x)

# N = array_of_x[0].size
N = 12

p = 5 # вводимое пользователем число нейронов второго слоя (p <= 2*N)

matrix_of_weighs = np.random.rand(N, p)

print(matrix_of_weighs)

array_of_y = []

for x in array_of_x:
  array_of_y.append(np.dot(x, matrix_of_weighs))

print(array_of_y)



# import code; code.interact(local=dict(globals(), **locals()))
# for x in array_of_x:


# сначала нужно сделать разбиение картинки на маленькие прямоугольники
# делим на 256 квадратов 16x16 для этого нужно

# теперь нужно использовать матрицу весов W для первого слоя
# количество нейронов второго слоя - p <= N*2
# N - кол-во элементов в X (12)
