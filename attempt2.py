import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np

# загружаю картинку
dir_path = os.path.dirname(os.path.realpath(__file__))
filename = dir_path + "/jesus.jpg"
image = Image.open(filename)
image_array = np.asarray(image)

def random_img(output, width, height):
  array = np.random.random_integers(0,255, (height,width,3))
  array = np.array(array, dtype=np.uint8)
  img = Image.fromarray(array)
  img.save(output)
  plt.imshow(img)
  plt.show()

# размер кусочков
m = 2
n = 2

# делю на маленькие штуки
tiles = [image_array[x:x+m,y:y+n] for x in range(0,image_array.shape[0],m) for y in range(0,image_array.shape[1],n)]

array_of_x = []

for i, tile in enumerate(tiles):
  array_of_x.append([])

for i, tile in enumerate(tiles):
  for j, row in enumerate(tile):
    for k, pixel in enumerate(row):
      for p, color in enumerate(pixel):
        array_of_x[i].append(((2 * color) / 255) - 1)

N = m * n * 3 # 4 клеточки по три цвета
L = len(tiles)

p = 5 # вводимое пользователем число нейронов второго слоя (p <= 2*N)

error_max = 539.15
error_current = error_max + 1
iteration = 0

w1 = np.random.rand(N, p)
w2 = w1.transpose()

while error_current > error_max:
  error_current = 0
  iteration += 1
  for i in array_of_x:
    y = i @ w1
    x1 = y @ w2
    dx = x1 - i
    learning_ratio = 1 / np.dot(i, np.transpose(i)) / 1000
    learning_ratio_dash = 1 / np.dot(y, np.transpose(y)) / 1000
    # w1 -= learning_ratio * np.dot(np.dot(np.transpose(i), dx), w2.transpose())
    w1 -= learning_ratio * np.dot(np.dot(np.asarray(i)[np.newaxis].transpose(), dx[np.newaxis]), w2.transpose())
    w2 -= learning_ratio_dash * np.matmul(y[np.newaxis].transpose(), dx[np.newaxis])
  for i in array_of_x:
    dx = ((i @ w1) @ w2) - i
    error = (dx * dx).sum()
    error_current += error
  print('итерация ', iteration, '    ', 'ошибка: ', error_current)


# ===================================ВОССТАНОВЛЕНИЕ

new_x = []
for i in array_of_x:
  y = i @ w1
  x1 = y @ w2
  new_x.append(x1)


new_pixels = []
for i in new_x:
  new_pixels.append([])

for i, x in enumerate(new_x):
  for px in x:
    color = 255 * (px + 1) / 2
    new_pixels[i].append(color)

# это можно написать в одном цикле

height = len(image_array)

new_image_array = []
for i in range(height):
  new_image_array.append([])

def check_counter(counter):
  if counter % 2 == 0:
    if (counter / 2) % 2 != 0:
      return True
    else:
      return False
  else:
    if ((counter + 1) / 2) % 2 != 0:
      return True
    else:
      return False

upper_side = 0
lower_side = 1
counter = 1

for three_px in np.asarray(new_pixels).reshape(-1, 3):
  if check_counter(counter):
    new_image_array[upper_side].append(np.rint(three_px))
  else:
    new_image_array[lower_side].append(np.rint(three_px))
  if counter == height * 2:
    counter = 1
    upper_side += 2
    lower_side += 2
  else:
    counter += 1


img = Image.fromarray(np.array(new_image_array).astype('uint8'))
plt.imshow(img)
plt.show()

Z = (N * L) / ((N + L) * p + 2)

print("Коэффициент сжатия: ", Z)
