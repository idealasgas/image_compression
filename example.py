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

random_img('random.png', 100, 50)

# Create a TensorFlow Variable
# x = tf.Variable(image, name='x')

# model = tf.global_variables_initializer()

# with tf.Session() as session:
#     x = tf.transpose(x, perm=[1, 0, 2])
#     session.run(model)
#     result = session.run(x)


# plt.imshow(result)
# plt.show()
