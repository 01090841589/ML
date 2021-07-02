from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

im = Image.open('3.png')

print(im.size)

im = np.array(im)

a = model.predict(np.reshape(im, (1, 28, 28)))