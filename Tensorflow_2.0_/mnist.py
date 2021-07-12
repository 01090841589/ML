from __future__ import absolute_import, division, print_function, unicode_literals
# source ~/anaconda3/etc/profile.d/conda.sh
#!pip install -q tensorflow-gpu==2.0.0-rc1
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(len(x_train), len(y_train))
print(len(x_test), len(y_test))

print(x_train[0])

x_train, x_test = x_train / 255.0, x_test / 255.0

plt.imshow(x_test[0])

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test,  y_test, verbose=2)

from PIL import Image

im = Image.open('3.png')
plt.imshow(im)
plt.show()
print(im.size)

im = np.array(im)

a = model.predict(np.reshape(im, (1, 28, 28)))

print(a)

print(max(a[0]))

print(list(a[0]).index(max(a[0])))

print("예측된 결과는 {} 입니다".format(list(a[0]).index(max(a[0]))))