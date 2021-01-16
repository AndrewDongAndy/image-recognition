"""
Run a trained model
"""


import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf


MODEL_FILE = 'siamese1/siamese_gpu.h5'

def get_model(filename):
    return tf.keras.models.load_model(filename)

def print_image(a):
    for row in a:
        for i in row:
            c = '#' if i > 0.5 else ' '
            print(c, end='')
        print()

def get_greyscale_image(filename, height, width, from_paint=True):
    # from_paint: flip black and white
    a = np.zeros((height, width))
    with Image.open(filename).convert('L') as im:
        loaded = im.load()
        for i in range(height):
            for j in range(width):
                a[i, j] = loaded[j, i]
                if from_paint:
                    a[i, j] = 255 - a[i, j]
    return a / 255

S = 28
a = get_greyscale_image('my_data/test1.png', S, S)
b = get_greyscale_image('my_data/test2.png', S, S)

print_image(a)
print_image(b)

a = np.reshape(a, (S, S, 1))
b = np.reshape(b, (S, S, 1))


x1 = np.array([a])
x2 = np.array([b])
input_data = [x1, x2]

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# reshape to Keras API requirement (data points, rows, columns, channels)
# rescale to values in [0.0, 1.0]
x_train = x_train[..., np.newaxis] / 255
x_test = x_test[..., np.newaxis] / 255

model = get_model(MODEL_FILE)
model.summary()

outputs = model.predict(x_test, batch_size=5000, verbose=1)
predictions = tf.math.argmax(outputs, axis=1)

wrong = 0
total = len(x_test)
for y_pred, y_true in zip(predictions, y_test):
    if y_pred != y_true:
        wrong += 1
correct = total - wrong
print(f'accuracy: {correct} / {total} = {correct / total}')

print(f'probability of being the same: {model(input_data)[0]}')
