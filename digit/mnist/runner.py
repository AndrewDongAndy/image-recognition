"""
Run a trained model
"""


import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf


MODEL_FILE = 'conv.h5'

def get_model(filename) -> tf.keras.Sequential:
    return tf.keras.models.load_model(filename)

def print_image(a):
    for row in a:
        for i in row:
            c = '#' if i > 127 else ' '
            print(c, end='')
        print()


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# # reshape to Keras API requirement (data points, rows, columns, channels)
# # rescale to values in [0.0, 1.0]
# x_train = x_train[..., np.newaxis] / 255
# x_test = x_test[..., np.newaxis] / 255

model = get_model(MODEL_FILE)
# outputs = model.predict(x_test, batch_size=5000, verbose=1)
# predictions = tf.math.argmax(outputs, axis=1)

# wrong = 0
# total = len(x_test)
# for y_pred, y_true in zip(predictions, y_test):
#     if y_pred != y_true:
#         wrong += 1
# correct = total - wrong
# print(f'accuracy: {correct} / {total} = {correct / total}')

S = 28
a = np.array([[0 for j in range(S)] for i in range(S)])
with Image.open('test.png').convert('L') as im:
    loaded = im.load()
    for i in range(S):
        for j in range(S):
            a[i, j] = 255 - loaded[j, i]


def test_array(a):
    b = a[np.newaxis, ..., np.newaxis] / 255
    return tf.math.argmax(model(b)[0])


print_image(a)
print(test_array(a))

for x, y in zip(x_train, y_train):
    if y == 9:
        print_image(x)
        print(test_array(x))
        break