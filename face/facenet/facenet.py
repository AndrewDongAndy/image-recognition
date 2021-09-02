"""
Run the trained model stored in MODEL_FILE.
"""


import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf


MODEL_FILE = 'facenet_keras.h5'


S = 28

def get_model(filename) -> tf.keras.Model:
    return tf.keras.models.load_model(filename)

def print_image(a):
    for row in a:
        for pixel in row:
            c = '#' if pixel[0] > 0.5 else ' '
            print(c, end='')
        print()

def get_greyscale_image(filename, height=S, width=S, from_paint=True):
    # from_paint: flip black and white
    a = np.zeros((height, width))
    with Image.open(filename).convert('L') as im:
        loaded = im.load()
        for i in range(height):
            for j in range(width):
                a[i, j] = loaded[j, i]
                if from_paint:
                    a[i, j] = 255 - a[i, j]
    return np.reshape(a, (height, width, 1)) / 255


model = get_model(MODEL_FILE)
model.summary()
exit()

a = get_greyscale_image('my_data/test1.png')
b = get_greyscale_image('my_data/test2.png')

print_image(a)
print_image(b)

x1 = np.array([get_greyscale_image(f'my_data/{i}.png') for i in range(10)])
x2 = np.array([b for i in range(10)])
input_data = [x1, x2]

outputs = model(input_data)
print(outputs)
prediction = tf.math.argmax(outputs, axis=0)
print(f'prediction: {prediction}')
exit()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# reshape to Keras API requirement (data points, rows, columns, channels)
# rescale to values in [0.0, 1.0]
x_train = x_train[..., np.newaxis] / 255
x_test = x_test[..., np.newaxis] / 255


# outputs = model.predict(x_test, batch_size=5000, verbose=1)
# predictions = tf.math.argmax(outputs, axis=1)


p = np.zeros((10, len(x_test), 1))
for i in range(10):
    second = get_greyscale_image(f'my_data/{i}.png')
    compare_to = np.array([second for i in range(len(x_test))])
    inputs = [x_test, compare_to]
    p[i] = model.predict(inputs)
predictions = np.argmax(p, axis=0)

wrong = 0
total = len(x_test)
for y_pred, y_true in zip(predictions, y_test):
    if y_pred != y_true:
        wrong += 1
correct = total - wrong
print(f'accuracy: {correct} / {total} = {correct / total}')

outputs = model.predict(input_data)
answer = np.argmax(outputs)

print(outputs)
print(f'predicted digit: {answer}')

# print(f'probability of being the same: {model(input_data)[0]}')
