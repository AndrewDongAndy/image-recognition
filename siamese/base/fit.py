"""
Script for building and training a Siamese network.
"""


import numpy as np
import random
import tensorflow as tf


# problem parameters
INPUT_SHAPE = (28, 28, 1)

# model parameters
FEATURES = 128
DROPOUT_RATE = 0.3

# training parameters
POSITIVE = 1000000
NEGATIVE = 1000000
BATCH_SIZE = 32
EPOCHS = 100

# testing parameters
# POSITIVE = 1000
# NEGATIVE = 1000
# BATCH_SIZE = 32
# EPOCHS = 5


def generate_pairs(x, y, positive, negative):
    x1 = []
    x2 = []
    new_y = []
    unique = np.unique(y)
    d = dict((c, i) for i, c in enumerate(unique))
    classes = len(unique)
    ids = [[] for i in range(classes)]
    for c, label in enumerate(y):
        ids[d[label]].append(c)
    for _ in range(positive):
        label = random.randint(0, classes - 1)
        a = np.random.choice(ids[label])
        b = np.random.choice(ids[label])
        x1.append(x[a])
        x2.append(x[b])
        new_y.append([1])
    for _ in range(negative):
        label = random.randint(0, classes - 1)
        a = np.random.choice(ids[label])
        t = random.randint(0, classes - 2)
        if t == label:
            t = classes - 1
        b = np.random.choice(ids[t])
        x1.append(x[a])
        x2.append(x[b])
        new_y.append([0])
    return (np.array(x1), np.array(x2)), np.array(new_y)


def dist(vectors):
    a, b = vectors
    s = tf.keras.backend.sum(tf.math.square(a - b), axis=1, keepdims=True)
    return tf.math.sqrt(s)
    # return tf.math.sqrt(tf.keras.backend.maximum(s, tf.keras.backend.epsilon()))


inputs = tf.keras.Input(shape=INPUT_SHAPE)

# convolution 1
conv1 = tf.keras.layers.Conv2D(64, 5, activation='relu')(inputs)
pool1 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(conv1)
drop1 = tf.keras.layers.Dropout(DROPOUT_RATE)(pool1)

# convolution 2
conv2 = tf.keras.layers.Conv2D(64, 3, activation='relu')(drop1)
pool2 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(conv2)
drop2 = tf.keras.layers.Dropout(DROPOUT_RATE)(pool2)

# final pooling
global1 = tf.keras.layers.GlobalAveragePooling2D()(drop2)
outputs = tf.keras.layers.Dense(FEATURES)(global1)

feature_model = tf.keras.Model(inputs=inputs, outputs=outputs)
# feature_model.summary()

a = tf.keras.Input(INPUT_SHAPE)
b = tf.keras.Input(INPUT_SHAPE)
features_a = feature_model(a)
features_b = feature_model(b)
edist = tf.keras.layers.Lambda(dist)([features_a, features_b])
outputs2 = tf.keras.layers.Dense(1, activation='sigmoid')(edist)

# make the model from the inputs and outputs
model = tf.keras.Model(inputs=[a, b], outputs=outputs2)

# compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
    loss='binary_crossentropy',
    metrics=['accuracy'],
)


# prepare the data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# reshape to Keras API requirement (data points, rows, columns, channels)
# rescale to values in [0.0, 1.0]
x_train = x_train[..., np.newaxis] / 255
x_test = x_test[..., np.newaxis] / 255

# generate training data
(x1_train, x2_train), new_y_train = generate_pairs(x_train, y_train, POSITIVE, NEGATIVE)

# train (fit the data)
model.fit(
    [x1_train, x2_train], new_y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=2,  # avoid super long output files
)

# save the network
model.save('siamese.h5')
