"""
Script for building and training a Siamese network.
"""


import numpy as np
import random
import tensorflow as tf


# problem parameters
INPUT_SHAPE = (28, 28, 1)

# model parameters
FEATURES = 256
DROPOUT_RATE = 0.3

# training parameters
POSITIVE = 1000000
NEGATIVE = 1000000
BATCH_SIZE = 32
EPOCHS = 20
VERBOSITY_MODE = 2

# testing parameters
# POSITIVE = 1000
# NEGATIVE = 1000
# BATCH_SIZE = 32
# EPOCHS = 5
# VERBOSITY_MODE = 1


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

    # new_y = [1 for _ in range(positive)].extend([0 for _ in range(negative)])
    # x1_labels = np.random.randint(0, classes, positive + negative)
    # # x2_labels = [classes - 1 if random.randint(0, classes - 2) == i for i in x1_labels[:positive]]
    # # x2_labels = [classes - 1 if random.randint(0, classes - 2) == i for i in x1_labels[:positive]]
    # x1 = [x[i] for i in x1_labels]
    # x2 = [x[i] for i in x2_labels]
    # return (np.array(x1), np.array(x2)), np.array(new_y)

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


def dist_L2(vectors):
    a, b = vectors
    res = tf.math.sqrt(tf.keras.backend.sum(tf.math.square(a - b), axis=1, keepdims=True))
    return tf.maximum(res, tf.keras.backend.epsilon())


def dist_L1(vectors):
    a, b = vectors
    res = tf.keras.backend.sum(tf.math.abs(a - b), axis=1, keepdims=True)
    return tf.maximum(res, tf.keras.backend.epsilon())


feature_model = tf.keras.Sequential([
    tf.keras.Input(shape=INPUT_SHAPE),  # (28, 28, 1)

    # convolution 1
    tf.keras.layers.Conv2D(32, 5,
        activation='relu',
        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
        bias_initializer=tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.01),
    ),  # (24, 24, 32)
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),  # (12, 12, 32)
    tf.keras.layers.Dropout(DROPOUT_RATE),

    # convolution 2
    tf.keras.layers.Conv2D(64, 3,
        activation='relu',
        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
        bias_initializer=tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.01),
    ),  # (10, 10, 64)
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),  # (5, 5, 64)
    tf.keras.layers.Dropout(DROPOUT_RATE),

    # final pooling
    # tf.keras.layers.GlobalAveragePooling2D(),
    # tf.keras.layers.Dense(FEATURES, activation='sigmoid'),

    # convolution 3
    tf.keras.layers.Conv2D(128, 3,
        activation='relu',
        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
        bias_initializer=tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.01),
    ),  # (3, 3, 128)
    # tf.keras.layers.GlobalAveragePooling2D(),

    # feature extraction
    tf.keras.layers.Flatten(),  # (1152,)
    tf.keras.layers.Dense(FEATURES,
        activation='sigmoid',
        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.2),
        bias_initializer=tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.01),
    ),  # (FEATURES,)
])

feature_model.summary()
# exit()

a = tf.keras.Input(INPUT_SHAPE)
b = tf.keras.Input(INPUT_SHAPE)
features_a = feature_model(a)
features_b = feature_model(b)
dist_layer = tf.keras.layers.Lambda(dist_L1)([features_a, features_b])
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(dist_layer)

# make the model from the inputs and outputs
model = tf.keras.Model(inputs=[a, b], outputs=outputs)

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
    verbose=VERBOSITY_MODE,
)

# save the network
model.save('siamese.h5')
