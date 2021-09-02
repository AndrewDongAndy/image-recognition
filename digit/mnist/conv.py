"""
Don't know what I'm doing, but let's try this!
"""


import numpy as np
import tensorflow as tf


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# reshape to Keras API requirement (data points, rows, columns, channels)
# rescale to values in [0.0, 1.0]
x_train = x_train[..., np.newaxis] / 255
x_test = x_test[..., np.newaxis] / 255

print(f'number of training examples: {len(x_train)}')
print(f'number of testing examples: {len(x_test)}')


# build model
model = tf.keras.Sequential([
    # convolution layer 1
    tf.keras.layers.Conv2D(32, 5, activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

    # convolution layer 2
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

    # dense classifier
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])
# model.summary()  # show summary of layers

# compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

# train (fit the data)
model.fit(x_train, y_train, batch_size=1000, epochs=20)

# save the model
model.save('conv.h5')
