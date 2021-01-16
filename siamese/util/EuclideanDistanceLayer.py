"""
The layer to merge the outputs of the two sister networks
in the Siamese network model.
"""


import numpy as np
import tensorflow as tf


class EuclideanDistanceLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(EuclideanDistanceLayer, self).__init__()
    
    def call(self, inputs):
        a, b = inputs
        return tf.sqrt(tf.keras.backend.sum(tf.square(a - b), axis=1, keepdims=True))