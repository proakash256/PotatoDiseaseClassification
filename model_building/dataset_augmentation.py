# Data augmentation is used to make the model more robust by creating new training samples.
# Data augmentation involves applying different transformations (such as rotation, and contrast)
# to original images to generate new training samples
import tensorflow as tf


def data_augmentation():
    augmented_data = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
    ])
    return augmented_data
