import tensorflow as tf

IMAGE_SIZE = 256


def resize_rescale():
    resize_and_rescale = tf.keras.Sequential([
        tf.keras.layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
        tf.keras.layers.Rescaling(1. / 255),
    ])
    return resize_and_rescale
