import tensorflow as tf

BATCH_SIZE = 32
IMAGE_SIZE = 256


def load_data(path):
    data = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        seed=123,
        shuffle=True,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE
    )
    return data
