from typing import Tuple, List
import gzip
import os
import urllib.request

import numpy as np
import plotly.express as px
import plotly.io as pio
import tensorflow.compat.v1 as tf
from keras.layers import Activation, Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential
from tqdm import tqdm
from mnist.generate_data import _generate_data

tf.disable_v2_behavior()
pio.renderers.default = 'browser'
tf.disable_v2_behavior()


def generate_data(size: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate MNIST data

    :param size int: number of items to return
    :rtype Tuple[np.ndarray, np.ndarray, np.ndarray]: np.arrays of input images
    with shape (size, 28, 28, 1), randomly generated label which does not match the real and the real label
    """
    return _generate_data(size)


def pred(imgs: np.ndarray, use_log=True, v=False) -> List[np.ndarray]:
    """
    Return predictions for images

    :param imgs np.ndarray: array of input images of size (n, 28, 28, 1)
    :rtype List[np.ndarray]: return list of len n
    """
    # with tf_debug.TensorBoardDebugWrapsperSession(tf.Session(), 'localhost:6064') as sess:
    with tf.Session() as sess:
        model = _MNISTModel('mnist/models/mnist', sess, use_log)
        image_dim = 28
        image_channels = 1
        num_labels = 10

        test_in = tf.placeholder(
            tf.float32, (1, image_dim, image_dim, image_channels), 'x')
        test_pred = tf.argmax(model.predict(test_in), axis=1)
        test_pred = model.predict(test_in)

        orig_pred = np.array([sess.run(test_pred, feed_dict={
            test_in: [img]})[0] for img in tqdm(imgs, disable=not v)])

        # px.imshow(img.reshape((28, 28))).show()
        # print(orig_pred)
        return orig_pred

class _MNISTModel:
    def __init__(self, restore=None, session=None, use_log=False):
        self.num_channels = 1
        self.image_size = 28
        self.num_labels = 10

        model = Sequential()

        model.add(Conv2D(32, (3, 3),
                         input_shape=(28, 28, 1)))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(200))
        model.add(Activation('relu'))
        model.add(Dense(200))
        model.add(Activation('relu'))
        model.add(Dense(10))
        # output log probability, used for black-box attack
        if use_log:
            model.add(Activation('softmax'))
        if restore:
            model.load_weights(restore)

        self.model = model

    def predict(self, data):
        return self.model(data)

def _main():
    inputs, targets, reals = generate_data(10000)
    preds = pred(inputs)
    corr = [np.argmax(p) == r for p, r in zip(preds, reals)]
    d = {'type': ['Correct' if c else 'Wrong' for c in corr]}
    px.histogram(d, x='type').show()


if __name__ == "__main__":
    _main()
