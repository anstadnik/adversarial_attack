import gzip
from typing import Tuple, List
import numpy as np
import os
import urllib

###################
#  Dataset class  #
###################

class _MNIST:
    def __init__(self):
        if not os.path.exists("data"):
            os.mkdir("data")
            files = ["train-images-idx3-ubyte.gz",
                     "t10k-images-idx3-ubyte.gz",
                     "train-labels-idx1-ubyte.gz",
                     "t10k-labels-idx1-ubyte.gz"]
            for name in files:
                urllib.request.urlretrieve(
                    'http://yann.lecun.com/exdb/mnist/' + name, "data/"+name)

        train_data = _extract_data("data/train-images-idx3-ubyte.gz", 60000)
        train_labels = _extract_labels("data/train-labels-idx1-ubyte.gz", 60000)
        self.test_data = _extract_data("data/t10k-images-idx3-ubyte.gz", 10000)
        self.test_labels = _extract_labels(
            "data/t10k-labels-idx1-ubyte.gz", 10000)

        validation_size = 5000

        self.validation_data = train_data[:validation_size, :, :, :]
        self.validation_labels = train_labels[:validation_size]
        self.train_data = train_data[validation_size:, :, :, :]
        self.train_labels = train_labels[validation_size:]

####################
#  Work with gzip  #
####################

def _extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(num_images*28*28)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data / 255) - 0.5
        data = data.reshape(num_images, 28, 28, 1)
        return data


def _extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8)
    return (np.arange(10) == labels[:, None]).astype(np.float32)

###################
#  Generate data  #
###################

def _generate_data(size: int = 1000, data=_MNIST()) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate MNIST data

    :param data MNIST: Dataset
    :param size int: number of items to return
    :rtype Tuple[np.ndarray, np.ndarray, np.ndarray]: np.arrays of input images,
    randomly generated label which does not match the real and the real label
    """
    inputs, targets, reals = [], [], []
    num_labels = data.test_labels.shape[1]
    i = 0
    while i < size and i < len(data.test_data):
        inputs.append(data.test_data[i].astype(np.float32))
        reals.append(np.argmax(data.test_labels[i]))
        other_labels = [x for x in range(
            num_labels) if data.test_labels[i][x] == 0]
        random_target = [0 for _ in range(num_labels)]
        random_target[np.random.choice(other_labels)] = 1
        targets.append(random_target)
        i += 1
    inputs = np.array(inputs)
    targets = np.array(targets)
    reals = np.array(reals)
    return inputs, targets, reals
