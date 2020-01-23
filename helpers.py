"""This is the module with miscellaneous functions"""
import argparse

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse arguments from the command line

    This function parses arguments and returns a class-like object with values

    Returns:
        argparse.Namespace: [TODO:description]
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", default=None, help="Path to the image", required=True)
    parser.add_argument("-d", "--display", action="store_true", default=False, help="Display the image")
    return parser.parse_args()

def __resize(img: np.ndarray, resize):
    """Handy function to resize image

    This functions gets as an input the image, resizes it and return the changed image

    If resize argument is float, it specifies the scaling percent. If it's a tuple,
    it specifies the dimensions of the resulting image. Additionally, if the 2-nd parameter
    equals -1, it is calculated so the proportions of the image are preserved.

    Args:
        img (np.ndarray): Image to resize
        resize (float or tuple): 
    """
    if isinstance(resize, float):
        assert 0 < resize <= 1
        width = int(img.shape[1] * resize)
        height = int(img.shape[0] * resize)
        dim = (width, height)
    elif isinstance(resize, (tuple, list)):
        if resize[1] == -1:
            dim = (resize[0], img.shape[0] * resize[0] // img.shape[1])
        else:
            dim = resize
    else:
        raise RuntimeError("Wrong resize argument: " + str(resize))
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

def get_img(path: str, resize=None):
    """Read image and process it

    This function reads image from the disk, and resizes it

    Args:
        path (str): Path to the image
        resize (float or typle, optional): Look at the __resize documentation for details.
    """
    img = cv2.imread(path)

    if resize is not None:
        img = __resize(img, resize)

    return img

def display_image(img):
    cv2.imshow('image', img)
    while cv2.waitKey(0) != 27:
        continue
    cv2.destroyAllWindows()
