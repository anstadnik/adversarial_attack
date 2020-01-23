"""This module handles the image processing"""
from functools import wraps
import pytesseract
from PIL import Image
import numpy as np
import cv2

def __convert_to_img(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        """potato"""
        if 'img' in kwargs and isinstance(kwargs['img'], np.ndarray):
            kwargs['img'] = Image.fromarray(cv2.cvtColor(kwargs['img'], cv2.COLOR_BGR2RGB))
        return f(*args, **kwargs)
    return wrapper

@__convert_to_img
def get_data_from_image(*, img):
    """Get information from the image"""
    #  TODO: Check for rotation later <20-01-20, astadnik> #
    # data = pytesseract.image_to_data(img)
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    return data

@__convert_to_img
def get_str_from_image(*, img):
    """Generate string from image

    This function extracts text from the image

    Args:
        image (): [TODO:description]
    """
    #  TODO: Check for rotation later <20-01-20, astadnik> #
    data = pytesseract.image_to_string(img)
    return data
