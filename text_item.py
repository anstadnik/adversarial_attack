"""
This module contains the text_item class
"""
from dataclasses import dataclass
from img import Img
import numpy as np

@dataclass
class TextItem(Img):
    left: int
    top: int
    width: int
    height: int
    conf: float
    text: str
    img: np.array = None
