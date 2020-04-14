"""
This module contains the text_item class
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image

from img import Img


@dataclass
class TextItem(Img):
    left: int
    top: int
    width: int
    height: int
    conf: float
    text: str
    _img: np.array = field(init=False, repr=False)
    has_img: bool = field(init=False)
    noise: np.array = None

    def get_pil(self) -> Image:
        return Image.fromarray(self.img)

    @property
    def img(self) -> np.array:
        if self._img is None:
            raise RuntimeError("Img is not setted")
        if self.noise:
            return self._img + self.noise
        return self._img

    @img.setter
    def img(self, img: np.array):
        self._img = img
        self.has_img = True
