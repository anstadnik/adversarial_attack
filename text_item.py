"""
This module contains the text_item class
"""
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from img import Img


@dataclass
class TextItem(Img):
    """
    This class represents the text item got from the tesseract.

    This class has bbox data, text, confidence and image. Also it might have a
    noise which is automatically applied to image
    """
    left: int
    top: int
    width: int
    height: int
    conf: float
    text: str
    _img: np.array = field(default=None, init=False, repr=False)
    has_img: bool = field(init=False)
    _noise: np.array = field(default=None, init=False, repr=False)

    def get_pil(self) -> Image:
        """Return PIL.Image"""
        return Image.fromarray(self.img)

    @property
    def img(self) -> np.array:
        """Get image with noise if noise is set"""
        if self._img is None:
            raise RuntimeError("Img is not setted")
        if self.noise is not None:
            return (self._img + self.noise).astype(np.uint8)
        return self._img

    @img.setter
    def img(self, img: np.array):
        self._img = img
        self.has_img = True

    @property
    def noise(self) -> np.array:
        return self._noise

    @noise.setter
    def noise(self, noise: np.array):
        s = self._img + noise
        self._noise = np.clip(noise, -1 * self._img, (255 - self._img))
