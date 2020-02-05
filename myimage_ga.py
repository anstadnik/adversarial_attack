import cv2
import numpy as np

from myimage import MyImage


class MyImageGA(MyImage):
    _img = None

    def __init__(self, **kwargs):
        self._img = None
        super().__init__(**kwargs)

    @property
    def img(self):
        if self._img is None:
            return None
        return self._img + self.noise

    @img.setter
    def img(self, img):
        if img is not None:
            if img.ndim == 3:
                self._img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            self.noise = np.random.randint(10, size=self._img.shape, dtype=np.uint8)
            mask = (self._img + self.noise) > 255
            self.noise[mask] -= self._img[mask] - 255
        else:
            self._img = None
