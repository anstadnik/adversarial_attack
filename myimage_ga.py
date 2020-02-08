"""This is the overloaded MyImage specialized for the generic algorithm."""
import cv2
import numpy as np

from myimage import MyImage

class MyImageGA(MyImage):
    """This class implements MyImage with the noise"""
    _img = None

    def __init__(self, noise_val=10, **kwargs):
        """Initialize the class

        Args:
            noise (int, optional): Noise applied to the image
            kwargs (kwargs): arguments for the MyImage
        """
        self._img = None
        self.noise = None
        self.noise_val = noise_val
        super().__init__(**kwargs)

    @property
    def img(self):
        """Overloaded getter for the img

        It applies the noise to the image
        """
        if self._img is None:
            return None
        return self._img + self.noise

    @img.setter
    def img(self, img):
        """Sets the image

        This function sets the image and creates the noise.

        Args:
            img (numpy.ndarray): Image
        """
        if img is not None:
            if img.ndim == 3:
                self._img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                self._img = img
            if self.noise_val:
                self.noise = np.random.randint(self.noise_val, size=self._img.shape, dtype=np.uint8)
            else:
                self.noise = np.zeros(self._img.shape, dtype=np.uint8)
            mask = (self._img + self.noise) > 255
            self.noise[mask] -= self._img[mask] - 255
        else:
            self._img = None

def get_overlaps(self, item):
    """Calculate overlapping with the item

    Args:
        row (pd.Series): row from the data from the compute_data method
    """

    for _, row in data.iterrows():
        hoverlaps = (r1.left > r2.right) or (r1.right < r2.left):
            hoverlaps = False
        if (r1.top < r2.bottom) or (r1.bottom > r2.top):
            voverlaps = False
        return hoverlaps and voverlaps
    

    def compute_fitness(self, data, **kwargs):
        """Compute the fitness for the given data

        This function takes information about the initial text position, and
        computes the fitness score for that region using the noise.

        Args:
            data (pd.DataFrame): data from the compute_data method
            kwargs (kwars): arguments for the compute_data method
        """
        data_ = super().compute_data(**kwargs)
        rezults = []
        for _, row in data.iterrows():
            __import__('ipdb').set_trace()
        return rezults
