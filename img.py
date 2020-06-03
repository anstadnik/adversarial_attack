"""
This module contains a base class which implements simple image
"""
import cv2
import pixcat
from PIL import Image


class Img():
    """This class is capable of storing and managing the image"""

    def __init__(self, *, img=None, path=None):
        """
        Initialize the class

        :param img np.array: raw image
        :param path str: path to image
        """
        if img is not None:
            self.img = img
        elif path is not None:
            self.load_img(path)
        else:
            raise RuntimeError("You should provide the image")

    def load_img(self, path: str):
        """Load image from the disk"""
        self.img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    # pylint: disable=too-many-arguments
    def mouse_callback(self, event, x, y, flags, param):
        """Callback for using with cv2.imshow"""

    def update(self, *, img=None, method=None):
        img = img if img is not None else self.img
        cv2.imshow('image', img)

    def show(self, *, img=None, method=None):
        """Display the image

        This function displays images in a various ways

        Args:
            method (string, optional): kitty, jupyter or None
        """
        img = img if img is not None else self.img
        if img is None:
            return None
        if method == "kitty":
            pixcat.Image(Image.fromarray(img)).thumbnail(
                512).show(align="left")
        elif method == "jupyter":
            return Image.fromarray(img)
        else:
            cv2.namedWindow('image')
            cv2.setMouseCallback('image', self.mouse_callback)
            cv2.imshow('image', img)
            while True:
                key = cv2.waitKey(0)
                if key == 27:
                    break
            cv2.destroyAllWindows()
        return None

    def resize(self, resize):
        """
        Handy function to resize image

        If resize argument is float, it specifies the scaling percent. If it's a tuple,
        it specifies the dimensions of the resulting image. Additionally, if the 2-nd parameter
        equals -1, it is calculated so the proportions of the image are preserved.

        Args:
            resize (float or tuple, optional): look at the description
        """
        if isinstance(resize, float):
            assert 0 < resize <= 1
            width = int(self.img.shape[1] * resize)
            height = int(self.img.shape[0] * resize)
            dim = (width, height)
        elif isinstance(resize, (tuple, list)):
            if resize[1] == -1:
                dim = (resize[0], self.img.shape[0] *
                       resize[0] // self.img.shape[1])
            else:
                dim = resize
        else:
            raise RuntimeError("Wrong resize argument: " + str(resize))
        self.img = cv2.resize(self.img, dim)
