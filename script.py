import cv2

from helpers import parse_args
from myimage import MyImage


path = "test.png"

img = MyImage(path, (1280, -1))
img.compute_data()
img.show()
