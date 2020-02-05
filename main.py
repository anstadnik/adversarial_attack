"""This is the main module"""

import cv2

from helpers import parse_args
from myimage import MyImage
from myimage_ga import MyImageGA


def main():
    """Main function"""
    args = parse_args()

    img = MyImageGA(path=args.image, resize=(1280, -1))
    img.compute_data()
    data = img.data

    if len(data['img']) > 8:
        img_ = MyImage(img=data['img'][8])
    img_.show()

if __name__ == "__main__":
    main()
