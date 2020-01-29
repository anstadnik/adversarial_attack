"""This is the main module"""

import cv2

from helpers import parse_args
from myimage import MyImage


def main():
    """Main function"""
    args = parse_args()

    img = MyImage(args.image, (1280, -1), kitty=True)
    img.compute_data()
    img.show()

if __name__ == "__main__":
    main()
