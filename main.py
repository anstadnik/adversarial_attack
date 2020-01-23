"""This is the main module"""

import cv2

from helpers import parse_args, get_img, display_image
from image import get_data_from_image, get_str_from_image


def main():
    """Main function"""
    args = parse_args()

    img = get_img(args.image, (1280, -1))

    if args.display:
        display_image(img)

    data = get_data_from_image(img=img)
    print(data)
    __import__('ipdb').set_trace()

if __name__ == "__main__":
    main()
