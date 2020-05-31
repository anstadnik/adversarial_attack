#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This is the main module"""

from helpers import parse_args
from img_ocr import ImgOCR
# from attack import gen_noise

def main():
    """Main function"""
    args = parse_args()

    img = ImgOCR(path=args.image)
    img.resize((1280, -1))
    data = img.compute_text_data()
    # img.process(pop_size_=int(args.pop_size), mutation_rate_=float(args.mutation_rate))
    img.show(annotate=True)

    # with open('img.pickle', 'wb') as f:
    #     pickle.dump(img, f)


if __name__ == "__main__":
    # with launch_ipdb_on_exception():
    main()
