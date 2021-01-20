#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This is the main module"""

from helpers import parse_args
from img_ocr import ImgOCR
from text_item import TextItem
from genetic_algorithm import GeneticAlgorithm
from ipdb import launch_ipdb_on_exception


def main():
    """Main function"""
    args = parse_args()

    img = ImgOCR(path=args.image)
    img.resize((1280, -1))
    data = img.compute_text_data()
    img.show(annotate=True)

    # d = data[1]
    # model = GeneticAlgorithm(d)
    # model.run()


if __name__ == "__main__":
    with launch_ipdb_on_exception():
        main()
