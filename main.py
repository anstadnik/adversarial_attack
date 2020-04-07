#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This is the main module"""

from helpers import parse_args
from img import ImgOCR
# from genetic_algorithm import GeneticAlgorithm


def main():
    """Main function"""
    args = parse_args()

    img = ImgOCR(path=args.image)
    img.resize((1280, -1))
    # img.show(annotate=True)
    img.compute_text_item(filter_data=True)
    img.show(annotate=args.annotate)


if __name__ == "__main__":
    main()
