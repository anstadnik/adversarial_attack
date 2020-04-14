#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This is the main module"""

from multiprocessing import Pool
from typing import List

import pickle
import pytesseract
from tqdm import tqdm
from functools import partial

from helpers import parse_args
from img_ocr import ImgOCR
from text_item import TextItem


def count_matches(c, data: List[TextItem]):
    rez = []
    for i, d in enumerate(tqdm(data, position=c)):
        entry = []
        try:
            d_ = pytesseract.image_to_data(
                d.get_pil(), output_type=pytesseract.Output.DICT, config=f'--psm {c}')
        except Exception as e:
            return None

        for i, t in enumerate(d_['text']):
            conf = d_['conf'][i]
            if t == d.text and isinstance(conf, int):
                entry.append(abs(conf - d.conf))
                break
        else:
            entry.append(-1 * min([0] + [abs(conf - d.conf) for conf in d_['conf'] if
                                         isinstance(conf, int)]))
            if entry[-1] == 0:
                entry[-1] = -1

        entry.append(len(d_['text']))
        rez.append(entry)

    return rez


def main():
    """Main function"""
    args = parse_args()

    img = ImgOCR(path=args.image)
    img.resize((1280, -1))
    # img.show(annotate=True)
    data = img.compute_text_data()
    fun = partial(count_matches, data=data)

    # psms = [3, 6, 7, 8, 9, 13]
    with Pool(7, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as p:
        rez = p.map(fun, range(14))
    print(rez)

    with open('rez.pickle', 'wb') as f:
        return pickle.dump(rez, f)
    # embed(colors="neutral")

    # img.show(annotate=args.annotate)


if __name__ == "__main__":
    main()
