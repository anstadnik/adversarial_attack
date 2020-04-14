#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module compared different psm arguments for tesseract"""

import pickle
from functools import partial
from multiprocessing import Pool
from typing import List

import plotly.express as px
import pytesseract
from pandas import DataFrame
from plotly.offline import plot
from tqdm import tqdm

from helpers import parse_args
from img_ocr import ImgOCR
from text_item import TextItem


def count_matches(c, data: List[TextItem]):
    rez = []
    for d in tqdm(data, position=c):
        entry = []
        try:
            d_ = pytesseract.image_to_data(
                d.get_pil(), output_type=pytesseract.Output.DICT, config=f'--psm {c}')
        except Exception as e:
            return None

        # __import__('ipdb').set_trace()
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

    ###########################################################################
    #                Generate data for different psm arguments                #
    ###########################################################################

    with Pool(7, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as p:
        rez = p.map(fun, range(14))

    with open('rez.pickle', 'wb') as f:
        return pickle.dump(rez, f)

    ###########################################################################
    #                     Plot resulst of the recognition                     #
    ###########################################################################

    # with open('rez.pickle', 'rb') as f:
    #     rez = pickle.load(f)

    l = [[i, *v_] for i, v in enumerate(rez) if v is not None for v_ in v]
    d = {k:v for k, v in zip(['s', 'diff', 'len'], zip(*sorted(l, key=lambda arr: arr[1])))}
    df = DataFrame(d)
    df.loc[df['diff'] == -1, 'diff'] = 100
    plot(px.line(df.sort_values(by=['diff']), y='diff', color='s'))

    # The best psm is 6


if __name__ == "__main__":
    main()
