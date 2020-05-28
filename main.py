#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This is the main module"""

from functools import partial
from tqdm import tqdm
import plotly.express as px
import plotly.io as pio
# from ipdb import launch_ipdb_on_exception
import numpy as np
import pickle

from genetic_algorithm import GeneticAlgorithm
from mnist.mnist import generate_data
from multiprocessing.pool import Pool
from multiprocessing import cpu_count

pio.renderers.default = 'browser'

def pred(kwargs):
    v = kwargs.pop('v', None)
    return GeneticAlgorithm(**kwargs).run(v=v)

def main():
    """Main function"""
    inputs, targets, reals = generate_data(1)

    # d = data[1]
    # px.imshow(inputs[0].reshape(28, 28)).show()
    params = []
    img = inputs[0]
    target = np.nonzero(targets[0])[0][0]
    for pool_size in 2 ** np.arange(2, 12, step=3):
        for mutation_rate in [0.3, 0.1, 0.05, 0.01]:
            for n_of_elite in [1, 3]:
    # for pool_size in 2 ** np.arange(2, 6, step=1):
    #     for mutation_rate in [0.3]:
    #         for n_of_elite in [1]:
                params.append({'img': img,
                               'target': target,
                               'pool_size': pool_size,
                               'mutation_rate': mutation_rate,
                               'n_of_elite': n_of_elite,
                               'n_iter': 2,
                               'v': False})

    with Pool(cpu_count(), initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as p:
        rez = list(tqdm(p.imap(pred, params), total=len(params)))

    rez = [{**{k: v for k, v in p.items() if k not in {'img', 'target', 'v'}},
            **{'iterations_needed': r[0], 'best_confidence': r[1]}}
            for p, r in zip(params, rez)]

    with open('rezults.pickle', 'wb') as f:
        pickle.dump(rez, f)

    # model = GeneticAlgorithm(inputs[0], np.nonzero(targets[0])[0][0], n_iter=10)
    # rez = model.run(v=True)
    # px.imshow(rez.reshape(28, 28)).show()


if __name__ == "__main__":
    main()
