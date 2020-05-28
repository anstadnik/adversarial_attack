"""
This module implements genetic algorithm which is intended to find the noise
such text on that image with this noise will not be found.
"""

from copy import copy
from statistics import mean

import numpy as np
from tqdm import tqdm, trange
import pickle

from multiprocessing.pool import Pool
from multiprocessing import cpu_count
from mnist.mnist import pred
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'


def crossover(n_1, n_2, mr: float):
    """
    Crossover for genetic algorithm.

    This function performs crossover for given mother and father and mutation
    for the child after

    :param n_1 TextItem: Father
    :param n_2 TextItem: Mother
    :param mr float: mutation rate
    :rtype TextItem: Child
    """
    # Probably should resize here
    # child = copy(f)
    s = n_1.shape
    child = np.empty(s)
    b_opts = [True, False]
    child = np.where(np.random.choice(b_opts, size=s), n_1, n_2)
    # child = (n1 + n2) / 2
    mask = np.random.choice(b_opts, size=s, p=((1 - mr), mr))
    # __import__('ipdb').set_trace()
    child[mask] = (np.random.rand(*s) / 5 - 0.1)[mask]
    child = np.minimum(np.full(child.shape, 0.2), child)
    child = np.maximum(np.full(child.shape, -0.2), child)
    return child


class GeneticAlgorithm():
    """
    This is the class which implements the genetic algorithm.
    """

    def __init__(self, img, target=0, n_of_elite: int = 2, pool_size: int = 1000,
                 mutation_rate: float = 0.01, n_iter: int = int(1e6)):
        self.pool_size = pool_size
        self.target = target
        self.mutation_rate = mutation_rate
        self.n_iter = n_iter
        self.scores = None
        self.img = img
        self.n_of_elite = n_of_elite
        self.needed_iterations = None
        self.pool = [np.random.rand(*img.shape) / 5 - 0.1
                     for _ in range(pool_size)]

    def finish(self, i):
        self.iterations_needed = i
        self.best_confidence = max(self.scores)
        # return self.img + self.pool[np.argmax(self.scores)]
        return self.iterations_needed, self.best_confidence, self.img + self.pool[np.argmax(self.scores)]

    def run(self, v=False):
        scores = []
        for i in (t := trange(self.n_iter, disable=not v)):
            self.comp_score(v=v)
            if max(self.scores) > 0.95:
                t.write(f'Got {max(self.scores)} confidence at {i} iteration')
                return self.finish(i)
            t.set_description(
                f'Score: mean = {mean(self.scores)}, '
                f'max = {max(self.scores)}, std = {np.std(self.scores)}')
            self.evolve()
            if not i % 10 and v:
                t.write(f'Score at {i} iteration: mean = {mean(self.scores)}, '
                        f'max = {max(self.scores)}, std = {np.std(self.scores)}')
                # with open('pool.pickle', 'wb') as f:
                #     pickle.dump(self.pool, f)
            scores.append(mean(self.scores))
            if i and not i % 100:
                px.line(y=scores).show()
        return self.finish(i)

    def comp_score(self, v=True):
        """
        Compute the confidence of the recognized text for the images
        """
        self.scores = [r[self.target]
                       for r in pred(self.img + self.pool, v)]

    def evolve(self):
        pool = []
        # scores = (100 - np.array(self.scores)) ** 3
        scores = self.scores
        scores = scores / sum(scores)
        # for _ in trange(self.pool_size - n_min_to_save, desc='Updating pool', leave=False):
        for _ in range(self.pool_size - self.n_of_elite):
            f, m = np.random.choice(range(self.pool_size), size=2, replace=False,
                                    p=scores)
            f, m = self.pool[f], self.pool[m]
            child = crossover(f, m, self.mutation_rate)
            pool.append(child)
        # pool.extend(sorted(self.pool, key = lambda ti: ti.score)[:20])
        pool.extend([self.pool[i]
                     for i in np.argsort(scores)[-self.n_of_elite:]])
        assert (len(pool) == self.pool_size)
        self.pool = pool
