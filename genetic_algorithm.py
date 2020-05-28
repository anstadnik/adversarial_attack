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
                 mutation_rate: float = 0.01, n_iter: int = int(1e6),
                 tqdm_pos=None):
        self.pool_size = pool_size
        self.target = target
        self.mutation_rate = mutation_rate
        self.n_iter = n_iter
        self.scores = None
        self.img = img
        self.n_of_elite = n_of_elite
        self.needed_iterations = None
        self.tqdm_pos = tqdm_pos
        self.pool = [np.random.rand(*img.shape) / 5 - 0.1
                     for _ in range(pool_size)]
        self.comp_score(v=False)
        self.first_score = mean(self.scores)

    def finish(self):
        self.best_confidence = max(self.scores)
        # return self.img + self.pool[np.argmax(self.scores)]
        return self.first_score, self.iterations_needed, self.best_confidence, self.img + self.pool[np.argmax(self.scores)]

    def run(self, v=0):
        scores = []
        with trange(self.n_iter, disable=not v>=1, position=self.tqdm_pos) as t:
            for i in t:
                self.comp_score(v=v>=2)
                t.set_description(
                        f'Score: mean={mean(self.scores):.3e}, '
                        f'max={max(self.scores):.3e}, '
                        f'std={np.std(self.scores):.3e}, '
                        f'diff={self.first_score - mean(self.scores):.3e}')
                if max(self.scores) > 0.95:
                    self.iterations_needed = i
                    return self.finish()
                self.evolve()
                scores.append(mean(self.scores))
                self.iterations_needed = i
        return self.finish()

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
