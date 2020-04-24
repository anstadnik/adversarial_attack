"""
This module implements genetic algorithm which is intended to find the noise
such text on that image with this noise will not be found.
"""

from copy import copy
from statistics import mean

import numpy as np
import pytesseract
from tqdm import tqdm, trange
import pickle

from text_item import TextItem
from multiprocessing.pool import Pool
from multiprocessing import cpu_count


def score(ti: TextItem, c=6) -> int:
    """
    Compute the score for the image

    The score is the confidence of the recognized text

    :param img TextItem: Input text item with image
    :param c int: psm argument for tesseract
    :rtype int: score
    """
    img = ti.get_pil()
    d_ = pytesseract.image_to_data(
        img, output_type=pytesseract.Output.DICT, config=f'--psm {c} --oem 0')
    # Add normalization by noise sum
    return max([c for c in d_['conf'] if isinstance(c, int)] + [0])


def crossover(f: TextItem, m: TextItem, mr: float) -> TextItem:
    """
    Crossover for genetic algorithm.

    This function performs crossover for given mother and father and mutation
    for the child after

    :param f TextItem: Father
    :param m TextItem: Mother
    :param mr float: mutation rate
    :rtype TextItem: Child
    """
    # Probably should resize here
    child = copy(f)
    b_opts = [True, False]
    s = f.noise.shape
    child.noise = np.where(np.random.choice(b_opts, size=s), f.noise, m.noise)
    mask = np.random.choice(b_opts, size=s, p=((1 - mr), mr))
    child.noise[mask] = np.random.randint(0, 10, size=s, dtype='uint8')[mask]
    return child


class GeneticAlgorithm():
    """
    This is the class which implements the genetic algorithm.
    """

    def __init__(self, ti: TextItem, pool_size: int = 1000, mutation_rate: float
                 = 0.0001, n_iter: int = int(1e6)):
        self.pool_size = pool_size
        self.mutation_rate = mutation_rate
        self.n_iter = n_iter
        self.pool = []
        self.scores = None
        self.setup(ti)

    def run(self):
        with trange(self.n_iter) as t:
            for i in t:
                self.comp_score()
                if min(self.scores) == 0:
                    return self.pool[np.argmin(self.scores)]
                t.set_description(
                    f'Score: mean = {mean(self.scores)}, min = {min(self.scores)}')
                self.evolve()
                if not i % 10:
                    tqdm.write(
                        f'Score at {i} iteration: mean = {mean(self.scores)}, min = {min(self.scores)}')
                    with open('pool.pickle', 'wb') as f:
                        pickle.dump(self.pool, f)

    def setup(self, ti: TextItem):
        for _ in range(self.pool_size):
            c = copy(ti)
            c.noise = np.random.randint(0, 10, size=c.img.shape, dtype='uint8')
            self.pool.append(c)

    def comp_score(self):
        n_cpu = cpu_count() - 4
        with Pool(n_cpu, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as p:
            self.scores = list(tqdm(
                p.imap(score, self.pool),
                desc=f'Computing scores with {n_cpu} cores',
                total=len(self.pool), leave=False))

    def evolve(self):
        pool = []
        scores = (100 - np.array(self.scores)) ** 3
        scores = scores / sum(scores)
        n_min_to_save = 20
        for _ in trange(self.pool_size - n_min_to_save, desc='Updating pool', leave=False):
            f, m = np.random.choice(self.pool, size=2, replace=False,
                                    p=scores)
            child = crossover(f, m, self.mutation_rate)
            pool.append(child)
        pool.extend(sorted(self.pool, key = lambda ti: ti.score)[:20])
        assert (len(pool) == self.pool_size)
        self.pool = pool
