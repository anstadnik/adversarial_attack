"""This is the main class for genetic algorithm."""
from tqdm import trange

from myimage_ga import MyImageGA


class GeneticAlgorithm():
    """This class implements the high-level genetic algorithm."""

    def __init__(self, path=None, img=None, n=100, noise_val=10, **kwargs):
        """Initialize the class

        Args:
            n (int, optional): Size of the population
            img (str, nd.array): Image
            noise_val (int, optional): Noise applied to the image
            kwargs (kwargs): Parameters for the MyImageGA
        """
        self.n = n
        self.img = MyImageGA(img=img, path=path, noise_val=0, **kwargs)
        self.data = self.img.compute_data()
        self.population = [MyImageGA(noise_val=noise_val, img=self.img.img, **kwargs)
                           for _ in range(self.n)]

    def compute_fitness(self):
        """Compute the fitness score for every item in the population
        """
        for item in self.population:
            item.compute_fitness(self.data)

    def run(self, n_iter=100):
        """Run the genetic algorithm

        Run genetic algorithm n_iter times

        Args:
            n_iter (int, optional): Amount of the iterations
        """
        for _ in trange(n_iter):
            self.compute_fitness()
            quit()
            # Make a pool
            # pool = ga.Pool(MyImage, img, random, 100)

            # Perform the evolution algorithm.
            # pool.compute_fitness()
            # for candidate in pool:
            #   candidate.compute_data_with_noise()
            #   candidate.get_fitness()

            # Update the pool.
