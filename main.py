"""This is the main module"""

from helpers import parse_args
from genetic_algorithm import GeneticAlgorithm


def main():
    """Main function"""
    args = parse_args()

    model = GeneticAlgorithm(path=args.image, noise_val=0, resize=(1280, -1), n=10)
    model.run()
    # img = MyImageGA(noise_val=0, path=args.image, resize=(1280, -1))
    # img.compute_data()

    # data = img.data


if __name__ == "__main__":
    main()
