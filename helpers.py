"""This is the module with miscellaneous functions"""
import argparse


def parse_args() -> argparse.Namespace:
    """Parse arguments from the command line

    This function parses arguments and returns a class-like object with values

    Returns:
        argparse.Namespace: [TODO:description]
    """
    parser = argparse.ArgumentParser(description="This program allows user to detect\
                                     and recognize printed text on images, as well as\
                                     get the contents of the image as a text")
    parser.add_argument("-i", "--image", default=None, help="Path to the image", required=True)
    # parser.add_argument("-d", "--display", action="store_true",
    #                     default=False, help="Display the image")
    parser.add_argument("-a", "--annotate", action="store_true",
                        default=False, help="Annotate the image")
    parser.add_argument("-p", "--pop_size", default=None, help="Population size")
    parser.add_argument("-m", "--mutation_rate", default=None,
                        help="Mutation rate")
    return parser.parse_args()
