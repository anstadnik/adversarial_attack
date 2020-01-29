"""This is the module with miscellaneous functions"""
import argparse


def parse_args() -> argparse.Namespace:
    """Parse arguments from the command line

    This function parses arguments and returns a class-like object with values

    Returns:
        argparse.Namespace: [TODO:description]
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", default=None, help="Path to the image", required=True)
    parser.add_argument("-d", "--display", action="store_true",
                        default=False, help="Display the image")
    return parser.parse_args()
