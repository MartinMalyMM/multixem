# coding: utf-8
import argparse
from . import __version__


def create_parser():
    """
    Create the argument parser for the command-line interface.

    Returns:
        argparse.ArgumentParser: The argument parser object.
    """
    parser = argparse.ArgumentParser(
        prog='multixem',
        description='Refinement pipeline for multiple data sets in structure biology.',
        )
    parser.add_argument('-v', '--version', action='version',
                        version=__version__, help="show version and exit")
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    print("Arguments parsed:", args)


if __name__ == '__main__':
    main()