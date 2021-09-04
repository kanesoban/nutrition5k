import argparse


def parse_args():
    """ Parse the arguments."""
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config_path', help='Name of the base config file without extension.', required=True)
    return parser.parse_args()
