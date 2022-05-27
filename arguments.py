import argparse

# get all the arguments
def get_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    args = parser.add_argument_group(
        title='Test parameters', description='Parameters for building model'
    )
    args.add_argument(
        '-e', '--epoch', help='epoch number for model training', default=50, type=int
    )
    args.add_argument(
        '-l', '--learning', help='learning rate for model training', default=0.0001, type=float
    )
    args.add_argument(
        '-c', '--nconv', help='number of convolution layers', default=6, type=int
    )
    args.add_argument(
        '-n', '--nheads', help='number of attention heads', default=3, type=int
    )
    parser.add_argument(
        "-o", "--out", help='model output directory', required=True
    )
    args.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        required=False,
        help="Print training progress",
    )

    args = parser.parse_args()
    
    return args