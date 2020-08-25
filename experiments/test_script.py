from experiments.utils import ExperimentDirectory
import argparse

if __name__ == '__main__':
    # Specify Arguments
    parser = argparse.ArgumentParser(description="Model Training")

    # directory parameters
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        required=True,
        metavar="DIRECTORY",
        help="path to experiment directory (default: None)",
    )

    parser.add_argument(
        "--name",
        type=str,
        default=None,
        required=True,
        metavar="NAME",
        help="experiment name (default: None)",
    )

    args = parser.parse_args()


def experiment(args):
    experiment = ExperimentDirectory(args.dir, args.name)
    experiment.add_folder('predictions')
    #experiment.add_table('predictions')

if __name__ == '__main__':
    experiment(args)
