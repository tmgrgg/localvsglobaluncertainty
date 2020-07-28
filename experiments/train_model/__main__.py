import argparse
from experiments.train_model import _default_models
from experiments.train_model.experiment import experiment

parser = argparse.ArgumentParser(description="Model Training")

# model, dataset parameters
parser.add_argument(
    "--model",
    type=str,
    default=None,
    required=True,
    choices=_default_models.keys(),
    metavar="MODEL",
    help="name of model class (default: None)",
)

parser.add_argument(
    "--dataset",
    type=str,
    default=None,
    required=True,
    metavar="DATASET",
    help="name of dataset (default: None)",
)


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
    help="name to give to saved model (default: None)",
)


# training parameters
parser.add_argument(
    "--no_validation",
    action="store_true",
    help="don't validate (i.e. use all data for training)",
)

parser.add_argument(
    "--val_ratio",
    type=float,
    default=0.2,
    required=False,
    metavar="RATIO",
    help="ratio of dataset to use for validation (default: 0.2, ignored if no_validation is True)",
)

parser.add_argument(
    "--epochs",
    type=int,
    required=False,
    default=100,
    metavar="EPOCHS",
    help="number of epochs to train for (default: 100)",
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=128,
    required=False,
    metavar="BATCH_SIZE",
    help="number of examples in a batch (default: 128)",
)

parser.add_argument(
    "--verbose",
    action="store_true",
    help="show live training progress",
)

parser.add_argument(
    "--save_graph",
    action="store_true",
    help="save final training graph",
)


# optimization params
parser.add_argument(
    "--optimizer",
    type=str,
    default='SGD',
    required=False,
    metavar="OPTIMIZER",
    choices=['SGD', 'Adam'],
    help="optimizer to use for gradient descent (default: SGD)",
)

parser.add_argument(
    "--criterion",
    type=str,
    default='CrossEntropyLoss',
    required=False,
    metavar="CRITERION",
    choices=['CrossEntropyLoss'],
    help="optimization criterion to use for gradient descent (default: SGD)",
)

parser.add_argument(
    "--lr_init",
    type=float,
    default=0.1,
    required=False,
    metavar="LR_INIT",
    help="initial learning rate for optimizer (default: 0.1)",
)

parser.add_argument(
    "--lr_final",
    type=float,
    default=0.005,
    required=False,
    metavar="LR_FINAL",
    help="final learning rate for optimizer (default: 0.005)",
)

parser.add_argument(
    "--l2",
    type=float,
    default=1e-4,
    required=False,
    metavar="L2",
    help="l2 regularization for optimizer (default: 0.0001)",
)


parser.add_argument(
    "--momentum",
    type=float,
    default=0.85,
    required=False,
    metavar="MOMENTUM",
    help="momentum for SGD optimizer (default: 0.85)",
)

parser.add_argument(
    "--beta_1",
    type=float,
    default=0.9,
    required=False,
    metavar="BETA_1",
    help="beta_1 for ADAM optimizer (default: 0.9)",
)

parser.add_argument(
    "--beta_2",
    type=float,
    default=0.999,
    required=False,
    metavar="BETA_2",
    help="beta_2 for ADAM optimizer (default: 0.999)",
)


args = parser.parse_args()
experiment(args)
