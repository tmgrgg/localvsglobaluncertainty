from experiments.train_swag_from_scratch import *
import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model Training")

    # model, dataset parameters
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        required=True,
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
        "--training_epochs",
        type=int,
        required=False,
        default=100,
        metavar="TRAINING_EPOCHS",
        help="number of epochs to train for (default: 100)",
    )

    parser.add_argument(
        "--swag_epochs",
        type=int,
        required=False,
        default=100,
        metavar="SWAG_EPOCHS",
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

    parser.add_argument(
        "--cuda",
        action="store_true",
        help="use GPU device for training if available",
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
        default=0.9,
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

    parser.add_argument(
        "--rank",
        type=int,
        default=30,
        required=False,
        metavar="RANK",
        help="rank of SWAG subspace",
    )

    parser.add_argument(
        "--sample_rate",
        type=float,
        default=1.0,
        required=False,
        metavar="SWAG_SAMPLERATIO",
        help="number of swag samples to draw per epoch (default: 1.0)",
    )

    args = parser.parse_args()


def experiment(args):
    exp_dir = args.dir + '/' + args.name
    if not os.path.isdir(exp_dir):
        os.mkdir(exp_dir)
    args.dir = exp_dir
    args.name = 'train_model'
    args.epochs = args.training_epochs
    exp1 = train_model(args)
    args.name = 'train_swag_from_pretrained'
    args.epochs = args.swag_epochs
    args.model_path = exp1._table.get_model_path(0)
    args.optimizer_path = exp1._table.get_optim_path(0)
    exp2 = train_swag_from_pretrained(args)
    return exp2


if __name__ == '__main__':
    experiment(args)
