from localvglobal.training.utils import seed
from localvglobal.probabilistic.models.swag import SWAGPosterior, SWAGSampler
from experiments.utils import ExperimentDirectory
from localvglobal.data import loaders
import localvglobal.models as models
import torch
import numpy as np
import argparse
from localvglobal.training.utils import run_training_epoch

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

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        required=True,
        metavar="DATASET",
        help="name of dataset (default: None)",
    )

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
        "--criterion",
        type=str,
        default='CrossEntropyLoss',
        required=False,
        metavar="CRITERION",
        choices=['CrossEntropyLoss'],
        help="optimization criterion to use for gradient descent (default: SGD)",
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
        "--batch_size",
        type=int,
        default=128,
        required=False,
        metavar="BATCH_SIZE",
        help="number of examples in a batch (default: 128)",
    )

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
        "--training_epochs",
        type=int,
        required=False,
        default=100,
        metavar="EPOCHS",
        help="number of epochs to train initiaal solution for (default: 100)",
    )

    parser.add_argument(
        "--swag_epochs",
        type=int,
        required=False,
        default=100,
        metavar="EPOCHS",
        help="number of epochs to train SWAG solution for (default: 100)",
    )

    parser.add_argument(
        "--sample_rate",
        type=float,
        required=False,
        default=1.0,
        metavar="SAMPLE RAATE",
        help="samples drawn per SWAG epoch (default: 1.0)",
    )

    parser.add_argument(
        "--rank",
        type=int,
        required=False,
        default=30,
        metavar="RANK",
        help="rank of SWAG solutions (default: 30)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        metavar="SEED",
        help="seed with which to train model",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="show live training progress",
    )

    parser.add_argument(
        "--cuda",
        action="store_true",
        help="use GPU device for training if available",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        required=False,
        metavar="NUM",
        default=0,
        help="number of workers",
    )

    args = parser.parse_args()


def experiment(args):
    experiment = ExperimentDirectory(args.dir, args.name)

    # parse optimizer and criterion, model_cfg and criterion
    optimizer_cls = getattr(torch.optim, args.optimizer)
    criterion = getattr(torch.nn, args.criterion)()
    model_cfg = getattr(models, args.model)

    # load data
    data_loaders = loaders(args.dataset)(
        dir=experiment.path,
        use_validation=not args.no_validation,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        test_transforms=model_cfg.transform_test,
        train_transforms=model_cfg.transform_train,
        num_workers=args.num_workers
    )

    train_loader = data_loaders['train']
    valid_loader = data_loaders['valid']
    num_classes = len(np.unique(train_loader.dataset.targets))

    model = model_cfg.model(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
    posterior = SWAGPosterior(model, rank=args.rank)

    if args.cuda:
        posterior.cuda()

    posterior_name = 'posterior_{}'.format(args.seed)
    print('predicting with {}'.format(posterior_name))
    posterior_state_dict, _ = experiment.cached_state_dict(posterior_name, folder='posteriors')
    posterior.load_state_dict(posterior_state_dict)
    posterior.expected()
    posterior.renormalize(train_loader)
    print(run_training_epoch(
        valid_loader,
        posterior,
        criterion,
        None,
        train=False,
        using_cuda=True,
    ))


if __name__ == '__main__':
    experiment(args)
