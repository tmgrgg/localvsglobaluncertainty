import argparse
from experiments import _default_models
from experiments.train_swag import train_swag

parser = argparse.ArgumentParser(description="SWA-Gaussian Training")

# model, dataset parameters
parser.add_argument(
    "--model",
    type=str,
    default=None,
    required=True,
    choices=_default_models.keys(),
    metavar="MODEL",
    help="name of pretrained model class (default: None)",
)

parser.add_argument(
    "--model_path",
    type=str,
    default=None,
    required=True,
    metavar="MODEL_PATH",
    help="path from which to load pretrained model (default: None)",
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
    help="path to results/storage directory (default: None)",
)

parser.add_argument(
    "--name",
    type=str,
    default=None,
    required=True,
    metavar="NAME",
    help="name attributed to this call (default: None)",
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
    metavar="VALIDATION RATIO",
    help="ratio of dataset to use for validation (default: 0.2, ignored if no_validation is True)",
)

parser.add_argument(
    "--epochs",
    type=int,
    required=False,
    default=100,
    metavar="EPOCHS",
    help="number of epochs to train SWA-Gaussian for (default: 100)",
)

parser.add_argument(
    "--swag_sample_rate",
    type=float,
    default=1.0,
    required=False,
    metavar="SWAG_SAMPLERATIO",
    help="number of swag samples to draw per epoch (default: 1.0)",
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
    "--swag_lr",
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

from experiments import default_model
from localvglobal.data import loaders
from localvglobal.probabilistic.models.swag import SWAGSampler, SWAGPosterior
from torch.optim import SGD
from experiments.utils import track, ExperimentTable, CachedExperiment
from localvglobal.probabilistic.utils import bayesian_model_averaging
import torch
import numpy as np


def run(
    posterior,
    sampler,
    criterion,
    train_loader,
    valid_loader,
    swag_epochs,
    verbose,
    using_cuda,
    save_graph,
    N,
    rank,
    swa_lr,
    call,
):
    posterior, tracker = train_swag(
        posterior=posterior,
        sampler=sampler,
        criterion=criterion,
        train_loader=train_loader,
        valid_loader=valid_loader,
        swag_epochs=swag_epochs,
        verbose=verbose,
        using_cuda=using_cuda
    )

    res_train = bayesian_model_averaging(
        posterior=posterior,
        data_loader=train_loader,
        criterion=criterion,
        using_cuda=using_cuda,
        N=N
    )
    res_valid = bayesian_model_averaging(
        posterior=posterior,
        data_loader=valid_loader,
        criterion=criterion,
        using_cuda=using_cuda,
        N=N
    )

    res = {
        'posterior': posterior,
        'loss_train': res_train['loss'],
        'acc_train': res_train['accuracy'],
        'loss_valid': res_valid['loss'],
        'acc_valid': res_valid['accuracy'],
        'loss_test': res_valid['loss'],
        'acc_test': res_valid['accuracy'],
        'rank': rank
        'swa_lr': swa_lr,
    }

    if save_graph:
        res.update({'training_graph': tracker.get_image()})

    res.update({'call': call})

    return res

    table = ExperimentTable(args.dir, args.name)
    exp = CachedExperiment(table, run)

    # load data
    data_loaders = loaders(args.dataset)(
        dir=args.dir,
        use_validation=not args.no_validation,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
    )

    train_loader = data_loaders['train']
    valid_loader = data_loaders['valid']

    # parse model and posteiror
    num_classes = len(np.unique(train_loader.dataset.targets))
    model = default_model(args.model, num_classes)
    posterior = SWAGPosterior(model, args.rank)

    # parse optimizer/sampler
    optimizer_cls = getattr(torch.optim, args.optimizer)
    if optimizer_cls
    sampler = SWAGSampler(posterior, optimizer_cls,  lr=args.lr_init, weight_decay=args.l2, momentum=args.momentum,  betas=(args.beta_1, args.beta_2)))

# load in pretrained model
model = DenseNet3(depth, num_classes).cuda()
model.load_state_dict(torch.load(MODEL_PATH))

posterior = SWAGPosterior(model, rank=RANK)
sampler = SWAGSampler(posterior, SGD, SAMPLE_FREQ, lr=SWA_LR, weight_decay=L2, momentum=SWA_MOMENTUM)


# Get pre-training metrics
res_train = run_training_epoch(train_loader, posterior, criterion,
                      None, train=False, using_cuda=True)
res_valid = run_training_epoch(valid_loader, posterior, criterion,
                      None, train=False, using_cuda=True)
track(tracker, res_train, res_valid, 'SWA')

