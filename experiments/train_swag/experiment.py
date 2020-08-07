import argparse

from experiments.train_swag import train_swag
from experiments import default_model
from localvglobal.data import loaders
from localvglobal.probabilistic.models.swag import SWAGSampler, SWAGPosterior
from experiments.utils import track, ExperimentTable, CachedExperiment
from localvglobal.training.utils import run_training_epoch
import torch
import numpy as np


parser = argparse.ArgumentParser(description="SWA-Gaussian Training")

# model, dataset parameters
parser.add_argument(
    "--model",
    type=str,
    default=None,
    required=True,
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
    "--rank",
    type=int,
    default=30,
    required=False,
    metavar="RANK",
    help="rank of SWAG subspace",
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
    "--sample_rate",
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
    "--optimizer_path",
    type=str,
    default=None,
    required=True,
    metavar="OPTIMIZER_PATH",
    help="path from which to load pretrained model's optimizer state (default: None)",
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

args = parser.parse_args()


def run(
    posterior_model,
    optimizer,
    criterion,
    train_loader,
    valid_loader,
    swag_epochs,
    sample_rate,
    using_cuda,
    save_graph,
    verbose,
    call
):
    sample_freq = int(len(train_loader.dataset)/(sample_rate*train_loader.batch_size))
    sampler = SWAGSampler(posterior_model, optimizer, sample_freq=sample_freq)

    posterior_model, tracker = train_swag(
        posterior=posterior_model,
        sampler=sampler,
        criterion=criterion,
        train_loader=train_loader,
        valid_loader=valid_loader,
        swag_epochs=swag_epochs,
        verbose=verbose,
        using_cuda=using_cuda
    )

    posterior_model.expected()
    posterior_model.renormalize(train_loader)
    res_train = run_training_epoch(train_loader, posterior_model, criterion,
                                   None, train=False, using_cuda=using_cuda)
    res_valid = run_training_epoch(valid_loader, posterior_model, criterion,
                                   None, train=False, using_cuda=using_cuda)

    res = {
        'posterior_model': posterior_model,
        'loss_train': res_train['loss'],
        'acc_train': res_train['accuracy'],
        'loss_valid': res_valid['loss'],
        'acc_valid': res_valid['accuracy'],
        'loss_test': res_valid['loss'],
        'acc_test': res_valid['accuracy'],
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

# parse model
num_classes = len(np.unique(train_loader.dataset.targets))
model = default_model(args.model, num_classes)
model.load_state_dict(torch.load(args.model_path))
posterior_model = SWAGPosterior(model, rank=args.rank)

if args.cuda:
    posterior_model.cuda()

# parse optimizer
optimizer_cls = getattr(torch.optim, args.optimizer)
if optimizer_cls == torch.optim.SGD:
    # the actual lr passed here will get overwritten by optimizer loading, so ignore it
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
else: #optimizer_cls == torch.optim.Adam:
    optimizer = torch.optim.Adam(model.parameters())

optimizer.load_state_dict(torch.load(args.optimizer_path))

# parse criterion
criterion = getattr(torch.nn, args.criterion)()

exp.run(
    posterior_model=posterior_model,
    optimizer=optimizer,
    criterion=criterion,
    train_loader=train_loader,
    valid_loader=valid_loader,
    swag_epochs=args.epochs,
    sample_rate=args.sample_rate,
    using_cuda=args.cuda,
    save_graph=args.save_graph,
    verbose=args.verbose,
    call=str(args),
)
