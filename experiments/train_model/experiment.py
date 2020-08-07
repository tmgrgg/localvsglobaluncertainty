import argparse
from experiments.train_model import train_model

from localvglobal.data import loaders
import localvglobal.models as models
import torch.nn
import numpy as np
from experiments.utils import ExperimentTable, CachedExperiment

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


def run(
    model,
    name,
    optimizer,
    criterion,
    train_loader,
    valid_loader,
    lr_init,
    lr_final,
    epochs,
    using_cuda,
    verbose,
    save_graph,
    call='',
):
    model, optimizer, tracker, res_train, res_valid = train_model(
        model=model,
        name=name,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        valid_loader=valid_loader,
        lr_init=lr_init,
        lr_final=lr_final,
        epochs=epochs,
        using_cuda=using_cuda,
        verbose=verbose,
    )

    res = {
        'model': model,
        'optimizer': optimizer,
        'name': name,
        'loss_train': res_train['loss'],
        'loss_valid': res_valid['loss'],
        'loss_acc': res_train['accuracy'],
        'loss_acc': res_valid['accuracy'],
        'call': call # for documentation
    }

    if save_graph:
        res.update({'training_graph': tracker.get_image()})

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
model_cfg = getattr(models, args.model)
model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)

# parse optimizer
optimizer_cls = getattr(torch.optim, args.optimizer)
if optimizer_cls == torch.optim.SGD:
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_init, weight_decay=args.l2, momentum=args.momentum)
else: #optimizer_cls == torch.optim.Adam:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_init, weight_decay=args.l2, betas=(args.beta_1, args.beta_2))

# parse criterion
criterion = getattr(torch.nn, args.criterion)()

exp.run(
    model=model,
    name=args.name,
    optimizer=optimizer,
    criterion=criterion,
    train_loader=train_loader,
    valid_loader=valid_loader,
    lr_init=args.lr_init,
    lr_final=args.lr_final,
    epochs=args.epochs,
    using_cuda=args.cuda,
    verbose=args.verbose,
    save_graph=args.save_graph,
    call=str(args),
)
