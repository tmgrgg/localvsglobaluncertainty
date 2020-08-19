from localvglobal.training.utils import seed
from experiments.train_model import train_model
from experiments.train_swag_from_pretrained import train_swag_from_pretrained
from tqdm import tqdm
from localvglobal.probabilistic.models.swag import SWAGPosterior, SWAGSampler
from experiments.utils import ExperimentDirectory
from localvglobal.data import loaders
import localvglobal.models as models
import torch
import numpy as np
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

    args = parser.parse_args()


def experiment(args):
    experiment = ExperimentDirectory(args.dir, args.name)

    # load data
    data_loaders = loaders(args.dataset)(
        dir=experiment.path,
        use_validation=not args.no_validation,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
    )

    train_loader = data_loaders['train']
    valid_loader = data_loaders['valid']
    num_classes = len(np.unique(train_loader.dataset.targets))

    # parse optimizer and criterion
    optimizer_cls = getattr(torch.optim, args.optimizer)
    criterion = getattr(torch.nn, args.criterion)()
    model_cfg = getattr(models, args.model)

    # parse optimizer and criterion
    optimizer_cls = getattr(torch.optim, args.optimizer)
    criterion = getattr(torch.nn, args.criterion)()


    model = model_cfg.model(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)

    if args.cuda:
        model.cuda()

    if optimizer_cls == torch.optim.SGD:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_init, weight_decay=args.l2,
                                    momentum=args.momentum)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_init, weight_decay=args.l2,
                                     betas=(args.beta_1, args.beta_2))

    model_name = 'model_{}'.format(args.seed)
    model_state_dict, cache_model = experiment.cached_state_dict(model_name)
    optim_name = 'optim_{}'.format(args.seed)
    optim_state_dict, cache_optim = experiment.cached_state_dict(optim_name, folder='optims')

    if cache_model or cache_optim:
        seed(args.seed)

        model, optimizer, tracker, res_train, res_valid = train_model(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            valid_loader=valid_loader,
            lr_init=args.lr_init,
            lr_final=args.lr_final,
            epochs=args.training_epochs,
            using_cuda=args.cuda,
            verbose=args.verbose
        )
        cache_model(model.state_dict())
        cache_optim(optimizer.state_dict())
    else:
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optim_state_dict)

    posterior = SWAGPosterior(model, rank=args.rank)

    if args.cuda:
        posterior.cuda()

    posterior_name = 'posterior_{}'.format(args.seed)
    posterior_state_dict, cache_posterior = experiment.cached_state_dict(posterior_name, folder='posteriors')

    if cache_posterior:
        sample_freq = int(len(train_loader.dataset) / (train_loader.batch_size * args.sample_rate))
        sampler = SWAGSampler(posterior, optimizer, sample_freq=sample_freq)
        posterior, tracker = train_swag_from_pretrained(
            posterior=posterior,
            sampler=sampler,
            criterion=criterion,
            train_loader=train_loader,
            valid_loader=valid_loader,
            swag_epochs=args.swag_epochs,
            using_cuda=args.cuda,
            verbose=args.verbose,
        )
        cache_posterior(posterior.state_dict())


if __name__ == '__main__':
    experiment(args)
