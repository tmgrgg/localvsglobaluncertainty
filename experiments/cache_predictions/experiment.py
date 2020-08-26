import matplotlib.pyplot as plt
from tqdm import tqdm
from localvglobal.probabilistic.models.swag import SWAGPosterior, SWAGSampler
from experiments.utils import ExperimentDirectory
from localvglobal.data import loaders
import localvglobal.models as models
import torch
import numpy as np
import argparse
from localvglobal.training.utils import bn_update
from localvglobal.models.utils import Ensembler
import torch.nn
import os


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

    parser.add_argument(
        "--train",
        action="store_true",
        help="get predictions on training set",
    )

    parser.add_argument(
        "--validation",
        action="store_true",
        help="get predictions on validation set ",
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
        "-test",
        action="store_true",
        help="get predictions on test set ",
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
        "--rank",
        type=int,
        required=False,
        default=-1,
        metavar="RANK",
        help="If caching a single rakn value, rank of SWAG solutions (default: -1)",
    )

    parser.add_argument(
        "--min_rank",
        type=int,
        required=False,
        default=2,
        metavar="MIN RANK",
        help="If caching a range, min rank of SWAG solutions (default: 2)",
    )

    parser.add_argument(
        "--step_rank",
        type=int,
        required=False,
        default=1,
        metavar="STEP RANK",
        help="If caching a range, incremental step for ranks of SWAG solutions (default: 1)",
    )

    parser.add_argument(
        "--max_rank",
        type=int,
        required=True,
        default=30,
        metavar="MAX RANK",
        help="Max rank of SWAG solutions (default: 30)",
    )

    parser.add_argument(
        "--local_samples",
        type=int,
        required=False,
        default=30,
        metavar="STEP",
        help="Number of local samples to draw for SWAG",
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
    experiment.add_folder('predictions')
    experiment.add_table('predictions')

    # check args
    if not (args.train or args.test or args.validation):
        print('Missing --train, --test, or --validation flag/flags.')
        return

    # load data
    data_loaders = loaders(args.dataset)(
        dir=experiment.path,
        use_validation=True,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
    )

    train_loader = data_loaders['train']
    valid_loader = data_loaders['valid']
    test_loader = data_loaders['test']

    if args.train:
        loader = train_loader
    if args.validation:
        loader = valid_loader
    if args.test:
        loader = test_loader
    num_classes = len(np.unique(train_loader.dataset.targets))

    model_cfg = getattr(models, args.model)
    criterion = getattr(torch.nn, args.criterion)()

    if args.rank == -1:
        ranks = list(range(args.min_rank, args.max_rank + 1, args.step_rank))
    else:
        ranks = list(range(args.rank, args.rank + 1))

    # SGD
    print('Predicting with SGD solutions')
    for model_name in tqdm(os.listdir(experiment.models_path)):
        if not model_name.endswith('.pt'):
            continue

        model_name = model_name[:-3]
        _, cache_row = experiment.cached_table_row(
            {'model': model_name,
             'type': 'SGD',
             'rank': None,
             'sample': None},
            table_name='predictions'
        )
        if cache_row:
            path = os.path.join(experiment.predictions_path, model_name + '_SGD.pt')
            model = model_cfg.model(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
            model.load_state_dict(experiment.cached_state_dict(model_name, folder='models')[0])
            if args.cuda:
                model.cuda()
            ensembler = Ensembler(loader)
            bn_update(train_loader, model)
            ensembler.add_model(model)
            torch.save(ensembler.predictions, path)
            cache_row({'path': path})

    # SWA
    print('Predicting with SWA solutions')
    for posterior_name in tqdm(os.listdir(experiment.posteriors_path)):
        if not posterior_name.endswith('.pt'):
            continue

        posterior_name = posterior_name[:-3]
        _, cache_row = experiment.cached_table_row(
            {'model': posterior_name,
             'type': 'SWA',
             'rank': None,
             'sample': None},
            table_name='predictions'
        )
        if cache_row:
            path = os.path.join(experiment.predictions_path, posterior_name + '_SWA.pt')
            model = model_cfg.model(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
            posterior = SWAGPosterior(model, rank=args.max_rank)
            posterior.load_state_dict(experiment.cached_state_dict(posterior_name, folder='posteriors')[0])
            if args.cuda:
                posterior.cuda()
            ensembler = Ensembler(loader)
            posterior.expected()
            posterior.renormalize(train_loader)
            ensembler.add_model(posterior)
            torch.save(ensembler.predictions, path)
            cache_row({'path': path})

    # SWAG
    print('Predicting with SWAG solutions')
    for posterior_name in tqdm(os.listdir(experiment.posteriors_path)):
        if not posterior_name.endswith('.pt'):
            continue

        posterior_name = posterior_name[:-3]
        model = model_cfg.model(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
        posterior = SWAGPosterior(model, rank=args.max_rank)
        posterior.load_state_dict(experiment.cached_state_dict(posterior_name, folder='posteriors')[0])
        if args.cuda:
            posterior.cuda()

        for rank in ranks:
            for k in range(1, args.local_samples + 1):
                _, cache_row = experiment.cached_table_row({
                    'model': posterior_name,
                    'type': 'SWAG',
                    'rank': rank,
                    'sample': k
                },
                    table_name='predictions'
                )

                if cache_row:
                    path = os.path.join(experiment.predictions_path,
                                        posterior_name + '_SWAG_rank_{}_sample_{}.pt'.format(rank, k))
                    ensembler = Ensembler(loader)
                    posterior.sample(rank=rank)
                    posterior.renormalize(train_loader)
                    ensembler.add_model(posterior)
                    torch.save(ensembler.predictions, path)
                    cache_row({'path': path})


if __name__ == '__main__':
    experiment(args)
