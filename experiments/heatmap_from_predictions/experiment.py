from localvglobal.models.utils import Ensembler
from experiments.utils import ExperimentDirectory
from tqdm import tqdm
from localvglobal.training.utils import accuracy
from localvglobal.data import loaders
import localvglobal.models as models
import os
import numpy as np
from localvglobal.probabilistic.models.swag import SWAGPosterior
import torch.nn
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
        "--rank",
        type=int,
        required=False,
        default=-1,
        metavar="RANK",
        help="If this is set, overrides min rank and max rank and computes heatmap row for given RANK of SWAG solutions (default: -1)",
    )

    parser.add_argument(
        "--min_rank",
        type=int,
        required=False,
        default=2,
        metavar="MIN RANK",
        help="Min rank of SWAG solutions (default: 30)",
    )

    parser.add_argument(
        "--max_rank",
        type=int,
        required=False,
        default=30,
        metavar="MAX RANK",
        help="Max rank of SWAG solutions (default: 30)",
    )

    parser.add_argument(
        "--max_num_models",
        type=int,
        required=False,
        default=30,
        metavar="MAX NUMBER OF MODELS",
        help="maximum number of independent solutions (default: 30)",
    )

    parser.add_argument(
        "--step_rank",
        type=int,
        required=False,
        default=1,
        metavar="STEP",
        help="Number by which to increase rank in heat map (default: 1)",
    )

    parser.add_argument(
        "--local_samples",
        type=int,
        required=False,
        default=30,
        metavar="STEP",
        help="Number of local samples to draw for SWAG",
    )

    # parser.add_argument(
    #     "--step_ensemble",
    #     type=int,
    #     required=False,
    #     default=1,
    #     metavar="STEP",
    #     help="Number by which to increase number of ensembled models in heat map (default: 1)",
    # )

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
    experiment.add_table('heatmap')

    model_cfg = getattr(models, args.model)
    criterion = getattr(torch.nn, args.criterion)()

    # load data
    data_loaders = loaders(args.dataset)(
        dir=experiment.path,
        use_validation=not args.no_validation,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        test_transforms=model_cfg.transform_test,
        train_transforms=model_cfg.transform_train,
    )

    train_loader = data_loaders['train']
    valid_loader = data_loaders['valid']
    num_classes = len(np.unique(train_loader.dataset.targets))

    num_models = range(args.max_num_models)
    if args.rank == -1:
        ranks = range(args.min_rank, args.max_rank + 1, args.step_rank)
    else:
        ranks = range(args.rank, args.rank + 1)

    # load posteriors
    posteriors = []
    for posterior_name in tqdm(os.listdir(experiment.posteriors_path)):
        if posterior_name.endswith('.pt'):
            posterior_name = posterior_name[:-3]
            model = model_cfg.model(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
            posterior = SWAGPosterior(model, rank=args.max_rank)
            posterior.load_state_dict(experiment.cached_state_dict(posterior_name, folder='posteriors')[0])
            posteriors.append(posterior)

    for rank in tqdm(ranks):
        _, cache_row = experiment.cached_table_row({'rank': rank}, table_name='heatmap')
        if cache_row:
            loss_valids = []
            accu_valids = []
            ensembler = Ensembler(valid_loader)
            for n in tqdm(num_models):
                # n^th global model
                posterior = posteriors[n]
                if args.cuda:
                    posterior.cuda()
                # add local models
                for k in tqdm(list(range(args.local_samples))):
                    posterior.sample()
                    posterior.renormalize(train_loader)
                    ensembler.add_model(posterior)
                loss_valids.append(ensembler.evaluate(criterion).item())
                accu_valids.append(ensembler.evaluate(accuracy))
            cache_row({
                'losses_valid': loss_valids,
                'accues_valid': accu_valids
            })


if __name__ == '__main__':
    experiment(args)