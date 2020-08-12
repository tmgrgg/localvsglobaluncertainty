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


def experiment(args):
    experiment = ExperimentDirectory(args.dir, args.name)
    experiment.add_table('heatmap_results')

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

    criterion = getattr(torch.nn, args.criterion)()

    num_models = range(args.max_num_models)
    ranks = range(2, args.max_rank + 1)

    # load posteriors
    posteriors = []
    for posterior_name in tqdm(os.listdir(experiment.posteriors_path)):
      if posterior_name.endswith('.pt'):
        posterior_name = posterior_name[:-3]
        model_cfg = getattr(models, args.model)
        model = model_cfg.model(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
        posterior = SWAGPosterior(model, rank=args.max_rank)
        posterior.load_state_dict(experiment.cached_state_dict(posterior_name, folder='posteriors')[0])
        posteriors.append(posterior)

    for rank in tqdm(ranks):
      _, cache_row = experiment.cached_table_row({'rank': rank}, table_name='heatmap_results')
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
          for k in tqdm(list(range(args.local_model_samples))):
            posterior.sample()
            posterior.renormalize(train_loader)
            ensembler.add_model(posterior)
          loss_valids.append(ensembler.evaluate(criterion).item())
          accu_valids.append(ensembler.evaluate(accuracy))
        cache_row({
            'losses_valid': loss_valids,
            'accues_valid': accu_valids
        })