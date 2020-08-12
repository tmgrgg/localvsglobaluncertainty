import localvglobal.models as models
from localvglobal.training.utils import seed
from localvglobal.data import loaders
import torch
from experiments.train_model import train_model
from experiments.train_swag_from_pretrained import train_swag_from_pretrained
import numpy as np
from tqdm import tqdm
from localvglobal.probabilistic.models.swag import SWAGPosterior, SWAGSampler
from experiments.utils import ExperimentDirectory


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

    for model_num in tqdm(list(range(1, args.max_num_models + 1))):
        model_cfg = getattr(models, args.model)
        model = model_cfg.model(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)

        if args.cuda:
            model.cuda()

        if optimizer_cls == torch.optim.SGD:
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_init, weight_decay=args.l2,
                                        momentum=args.momentum)
        else:  # optimizer_cls == torch.optim.Adam:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_init, weight_decay=args.l2,
                                         betas=(args.beta_1, args.beta_2))

        model_name = 'model_{}'.format(model_num)
        model_state_dict, cache_model = experiment.cached_state_dict(model_name)
        optim_name = 'optim_{}'.format(model_num)
        optim_state_dict, cache_optim = experiment.cached_state_dict(optim_name, folder='optims')
        if cache_model or cache_optim:
            seed(model_num)

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

        posterior = SWAGPosterior(model, rank=args.max_rank)

        if args.cuda:
            posterior.cuda()

        posterior_name = 'posterior_{}'.format(model_num)
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
