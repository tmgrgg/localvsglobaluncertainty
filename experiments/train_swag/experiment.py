from experiments import default_model
from localvglobal.data import loaders
from localvglobal.probabilistic.models.swag import SWAGSampler, SWAGPosterior
from torch.optim import SGD
from localvglobal.training.utils import TrainingTracker, Timer, run_training_epoch
from experiments.utils import track, ExperimentTable, CachedExperiment
from localvglobal.probabilistic.utils import bayesian_model_averaging
import torch
import numpy as np

def train_swag(
    posterior,
    sampler,
    criterion,
    train_loader,
    valid_loader,
    swag_epochs,
    verbose,
    using_cuda,
):
    # NOTE: measure the SWA solution since it's cheaper than doing BMA every epoch
    tracker = TrainingTracker()
    timer = Timer()

    if using_cuda:
        posterior.cuda()

    for epoch in range(swag_epochs):
        timer.start()
        res_train = run_training_epoch(train_loader, posterior, criterion,
                                       sampler, train=True, using_cuda=True)
        sampler.collect()
        res_valid = run_training_epoch(valid_loader, posterior, criterion,
                                       None, train=False, using_cuda=True)

        # note that we don't track the SWAG solution - just the SGD iterates
        track(tracker, res_train, res_valid)
        if verbose:
            print('Epoch completed in {} seconds'.format(timer.elapsed_seconds()))
            print('::: Train :::\n', res_train)
            print('::: Valid :::\n', res_valid)

    return posterior, tracker

# run should not be responsible for state... that's experiments job
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

def experiment(args):
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

