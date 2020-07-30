import torch
from localvglobal.training.utils import TrainingTracker, Timer, run_training_epoch
from localvglobal.data import loaders
from experiments import default_model
import torch.optim
import torch.nn
import numpy as np
from experiments.utils import ExperimentTable, CachedExperiment


# I'm implementing experiments with a run method that should be
# 1. Runnable outside of the context of a listed experiment, i.e. parameterised so that I can easily play around beyond
#        the inherently limiting scope of having to write these as command-line runnable
# 2. Cachable via CachedExperiment so that I can make use of ExperimentTable (if necessary)
# 3. experiment() method will handle the "business" of turning run method into a
#  cachable, command-line runnable experiment


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def schedule(lr_init, lr_final, epoch, max_epochs,):
    t = epoch / max_epochs
    lr_ratio = lr_final / lr_init
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return lr_init * factor


def train_model(
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
        verbose
):
    tracker = TrainingTracker()
    timer = Timer()

    if using_cuda:
        model.cuda()

    # Get pre-training metrics
    res_train = run_training_epoch(train_loader, model, criterion,
                                   None, train=False, using_cuda=using_cuda)
    res_valid = run_training_epoch(valid_loader, model, criterion,
                                   None, train=False, using_cuda=using_cuda)
    track(tracker, res_train, res_valid, name, plot=verbose)

    # TRAINING LOOP
    for epoch in range(epochs):
        # adjust learning rate
        lr = schedule(lr_init, lr_final, epoch, epochs)
        adjust_learning_rate(optimizer, lr)

        timer.start()
        res_train = run_training_epoch(train_loader, model, criterion,
                                       optimizer, train=True, using_cuda=using_cuda)
        res_valid = run_training_epoch(valid_loader, model, criterion,
                                       None, train=False, using_cuda=using_cuda)
        track(tracker, res_train, res_valid, name, plot=verbose)

        if verbose:
            print('Epoch completed in {} seconds'.format(timer.elapsed_seconds()))
            print('res_train:', res_train)
            print('res_valid:', res_valid)

    return model, tracker, res_train, res_valid


# implementing this so that run is parametrised by python objects, i.e. so it can be pulled into e.g. a notebook
# environment and played with more dynamically. The __main__.py script just defines a relatively simple command
# line interface for doing "common" things with the experiment.
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
    model, tracker, res_train, res_valid = train_model(
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

    # parse model
    num_classes = len(np.unique(train_loader.dataset.targets))
    model = default_model(args.model, num_classes)

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
