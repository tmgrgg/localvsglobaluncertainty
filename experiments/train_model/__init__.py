from experiments.utils import track
from localvglobal.training.utils import TrainingTracker, Timer, run_training_epoch

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

    # this doesn't work because it creates a new object if the model is not already on CUDA
    # and the model parameters assigned in the optimizer won't corrspond to the model on GPU.
    #if using_cuda:
    #    model.cuda()

    # Get pre-training metrics
    res_train = run_training_epoch(train_loader, model, criterion,
                                   None, train=False, using_cuda=using_cuda)
    res_valid = run_training_epoch(valid_loader, model, criterion,
                                   None, train=False, using_cuda=using_cuda)
    track(tracker, res_train, res_valid, plot=verbose)

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
        track(tracker, res_train, res_valid, plot=verbose)

        if verbose:
            print('Epoch completed in {} seconds'.format(timer.elapsed_seconds()))
            print('res_train:', res_train)
            print('res_valid:', res_valid)

    return model, optimizer, tracker, res_train, res_valid