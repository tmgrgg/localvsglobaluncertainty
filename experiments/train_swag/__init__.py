from localvglobal.training.utils import TrainingTracker, Timer, run_training_epoch
from copy import deepcopy


def train_swag(
        posterior,
        sampler,
        criterion,
        train_loader,
        valid_loader,
        swag_epochs,
        using_cuda,
        verbose,
        eval_freq=5,
):
    tracker = TrainingTracker()
    timer = Timer()

    if using_cuda:
        posterior.cuda()

    for epoch in range(swag_epochs):
        timer.start()
        res_train = run_training_epoch(train_loader, posterior, criterion,
                                       sampler, train=True, using_cuda=True)
        sampler.collect()

        if (epoch == 0) or (epoch == swag_epochs - 1) or (epoch % eval_freq == 0):
            res_valid_sgd = run_training_epoch(valid_loader, posterior, criterion,
                                               None, train=False, using_cuda=True)

            # save current position in weight space
            tmp = deepcopy(posterior.state_dict())

            # set posterior parameters to current SWA solution and measure
            posterior.expected()
            posterior.renormalize(train_loader)
            res_valid_swa = run_training_epoch(valid_loader, posterior, criterion,
                                            None, train=False, using_cuda=True)
            tracker.log(res_valid_swa['loss'], metric='loss', setting='valid')
            tracker.log(res_valid_swa['accuracy'], metric='accuracy', setting='valid')

            # revert to previous position in weight space
            posterior.load_state_dict(tmp)
            posterior.renormalize(train_loader)

            if verbose:
                print("\n:::")
                print("SWA:::", res_valid_swa)
                print("SGD:::", res_valid_sgd)
                print(":::\n")
        if verbose:
            print(timer.elapsed_seconds(), 's for epoch {}'.format(epoch))

    return posterior, tracker
