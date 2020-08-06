from localvglobal.training.utils import TrainingTracker, Timer, run_training_epoch


def train_swag(
        posterior,
        sampler,
        criterion,
        train_loader,
        valid_loader,
        swag_epochs,
        verbose,
        using_cuda,
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
            torch.save(posterior.state_dict(), '/content/tmp')
            posterior.expected()
            posterior.renormalize(train_loader)
            res_valid_swa = run_training_epoch(valid_loader, posterior, criterion,
                                               None, train=False, using_cuda=True)
            posterior.load_state_dict(torch.load('/content/tmp'))
            posterior.renormalize(train_loader)

            print("\n:::")
            print("SWA:::", res_valid_swa)
            print("SGD:::", res_valid_sgd)
            print(":::\n")
        print(timer.elapsed_seconds(), 's for epoch {}'.format(epoch))
        # note that we don't track the SWAG solution - just the SGD iterates

    return posterior, tracker