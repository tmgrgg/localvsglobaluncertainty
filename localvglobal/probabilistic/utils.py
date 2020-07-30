import torch
import numpy as np


@torch.no_grad()
def bayesian_model_averaging(
        posterior,
        data_loader,
        criterion,
        using_cuda=True,
        N=50,
        predict=lambda output: output.data.argmax(1, keepdim=True)
):
    assert N >= 1
    loss_sum = 0.0
    correct = 0.0
    example_count = 0
    posterior.eval()

    num_datapoints = len(data_loader.dataset)
    num_classes = len(np.unique(data_loader.dataset.targets))
    targets = torch.zeros(size=(num_datapoints,)).long()
    outputs = torch.zeros(size=(num_datapoints, num_classes))
    for k in tqdm(list(range(N))):
        posterior.cpu()
        posterior.sample()
        posterior.cuda()
        bn_update(train_loader, posterior)

        start_idx = 0
        for i, (input, target) in enumerate(data_loader):
            batch_size = input.size()[0]
            end_idx = start_idx + batch_size

            if using_cuda:
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            output = posterior(input).cpu()
            outputs[start_idx:end_idx] = (k * outputs[start_idx:end_idx] + output) / (k + 1)

            if k == 0:
                targets[start_idx:end_idx] = target

            start_idx = end_idx

    loss = criterion(outputs, targets).item()
    preds = predict(outputs)
    correct = preds.eq(targets.data.view_as(preds)).sum().item()

    return {
        "loss": loss,
        "accuracy": 100.0 * correct / num_datapoints
    }