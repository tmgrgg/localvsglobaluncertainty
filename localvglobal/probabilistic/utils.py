import torch
import numpy as np
from tqdm import tqdm
from torch.nn import NLLLoss
import torch.nn.functional as F

def evaluate(outputs, targets):
    outputs = torch.log(outputs)
    num_datapoints = targets.size(0)
    loss = NLLLoss()(outputs, targets).item()
    preds = outputs.data.argmax(1, keepdim=True)
    correct = preds.eq(targets.data.view_as(preds)).sum().item()
    return {
        "loss": loss,
        "accuracy": 100.0 * correct / num_datapoints
    }


@torch.no_grad()
def bayesian_model_averaging(
        posterior,
        data_loader,
        train_loader,
        using_cuda=True,
        N=1,
        verbose=False,
):
    assert N >= 1

    num_datapoints = len(data_loader.dataset)
    num_classes = len(np.unique(data_loader.dataset.targets))
    targets = torch.zeros(size=(num_datapoints,)).long()
    probs = torch.zeros(size=(num_datapoints, num_classes))

    if verbose:
        ns = tqdm(list(range(N)))
    else:
        ns = range(N)
    for k in ns:
        posterior.sample()
        posterior.renormalize(train_loader)
        # TODO: shouldn't really have to reset to eval mode?
        posterior.eval()

        start_idx = 0
        for i, (input, target) in enumerate(data_loader):
            batch_size = input.size()[0]
            end_idx = start_idx + batch_size

            if using_cuda:
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            output = posterior(input).cpu()
            probs[start_idx:end_idx] = (k * probs[start_idx:end_idx] + F.softmax(output, dim=1)) / (k + 1)
            targets[start_idx:end_idx] = target
            start_idx = end_idx

        if verbose:
            print('{} BMA samples:'.format(k), evaluate(probs, targets))

    return evaluate(probs, targets)
