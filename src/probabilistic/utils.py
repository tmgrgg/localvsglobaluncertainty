import torch


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
    for i, (input, target) in enumerate(data_loader):
        if using_cuda:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        output = 0
        for k in range(N):
            posterior.sample()
            output = (output * k + posterior(input)) / (k + 1)
        loss = criterion(output, target)

        # update qois
        loss_sum += loss.data.item() * input.size(0)
        preds = predict(output)
        correct += preds.eq(target.data.view_as(preds)).sum().item()
        example_count += input.size(0)

        return {
            "loss": loss_sum / example_count,
            "accuracy": (correct / example_count) * 100.0,
        }
