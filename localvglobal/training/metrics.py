import torch


def matrix_kl(P, Q, reduce=None):
    # P.size: [num_examples, num_classes] (holding probability predictions)
    # Q.size: [num_examples, num_classes] (holding probabilitiy predictions)
    # assuming same order:
    log_p_over_q = torch.log(P / Q)
    # stop this from being negative infinity when P = 0
    log_p_over_q[P == 0] = 0.0
    # elementwise multiply
    res = P * log_p_over_q
    if reduce is 'mean':
        return res.sum(axis=1).mean().item()
    if reduce is 'sum':
        return res.sum(axis=1).sum().item()

    return res.sum(axis=1)


def reverse_matrix_kl(P, Q):
    return matrix_kl(Q, P)


def accuracy(outputs, targets):
    num_datapoints = targets.size(0)
    preds = outputs.data.argmax(1, keepdim=True)
    correct = preds.eq(targets.data.view_as(preds)).sum().item()
    return 100.0 * correct / num_datapoints