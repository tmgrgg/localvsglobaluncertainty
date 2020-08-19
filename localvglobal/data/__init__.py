from localvglobal.data import datasets


_loaders = {
    'fashionmnist': datasets.load_FashionMNIST,
    'cifar10': datasets.load_CIFAR10
}


def loaders(name):
    return _loaders[name.lower()]