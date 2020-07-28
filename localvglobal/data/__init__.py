from localvglobal.data import datasets


_loaders = {'fashionmnist': datasets.load_FashionMNIST}


def loaders(name):
    return _loaders[name.lower()]