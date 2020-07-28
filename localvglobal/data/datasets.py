from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.utils import data


DEFAULT_TRANSFORM = transforms.Compose([transforms.ToTensor()])


def load_FashionMNIST(
        dir,
        use_validation=True,
        val_ratio=0.2,
        train_transforms=DEFAULT_TRANSFORM,
        test_transforms=DEFAULT_TRANSFORM,
        pin_memory=True,
        batch_size=128,
        num_workers=1
):
    path = dir + '/data/FashionMNIST'
    train_set = FashionMNIST(path, train=True, download=True, transform=train_transforms)
    test_set = FashionMNIST(path, train=False, download=True, transform=test_transforms)

    if use_validation:
        val_size = int(val_ratio * len(train_set))
        train_set.data = train_set.data[:-val_size]
        train_set.targets = train_set.targets[:-val_size]
        val_set = FashionMNIST(path, train=True, download=True, transform=test_transforms)
        val_set.train = False
        val_set.data = val_set.data[-val_size:]
        val_set.targets = val_set.targets[-val_size:]

    train_loader = data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    valid_loader = None
    if use_validation:
        valid_loader = data.DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    test_loader = data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return {'train': train_loader, 'valid': valid_loader, 'test': test_loader}
