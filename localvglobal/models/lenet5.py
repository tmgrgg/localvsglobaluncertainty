import math
import torch.nn as nn
import torchvision.transforms as transforms


class LeNet5Base(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5Base, self).__init__()
        self.conv_part = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc_part = nn.Sequential(
            nn.Linear(800, 500), nn.ReLU(True), nn.Linear(500, num_classes.item())
        )

        # Initialize weights
        for m in self.conv_part.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv_part(x)
        x = x.view(x.size(0), -1)
        x = self.fc_part(x)
        return x


class LeNet5:
    model = LeNet5Base
    args = list()
    kwargs = {}

    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
        ]
    )

    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=(0.1307,), std=(0.3081,))]
    )
