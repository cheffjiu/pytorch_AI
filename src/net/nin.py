import torch.nn as nn


class NiN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            self._nin_block(1, 96, kernel_size=11, stride=4, padding=0),
            nn.MaxPool2d(3, stride=2),
            self._nin_block(96, 256, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(3, stride=2),
            self._nin_block(256, 384, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(0.5),
            self._nin_block(384, num_classes, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        return x

    def _nin_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )
