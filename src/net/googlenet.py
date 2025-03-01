import torch
import torch.nn as nn


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super().__init__()
        # 分支1：1x1卷积
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        # 分支2：1x1降维 + 3x3卷积
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1),
        )

        # 分支3：1x1降维 + 5x5卷积
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2),
        )

        # 分支4：3x3池化 + 1x1卷积
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1),
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        return torch.cat([branch1, branch2, branch3, branch4], 1)


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class AuxClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(0.7)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = nn.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True):
        super().__init__()
        self.aux_logits = aux_logits

        # 初始卷积层
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        # Inception模块堆叠
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        # 中间层辅助分类器
        if self.aux_logits:
            self.aux1 = AuxClassifier(512, num_classes)
            self.aux2 = AuxClassifier(528, num_classes)

        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        # 前向传播流程
        x = self.conv1(x)
        x = self.maxpool1(x)
        # ...（中间层前向传播）

        # 辅助分类器输出
        if self.training and self.aux_logits:
            aux1 = self.aux1(x)
            aux2 = self.aux2(x)

        # 主分类器
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, aux1, aux2 if self.training else x
