import torch.nn as nn


class SoftMaxRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.flatten(x)
        return self.linear(x)
