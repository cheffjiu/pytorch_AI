import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.container = nn.ModuleList()
        prev_dim = input_dim
        for size in hidden_dim:
            self.container.append(nn.Linear(prev_dim, size))
            self.container.append(nn.ReLU())
            prev_dim = size
        self.container.append(nn.Linear(prev_dim, output_dim))

    def forward(self, x):
        x = nn.Flatten()(x)
        for layer in self.container:
            x = layer(x)
        return x
