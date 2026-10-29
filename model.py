"""Model architecture matching the LogisticRegression class in Assignment 1 train.py.

The class name follows the course skeleton. With hidden layers it is an MLP.
"""

import torch
from torch import nn


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=(128, 64), dropout_rate=0.2):
        super().__init__()
        layers = []
        previous_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.extend([nn.Linear(previous_dim, hidden_dim), nn.ReLU()])
            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        return self.network(x)
