from torch import nn


def fish(dim: int, mult: int = 4):
    return nn.Sequential(
        nn.Linear(dim, dim * mult),
        nn.Softplus(),
        nn.Dropout(0.1),
        nn.LayerNorm(dim * mult),
        nn.Softmax(dim=-1),  # change this line
        nn.Linear(dim * mult, dim),
    )
