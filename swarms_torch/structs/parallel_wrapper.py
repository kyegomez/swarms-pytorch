import torch
from torch import nn
from typing import List


class ParallelSwarm(nn.Module):
    def __init__(
        self,
        models: List[nn.Module],
    ):
        """
        Initializes a parallel swarm of models.

        Args:
            models (List[nn.Module]): A list of PyTorch models.

        """
        super().__init__()
        self.models = models

        for model in models:
            self.model = model

    def forward(self, x: torch.Tensor, *args, **kwargs):
        """Forward pass of the swarm

        Args:
            x (torch.Tensor): _description_
            weights (_type_, optional): _description_. Defaults to None.

        Returns:
            torch.Tensor: _description_
        """
        outputs = []
        for model in self.models:
            outputs.append(model(x, *args, **kwargs))
        return outputs
