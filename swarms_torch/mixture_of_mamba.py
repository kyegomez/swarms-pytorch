import torch
from torch import nn

try:
    from mamba_ssm import Mamba
except ImportError:
    print("Mamba not installed")


class MixtureOfMambas(nn.Module):
    """Mixture of Mambas

    Args:
        num_mambas (int): _description_
        dim (int): _description_
        d_state (int): _description_
        d_conv (_type_): _description_
        expand (int): _description_
        aggregation_method (str, optional): _description_. Defaults to "average".

    Example::
    model = MixtureOfMambas(
        num_mambas=4,
        dim=512,
        d_state=1024,
        d_conv=1024,
        expand=4,
        aggregation_method="average",

    )




    """

    def __init__(
        self,
        num_mambas: int,
        dim: int,
        d_state: int,
        d_conv,
        expand: int,
        aggregation_method: str = "average",
    ):
        super(MixtureOfMambas, self).__init__()
        self.num_mambas = num_mambas
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.aggregation_method = aggregation_method

        self.models = nn.ModuleList()
        for _ in range(num_mambas):
            mamba_model = Mamba(dim, d_state, d_conv, d_conv, expand)
            self.models.append(mamba_model)

    def forward(self, x: torch.Tensor, weights=None):
        """Forward pass of the swarm

        Args:
            x (torch.Tensor): _description_
            weights (_type_, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        outputs = [model(x) for model in self.models]

        if self.aggregation_method == "average":
            return torch.mean(torch.stack(outputs), dim=0)
        elif self.aggregation_method == "weighted":
            return self.weighted_aggregate(outputs, weights)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

    def average_aggregate(self, outputs):
        """Average the outputs of the models in the swarm

        Args:
            outputs (_type_): _description_

        Returns:
            _type_: _description_
        """
        return torch.mean(torch.stack(outputs), dim=0)

    def weighted_aggegrate(self, outputs, weights):
        """Weighted average the outputs of the models in the swarm

        Args:
            outputs (_type_): _description_
            weights (_type_): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        if weights is None or len(weights) != len(outputs):
            raise ValueError("Weights must be the same length as outputs")
        weighted_outputs = [weight * output for weight, output in zip(weights, outputs)]
        return sum(weighted_outputs)
