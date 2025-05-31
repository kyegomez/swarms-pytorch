import torch
from torch import nn, Tensor
from zeta.nn import MambaBlock


def router(
    x: Tensor,
    k: int,
    largest: bool = True,
    experts: nn.ModuleList = None,
    limit_of_experts: int = None,
    dropout_on: bool = False,
    dropout_p: float = 0.2,
    dim: int = -1,
    *args,
    **kwargs,
):
    # If experts is None, then we use the default topk function
    topk = torch.topk(x, k, largest=largest, *args, **kwargs)

    # Adaptive log softmax with loss
    # softmax = nn.LogSoftmax(dim)
    # topk = softmax(x)

    # Dropout
    if dropout_on:
        dropout = nn.Dropout(dropout_p)
        topk = dropout(topk)

    # If limit_of_experts is not None, then we only send the topk to the
    # experts. This is useful when we want to limit the number of experts
    # that we send the topk to.
    if limit_of_experts is not None:
        experts = experts[:limit_of_experts]

    # Send the topk to the experts
    if experts is not None:
        topk = [expert(topk) for expert in experts]
    return topk


class MixtureOfMambas(nn.Module):
    """
    Mixtures of Mamba is a swarm of Mamba models. The swarm can be aggregated
    using a weighted average or a simple average. We plan to add more aggregation
    methods in the future like a gating mechanism or a neural network or a
    transformer.

    Args:
        num_mambas (int): _description_
        dim (int): _description_
        d_state (int): _description_
        d_conv (_type_): _description_
        expand (int): _description_
        fusion_method (str, optional): _description_. Defaults to "average".

    Example::
    >>> model = MixtureOfMambas(
    ...     num_mambas=2,
    ...     dim=512,
    ...     d_state=1024,
    ...     depth=4,
    ...     d_conv=1024,
    ...     expand=4,
    ...     fusion_method="average",
    ... )
    >>> x = torch.rand(1, 512, 512)
    >>> model(x).shape
    torch.Size([1, 512, 512])
    """

    def __init__(
        self,
        num_mambas: int,
        dim: int,
        d_state: int,
        depth: int,
        d_conv,
        expand: int,
        fusion_method: str = "average",
        custom_fusion_func: callable = None,
        *args,
        **kwargs,
    ):
        super(MixtureOfMambas, self).__init__()
        self.num_mambas = num_mambas
        self.dim = dim
        self.d_state = d_state
        self.depth = depth
        self.d_conv = d_conv
        self.expand = expand
        self.fusion_method = fusion_method
        self.custom_fusion_func = custom_fusion_func

        self.models = nn.ModuleList()
        for _ in range(num_mambas):
            mamba_model = MambaBlock(
                dim, depth, d_state, expand, d_conv, *args, **kwargs
            )
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

        if self.fusion_method == "average":
            return self.average_aggregate(outputs)
        elif self.fusion_method == "weighted":
            return self.weighted_aggregate(outputs, weights)
        elif self.fusion_method == "absmax":
            return self.absmax_aggregate(outputs, weights)
        elif self.fusion_method == "softmax":
            return self.softmax_aggregate(outputs, weights)
        elif self.fusion_method == "custom":
            if self.custom_fusion_func is None:
                raise ValueError(
                    "custom_fusion_func must be provided if fusion_method is"
                    " custom"
                )
            return self.custom_fusion_func(outputs, weights)
        else:
            raise ValueError(
                f"Unknown aggregation method: {self.fusion_method}"
            )

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
        weighted_outputs = [
            weight * output for weight, output in zip(weights, outputs)
        ]
        return sum(weighted_outputs)

    def softmax_aggregate(self, outputs, weights):
        """Weighted average the outputs of the models in the swarm

        Args:
            outputs (_type_): _description_
            weights (_type_): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        # if weights is None or len(weights) != len(outputs):
        #     raise ValueError("Weights must be the same length as outputs")
        if weights:
            weighted_outputs = [
                weight * output for weight, output in zip(weights, outputs)
            ]
            out = sum(weighted_outputs)
            out = torch.softmax(out, dim=1)
        else:
            out = torch.softmax(outputs, dim=1)

        return out

    def absmax(self, outputs):
        """Absolute maximum of the outputs of the models in the swarm

        Args:
            outputs (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Absolute maximum of the outputs of the models in the swarm
        return torch.max(torch.abs(torch.stack(outputs)), dim=0)[0]

    def absmax_aggregate(self, outputs, weights=None):
        """
        Weighted average the outputs of the models in the swarm

        Args:
            outputs (_type_): _description_
            weights (_type_): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        # if weights is not None or len(weights) != len(outputs):
        #     raise ValueError("Weights must be the same length as outputs")

        if weights:
            weighted_outputs = [
                weight * output for weight, output in zip(weights, outputs)
            ]
            return self.absmax(weighted_outputs)
        else:
            return self.absmax(outputs)
