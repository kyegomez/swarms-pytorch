import torch
from torch import nn, Tensor
import torch.nn.functional as F


class FeedForward(nn.Module):
    """
    Simple FeedForward module.
    
    Args:
        dim (int): Input dimension
        hidden_dim (int): Hidden dimension
        mult (int): Multiplier for hidden dimension
        dropout (float): Dropout rate
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: int = None,
        mult: int = 4,
        dropout: float = 0.0,
        *args,
        **kwargs,
    ):
        super().__init__()
        hidden_dim = hidden_dim or dim * mult
        
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class GatingMechanism(nn.Module):
    def __init__(
        self,
        dim: int,
        num_experts: int,
    ):
        """
        GatingMechanism is a class that represents the gating mechanism in a mixture of experts model.

        Args:
            dim (int): The input dimension.
            num_experts (int): The number of experts in the mixture.

        """
        super().__init__()
        self.gate = nn.Linear(dim, num_experts)

    def forward(self, x: Tensor):
        """
        Forward pass of the gating mechanism.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after applying the gating mechanism.

        """
        return F.softmax(self.gate(x), dim=-1)


class SimpleMoE(nn.Module):
    """
    Simple Mixture of Experts (MoE) model.

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension of the feedforward network.
        output_dim (int): Output dimension.
        num_experts (int): Number of experts in the MoE.
        mult (int, optional): Multiplier for the hidden dimension. Defaults to 4.
    """

    def __init__(
        self,
        dim,
        hidden_dim: int,
        output_dim: int,
        num_experts: int,
        mult: int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.mult = mult

        self.experts = nn.ModuleList(
            [FeedForward(dim, dim, mult) for _ in range(num_experts)]
        )
        self.gate = GatingMechanism(dim, num_experts)

    def forward(self, x: Tensor):
        """
        Forward pass of the SimpleMoE model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            Tensor: Output tensor of shape (batch_size, sequence_length, output_dim).
        """
        gating_scores = self.gate(x)

        expert_outputs = torch.stack(
            [expert(x) for expert in self.experts], dim=-1
        )

        output = torch.sum(gating_scores.unsqueeze(2) * expert_outputs, dim=-1)

        return output
