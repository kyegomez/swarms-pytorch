import torch
from torch import nn


class TransformerCell(nn.Module):
    def __init__(
        self,
        input_dim,
        nhead,
        num_layers=1,
        neighborhood_size=3,
    ):
        super(TransformerCell, self).__init__()
        self.transformer = nn.Transformer(
            input_dim, nhead=nhead, num_encoder_layers=num_layers
        )
        self.neighborhood_size = neighborhood_size

    def forward(self, x, neigbors):
        x = self.transformer(x, x)

        out = torch.cat([x] + neigbors, dim=0)

        return out


class CellularSwarm(nn.Module):
    """
    CellularSwarm

    Architecture:
        - Input -> TransformerCell -> TransformerCell -> ... -> Output

    Overview:
    CellularSwarm is a cellular neural network that uses a transformer cell
    to process the input.

    Args:
        cell_count (int): Number of transformer cells
        input_dim (int): Input dimension
        nhead (int): Number of heads in the transformer cell
        time_steps (int): Number of time steps to run the network

    Returns:
        torch.Tensor: Output tensor

    Usage:
        >>> x = torch.randn(10, 32, 512)
        >>> model = CellularSwarm(cell_count=5, input_dim=512, nhead=8)
        >>> output = model(x)
        >>> print(output)


    """

    def __init__(self, cell_count, input_dim, nhead, time_steps=4):
        super(CellularSwarm, self).__init__()
        self.cells = nn.ModuleList(
            [TransformerCell(input_dim, nhead) for _ in range(cell_count)]
        )
        self.time_steps = time_steps

    def forward(self, x):
        for _ in range(self.time_steps):
            for i, cell in enumerate(self.cells):
                # get neighboring cells states
                start_idx = max(0, i - cell.neighborhood_size)

                end_idx = min(len(self.cells), i + cell.neighborhood_size + 1)

                neighbors = [
                    self.cells[j].transformer(x, x)
                    for j in range(start_idx, end_idx)
                    if j != i
                ]

                x = cell(x, neighbors)
        return x
