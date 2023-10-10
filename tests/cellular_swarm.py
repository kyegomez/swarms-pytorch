import pytest
import torch
from swarms_torch import TransformerCell, CellularSwarm


def test_transformercell_initialization():
    transformercell = TransformerCell(input_dim=512, nhead=8)
    assert isinstance(transformercell, TransformerCell)
    assert transformercell.neighborhood_size == 3


def test_transformercell_forward():
    transformercell = TransformerCell(input_dim=512, nhead=8)
    x = torch.randn(10, 32, 512)
    neighbors = [torch.randn(10, 32, 512)]
    output = transformercell(x, neighbors)
    assert output.shape == torch.Size([20, 32, 512])


def test_cellularswarm_initialization():
    cellularswarm = CellularSwarm(cell_count=5, input_dim=512, nhead=8)
    assert isinstance(cellularswarm, CellularSwarm)
    assert len(cellularswarm.cells) == 5
    assert cellularswarm.time_steps == 4


def test_cellularswarm_forward():
    cellularswarm = CellularSwarm(cell_count=5, input_dim=512, nhead=8)
    x = torch.randn(10, 32, 512)
    output = cellularswarm(x)
    assert output.shape == torch.Size([10, 32, 512])
