import pytest
import torch
from swarms_torch.spiral_optimization import SPO

def test_spo_initialization():
    spo = SPO(goal="Hello SPO", m=100, k_max=1000)
    assert isinstance(spo, SPO)
    assert spo.goal.shape == torch.Size([9])
    assert spo.points.shape == torch.Size([100, 9])
    assert spo.center.shape == torch.Size([9])

def test_spo_step_rate():
    spo = SPO(goal="Hello SPO", m=100, k_max=1000)
    step_rate = spo._step_rate(1)
    assert step_rate == 0.5

def test_spo_update_points():
    spo = SPO(goal="Hello SPO", m=100, k_max=1000)
    spo._update_points(1)
    assert spo.points.shape == torch.Size([100, 9])

def test_spo_update_center():
    spo = SPO(goal="Hello SPO", m=100, k_max=1000)
    spo._update_center()
    assert spo.center.shape == torch.Size([9])

def test_spo_optimize():
    spo = SPO(goal="Hello SPO", m=100, k_max=1000)
    spo.optimize()
    assert spo.center.shape == torch.Size([9])

def test_spo_best_string():
    spo = SPO(goal="Hello SPO", m=100, k_max=1000)
    spo.optimize()
    best_string = spo.best_string()
    assert isinstance(best_string, str)
    assert len(best_string) == 9