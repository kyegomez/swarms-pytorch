import pytest
import torch
from swarms_torch.hivemind_swarm_transformer import HivemindSwarm


# Create a fixture for the HivemindSwarm model
@pytest.fixture
def swarm_model():
    return HivemindSwarm(
        dim=512, max_seq_len=32, depth=6, heads=8, dim_head=64, num_models=3
    )


# Test the basic functionality of HivemindSwarm
def test_hivemind_swarm_forward(swarm_model):
    x = torch.randint(0, 20000, (1, 32))
    y = swarm_model(x)
    assert y.shape == (1, 32, 512)


# Test if the swarm consists of the correct number of transformers
def test_num_transformers_in_swarm(swarm_model):
    assert len(list(swarm_model.experts)) == 3


# Test if the gate mechanism works as expected
def test_gate_mechanism(swarm_model):
    x = torch.randint(0, 20000, (1, 32))
    outputs = torch.stack([expert(x) for expert in swarm_model.experts], dim=1)
    gate = swarm_model.gate_activation(
        swarm_model.gate_bias + swarm_model.gate(outputs)
    )

    # Check if the gate values sum to 1 along the transformer dimension
    assert torch.allclose(gate.sum(dim=-1), torch.ones(1, 3))


# Test if the model can handle different input shapes
def test_different_input_shapes(swarm_model):
    x1 = torch.randint(0, 20000, (1, 32))
    x2 = torch.randint(0, 20000, (1, 16))
    y1 = swarm_model(x1)
    y2 = swarm_model(x2)
    assert y1.shape == (1, 32, 512)
    assert y2.shape == (1, 16, 512)


# Test if the model can handle different numbers of models in the swarm
def test_different_num_models():
    swarm_model_1 = HivemindSwarm(
        dim=512, max_seq_len=32, depth=6, heads=8, dim_head=64, num_models=1
    )
    swarm_model_2 = HivemindSwarm(
        dim=512, max_seq_len=32, depth=6, heads=8, dim_head=64, num_models=5
    )

    x = torch.randint(0, 20000, (1, 32))
    y1 = swarm_model_1(x)
    y2 = swarm_model_2(x)

    assert y1.shape == (1, 32, 512)
    assert y2.shape == (1, 32, 512)


# Test if the model works with different configurations
def test_different_configurations():
    model_1 = HivemindSwarm(
        dim=256, max_seq_len=16, depth=4, heads=4, dim_head=64, num_models=2
    )
    model_2 = HivemindSwarm(
        dim=1024, max_seq_len=64, depth=8, heads=16, dim_head=128, num_models=4
    )

    x = torch.randint(0, 20000, (1, 16))
    y1 = model_1(x)
    y2 = model_2(x)

    assert y1.shape == (1, 16, 256)
    assert y2.shape == (1, 16, 1024)
