import pytest
import torch
from swarms_torch.mixture_of_mamba import MixtureOfMambas


@pytest.fixture
def mixture():
    num_mambas = 5
    dim = 10
    d_state = 20
    d_conv = 30
    expand = 40
    return MixtureOfMambas(num_mambas, dim, d_state, d_conv, expand)


def test_init(mixture):
    assert mixture.num_mambas == 5
    assert mixture.dim == 10
    assert mixture.d_state == 20
    assert mixture.d_conv == 30
    assert mixture.expand == 40
    assert len(mixture.models) == 5


def test_forward_average(mixture):
    x = torch.rand((1, 10))
    output = mixture.forward(x)
    assert output.shape == (1, 10)


def test_forward_weighted(mixture):
    x = torch.rand((1, 10))
    weights = torch.ones(5)
    mixture.aggregation_method = "weighted"
    output = mixture.forward(x, weights)
    assert output.shape == (1, 10)


def test_forward_invalid_aggregation(mixture):
    x = torch.rand((1, 10))
    mixture.aggregation_method = "invalid"
    with pytest.raises(ValueError):
        mixture.forward(x)


def test_average_aggregate(mixture):
    outputs = [torch.rand((1, 10)) for _ in range(5)]
    output = mixture.average_aggregate(outputs)
    assert output.shape == (1, 10)


def test_weighted_aggregate(mixture):
    outputs = [torch.rand((1, 10)) for _ in range(5)]
    weights = torch.ones(5)
    output = mixture.weighted_aggregate(outputs, weights)
    assert output.shape == (1, 10)


def test_weighted_aggregate_invalid_weights(mixture):
    outputs = [torch.rand((1, 10)) for _ in range(5)]
    weights = torch.ones(4)
    with pytest.raises(ValueError):
        mixture.weighted_aggregate(outputs, weights)


def test_forward_different_dimensions(mixture):
    x = torch.rand((2, 10))
    with pytest.raises(ValueError):
        mixture.forward(x)


def test_forward_no_weights(mixture):
    x = torch.rand((1, 10))
    mixture.aggregation_method = "weighted"
    with pytest.raises(ValueError):
        mixture.forward(x)


def test_forward_extra_weights(mixture):
    x = torch.rand((1, 10))
    weights = torch.ones(6)
    mixture.aggregation_method = "weighted"
    with pytest.raises(ValueError):
        mixture.forward(x, weights)
