import torch
from swarms_torch.neuronal_transformer import (
    TransformerLayer,
    Neuron,
    SynapseTransformer,
    NNTransformer,
)


def test_transformerlayer_initialization():
    transformerlayer = TransformerLayer(input_dim=512, output_dim=256, nhead=8)
    assert isinstance(transformerlayer, TransformerLayer)


def test_transformerlayer_forward():
    transformerlayer = TransformerLayer(input_dim=512, output_dim=256, nhead=8)
    x = torch.randn(10, 32, 512)
    output = transformerlayer(x)
    assert output.shape == torch.Size([10, 32, 256])


def test_neuron_initialization():
    neuron = Neuron(num_states=10)
    assert isinstance(neuron, Neuron)
    assert neuron.states.shape == torch.Size([10])


def test_synapsetransformer_initialization():
    synapsetransformer = SynapseTransformer(
        input_dim=512, output_dim=256, nhead=8
    )
    assert isinstance(synapsetransformer, SynapseTransformer)


def test_synapsetransformer_forward():
    synapsetransformer = SynapseTransformer(
        input_dim=512, output_dim=256, nhead=8
    )
    x = torch.randn(10, 32, 512)
    output = synapsetransformer(x)
    assert output.shape == torch.Size([10, 32, 256])


def test_nntransformer_initialization():
    nntransformer = NNTransformer(
        neuron_count=5, num_states=10, input_dim=512, output_dim=256, nhead=8
    )
    assert isinstance(nntransformer, NNTransformer)
    assert len(nntransformer.neurons) == 5
    assert len(nntransformer.synapses) == 5


def test_nntransformer_forward():
    nntransformer = NNTransformer(
        neuron_count=5, num_states=10, input_dim=512, output_dim=256, nhead=8
    )
    x = torch.randn(1, 10)
    output = nntransformer(x)
    assert output.shape == torch.Size([10])
