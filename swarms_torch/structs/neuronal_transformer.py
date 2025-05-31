"""
Cellular neural network

Architecture:
    - Input -> Linear -> ReLU -> Linear -> ReLU -> Output
    - Neuron states are updated after each synapse
    - Softmax is applied after each synapse
    - Layer normalization is applied after each synapse

"""

import torch
from torch import nn


class TransformerLayer(nn.Module):
    """
    Transformer Layer

    Architecture:
        - Input -> Linear -> ReLU -> Linear -> ReLU -> Output

    Args:
        input_dim (int): Input dimension
        output_dim (int): Output dimension

    Returns:
        torch.Tensor: Output tensor

    Usage

    """

    def __init__(
        self,
        input_dim,
        output_dim,
        nhead: int,
    ):
        super(TransformerLayer, self).__init__()
        self.transformer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
        )
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(self.transformer(x))


class Neuron(nn.Module):
    def __init__(self, num_states):
        super(Neuron, self).__init__()
        self.states = nn.Parameter(torch.randn(num_states))


class SynapseTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, nhead: int):
        super(SynapseTransformer, self).__init__()
        self.transformer = TransformerLayer(input_dim, output_dim, nhead)

    def forward(self, x):
        return self.transformer(x)


class NNTransformer(nn.Module):
    """
    Neural Network NNTransformer

    Args:
        neuron_count (int): Number of neurons
        num_states (int): Number of states
        input_dim (int): Input dimension
        output_dim (int): Output dimension
        nhead (int): Number of heads in transformer layer

    Returns:
        torch.Tensor: Output tensor

    Architecture:
        - Input -> Linear -> ReLU -> Linear -> ReLU -> Output
        - Neuron states are updated after each synapse
        - Softmax is applied after each synapse
        - Layer normalization is applied after each synapse

    Usage:
        network = CellularNN(5, 10, 10, 10, 2)
        output = network(torch.randn(1, 10))
        print(output)


    Training:
    network = NNTransformer(5, 10, 10, 10, 2)
    output = network(torch.randn(1, 10))
    print(output)


    # Test the network
    import torch
    import torch.optim as optim
    import torch.nn.functional as F

    # Random dataset
    batch_size = 64
    input_size = 10
    output_size = 10

    x = torch.randn(batch_size, input_size)  # Random inputs
    y = torch.randn(batch_size, output_size)  # Random targets

    # Hyperparameters
    neuron_count = 5
    num_states = 10
    input_dim = input_size
    output_dim = output_size
    n_head = 2

    # Initialize the network
    network = CellularNN(neuron_count, num_states, input_dim, output_dim, n_head)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(network.parameters(), lr=0.001)

    # Training loop
    epochs = 1000
    for epoch in range(epochs):
        # Forward pass
        outputs = network(x)

        # Compute loss
        loss = criterion(outputs, y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss every 100 epochs
        if (epoch+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # Test the network with a new random input
    test_input = torch.randn(1, input_size)
    test_output = network(test_input)
    print(test_output)


    """

    def __init__(self, neuron_count, num_states, input_dim, output_dim, nhead):
        super(NNTransformer, self).__init__()

        # Initialize neurons and synapses
        self.neurons = nn.ModuleList(
            [Neuron(num_states) for _ in range(neuron_count)]
        )
        self.synapses = nn.ModuleList(
            [
                SynapseTransformer(input_dim, output_dim, nhead)
                for _ in range(neuron_count)
            ]
        )

        self.norm = nn.LayerNorm(output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for neuron, synapse in zip(self.neurons[:-1], self.synapses):
            # norm before synapse
            x = self.norm(x)

            # synapse
            x = synapse(x)

            # softmax after synapse
            x = self.softmax(x)

            neuron.states.data = x
        return self.neurons[-1].states
