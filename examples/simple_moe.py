import torch
from swarms_torch import SimpleMoE

# Example usage:
input_dim = 512  # Dimension of input tokens
hidden_dim = 1024  # Hidden dimension of experts
output_dim = 512  # Output dimension, should match input dimension for residual connection
num_experts = 4  # Number of experts

moe = SimpleMoE(input_dim, hidden_dim, output_dim, num_experts)

# Create a sample input tensor (batch_size, seq_len, input_dim)
x = torch.rand(10, 16, input_dim)

# Forward pass through the MoE layer
output = moe(x)
print(output)
