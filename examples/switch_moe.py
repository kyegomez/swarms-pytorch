import torch
from swarms_torch import SwitchMoE

# Example usage:
input_dim = 768  # Dimension of input tokens
hidden_dim = 2048  # Hidden dimension of experts
output_dim = 768  # Output dimension, should match input dimension for residual connection
num_experts = 16  # Number of experts

moe_layer = SwitchMoE(
    dim=input_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    num_experts=num_experts,
    use_aux_loss=False,
)

# Create a sample input tensor (batch_size, seq_len, input_dim)
x = torch.rand(32, 128, input_dim)

# Forward pass through the MoE layer with auxiliary loss computation
output, auxiliary_loss = moe_layer(x)

# Now, 'output' contains the MoE output, and 'auxiliary_loss' contains the load balancing loss.
# This auxiliary loss should be added to the main loss function during training.

print(output)
print(auxiliary_loss)
