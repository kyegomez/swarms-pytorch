import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SwiGLU, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(F.silu(self.fc1(x)))


class TopKGate(nn.Module):
    def __init__(self, model_dim, num_experts, top_k):
        super(TopKGate, self).__init__()
        self.w_gate = nn.Linear(model_dim, num_experts)
        self.top_k = top_k

    def forward(self, x):
        gate_logits = self.w_gate(x)
        top_logits, top_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        top_k_logits = torch.full_like(gate_logits, float("-inf"))
        top_k_logits.scatter_(1, top_indices, top_logits)
        return F.softmax(top_k_logits, dim=-1)


class MoE(nn.Module):
    def __init__(self, model_dim, hidden_dim, num_experts, top_k):
        super(MoE, self).__init__()
        self.experts = nn.ModuleList(
            [
                SwiGLU(model_dim, hidden_dim, model_dim)
                for _ in range(num_experts)
            ]
        )
        self.gate = TopKGate(model_dim, num_experts, top_k)

    def forward(self, x):
        gate_scores = self.gate(x)
        expert_outputs = torch.stack(
            [expert(x) for expert in self.experts], dim=2
        )
        weighted_expert_outputs = gate_scores.unsqueeze(-1) * expert_outputs
        return weighted_expert_outputs.sum(dim=2)


# Model architecture parameters
model_dim = 4096
n_layers = 32
head_dim = 128
hidden_dim = 14336
n_heads = 32
context_len = 32768
vocab_size = 32000
num_experts = 8
top_k_experts = 2

# Create a single MoE layer as a demonstration
moe_layer = MoE(model_dim, hidden_dim, num_experts, top_k_experts)

# Example input tensor
x = torch.rand(1, context_len, model_dim)  # (batch_size, seq_len, model_dim)

# Forward pass through the MoE layer
output = moe_layer(x)

print(output)
