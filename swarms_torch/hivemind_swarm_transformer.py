import torch
from torch import nn
from zeta.structs.transformer import (
    Decoder,
    Transformer,
)


class HivemindTransformer(nn.Module):
    def __init__(
        self,
        dim: int = None,
        max_seq_len: int = None,
        depth: int = None,
        heads: int = None,
        dim_head: int = None,
        num_tokens: int = None,
    ):
        super(HivemindTransformer, self).__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.num_tokens = num_tokens

        self.model = Transformer(
            num_tokens=num_tokens,
            max_seq_len=max_seq_len,
            attn_layers=Decoder(
                dim=dim,
                depth=depth,
                dim_head=dim_head,
                heads=heads,
            ),
        )

    def forward(self, x):
        return self.model(x)


class HivemindSwarm(nn.Module):
    """
    HiveMind Swarm Transformer

    This is a transformer that is composed of a swarm of transformers where each transformer shares the same weights.

    Args:
        dim: dimension of the model
        max_seq_len: maximum sequence length
        depth: depth of the model
        heads: number of heads
        dim_head: dimension of each head
        num_models: number of models in the swarm
        base_transformer: the base transformer to be used in the swarm


    Example::
    model = HivemindSwarm(
        dim=512,
        max_seq_len=1024,
        depth=6,
        heads=8,
        dim_head=64,
        num_models=4,
    )

    x = torch.randn(1, 1024, 512)
    y = model(x)
    print(y.shape)


    """

    def __init__(
        self,
        dim: int = None,
        max_seq_len: int = None,
        num_tokens: int = None,
        depth: int = None,
        heads: int = None,
        dim_head: int = None,
        num_models: int = 1,
        **kwargs,
    ):
        super(HivemindSwarm, self).__init__()

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.depth = depth
        self.heads = heads
        self.num_tokens = num_tokens
        self.dim_head = dim_head
        self.num_models = num_models
        self.base_transformer = HivemindTransformer(
            dim=dim,
            num_tokens=num_tokens,
            max_seq_len=max_seq_len,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
        )
        # Create a list of transformers sharing the same weights
        self.experts = nn.ModuleList(
            [self.base_transformer for _ in range(num_models)]
        )

        # Gating mechniams allows the model to dynamically weight the contribution of each transformer
        # in the swarm. This is done by learning a weight for each transformer and then using a softmax
        # to normalize the weights.
        self.gate = nn.Linear(num_models, num_models)
        self.gate_activation = nn.Softmax(dim=-1)
        self.gate_bias = nn.Parameter(torch.zeros(num_models))

    def forward(self, x):
        logits = []
        for expert in self.experts:
            output = expert(x)
            logits.append(output)
        # Run each transformer on the input
        # outputs = [expert(x) for expert in self.experts]

        # stack outputs
        outputs = torch.stack(logits, dim=1)

        # Compute the gate
        gate = self.gate_activation(self.gate_bias + self.gate(outputs))

        # Weight the outputs
        outputs = torch.sum(outputs * gate.unsqueeze(-1), dim=1)
        return outputs


# model = HivemindSwarm(
#     dim=512,
#     max_seq_len=1024,
#     num_tokens=20000,
#     depth=6,
#     heads=8,
#     dim_head=64,
#     num_models=4,
# )

# x = torch.randn(1, 1024, 512)
# y = model(x)
# print(y.shape)
