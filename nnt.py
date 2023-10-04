import torch
from swarms_torch.nnt import NNTransformer

network = NNTransformer(5, 10, 10, 10, 2)
output = network(torch.randn(1, 10))
print(output)