import torch
from swarms_torch.structs.mixture_of_mamba import MixtureOfMambas

# Example Usage
num_models = 3
dim = 16
state_range = (1, 20)
conv_range = (1, 10)
expand_range = (1, 5)

mixture_model = MixtureOfMambas(
    num_models, dim, state_range, conv_range, expand_range
)
x = torch.randn(2, 64, dim).to("cuda")
output = mixture_model(
    x, fusion_method="average"
)  # Or use 'weighted' with weights
