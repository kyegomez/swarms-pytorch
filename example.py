import torch
from swarms_torch import MixtureOfMambas

# 3D Tensor for text
x = torch.rand(1, 512, 512)
model = MixtureOfMambas(
    num_mambas=2,
    dim=512,
    d_state=1024,
    depth=4,
    d_conv=1024,
    expand=4,
    fusion_method="average",
    custom_fusion_func=None,
)
print(model(x).shape)
