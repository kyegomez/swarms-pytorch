import torch
from swarms_torch import MixtureOfMambas

# Create a 3D tensor for text
x = torch.rand(1, 512, 512)

# Create an instance of the MixtureOfMambas model
model = MixtureOfMambas(
    num_mambas=2,  # Number of Mambas in the model
    dim=512,  # Dimension of the input tensor
    d_state=1024,  # Dimension of the hidden state
    depth=4,  # Number of layers in the model
    d_conv=1024,  # Dimension of the convolutional layers
    expand=4,  # Expansion factor for the model
    fusion_method="absmax",  # Fusion method for combining Mambas' outputs
    custom_fusion_func=None,  # Custom fusion function (if any)
)

# Pass the input tensor through the model and print the output shape
print(model(x).shape)
