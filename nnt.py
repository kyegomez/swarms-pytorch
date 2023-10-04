import torch
from swarms_torch.nnt import NNTransformer

x = torch.randn(1, 10)

network = NNTransformer(
    #transformer cells
    neuron_count = 5, 
    
    #num states
    num_states = 10,

    #input dim
    input_dim = 10,

    #output dim
    output_dim = 10,

    #nhead
    nhead = 2,
)



output = network(x)
print(output)