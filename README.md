[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Swarms in Torch
Swarms in Torch exclusivley hosts a vast array of 100% novel swarming models. Our purpose for this repo is to create, optimize, and train novel foundation models that outperform the status quo of model architectures such as the Transformer and SSM model architectures. We provide implementations of various novel models like PSO with transformers as particles, ant colony with transformers as ants, a basic NN with transformers as neurons, Mixture of Mambas and many more. If you would like to help contribute to the future of AI model architecture's please join Agora, the open source lab here. And, if you have any idea's please submit them as issues and notify me.


## Installation

```bash
pip3 install swarms-torch
```

# Usage

### ParticleSwarmOptimization

```python
from swarms_torch import ParticleSwarmOptimization


pso = ParticleSwarmOptimization(goal="Attention is all you need", n_particles=100)

pso.optimize(iterations=1000)
```

### Ant Colony Optimization
```python
from swarms_torch.ant_colony_swarm import AntColonyOptimization

# Usage:
goal_string = "Hello ACO"
aco = AntColonyOptimization(goal_string, num_iterations=1000)
best_solution = aco.optimize()
print("Best Matched String:", best_solution)

```

### Neural Network with Transformers as synapases
```python
import torch
from swarms_torch.nnt import NNTransformer

x = torch.randn(1, 10)

network = NNTransformer(
    neuron_count = 5, 
    num_states = 10,
    input_dim = 10,
    output_dim = 10,
    nhead = 2,
)
output = network(x)
print(output)
```

### CellularSwarm
a Cellular Neural Net with transformers as cells, time simulation, and a local neighboorhood!

```python
from swarms_torch import CellularSwarm 

x = torch.randn(10, 32, 512)  # sequence length of 10, batch size of 32, embedding size of 512
model = CellularSwarm(cell_count=5, input_dim=512, nhead=8)
output = model(x)

```
### Fish School/Sakana
- An all-new innovative approaches to machine learning that leverage the power of the Transformer model architecture. These systems are designed to mimic the behavior of a school of fish, where each fish represents an individual Transformer model. The goal is to optimize the performance of the entire school by learning from the best-performing fish.

```python
import torch
from swarms_torch.fish_school import Fish, FishSchool

# Create random source and target sequences
src = torch.randn(10, 32, 512)
tgt = torch.randn(10, 32, 512)

# Create random labels
labels = torch.randint(0, 512, (10, 32))

# Create a fish and train it on the random data
fish = Fish(512, 8, 6)
fish.train(src, tgt, labels)
print(fish.food)  # Print the fish's food

# Create a fish school and optimize it on the random data
school = FishSchool(10, 512, 8, 6, 100)
school.forward(src, tgt, labels)
print(school.fish[0].food)  # Print the first fish's food

```

### Swarmalators
```python
from swarms_torch import visualize_swarmalators, simulate_swarmalators

# Init for Swarmalator
# Example usage:
N = 100
J, alpha, beta, gamma, epsilon_a, epsilon_r, R = [0.1] * 7
D = 3  # Ensure D is an integer
xi, sigma_i = simulate_swarmalators(
    N, J, alpha, beta, gamma, epsilon_a, epsilon_r, R, D
)


# Call the visualization function
visualize_swarmalators(xi)
```

### Mixture of Mambas
- An 100% novel implementation of a swarm of MixtureOfMambas.
- Various fusion methods through averages, weighted_aggegrate, and more to come like a gating mechanism or other various methods.
- fusion methods: average, weighted, absmax, weighted_softmax, or your own custom function

```python
import torch
from swarms_torch import MixtureOfMambas

# Create a 3D tensor for text
x = torch.rand(1, 512, 512)

# Create an instance of the MixtureOfMambas model
model = MixtureOfMambas(
    num_mambas=2,            # Number of Mambas in the model
    dim=512,                 # Dimension of the input tensor
    d_state=1024,            # Dimension of the hidden state
    depth=4,                 # Number of layers in the model
    d_conv=1024,             # Dimension of the convolutional layers
    expand=4,                # Expansion factor for the model
    fusion_method="absmax",  # Fusion method for combining Mambas' outputs
    custom_fusion_func=None  # Custom fusion function (if any)
)

# Pass the input tensor through the model and print the output shape
print(model(x).shape)

```


### `SwitchMoE`

```python
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
```
### SimpleMoE
A very simple Mixture of Experts with softmax as a gating mechanism.

```python
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
```


# Documentation
- [Click here for documentation](https://swarmstorch.readthedocs.io/en/latest/swarms/)

# Examples
- There are various scripts in the playground folder with various examples for each swarm, like ant colony and fish school and spiral optimization.

# Todo
- [Check out the project board](https://github.com/users/kyegomez/projects/9/views/1)
- Make training scripts for each model on enwiki 8
- Create hivemind model for robotics, 1 model that takes in X inputs from N robots and distributes tasks to individual or many robots
- Swarm of liquid nets for text sequence modeling or vision
- Swarm of Convnets for facial recognition?
- Swarm of resnet
- Make various swarm architecture backbone's so it's plug in and play like parallel, basic neural network style, fish school.
- Swarm of transformers where each transformer is an expert and they all share the same weights so they can see each others knowledge, but inference is local.

# License
MIT
