[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Swarms in Torch
Swarms in Torch is an experimental repository designed to cater to your swarming algorithm needs. With a range of useful algorithms including Particle Swarm Optimization (PSO), Ant Colony, Sakana, and more, all implemented using PyTorch primitives, you can easily leverage the power of swarming techniques in your projects.


## Installation

You can install the package using pip

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
- Mixture of Mamba models of SSMs, could be improved with a router of some kind or better aggregation methods!

```python
import torch
from swarms_torch.mixture_of_mamba import MixtureOfMambas

# Example Usage
num_models = 3
dim = 16
state_range = (1, 20)
conv_range = (1, 10)
expand_range = (1, 5)

mixture_model = MixtureOfMambas(num_models, dim, state_range, conv_range, expand_range)
x = torch.randn(2, 64, dim).to("cuda")
output = mixture_model(
    x, aggregation_method="average"
)  # Or use 'weighted' with weights


```

# Documentation
- [Click here for documentation](https://swarmstorch.readthedocs.io/en/latest/swarms/)

# Playground
- There are various scripts in the playground folder with various examples for each swarm, like ant colony and fish school and spiral optimization.

# Todo
- [Check out the project board](https://github.com/users/kyegomez/projects/9/views/1)
- make training script ready for fish school with autoregressive,
- practical examples of training models and conduct interence
- upload to huggingface a small model
- train fish school, neural transformer, and pso transformer, 
- Create hivemind model for robotics, 1 model that takes in X inputs from N robots and distributes tasks to individual or many robots
- Swarm of liquid nets for text sequence modeling or vision
- Swarm of Convnets for facial recognition?
- Swarm of transformers where each transformer is an expert and they all share the same weights so they can see each others knowledge, but inference is local.


# License
MIT
