# Novel Swarm Intelligence Model Architectures

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/jM3Z6M9uMq) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@kyegomez3242) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kye-g-38759a207/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/kyegomezb)


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

### Firefly

Exploration into the Firefly algorithm (a generalized version of particle swarm optimization) in Pytorch. In particular interested in hybrid <a href="https://academic.oup.com/jcde/article/9/2/706/6566441">firefly + genetic algorithms</a>, or ones that are <a href="https://www.sciencedirect.com/science/article/abs/pii/S0957417423005298">gender-based</a>. This code was adapted from lucidrains.

```python
from swarms_torch.firefly import FireflyOptimizer
from torch import Tensor


def rosenbrock(x: Tensor) -> Tensor:
    return (
        100 * (x[..., 1:] - x[..., :-1] ** 2) ** 2 + (1 - x[..., :-1]) ** 2
    ).sum(dim=-1)


if __name__ == "__main__":
    optimizer = FireflyOptimizer(cost_function=rosenbrock)
    optimizer.optimize()
    best_solution = optimizer.get_best_solution()
    print(f"Best solution: {best_solution}")

```


# Merging Multiple Models

- `hyperslice_merge` Merge models by selecting specific slices of weight tensors across the models.

- `random_subspace_merge` Merge models by randomly selecting subspaces of weights and averaging them.

- `dimensional_cross_fusion` Merge models by fusing weights along a specific axis (dimension) across models.

- `weighted_evolutionary_crossover` Merge models using a weighted crossover based on performance scores.

- `permutation_weight_swapping` Merge models by permuting weight matrices and swapping them between models.


```python
import torch 
from swarms_torch.mergers.all_new_evo_mergers import (
    hyperslice_merge,
    random_subspace_merge,
    dimensional_cross_fusion,
    weighted_evolutionary_crossover,
    permutation_weight_swapping,
)

# Example of how to use the logger and merge methods
if __name__ == "__main__":
    # Example models, replace with actual model instances
    model_1 = torch.nn.Linear(10, 10)
    model_2 = torch.nn.Linear(10, 10)
    model_3 = torch.nn.Linear(10, 10)

    # Perform HyperSlice merge
    merged_model_hs = hyperslice_merge(
        [model_1, model_2, model_3], slice_indices=[0, 2, 4]
    )

    # Perform Random Subspace merge
    merged_model_rs = random_subspace_merge(
        [model_1, model_2, model_3], subspace_fraction=0.5
    )

    # Perform Dimensional Cross-fusion merge
    merged_model_dc = dimensional_cross_fusion([model_1, model_2], cross_axis=0)

    # Perform Weighted Evolutionary Crossover merge
    merged_model_wc = weighted_evolutionary_crossover(
        [model_1, model_2, model_3], performance_scores=[0.7, 0.85, 0.65]
    )

    # Perform Permutation-based Weight Swapping
    merged_model_pw = permutation_weight_swapping(
        [model_1, model_2], permutation_seed=42
    )

```




# Documentation
- [Click here for documentation](https://swarmstorch.readthedocs.io/en/latest/swarms/)

# Examples
- There are various scripts in the playground folder with various examples for each swarm, like ant colony and fish school and spiral optimization.


## 🫶 Contributions:

The easiest way to contribute is to pick any issue with the `good first issue` tag 💪. Read the Contributing guidelines [here](/CONTRIBUTING.md). Bug Report? [File here](https://github.com/swarms/gateway/issues) | Feature Request? [File here](https://github.com/swarms/gateway/issues)

Swarms is an open-source project, and contributions are VERY welcome. If you want to contribute, you can create new features, fix bugs, or improve the infrastructure. Please refer to the [CONTRIBUTING.md](https://github.com/kyegomez/swarms-pytorch/blob/master/CONTRIBUTING.md) and our [contributing board](https://github.com/users/kyegomez/projects/9) to participate in Roadmap discussions!

<a href="https://github.com/kyegomez/swarms-pytorch/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=kyegomez/swarms-pytorch" />
</a>

----

## Community

Join our growing community around the world, for real-time support, ideas, and discussions on Swarms 😊 

- View our official [Blog](https://swarms.apac.ai)
- Follow us on [Twitter](https://twitter.com/kyegomez)
- Connect with us on [LinkedIn](https://www.linkedin.com/company/the-swarm-corporation)
- Visit us on [YouTube](https://www.youtube.com/channel/UC9yXyitkbU_WSy7bd_41SqQ)
- [Join the Swarms community on Discord!](https://discord.gg/jM3Z6M9uMq)
- Join our Swarms Community Gathering every Thursday at 1pm NYC Time to unlock the potential of autonomous agents in automating your daily tasks [Sign up here](https://lu.ma/5p2jnc2v)


# License
MIT


## Citations

```bibtex
@article{Yang2018WhyTF,
    title   = {Why the Firefly Algorithm Works?},
    author  = {Xin-She Yang and Xingshi He},
    journal = {ArXiv},
    year    = {2018},
    volume  = {abs/1806.01632},
    url     = {https://api.semanticscholar.org/CorpusID:46940737}
}
```

```bibtex
@article{article,
    author  = {El-Shorbagy, M. and Elrefaey, Adel},
    year    = {2022},
    month   = {04},
    pages   = {706-730},
    title   = {A hybrid genetic-firefly algorithm for engineering design problems},
    volume  = {Journal of Computational Design and Engineering, Volume 9},
    journal = {Journal of Computational Design and Engineering},
    doi     = {10.1093/jcde/qwac013}
}
```