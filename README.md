# Swarms-Torch: Enterprise-Grade Swarm Intelligence Architectures

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/jM3Z6M9uMq) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@kyegomez3242) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kye-g-38759a207/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/kyegomezb)

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [Key Features](#key-features)
- [Architecture Overview](#architecture-overview)
- [Installation](#installation)
- [Quick Start Guide](#quick-start-guide)
- [Model Implementations](#model-implementations)
- [Model Merging Techniques](#model-merging-techniques)
- [Community & Support](#community--support)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Citations](#citations)

---

## Executive Summary

**Swarms-Torch** is a cutting-edge PyTorch library that implements novel swarm intelligence architectures for next-generation AI systems. Our platform delivers 100% original swarming models designed to surpass traditional architectures like Transformers and State Space Models (SSMs). 

Built for enterprise-scale applications, Swarms-Torch provides production-ready implementations of bio-inspired algorithms including Particle Swarm Optimization with Transformers, Ant Colony systems, Neural Networks with Transformer synapses, and advanced Mixture of Experts architectures.

---

## Key Features

### 🔬 **Novel Architectures**
- Particle Swarm Optimization with Transformer particles
- Ant Colony Optimization with intelligent agents
- Cellular Neural Networks with Transformer cells
- Fish School/Sakana collective intelligence systems
- Swarmalator dynamics simulation

### 🏗️ **Enterprise Components**
- Mixture of Mambas with configurable fusion methods
- Switch Mixture of Experts (SwitchMoE)
- Simplified MoE implementations
- Firefly optimization algorithms

### 🔧 **Advanced Model Merging**
- HyperSlice merge techniques
- Random subspace merging
- Dimensional cross-fusion
- Weighted evolutionary crossover
- Permutation-based weight swapping

### 📈 **Production Ready**
- Optimized for large-scale deployment
- Comprehensive documentation
- Extensive test coverage
- Enterprise support available

---

## Architecture Overview

Swarms-Torch implements bio-inspired collective intelligence patterns that leverage the emergent behaviors of natural swarms. Our architectures combine:

- **Distributed Processing**: Multiple specialized agents working in parallel
- **Emergent Intelligence**: Complex behaviors arising from simple interaction rules
- **Adaptive Learning**: Dynamic optimization through collective feedback
- **Scalable Design**: Efficient scaling from prototype to production

---

## Installation

### Requirements
- Python 3.8+
- PyTorch 1.12+
- CUDA support recommended for optimal performance

### Install from PyPI
```bash
pip install swarms-torch
```

### Development Installation
```bash
git clone https://github.com/kyegomez/swarms-pytorch.git
cd swarms-pytorch
pip install -e .
```

---

## Quick Start Guide

### Basic Particle Swarm Optimization
```python
from swarms_torch import ParticleSwarmOptimization

# Initialize PSO with target optimization goal
pso = ParticleSwarmOptimization(
    goal="Attention is all you need", 
    n_particles=100
)

# Execute optimization process
pso.optimize(iterations=1000)
```

### Neural Network with Transformer Synapses
```python
import torch
from swarms_torch.nnt import NNTransformer

# Create input tensor
x = torch.randn(1, 10)

# Initialize network architecture
network = NNTransformer(
    neuron_count=5, 
    num_states=10,
    input_dim=10,
    output_dim=10,
    nhead=2,
)

# Forward pass
output = network(x)
```

---

## Model Implementations

### 1. Particle Swarm Optimization
**Use Case**: Hyperparameter optimization, neural architecture search
```python
from swarms_torch import ParticleSwarmOptimization

pso = ParticleSwarmOptimization(goal="Attention is all you need", n_particles=100)
pso.optimize(iterations=1000)
```

### 2. Ant Colony Optimization
**Use Case**: Combinatorial optimization, routing problems
```python
from swarms_torch.ant_colony_swarm import AntColonyOptimization

goal_string = "Hello ACO"
aco = AntColonyOptimization(goal_string, num_iterations=1000)
best_solution = aco.optimize()
```

### 3. Cellular Swarm Networks
**Use Case**: Distributed computing, parallel processing
```python
from swarms_torch import CellularSwarm 

x = torch.randn(10, 32, 512)
model = CellularSwarm(cell_count=5, input_dim=512, nhead=8)
output = model(x)
```

### 4. Fish School Intelligence
**Use Case**: Collective decision making, ensemble learning
```python
import torch
from swarms_torch.fish_school import FishSchool

src = torch.randn(10, 32, 512)
tgt = torch.randn(10, 32, 512)
labels = torch.randint(0, 512, (10, 32))

school = FishSchool(10, 512, 8, 6, 100)
school.forward(src, tgt, labels)
```

### 5. Mixture of Mambas
**Use Case**: Large language models, sequence processing
```python
import torch
from swarms_torch import MixtureOfMambas

x = torch.rand(1, 512, 512)
model = MixtureOfMambas(
    num_mambas=2,
    dim=512,
    d_state=1024,
    depth=4,
    fusion_method="absmax"
)
output = model(x)
```

### 6. Switch Mixture of Experts
**Use Case**: Sparse expert routing, efficient scaling
```python
import torch 
from swarms_torch import SwitchMoE

moe_layer = SwitchMoE(
    dim=768,
    hidden_dim=2048,
    output_dim=768,
    num_experts=16,
    use_aux_loss=False,
)

x = torch.rand(32, 128, 768)
output, auxiliary_loss = moe_layer(x)
```

### 7. Firefly Optimization
**Use Case**: Function optimization, genetic algorithms
```python
from swarms_torch.firefly import FireflyOptimizer
from torch import Tensor

def rosenbrock(x: Tensor) -> Tensor:
    return (100 * (x[..., 1:] - x[..., :-1] ** 2) ** 2 + (1 - x[..., :-1]) ** 2).sum(dim=-1)

optimizer = FireflyOptimizer(cost_function=rosenbrock)
optimizer.optimize()
best_solution = optimizer.get_best_solution()
```

---

## Model Merging Techniques

### Advanced Fusion Methods
```python
import torch 
from swarms_torch.mergers.all_new_evo_mergers import (
    hyperslice_merge,
    random_subspace_merge,
    dimensional_cross_fusion,
    weighted_evolutionary_crossover,
    permutation_weight_swapping,
)

# Initialize example models
model_1 = torch.nn.Linear(10, 10)
model_2 = torch.nn.Linear(10, 10)
model_3 = torch.nn.Linear(10, 10)

# HyperSlice merge
merged_model_hs = hyperslice_merge(
    [model_1, model_2, model_3], 
    slice_indices=[0, 2, 4]
)

# Random Subspace merge
merged_model_rs = random_subspace_merge(
    [model_1, model_2, model_3], 
    subspace_fraction=0.5
)

# Weighted Evolutionary Crossover
merged_model_wc = weighted_evolutionary_crossover(
    [model_1, model_2, model_3], 
    performance_scores=[0.7, 0.85, 0.65]
)
```

---

## Community & Support

### Official Resources

| Resource | Description | Link |
|----------|-------------|------|
| **Documentation** | Comprehensive API documentation and tutorials | [swarmstorch.readthedocs.io](https://swarmstorch.readthedocs.io/en/latest/swarms/) |
| **Discord Community** | Real-time support and discussions | [Join Discord](https://discord.gg/jM3Z6M9uMq) |
| **Official Blog** | Latest updates and technical insights | [swarms.apac.ai](https://swarms.apac.ai) |
| **Weekly Gatherings** | Community meetings every Thursday 1pm NYC | [Sign up here](https://lu.ma/5p2jnc2v) |

### Social Media & Updates

| Platform | Purpose | Link |
|----------|---------|------|
| **Twitter/X** | Latest announcements and updates | [@swarms_corp](https://twitter.com/swarms_corp) |
| **LinkedIn** | Professional network and company updates | [The Swarm Corporation](https://www.linkedin.com/company/the-swarm-corporation) |
| **YouTube** | Video tutorials and demonstrations | [Swarms Channel](https://www.youtube.com/channel/UC9yXyitkbU_WSy7bd_41SqQ) |
| **Personal Twitter** | Creator insights and development updates | [@kyegomezb](https://x.com/kyegomezb) |

### Getting Help

| Type | Where to Go | Response Time |
|------|-------------|---------------|
| **Bug Reports** | [GitHub Issues](https://github.com/swarms/gateway/issues) | 24-48 hours |
| **Feature Requests** | [GitHub Issues](https://github.com/swarms/gateway/issues) | 1-2 weeks |
| **General Questions** | [Discord #help](https://discord.gg/jM3Z6M9uMq) | Real-time |
| **Enterprise Support** | Contact via LinkedIn | 24 hours |

---

## Documentation

- **[API Reference](https://swarmstorch.readthedocs.io/en/latest/swarms/)**: Complete documentation of all classes and methods
- **[Examples](./playground/)**: Practical examples and implementation guides
- **[Contributing Guide](./CONTRIBUTING.md)**: Guidelines for contributing to the project
- **[Roadmap](https://github.com/users/kyegomez/projects/9)**: Development roadmap and future features

---

## Contributing

We welcome contributions from the community! Swarms-Torch is an open-source project that thrives on collaboration.

### How to Contribute
1. **Pick an Issue**: Look for issues tagged with `good first issue`
2. **Fork the Repository**: Create your own fork of the project
3. **Make Changes**: Implement your feature or bug fix
4. **Submit PR**: Create a pull request with detailed description
5. **Review Process**: Collaborate with maintainers on feedback

### Areas of Contribution
- **New Model Architectures**: Implement novel swarm intelligence patterns
- **Performance Optimization**: Improve computational efficiency
- **Documentation**: Enhance guides and API documentation
- **Testing**: Expand test coverage and validation
- **Bug Fixes**: Resolve existing issues

**Read our full [Contributing Guidelines](./CONTRIBUTING.md)**

### Contributors

<a href="https://github.com/kyegomez/swarms-pytorch/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=kyegomez/swarms-pytorch" />
</a>

---

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

---

## Citations

If you use Swarms-Torch in your research, please cite:

### Firefly Algorithm
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

### Hybrid Genetic-Firefly Algorithm
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

---

**© 2024 The Swarm Corporation. All rights reserved.**
