[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Swarms in Torch
Swarming algorithms like PSO, Ant Colony, Sakana, and more in PyTorch primitives😊


## Installation

You can install the package using pip

```bash
pip3 install swarms-torch
```

# Usage
- We have just PSO now, but we're adding in ant colony and others!

```python
from swarms_torch import ParticleSwarmOptimization

#test
pso = ParticleSwarmOptimization(goal="Attention is all you need", n_particles=100)
pso.optimize(iterations=1000)

```

# License
MIT



