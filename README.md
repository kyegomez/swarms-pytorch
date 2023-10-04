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

- Ant Colony Optimization
```python
from swarms_torch.ant_colony_swarm import AntColonyOptimization

# Usage:
goal_string = "Hello ACO"
aco = AntColonyOptimization(goal_string, num_iterations=1000)
best_solution = aco.optimize()
print("Best Matched String:", best_solution)

```

- Neural Network with Transformers as synapases LMAO.
```python
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
```

- CellularSwarm, a Cellular Neural Net with transformers as cells, time simulation, and a local neighboorhood!

```python
from swarms_torch import CellularSwarm 

x = torch.randn(10, 32, 512)  # sequence length of 10, batch size of 32, embedding size of 512
model = CellularSwarm(cell_count=5, input_dim=512, nhead=8)
output = model(x)

```

# Documentation
- [Click here for documentation](https://swarmstorch.readthedocs.io/en/latest/swarms/)


# Todo
Here are 10 swarming neural network algorithms, with brief overviews, algorithmic pseudocode, and potential use cases for each:

1. **Particle Swarm Optimization (PSO)**
    - **Overview**: Simulates the social behavior of birds flocking or fish schooling. It adjusts trajectories of individual agents (particles) based on their own and their neighbors' best known positions.
    - **Pseudocode**:
        ```
        Initialize each particle with random position and velocity
        while not converged:
            for each particle:
                calculate fitness value
                if fitness better than its best, update its best
            end for
            for each particle:
                update velocity towards its best and global best
                update position
            end for
        end while
        ```
    - **Use Cases**: Function optimization, neural network training, feature selection.

2. **Ant Colony Optimization (ACO)**
    - **Overview**: Simulates the foraging behavior of ants to find paths through graphs. Uses pheromones to mark good paths, which evaporate over time.
    - **Pseudocode**:
        ```
        Initialize pheromones on paths
        while not converged:
            deploy ants to find paths based on pheromone strengths
            update pheromone strengths based on quality of paths
            evaporate some pheromone
        end while
        ```
    - **Use Cases**: Path finding, traveling salesman problem, network routing.

3. **Bee Algorithm (BA)**
    - **Overview**: Simulates the behavior of honey bees finding optimal nectar sources.
    - **Pseudocode**:
        ```
        Initialize scout bees randomly
        while not converged:
            deploy scout bees to search for nectar
            recruit forager bees based on nectar quality
            more foragers for better nectar sources
        end while
        ```
    - **Use Cases**: Job scheduling, function optimization, image processing.

4. **Firefly Algorithm (FA)**
    - **Overview**: Based on the flashing behavior of fireflies. Fireflies are attracted to each other depending on the brightness of their flashing.
    - **Pseudocode**:
        ```
        Initialize fireflies randomly
        while not converged:
            for each firefly:
                move towards brighter fireflies
            end for
        end while
        ```
    - **Use Cases**: Multi-modal optimization, feature selection, clustering.

5. **Bat Algorithm (BA)**
    - **Overview**: Inspired by the echolocation behavior of bats. Bats fly randomly and adjust positions based on emitted and returned echoes.
    - **Pseudocode**:
        ```
        Initialize bats with random positions and velocities
        while not converged:
            for each bat:
                adjust velocity based on echolocation feedback
                move bat
            end for
        end while
        ```
    - **Use Cases**: Engineering design, tuning machine learning algorithms, scheduling.

6. **Wolf Search Algorithm (WSA)**
    - **Overview**: Models the hunting behavior of gray wolves.
    - **Pseudocode**:
        ```
        Initialize wolves
        while not converged:
            calculate fitness of all wolves
            rank wolves: alpha, beta, delta, and omega
            adjust position of omega wolves towards alpha, beta, and delta
        end while
        ```
    - **Use Cases**: Neural network training, function optimization, game AI.

7. **Fish School Search (FSS)**
    - **Overview**: Simulates the social behavior of fish schooling.
    - **Pseudocode**:
        ```
        Initialize fish randomly in search space
        while not converged:
            for each fish:
                if neighbor has better food, move towards it
                else explore randomly
            end for
            adjust school behavior based on total food
        end while
        ```
    - **Use Cases**: Load balancing, function optimization, robotics.

8. **Cuckoo Search (CS)**
    - **Overview**: Based on the reproduction strategy of cuckoos. They lay eggs in host bird nests and those nests with the best eggs (solutions) will carry on to the next generation.
    - **Pseudocode**:
        ```
        Initialize host nests with eggs (solutions)
        while not converged:
            lay new eggs by Levy flights
            if new egg better than the worst in nest, replace it
            discover a fraction of nests and lay new eggs
        end while
        ```
    - **Use Cases**: Engineering design optimization, image processing, numerical optimization.

9. **Whale Optimization Algorithm (WOA)**
    - **Overview**: Simulates the bubble-net hunting strategy of humpback whales.
    - **Pseudocode**:
        ```
        Initialize whales
        while not converged:
            for each whale:
                encircle prey or search using bubble-net approach
            end for


        end while
        ```
    - **Use Cases**: Structural optimization, neural network training, function optimization.

10. **Grasshopper Optimization Algorithm (GOA)**
    - **Overview**: Simulates the swarming behavior of grasshoppers towards food sources.
    - **Pseudocode**:
        ```
        Initialize grasshoppers
        while not converged:
            for each grasshopper:
                adjust position towards other grasshoppers based on distance and food source
            end for
        end while
        ```
    - **Use Cases**: Job scheduling, clustering, neural network training.

# License
MIT
