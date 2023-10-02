# ParticleSwarmOptimization Class

The `ParticleSwarmOptimization` class implements the Particle Swarm Optimization (PSO) algorithm. PSO is a computational method that optimizes a problem by iteratively trying to improve a candidate solution with regard to a given measure of quality. It solves a problem by having a population of candidate solutions, here dubbed particles, and moving these particles around in the search-space according to simple mathematical formulae over the particle's position and velocity.

## Attributes

-   `goal` (str): The goal string to be optimized.
-   `n_particles` (int): Number of particles.
-   `inertia` (float): Inertia weight.
-   `personal_best_weight` (float): Personal best weight.
-   `global_best_weight` (float): Global best weight.
-   `particles` (torch.Tensor): The particles' positions.
-   `velocities` (torch.Tensor): The particles' velocities.
-   `personal_best` (torch.Tensor): The personal best positions of each particle.
-   `global_best` (torch.Tensor): The global best position.

## Methods

### `__init__(self, goal: str = None, n_particles: int = 100, inertia: float = 0.5, personal_best_weight: float = 1.5, global_best_weight: float = 1.5, dim: int = 1)`

The constructor for the `ParticleSwarmOptimization` class. Initializes the particles with random positions and velocities, and the personal best and global best with the initial positions of the particles.

#### Parameters

-   `goal` (str, optional): The goal string to be optimized.
-   `n_particles` (int, optional): Number of particles. Default is 100.
-   `inertia` (float, optional): Inertia weight. Default is 0.5.
-   `personal_best_weight` (float, optional): Personal best weight. Default is 1.5.
-   `global_best_weight` (float, optional): Global best weight. Default is 1.5.
-   `dim` (int, optional): The dimension of the problem. Default is 1.

#### Example

```
pso = ParticleSwarmOptimization(goal="Attention is all you need", n_particles=100)
```


### `compute_fitness(self, particle)`

Computes the fitness value of a particle. The fitness value is the inverse of the Euclidean distance between the particle and the goal.

#### Parameters

-   `particle` (torch.Tensor): The particle to compute the fitness value for.

#### Returns

-   `fitness` (float): The fitness value of the particle.

#### Example

```
fitness = pso.compute_fitness(particle)
```


### `update(self)`

Updates the personal best and global best, and the velocity and position of each particle.

#### Example

```
pso.update()
```


### `optimize(self, iterations: int = 1000)`

Optimizes the goal string. Updates the particles for a given number of iterations and prints the best particle at each iteration.

#### Parameters

-   `iterations` (int, optional): The maximum number of iterations. Default is 1000.

#### Example

```
pso.optimize(iterations=1000)
```


Usage Examples
--------------

### Example 1: Optimize a String

In this example, we will optimize the string "Attention is all you need" using 100 particles.

```python
pso = ParticleSwarmOptimization(goal="Attention is all you need", n_particles=100)
pso.optimize(iterations=1000)
```
### Example 2: Optimize a Different String

In this example, we will optimize the string "Hello, World!" using 200 particles.

```python
pso = ParticleSwarmOptimization(goal="Hello, World!", n_particles=200)
pso.optimize(iterations=1000)
```


### Example 3: Using Different Weights

In this example, we will optimize the string "Particle Swarm Optimization" using 100 particles, an inertia weight of 0.8, a personal best weight of 2.0, and a global best weight of 2.0.

```python
pso = ParticleSwarmOptimization(goal="Particle Swarm Optimization", n_particles=100, inertia=0.8, personal_best_weight=2.0, global_best_weight=2.0)
pso.optimize(iterations=1000)
```


### Example 4: Using a Large Number of Particles

In this example, we will optimize the string "Large number of particles" using 1000 particles.

```python
pso = ParticleSwarmOptimization(goal="Large number of particles", n_particles=1000)
pso.optimize(iterations=1000)
```


### Example 5: Using a Small Number of Iterations

In this example, we will optimize the string "Small number of iterations" using 100 particles and 100 iterations.

```python
pso = ParticleSwarmOptimization(goal="Small number of iterations", n_particles=100)
pso.optimize(iterations=100)
```


### Example 6: Using a Large Number of Iterations

In this example, we will optimize the string "Large number of iterations" using 100 particles and 10000 iterations.

```python
pso = ParticleSwarmOptimization(goal="Large number of iterations", n_particles=100)
pso.optimize(iterations=10000)
```


### Example 7: Using Different Characters

In this example, we will optimize the string "1234567890" using 100 particles.

```python
pso = ParticleSwarmOptimization(goal="1234567890", n_particles=100)
pso.optimize(iterations=1000)
```


### Example 8: Using Special Characters

In this example, we will optimize the string "!@#$%^&*()" using 100 particles.

```python
pso = ParticleSwarmOptimization(goal="!@#$%^&*()", n_particles=100)
pso.optimize(iterations=1000)
```


### Example 9: Using a Long String

In this example, we will optimize a long string using 100 particles.

```python
pso = ParticleSwarmOptimization(goal="This is a very long string that we want to optimize using Particle Swarm Optimization.", n_particles=100)
pso.optimize(iterations=1000)
```


### Example 10: Using a Short String

In this example, we will optimize a short string using 100 particles.

```python
pso = ParticleSwarmOptimization(goal="Short", n_particles=100)
pso.optimize(iterations=1000)
```