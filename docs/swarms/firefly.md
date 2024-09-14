# FireflyOptimizer

```python
class FireflyOptimizer(cost_function, steps=5000, species=4, population_size=1000, dimensions=10, lower_bound=-4.0, upper_bound=4.0, mix_species_every=25, beta0=2.0, gamma=1.0, alpha=0.1, alpha_decay=0.995, use_genetic_algorithm=False, breed_every=10, tournament_size=100, num_children=500, use_cuda=True, verbose=True)
```

The `FireflyOptimizer` class implements the Firefly Algorithm to minimize a given objective function. It simulates the flashing behavior of fireflies to explore the search space efficiently.

## Parameters

- **cost_function** (callable):  
  The objective function to minimize. Should accept a `torch.Tensor` and return a `torch.Tensor` of costs.

- **steps** (int, optional):  
  Number of optimization steps. Default: `5000`.

- **species** (int, optional):  
  Number of species in the population. Default: `4`.

- **population_size** (int, optional):  
  Number of fireflies in each species. Default: `1000`.

- **dimensions** (int, optional):  
  Dimensionality of the search space. Default: `10`.

- **lower_bound** (float, optional):  
  Lower bound of the search space. Default: `-4.0`.

- **upper_bound** (float, optional):  
  Upper bound of the search space. Default: `4.0`.

- **mix_species_every** (int, optional):  
  Interval (in steps) to mix species. Default: `25`.

- **beta0** (float, optional):  
  Base attractiveness coefficient. Default: `2.0`.

- **gamma** (float, optional):  
  Light absorption coefficient controlling intensity decay. Default: `1.0`.

- **alpha** (float, optional):  
  Randomness scaling factor. Default: `0.1`.

- **alpha_decay** (float, optional):  
  Decay rate of `alpha` per step. Default: `0.995`.

- **use_genetic_algorithm** (bool, optional):  
  Whether to include genetic algorithm operations. Default: `False`.

- **breed_every** (int, optional):  
  Steps between breeding operations when using genetic algorithm. Default: `10`.

- **tournament_size** (int, optional):  
  Number of participants in each tournament selection. Default: `100`.

- **num_children** (int, optional):  
  Number of offspring produced during breeding. Default: `500`.

- **use_cuda** (bool, optional):  
  Use CUDA for computations if available. Default: `True`.

- **verbose** (bool, optional):  
  Print progress messages during optimization. Default: `True`.

## Attributes

| Attribute          | Type            | Description                                            |
|--------------------|-----------------|--------------------------------------------------------|
| `fireflies`        | `torch.Tensor`  | Positions of the fireflies in the search space.        |
| `device`           | `torch.device`  | Device used for computations (`cpu` or `cuda`).        |
| `current_alpha`    | `float`         | Current value of `alpha` during optimization.          |

## Methods

### `optimize()`

Runs the optimization loop for the specified number of steps.

**Example:**

```python
optimizer.optimize()
```

### `get_best_solution()`

Retrieves the best solution found by the optimizer.

**Returns:**

- **best_firefly** (`torch.Tensor`):  
  The best solution vector found.

**Example:**

```python
best_solution = optimizer.get_best_solution()
print(f"Best solution: {best_solution}")
```

### `generate()`

Generates a new set of fireflies, reinitializing their positions.

**Returns:**

- **fireflies** (`torch.Tensor`):  
  The new set of fireflies.

**Example:**

```python
optimizer.generate()
```

### `reset()`

Resets the optimizer to its initial state, including `alpha` and firefly positions.

**Example:**

```python
optimizer.reset()
```

---

**Note:** The Firefly Algorithm is inspired by the flashing behavior of fireflies and is suitable for continuous optimization problems. This implementation allows for customization and includes optional genetic algorithm operations for enhanced performance.

**Example Usage:**

```python
from swarms_torch.firefly import FireflyOptimizer
from torch import Tensor


def rosenbrock(x: Tensor) -> Tensor:
    return (
        100 * (x[..., 1:] - x[..., :-1] ** 2) ** 2 + (1 - x[..., :-1]) ** 2
    ).sum(dim=-1)


if __name__ == "__main__":
    optimizer = FireflyOptimizer(
        cost_function=rosenbrock,
        steps=100,
        species=10,
        population_size=100,
        dimensions=10,
        lower_bound=-4,
        upper_bound=4,
        # Many more parameters can be set, see the documentation for more details
    )
    optimizer.optimize()
    best_solution = optimizer.get_best_solution()
    print(f"Best solution: {best_solution}")
```