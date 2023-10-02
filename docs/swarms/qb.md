# QueenBeeGa Class

The `QueenBeeGa` class implements the Queen Bee Genetic Algorithm (GA). This GA is inspired by the evolution of bees, where the fittest solution is designated as the queen and the rest of the population contends to mate with it. The strong exploitation is balanced by a higher than normal mutation rate.

## Attributes
---
-   `goal` (str): The goal string to be optimized.
-   `pop_size` (int): Population size.
-   `mutation_prob` (float): Mutation probability.
-   `strong_mutation_rate` (float): Strong mutation rate.
-   `strong_mutation_prob` (float): Strong mutation probability.
-   `num_tournament_participants` (int): Number of tournament participants.
-   `gene_length` (int): Length of the gene.
-   `gene_midpoint` (int): Midpoint of the gene.
-   `target_gene` (torch.Tensor): The target gene.
-   `strong_mutate_pool_size` (float): Size of the strong mutate pool.
-   `num_code_mutate` (float): Number of code mutations.
-   `strong_num_code_mutate` (float): Number of strong code mutations.
-   `pool` (torch.Tensor): The pool of genes.
-   `queen` (torch.Tensor): The queen gene.
-   `queen_fitness` (float): The fitness of the queen.
-   `generation` (int): The current generation.

## Methods
-------

### `__init__(self, goal: str = "Attention is all you need", pop_size: int = 100, mutation_prob: float = 0.04, strong_mutation_rate: float = 0.1, strong_mutation_prob: float = 0.25, num_tournament_participants: int = 25)`

The constructor for the `QueenBeeGa` class. Initializes the pool of genes, the queen, and the queen's fitness.

#### Parameters

-   `goal` (str, optional): The goal string to be optimized. Default is "Attention is all you need".
-   `pop_size` (int, optional): Population size. Default is 100.
-   `mutation_prob` (float, optional): Mutation probability. Default is 0.04.
-   `strong_mutation_rate` (float, optional): Strong mutation rate. Default is 0.1.
-   `strong_mutation_prob` (float, optional): Strong mutation probability. Default is 0.25.
-   `num_tournament_participants` (int, optional): Number of tournament participants. Default is 25.

#### Example

```
optimizer = QueenBeeGa(goal="Attention is all you need", pop_size=100, mutation_prob=0.04, strong_mutation_rate=0.1, strong_mutation_prob=0.25, num_tournament_participants=25)
```


### `encode(s)`

Converts a string to its ASCII values.

#### Parameters

-   `s` (str): The string to encode.

#### Returns

-   `encoded` (torch.Tensor): The encoded string.

#### Example

```
encoded = QueenBeeGa.encode("Hello")
```


### `decode(t)`

Converts a tensor of ASCII values back to a string.

#### Parameters

-   `t` (torch.Tensor): The tensor to decode.

#### Returns

-   `decoded` (str): The decoded string.

#### Example

```
decoded = QueenBeeGa.decode(encoded)
```


### `run(self, max_generations: int = 1000)`

Runs the Queen Bee GA. Evolves the population for a given number of generations.

#### Parameters

-   `max_generations` (int, optional): The maximum number of generations. Default is 1000.

#### Example

```
optimizer.run(max_generations=100)
```


### `_evolve(self)`

Executes one step of the evolution process. Sorts the population by fitness, displays the queen and the population, and updates the queen and the population.

#### Example

```
optimizer._evolve()
```


### `_check_convergence(self)`

Checks if any of the solutions has achieved the goal.

#### Returns

-   `converged` (bool): Whether any of the solutions has achieved the goal.

#### Example

```
converged = optimizer._check_convergence()
```
------

## Usage Examples
--------------

### Example 1: Optimize a String

In this example, we will optimize the string "Attention is all you need" using a population size of 100, a mutation probability of 0.04, a strong mutation rate of 0.1, a strong mutation probability of 0.25, and 25 tournament participants.

```python
optimizer = QueenBeeGa(goal="Attention is all you need", pop_size=100, mutation_prob=0.04, strong_mutation_rate=0.1, strong_mutation_prob=0.25, num_tournament_participants=25)
optimizer.run(max_generations=100)
```


### Example 2: Using a Different Goal String

In this example, we will optimize the string "Hello, World!" using a population size of 100, a mutation probability of 0.04, a strong mutation rate of 0.1, a strong mutation probability of 0.25, and 25 tournament participants.

```python
optimizer = QueenBeeGa(goal="Hello, World!", pop_size=100, mutation_prob=0.04, strong_mutation_rate=0.1, strong_mutation_prob=0.25, num_tournament_participants=25)
optimizer.run(max_generations=100)
```


### Example 3: Using a Different Population Size

In this example, we will optimize the string "Attention is all you need" using a population size of 200, a mutation probability of 0.04, a strong mutation rate of 0.1, a strong mutation probability of 0.25, and 25 tournament participants.

```python
optimizer = QueenBeeGa(goal="Attention is all you need", pop_size=200, mutation_prob=0.04, strong_mutation_rate=0.1, strong_mutation_prob=0.25, num_tournament_participants=25)
optimizer.run(max_generations=100)
```


### Example 4: Using Different Mutation Probabilities

In this example, we will optimize the string "Attention is all you need" using a population size of 100, a mutation probability of 0.05, a strong mutation rate of 0.1, a strong mutation probability of 0.3, and 25 tournament participants.

```python
optimizer = QueenBeeGa(goal="Attention is all you need", pop_size=100, mutation_prob=0.05, strong_mutation_rate=0.1, strong_mutation_prob=0.3, num_tournament_participants=25)
optimizer.run(max_generations=100)
```


### Example 5: Using a Different Number of Tournament Participants

In this example, we will optimize the string "Attention is all you need" using a population size of 100, a mutation probability of 0.04, a strong mutation rate of 0.1, a strong mutation probability of 0.25, and 30 tournament participants.

```python
optimizer = QueenBeeGa(goal="Attention is all you need", pop_size=100, mutation_prob=0.04, strong_mutation_rate=0.1, strong_mutation_prob=0.25, num_tournament_participants=30)
optimizer.run(max_generations=100)
```