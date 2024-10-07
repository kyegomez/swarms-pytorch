import torch
from torch import nn


class AntColonyOptimization(nn.Module):
    """
    Ant Colony Optimization
    Overview: https://en.wikipedia.org/wiki/Ant_colony_optimization_algorithms

    How does it work?
    1. Initialize pheromone levels for each path
    2. For each ant, choose the next path based on the pheromone levels
    3. Update the pheromone levels
    4. Repeat step 2 to 3 until the maximum number of iterations is reached

    Parameters
    ----------
    goal: str
        The goal string to be optimized
    num_ants: int
        Number of ants
    evaporation_rate: float
        Evaporation rate

    Usage
    -----
    from swarms_torch import AntColonyOptimization

    goal_string = "Hello ACO"
    aco = AntColonyOptimization(goal_string, num_iterations=1000)
    best_solution = aco.optimize()

    print("Best Matched String:", best_solution)

    Features to implement
    --------
    1. Add a stopping criterion
    2. Add a callback function to track the progress
    3. Add a function to plot the pheromone levels
    4. Add a function to plot the ants
    5. Add a function to plot the best solution

    """

    def __init__(
        self,
        goal: str = None,
        num_ants: int = 10000,
        evaporation_rate: float = 0.1,
        alpha: int = 1,
        beta: int = 1,
        num_iterations: int = 10010,
    ):
        self.goal = torch.tensor([ord(c) for c in goal], dtype=torch.float32)
        self.num_ants = num_ants
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.num_iterations = num_iterations
        # Pheromone levels can be initialized for different paths
        # (architectures)
        self.pheromones = torch.ones(num_ants)
        self.solutions = []

    def fitness(self, solution):
        """Fitness of a solution"""
        return -torch.norm(solution - self.goal)

    def update_pheromones(self):
        """Update pheromone levels"""
        for i, solution in enumerate(self.solutions):
            self.pheromones[i] = (1 - self.evaporation_rate) * self.pheromones[
                i
            ] + self.fitness(solution)

    def choose_next_path(self):
        """Choose the next path based on the pheromone levels"""
        probabilities = (self.pheromones**self.alpha) * (
            (1.0 / (1 + self.pheromones)) ** self.beta
        )

        probabilities /= probabilities.sum()

        return torch.multinomial(probabilities, num_samples=1).item()

    def optimize(self):
        """Optimize the goal string"""
        for iteration in range(self.num_iterations):
            self.solutions = []
            for _ in range(self.num_ants):
                # This is a placeholder. Actual implementation will define how
                # ants traverse the search space.
                solution = torch.randint(
                    32, 127, (len(self.goal),), dtype=torch.float32
                )  # Random characters.
                self.solutions.append(solution)
            self.update_pheromones()

        best_solution_index = self.pheromones.argmax().item()
        best_solution = self.solutions[best_solution_index]
        return "".join([chr(int(c)) for c in best_solution])
