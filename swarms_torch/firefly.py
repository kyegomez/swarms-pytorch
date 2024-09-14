from typing import Callable

import einx  # s - species, p - population, i - population source, j - population target, t - tournament participants, d - dimension
import torch
from loguru import logger
from torch import Tensor
from torch import nn


class FireflyOptimizer(nn.Module):
    def __init__(
        self,
        cost_function: Callable[[Tensor], Tensor],
        steps: int = 5000,
        species: int = 4,
        population_size: int = 1000,
        dimensions: int = 10,
        lower_bound: float = -4.0,
        upper_bound: float = 4.0,
        mix_species_every: int = 25,
        beta0: float = 2.0,
        gamma: float = 1.0,
        alpha: float = 0.1,
        alpha_decay: float = 0.995,
        use_genetic_algorithm: bool = False,
        breed_every: int = 10,
        tournament_size: int = 100,
        num_children: int = 500,
        use_cuda: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize the FireflyOptimizer.

        Parameters
        ----------
        cost_function : Callable[[Tensor], Tensor]
            The objective function to minimize.
        steps : int, optional
            Number of optimization steps, by default 5000.
        species : int, optional
            Number of species, by default 4.
        population_size : int, optional
            Size of the population per species, by default 1000.
        dimensions : int, optional
            Dimensionality of the problem, by default 10.
        lower_bound : float, optional
            Lower bound for the variables, by default -4.0.
        upper_bound : float, optional
            Upper bound for the variables, by default 4.0.
        mix_species_every : int, optional
            Steps interval at which species mix, by default 25.
        beta0 : float, optional
            Base attractiveness, by default 2.0.
        gamma : float, optional
            Light absorption coefficient, by default 1.0.
        alpha : float, optional
            Randomness scaling factor, by default 0.1.
        alpha_decay : float, optional
            Decay rate of alpha per step, by default 0.995.
        use_genetic_algorithm : bool, optional
            Whether to use genetic algorithm operations, by default False.
        breed_every : int, optional
            Steps interval at which breeding occurs, by default 10.
        tournament_size : int, optional
            Size of the tournament for selection, by default 100.
        num_children : int, optional
            Number of children to produce during breeding, by default 500.
        use_cuda : bool, optional
            Whether to use CUDA if available, by default True.
        verbose : bool, optional
            Whether to print progress, by default True.
        """
        self.cost_function = cost_function
        self.steps = steps
        self.species = species
        self.population_size = population_size
        self.dimensions = dimensions
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.mix_species_every = mix_species_every
        self.beta0 = beta0
        self.gamma = gamma
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.use_genetic_algorithm = use_genetic_algorithm
        self.breed_every = breed_every
        self.tournament_size = tournament_size
        self.num_children = num_children
        self.use_cuda = use_cuda
        self.verbose = verbose

        # Additional initializations
        assert (
            self.tournament_size <= self.population_size
        ), "Tournament size must be less than or equal to population size."
        assert (
            self.num_children <= self.population_size
        ), "Number of children must be less than or equal to population size."

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.use_cuda else "cpu"
        )

        # Initialize fireflies
        self.fireflies = torch.zeros(
            (self.species, self.population_size, self.dimensions),
            device=self.device,
        ).uniform_(self.lower_bound, self.upper_bound)

        # Initialize alpha (in case we need to reset)
        self.current_alpha = self.alpha

    def optimize(self) -> None:
        """
        Run the Firefly optimization algorithm.
        """
        for step in range(self.steps):
            costs = self.cost_function(self.fireflies)

            if self.verbose:
                logger.info(f"Step {step}: Minimum cost {costs.amin():.5f}")

            # Fireflies with lower light intensity (high cost) move towards higher intensity (lower cost)
            move_mask = einx.greater("s i, s j -> s i j", costs, costs)

            # Get vectors of fireflies to one another
            delta_positions = einx.subtract(
                "s j d, s i d -> s i j d", self.fireflies, self.fireflies
            )

            distance = delta_positions.norm(dim=-1)

            betas = self.beta0 * torch.exp(-self.gamma * distance**2)

            # Calculate movements
            attraction = einx.multiply(
                "s i j, s i j d -> s i j d", move_mask * betas, delta_positions
            )
            random_walk = (
                self.current_alpha
                * (torch.rand_like(self.fireflies) - 0.5)
                * (self.upper_bound - self.lower_bound)
            )

            # Move the fireflies
            self.fireflies += (
                einx.sum("s i j d -> s i d", attraction) + random_walk
            )

            self.fireflies.clamp_(min=self.lower_bound, max=self.upper_bound)

            # Decay exploration factor
            self.current_alpha *= self.alpha_decay

            # Species mixing
            if self.species > 1 and (step % self.mix_species_every) == 0:
                midpoint = self.population_size // 2
                fireflies_a = self.fireflies[:, :midpoint]
                fireflies_b = self.fireflies[:, midpoint:]
                rotated_fireflies_b = torch.roll(
                    fireflies_b, shifts=1, dims=(0,)
                )
                self.fireflies = torch.cat(
                    (fireflies_a, rotated_fireflies_b), dim=1
                )

            # Genetic algorithm operations
            if self.use_genetic_algorithm and (step % self.breed_every) == 0:
                self._genetic_operations(costs)

    def _genetic_operations(self, costs: Tensor) -> None:
        """
        Perform genetic algorithm operations: selection, crossover, and replacement.

        Parameters
        ----------
        costs : Tensor
            Costs associated with each firefly.
        """
        fitness = 1.0 / costs

        batch_randperm = torch.randn(
            (self.species, self.num_children, self.population_size),
            device=self.device,
        ).argsort(dim=-1)
        tournament_indices = batch_randperm[..., : self.tournament_size]

        tournament_participants = einx.get_at(
            "s [p], s c t -> s c t", fitness, tournament_indices
        )
        winners_per_tournament = tournament_participants.topk(2, dim=-1).indices

        # Breed the top two winners of each tournament
        parent1, parent2 = einx.get_at(
            "s [p] d, s c parents -> parents s c d",
            self.fireflies,
            winners_per_tournament,
        )

        # Uniform crossover
        crossover_mask = torch.rand_like(parent1) < 0.5
        children = torch.where(crossover_mask, parent1, parent2)

        # Sort the fireflies by cost and replace the worst performing with the new children
        _, sorted_indices = costs.sort(dim=-1)
        sorted_fireflies = einx.get_at(
            "s [p] d, s sorted -> s sorted d", self.fireflies, sorted_indices
        )

        self.fireflies = torch.cat(
            (sorted_fireflies[:, : -self.num_children], children), dim=1
        )

    def get_best_solution(self) -> Tensor:
        """
        Retrieve the best solution found by the optimizer.

        Returns
        -------
        Tensor
            The best solution vector.
        """
        fireflies_flat = einx.rearrange("s p d -> (s p) d", self.fireflies)
        costs = self.cost_function(fireflies_flat)
        sorted_costs, sorted_indices = costs.sort(dim=-1)
        best_firefly = fireflies_flat[sorted_indices[0]]
        best_cost = sorted_costs[0]
        logger.info(f"Best solution found with cost {best_cost:.5f}")
        return best_firefly

    def generate(self) -> Tensor:
        """
        Generate a new set of fireflies.

        Returns
        -------
        Tensor
            The new set of fireflies.
        """
        self.fireflies = torch.zeros(
            (self.species, self.population_size, self.dimensions),
            device=self.device,
        ).uniform_(self.lower_bound, self.upper_bound)
        self.current_alpha = self.alpha
        return self.fireflies

    def reset(self) -> None:
        """
        Reset the optimizer to its initial state.
        """
        self.generate()
        self.current_alpha = self.alpha


# Example usage:

# def rosenbrock(x: Tensor) -> Tensor:
#     return (100 * (x[..., 1:] - x[..., :-1] ** 2) ** 2 + (1 - x[..., :-1]) ** 2).sum(dim=-1)

# if __name__ == "__main__":
#     optimizer = FireflyOptimizer(cost_function=rosenbrock)
#     optimizer.optimize()
#     best_solution = optimizer.get_best_solution()
#     print(f"Best solution: {best_solution}")
