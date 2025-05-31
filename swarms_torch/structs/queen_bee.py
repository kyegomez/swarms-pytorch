import torch
from torch import nn


class QueenBeeGa(nn.Module):
    """
    Queen Bee evolution for genetic algos

    Inspired by the evolution of bees, the fittest solution is designated
    and the rest of the population contends to mate with it.

    The strong exploitation is balanced by a higher than a normal mutation rate.

    Reference:
    ---------
    https://www.researchgate.net/publication/228961729_A_Queen_Bee_GA_for_optimization

    Usage
    -----
    optimizer = QueenBeeGa(
        goal="Attention is all you need",
        pop_size=100,
        mutation_prob=0.04,
        strong_mutation_rate=0.1,
        strong_mutation_prob=0.25,
        num_tournament_participants=25
    )
    optimizer.run(max_generations=100)
    """

    def __init__(
        self,
        goal: str = "Attention is all you need",
        pop_size: int = 100,
        mutation_prob: float = 0.04,
        strong_mutation_rate: float = 0.1,
        strong_mutation_prob: float = 0.25,
        num_tournament_participants: int = 25,
    ):
        """
        QueenBeeGa with params and initial configs

        Parameters
        ----------
        goal: str
            The goal string to be optimized
        pop_size: int
            Number of ants
        mutation_prob: float
            Mutation rate
        strong_mutation_rate: float
            Strong mutation rate
        strong_mutation_prob: float
            Strong mutation probability
        num_tournament_participants: int
            Number of tournament participants
        """
        self.goal = goal
        self.pop_size = pop_size
        self.mutation_prob = mutation_prob
        self.strong_mutation_rate = strong_mutation_rate
        self.strong_mutation_prob = strong_mutation_prob
        self.num_tournament_participants = num_tournament_participants

        self.gene_length = len(goal)
        self.gene_midpoint = self.gene_length // 2
        self.target_gene = self.encode(goal)

        self.strong_mutate_pool_size = strong_mutation_rate * pop_size
        self.num_code_mutate = mutation_prob * self.gene_length
        self.strong_num_code_mutate = strong_mutation_prob * self.gene_length

        self.pool = torch.randint(0, 255, (pop_size, self.gene_length))
        self.queen = None
        self.queen_fitness = None
        self.generation = 0

    @staticmethod
    def encode(s):
        """Convert string to it's values"""
        return torch.tensor([ord(c) for c in s])

    @staticmethod
    def decode(t):
        """Convert ASCII values tensor back to string"""
        return "".join([chr(i) for i in t.tolist()])

    def run(self, max_generations: int = 1000):
        """
        Run the queen genertic algorithm evolution

        Parameters:
        -----------
        max_generations: int
            Maximum number of generations
        """
        for _ in range(max_generations):
            self.generation += 1
            print(f"Generation: {self.generation}")
            self._evolve()
            if self._check_convergence():
                pass

    def _evolve(self):
        """
        Execute one step of the evolution process.
        """

        # Sort population by fitness
        fitnesses = 1.0 / torch.square(self.pool - self.target_gene).sum(dim=-1)
        indices = fitnesses.sort(descending=True).indices
        self.pool, fitnesses = self.pool[indices], fitnesses[indices]

        # Display every generation
        if self.queen is not None:
            print("queen:")
            print(
                f"{self.decode(self.queen)} ({self.queen_fitness.item():.3f})\n"
            )
        for gene, fitness in zip(self.pool, fitnesses):
            print(f"{self.decode(gene)} ({fitness.item():.3f})")

        # If one of the children has a better fitness than queen, that child becomes the new queen
        # and the queen replaces the worst bee in the population
        if self.queen is not None and self.queen_fitness < fitnesses[0]:
            self.pool = torch.cat((self.pool, self.queen[None, :]), dim=0)
            fitnesses = torch.cat((fitnesses, self.queen_fitness[None]), dim=0)
            self.queen = self.queen_fitness = None

        # Separate the queen bee from the rest of the population
        if self.queen is None:
            self.queen, self.pool = self.pool[0], self.pool[1:]
            self.queen_fitness, fitnesses = fitnesses[0], fitnesses[1:]

        # Deterministic tournament selection
        contender_ids = torch.randn(
            (self.pop_size - 1, self.pop_size - 1)
        ).argsort(dim=-1)[..., : self.num_tournament_participants]
        participants, tournaments = (
            self.pool[contender_ids],
            fitnesses[contender_ids],
        )
        top_winner = tournaments.topk(
            1, dim=-1, largest=True, sorted=False
        ).indices
        top_winner = top_winner.unsqueeze(-1).expand(-1, -1, self.gene_length)
        parents = participants.gather(1, top_winner).squeeze(1)

        # Cross over all chosen drones with the queen
        queen_parents = self.queen.unsqueeze(0).expand(
            self.pop_size - 1, self.gene_length
        )
        self.pool = torch.cat(
            (
                queen_parents[:, : self.gene_midpoint],
                parents[:, self.gene_midpoint :],
            ),
            dim=-1,
        )

        # Mutate genes in population
        mutate_mask = (
            torch.randn(self.pool.shape).argsort(dim=-1) < self.num_code_mutate
        )
        noise = torch.randint(0, 2, self.pool.shape) * 2 - 1
        mutated_pool = torch.where(mutate_mask, self.pool + noise, self.pool)

        strong_mutate_mask = (
            torch.randn(self.pool.shape).argsort(dim=-1)
            < self.strong_num_code_mutate
        )
        noise = torch.randint(0, 2, self.pool.shape) * 2 - 1
        strong_mutated_pool = torch.where(
            strong_mutate_mask, self.pool + noise, self.pool
        )

        strong_mutate_pool_mask = (
            torch.randn(self.pop_size - 1).argsort(dim=-1)
            < self.strong_mutate_pool_size
        )
        self.pool = torch.where(
            strong_mutate_pool_mask[:, None], strong_mutated_pool, mutated_pool
        )
        self.pool.clamp_(0, 255)

    def _check_convergence(self):
        """
        Check if any of the solutions has achieved the goal
        """
        fitnesses = 1.0 / torch.square(self.pool - self.target_gene).sum(dim=-1)
        return (fitnesses == float("inf")).any().item()


# # Usage:
# optimizer = QueenBeeGa()
# optimizer.run(max_generations=100)
