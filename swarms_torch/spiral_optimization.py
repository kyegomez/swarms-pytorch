import torch
from torch import nn


class SPO(nn.Module):
    """
    Spiral Optimization (SPO) Algorithm in PyTorch.

    Implements the SPO algorithm for optimization towards a target string.

    How does it work?
    ----------
    1. Initialize the search points randomly
    2. Initialize the center randomly
    3. Update the search points based on the spiral model
    4. Find the best search point and set as the new center
    5. Repeat step 3 to 4 until the maximum number of iterations is reached

    Usage
    -----
    from swarms_torch import SPO

    goaling = "Hello SPO"
    spo = SPO(goaling, m=100, k_max=1000)
    spo.optimize()

    print("Best Matched String:", spo.best_string())

    Future Features to implement
    --------
    1. Add a stopping criterion
    2. Add a callback function to track the progress
    3. Add a function to plot the search points
    4. Add a function to plot the best solution

    """

    def __init__(self, goal: str = None, m: int = 10, k_max: int = 1000):
        """
        Initialize the SPO class.

        Args:
        - goal: The target string.
        - m: Number of search points (strings).
        - k_max: Maximum number of iterations.
        """
        self.goal = torch.tensor(
            [ord(c) for c in goal], dtype=torch.float32
        )  # ASCII representation

        self.m = m
        self.k_max = k_max
        self.n_dim = len(goal)

        # Initializing the search points and center randomly
        # Note: 32-126 is the ASCII range for all printable characters
        self.points = torch.randint(
            32, 127, (self.m, self.n_dim), dtype=torch.float32
        )
        self.center = torch.randint(32, 127, (self.n_dim,), dtype=torch.float32)

    def _step_rate(self, k):
        """
        Define the step rate function.

        Args:
        - k: Current iteration.

        Returns: Step rate for the current iteration.
        """
        return 1 / (1 + k)

    def _update_points(self, k):
        """Update the search points based on the spiral model."""
        r = self._step_rate(k)
        R = torch.eye(self.n_dim)  # Identity for simplicity in n-dimensions
        for i in range(self.m):
            self.points[i] = self.center + r * torch.mv(
                R, (self.points[i] - self.center)
            )

    def _update_center(self):
        """Find the best search point and set as the new center."""
        fitnesses = torch.norm(self.points - self.goal, dim=1)
        best_idx = torch.argmin(fitnesses)
        self.center = self.points[best_idx]

    def optimize(self):
        """Run the optimization loop."""
        for k in range(self.k_max):
            self._update_points(k)
            self._update_center()
            if (
                torch.norm(self.center - self.goal) < 1e-5
            ):  # Example convergence condition
                break

    def best_string(self):
        """Convert the best found point to its string representation"""
        return "".join([chr(int(c)) for c in self.center.round()])


# # Example Usage
# goal = "Attention is all you need"
# optimizer = SPO(goal)
# optimizer.optimize()
# print(f"Optimized String: {optimizer.best_string()}")
