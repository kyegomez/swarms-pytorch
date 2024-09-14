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
