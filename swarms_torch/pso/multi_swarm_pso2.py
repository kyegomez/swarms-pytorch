import torch


class Particle:
    def __init__(self, dim, minx, maxx):
        self.position = torch.rand(dim) * (maxx - minx) + minx
        self.velocity = torch.rand(dim) * (maxx - minx) + minx
        self.best_position = self.position.clone()
        self.best_score = float("inf")

    def update_velocity(self, global_best, w=0.7, c1=1.5, c2=1.5):
        r1 = torch.rand(self.position.size())
        r2 = torch.rand(self.position.size())
        self.velocity = (
            w * self.velocity
            + c1 * r1 * (self.best_position - self.position)
            + c2 * r2 * (global_best - self.position)
        )

    def update_position(self, minx, maxx):
        self.position += self.velocity
        self.position = torch.clamp(self.position, minx, maxx)


class Swarm:
    def __init__(self, num_particles, dim, minx, maxx):
        self.particles = [
            Particle(dim, minx, maxx) for _ in range(num_particles)
        ]
        self.global_best = None
        self.global_best_score = float("inf")

    def update_global_best(self):
        for particle in self.particles:
            if particle.best_score < self.global_best_score:
                self.global_best = particle.best_position.clone()
                self.global_best_score = particle.best_score

    def move_particles(self, minx, maxx):
        for particle in self.particles:
            particle.update_velocity(self.global_best)
            particle.update_position(minx, maxx)


class MultiSwarm:
    def __init__(self, num_swarms, num_particles, dim, minx, maxx):
        self.swarms = [
            Swarm(num_particles, dim, minx, maxx) for _ in range(num_swarms)
        ]
        self.minx = minx
        self.maxx = maxx

    def optimize(self, func, max_iter):
        for _ in range(max_iter):
            for swarm in self.swarms:
                swarm.update_global_best()
                swarm.move_particles(self.minx, self.maxx)
        best_swarm = min(self.swarms, key=lambda s: s.global_best_score)
        return best_swarm.global_best, best_swarm.global_best_score


def rosenbrock(x, a=1, b=100):
    return (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2


# num_swarms = 5
# num_particles = 20
# dim = 2
# minx = -5
# maxx = 5
# max_iter = 100

# multi_swarm = MultiSwarm(num_swarms, num_particles, dim, minx, maxx)

# best_position, best_score = multi_swarm.optimize(rosenbrock, max_iter)

# print(f"Best position: {best_position}")
# print(f"Best score: {best_score}")
