import torch


class ParticleSwarmOptimization:
    """
    Particle Swarm Optimization
    Overview: https://en.wikipedia.org/wiki/Particle_swarm_optimization

    How does it work?
    1. Initialize particles with random positions and velocities
    2. For each particle, compute the fitness value
    3. Update the personal best and global best
    4. Update the velocity and position of each particle
    5. Repeat step 2 to 4 until the maximum number of iterations is reached



    Parameters
    ----------
    goal: str
        The goal string to be optimized
    n_particles: int
        Number of particles
    inertia: float
        Inertia weight
    personal_best_weight: float
        Personal best weight
    global_best_weight: float
        Global best weight

    Usage
    -----
    pso = ParticleSwarmOptimization(goal="Attention is all you need", n_particles=100)
    pso.optimize(iterations=1000)


    Future Improvements
    -------------------
    1. Add a stopping criterion
    2. Add a callback function to track the progress
    3. Add a function to plot the fitness value
    4. Add a function to plot the particles
    5. Add a function to plot the velocity
    6. Add a function to plot the position
    7. Add a function to plot the personal best
    8. Add a function to plot the global best
    9. Add a function to plot the personal best weight
    10. Add a function to plot the global best weight



    """

    def __init__(
        self,
        goal: str = None,
        n_particles: int = 100,
        inertia: float = 0.5,
        personal_best_weight: float = 1.5,
        global_best_weight: float = 1.5,
        dim: int = 1,
    ):
        self.goal = torch.tensor([ord(c) for c in goal])
        self.n_particles = n_particles
        self.inertia = inertia
        self.personal_best_weight = personal_best_weight
        self.global_best_weight = global_best_weight

        self.particles = torch.randint(0, 255, (n_particles, len(goal)))
        self.velocities = torch.zeros((n_particles, len(goal)))

        self.personal_best = self.particles.clone()
        self.global_best = self.particles[0].clone()

    def compute_fitness(
        self,
        particle,
    ):
        return 1.0 / (1.0 + torch.norm((particle - self.goal).float()))

    def update(
        self,
    ):
        """Update the particles"""
        for i in range(self.n_particles):
            fitness = self.compute_fitness(
                self.particles[i],
            )

            personal_best_fitness = self.compute_fitness(
                self.personal_best[i],
            )

            if fitness > personal_best_fitness:
                self.personal_best[i] = self.particles[i]

            global_best_fitness = self.compute_fitness(self.global_best)

            if fitness > global_best_fitness:
                self.global_best = self.particles[i]

            # update velocity
            personal_attraction = (
                self.personal_best_weight
                * torch.rand(self.goal.size())
                * (self.personal_best[i] - self.particles[i])
            )

            global_attraction = (
                self.global_best_weight
                * torch.rand(self.goal.size())
                * (self.global_best - self.particles[i])
            )

            self.velocities[i] = (
                self.inertia * self.velocities[i]
                + personal_attraction
                + global_attraction
            )

            # Update position
            self.particles[i] += self.velocities[i].int()
            self.particles[i].clamp_(0, 255)

    def optimize(
        self,
        iterations: int = 1000,
    ):
        """Optimize the goal string"""
        for _ in range(iterations):
            self.update()
            best_particle = self.global_best
            print(
                "Best Particle: ", "".join([chr(int(i)) for i in best_particle])
            )
