# import torch
# import torch.nn as nn
# from copy import deepcopy
# from swarms_torch.transformer_pso import Particle, TransformerParticleSwarmOptimization

# class MultiSwarm(nn.Module):
#     def __init__(
#         self,
#         num_swarms,
#         *args,
#         **kwargs
#     ):
#         #create multiple instances of the transformerparticle swarm optimization
#         self.swarms = [TransformerParticleSwarmOptimization(*args, **kwargs) for _ in range(num_swarms)]
#         self.num_swarms = num_swarms

#     def optimize(self, iterations):
#         for _ in range(iterations):
#             #update each swarm
#             for swarm in self.swarms:
#                 swarm.update()

#             #apply diversification strategy
#             self.diversification_method()

#     def diversification_strategy(self):
#         for i in range(self.num_swarms):
#             for j in range(i + 1, self.num_swarms):
#                 if self.is_collided(self.swarms[i].global_best, self.swarms[j].global_best):
#                     #handle collision by launching a new swarm or re init one of the swarms
#                     self.handle_collision(i, j)

#     def is_collided(self, global_best_1, global_best_2):
#         #Check if difference between the global bests or 2 swarms is below a threshold
#         diff = sum((global_best_1[key] - global_best_2[key]).abs().sum() for key in global_best_1.keys())
#         COLLISION_THRESHOLD = 0.1

#         return diff < COLLISION_THRESHOLD

#     def handle_collision(self, idx1, idx2):
#         #for simplicity re init 2nd swarm
#         self.swarms[idx2] = TransformerParticleSwarmOptimization(*self.swarms[idx2].model_args, **self.swarms[idx2].kwargs)

# import torch
# from torch.utils.data import DataLoader, TensorDataset

# # Generate random data
# num_samples = 1000
# input_dim = 50  # Length of input sequence
# num_classes = 2

# inputs = torch.randint(0, 1000, (num_samples, input_dim))
# targets = torch.randint(0, num_classes, (num_samples,))

# dataset = TensorDataset(inputs, targets)
# data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# # Define hyperparameters and model arguments
# model_args = (1000, 512, 8, 6, 2)  # (input_dim, d_model, nhead, num_layers, output_dim)
# optimizer_args = {
#     "model_constructor": Particle,
#     "model_args": model_args,
#     "device": "cpu",
#     "criterion": torch.nn.CrossEntropyLoss(),
#     "data_loader": data_loader
# }

# # Create MultiSwarmOptimizer
# num_swarms = 3
# mso = MultiSwarm(num_swarms, **optimizer_args)

# # Optimize
# mso.optimize(iterations=10)

# # Get the best model from the best-performing swarm
# best_swarm = max(mso.swarms, key=lambda s: s.compute_fitness(s.global_best))
# best_model = best_swarm.get_best_model()

import torch
import torch.nn as nn
from copy import deepcopy


class Particle(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Particle, self).__init__()
        self.transformer = nn.Transformer(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.transformer(x)
        x = self.fc(x)
        return x


class MultiSwarmOptimizer:
    def __init__(
        self,
        particle,
        num_particles,
        num_subswarms,
        fitness_func,
        bounds,
        num_epochs,
    ):
        self.particle = particle
        self.num_particles = num_particles
        self.num_subswarms = num_subswarms
        self.fitness_func = fitness_func
        self.bounds = bounds
        self.num_epochs = num_epochs

        self.subswarms = []
        for _ in range(num_subswarms):
            self.subswarms.append(
                [deepcopy(particle) for _ in range(num_particles)]
            )

    def optimize(self):
        for epoch in range(self.num_epochs):
            for subswarm in self.subswarms:
                for particle in subswarm:
                    fitness = self.fitness_func(particle)
                    if fitness > particle.best_fitness:
                        particle.best_fitness = fitness
                        particle.best_position = deepcopy(particle.position)

                best_particle = max(subswarm, key=lambda p: p.best_fitness)
                for particle in subswarm:
                    particle.velocity = (
                        particle.velocity
                        + 0.5 * (particle.best_position - particle.position)
                        + 0.5
                        * (best_particle.best_position - particle.position)
                    )
                    particle.position = particle.position + particle.velocity
                    particle.position = torch.clamp(
                        particle.position, *self.bounds
                    )

            best_subswarm = max(
                self.subswarms, key=lambda s: max(p.best_fitness for p in s)
            )
            best_particle = max(best_subswarm, key=lambda p: p.best_fitness)
            print(
                f"Epoch {epoch+1}/{self.num_epochs}, Best Fitness:"
                f" {best_particle.best_fitness}"
            )

        best_subswarm = max(
            self.subswarms, key=lambda s: max(p.best_fitness for p in s)
        )
        best_particle = max(best_subswarm, key=lambda p: p.best_fitness)
        return best_particle

    def get_best_model(self):
        return self.particle


# import torch
# import torch.nn as nn
# from random import random


# # Define the fitness function
# def fitness_func(particle):
#     # This is a dummy fitness function. Replace it with your own.
#     return random()


# # Define the bounds for the particle positions
# bounds = (-1.0, 1.0)

# # Define the number of particles, sub-swarms, and epochs
# num_particles = 10
# num_subswarms = 5
# num_epochs = 100

# # Define the dimensions for the transformer model
# input_dim = 10
# hidden_dim = 20
# output_dim = 2

# # Create a particle (transformer model)
# particle = Particle(input_dim, hidden_dim, output_dim)

# # Create the multi-swarm optimizer
# optimizer = MultiSwarmOptimizer(
#     particle, num_particles, num_subswarms, fitness_func, bounds, num_epochs
# )

# # Run the optimization
# best_particle = optimizer.optimize()

# # The best_particle is the model with the highest fitness score
# print(best_particle)
