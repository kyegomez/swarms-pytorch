from swarms_torch import ParticleSwarmOptimization

# test
pso = ParticleSwarmOptimization(goal="Attention is all you need", n_particles=100)
pso.optimize(iterations=1000)
