import unittest
import torch

from swarms_torch import ParticleSwarmOptimization  # Import your class here


class TestParticleSwarmOptimization(unittest.TestCase):

    def setUp(self):
        self.pso = ParticleSwarmOptimization(goal="Hello", n_particles=10)

    def test_initialization(self):
        self.assertEqual(self.pso.goal.tolist(), [ord(c) for c in "Hello"])
        self.assertEqual(self.pso.particles.size(), (10, 5))
        self.assertEqual(self.pso.velocities.size(), (10, 5))

    def test_compute_fitness(self):
        particle = torch.tensor([ord(c) for c in "Hello"])
        fitness = self.pso.compute_fitness(particle)
        self.assertEqual(fitness.item(), 1.0)

    def test_update(self):
        initial_particle = self.pso.particles.clone()
        self.pso.update()
        # After updating, particles should not remain the same (in most cases)
        self.assertFalse(torch.equal(initial_particle, self.pso.particles))

    def test_optimize(self):
        initial_best_particle = self.pso.global_best.clone()
        self.pso.optimize(iterations=10)
        # After optimization, global best should be closer to the goal
        initial_distance = torch.norm(
            (initial_best_particle - self.pso.goal).float()).item()
        final_distance = torch.norm(
            (self.pso.global_best - self.pso.goal).float()).item()
        self.assertLess(final_distance, initial_distance)


if __name__ == "__main__":
    unittest.main()
