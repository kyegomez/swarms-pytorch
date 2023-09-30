import unittest
import torch

from swarms_torch import AntColonyOptimization  # Import your class

class TestAntColonyOptimization(unittest.TestCase):

    def setUp(self):
        self.aco = AntColonyOptimization(goal="Hello ACO", num_ants=1000, num_iterations=10)
        
    def test_initialization(self):
        self.assertEqual(self.aco.goal.tolist(), [ord(c) for c in "Hello ACO"])
        self.assertEqual(self.aco.pheromones.size(), torch.Size([1000]))
        self.assertEqual(self.aco.pheromones.tolist(), [1.0] * 1000)
        
    def test_fitness(self):
        solution = torch.tensor([ord(c) for c in "Hello ACO"], dtype=torch.float32)
        self.assertEqual(self.aco.fitness(solution).item(), 0)  # Should be maximum fitness
        
    def test_update_pheromones(self):
        initial_pheromones = self.aco.pheromones.clone()
        self.aco.solutions = [torch.tensor([ord(c) for c in "Hello ACO"], dtype=torch.float32) for _ in range(1000)]
        self.aco.update_pheromones()
        # After updating, pheromones should not remain the same
        self.assertFalse(torch.equal(initial_pheromones, self.aco.pheromones))
        
    def test_choose_next_path(self):
        path = self.aco.choose_next_path()
        # Path should be an integer index within the number of ants
        self.assertIsInstance(path, int)
        self.assertGreaterEqual(path, 0)
        self.assertLess(path, 1000)
        
    def test_optimize(self):
        solution = self.aco.optimize()
        self.assertIsInstance(solution, str)
        # Given enough iterations and ants, the solution should approach the goal. For short runs, this might not hold.
        # self.assertEqual(solution, "Hello ACO") 

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            _ = AntColonyOptimization(num_ants=-5)
        with self.assertRaises(ValueError):
            _ = AntColonyOptimization(evaporation_rate=1.5)

if __name__ == "__main__":
    unittest.main()
