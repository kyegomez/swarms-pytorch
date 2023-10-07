import unittest
import torch
from swarms_torch.queen_bee import QueenBeeGa  # Import the class


class TestQueenBeeGa(unittest.TestCase):
    def setUp(self):
        self.optimizer = QueenBeeGa(goal="Hello QBGA", pop_size=50)

    def test_initialization(self):
        self.assertEqual(self.optimizer.goal, "Hello QBGA")
        self.assertEqual(self.optimizer.gene_length, len("Hello QBGA"))
        self.assertIsNone(self.optimizer.queen)
        self.assertIsNone(self.optimizer.queen_fitness)

    def test_encode_decode(self):
        encoded = QueenBeeGa.encode("Hello")
        decoded = QueenBeeGa.decode(encoded)
        self.assertEqual(decoded, "Hello")

    def test_evolution(self):
        initial_population = self.optimizer.pool.clone()
        self.optimizer._evolve()
        self.assertFalse(torch.equal(initial_population, self.optimizer.pool))

    def test_run(self):
        initial_population = self.optimizer.pool.clone()
        self.optimizer.run(max_generations=10)
        self.assertNotEqual(
            QueenBeeGa.decode(self.optimizer.queen),
            QueenBeeGa.decode(initial_population[0]),
        )

    def test_check_convergence(self):
        self.optimizer.pool = torch.stack([self.optimizer.target_gene] * 50)
        self.assertTrue(self.optimizer._check_convergence())

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            _ = QueenBeeGa(mutation_prob=1.5)
        with self.assertRaises(ValueError):
            _ = QueenBeeGa(strong_mutation_rate=-0.5)


if __name__ == "__main__":
    unittest.main()
