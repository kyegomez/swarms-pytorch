import torch
from torch.utils.data import DataLoader
from swarms_torch.pso.transformer_pso import (
    SimpleTransformer,
    TransformerParticleSwarmOptimization,
)


def test_simpletransformer_initialization():
    simpletransformer = SimpleTransformer(
        input_dim=10, d_model=512, nhead=8, num_layers=1, output_dim=2
    )
    assert isinstance(simpletransformer, SimpleTransformer)


def test_simpletransformer_forward():
    simpletransformer = SimpleTransformer(
        input_dim=10, d_model=512, nhead=8, num_layers=1, output_dim=2
    )
    x = torch.randint(0, 10, (10, 32))
    output = simpletransformer(x)
    assert output.shape == torch.Size([32, 2])


def test_TransformerParticleSwarmOptimization_initialization():
    model_constructor = SimpleTransformer
    model_args = (10, 512, 8, 1, 2)
    device = "cpu"
    criterion = torch.nn.CrossEntropyLoss()
    data_loader = DataLoader(
        [(torch.randint(0, 10, (10,)), torch.tensor(1)) for _ in range(100)],
        batch_size=32,
    )
    pso = TransformerParticleSwarmOptimization(
        model_constructor, model_args, device, criterion, data_loader
    )
    assert isinstance(pso, TransformerParticleSwarmOptimization)
    assert len(pso.particles) == 10
    assert len(pso.velocities) == 10
    assert len(pso.personal_best) == 10


def test_TransformerParticleSwarmOptimization_compute_fitness():
    model_constructor = SimpleTransformer
    model_args = (10, 512, 8, 1, 2)
    device = "cpu"
    criterion = torch.nn.CrossEntropyLoss()
    data_loader = DataLoader(
        [(torch.randint(0, 10, (10,)), torch.tensor(1)) for _ in range(100)],
        batch_size=32,
    )
    pso = TransformerParticleSwarmOptimization(
        model_constructor, model_args, device, criterion, data_loader
    )
    fitness = pso.compute_fitness(pso.particles[0].state_dict())
    assert isinstance(fitness, float)


def test_TransformerParticleSwarmOptimization_update():
    model_constructor = SimpleTransformer
    model_args = (10, 512, 8, 1, 2)
    device = "cpu"
    criterion = torch.nn.CrossEntropyLoss()
    data_loader = DataLoader(
        [(torch.randint(0, 10, (10,)), torch.tensor(1)) for _ in range(100)],
        batch_size=32,
    )
    pso = TransformerParticleSwarmOptimization(
        model_constructor, model_args, device, criterion, data_loader
    )
    pso.update()
    assert len(pso.particles) == 10
    assert len(pso.velocities) == 10
    assert len(pso.personal_best) == 10


def test_TransformerParticleSwarmOptimization_optimize():
    model_constructor = SimpleTransformer
    model_args = (10, 512, 8, 1, 2)
    device = "cpu"
    criterion = torch.nn.CrossEntropyLoss()
    data_loader = DataLoader(
        [(torch.randint(0, 10, (10,)), torch.tensor(1)) for _ in range(100)],
        batch_size=32,
    )
    pso = TransformerParticleSwarmOptimization(
        model_constructor, model_args, device, criterion, data_loader
    )
    pso.optimize(iterations=10)
    assert len(pso.particles) == 10
    assert len(pso.velocities) == 10
    assert len(pso.personal_best) == 10


def test_TransformerParticleSwarmOptimization_get_best_model():
    model_constructor = SimpleTransformer
    model_args = (10, 512, 8, 1, 2)
    device = "cpu"
    criterion = torch.nn.CrossEntropyLoss()
    data_loader = DataLoader(
        [(torch.randint(0, 10, (10,)), torch.tensor(1)) for _ in range(100)],
        batch_size=32,
    )
    pso = TransformerParticleSwarmOptimization(
        model_constructor, model_args, device, criterion, data_loader
    )
    pso.optimize(iterations=10)
    best_model = pso.get_best_model()
    assert isinstance(best_model, SimpleTransformer)
