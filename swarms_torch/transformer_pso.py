import torch
import torch.nn as nn
from copy import deepcopy


class Particle(nn.Module):
    """
    Simple Transformer model for classification.

    Parameters
    ----------
    input_dim : int
        The number of expected features in the input (required).
    d_model : int
        The number of expected features in the encoder/decoder inputs (required).
    nhead : int
        The number of heads in the multiheadattention models (required).
    num_layers : int
        The number of sub-encoder-layers in the encoder (required).
    output_dim : int
        The number of classes to predict (required).

    Usage:
    >>> model = SimpleTransformer(1000, 512, 8, 6, 10)
    >>> model(x)


    """

    def __init__(self, input_dim, d_model, nhead, num_layers, output_dim):
        super(Particle, self).__init__()
        self.embedding = nn.Embedding(input_dim, d_model)
        self.transformer = nn.Transformer(
            d_model, nhead, num_layers, num_layers
        )
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        """
        Forward pass through the model.

        """
        x = self.embedding(x)
        x = self.transformer(x, x)
        return self.fc(x[-1])


class TransformerParticleSwarmOptimization(nn.Module):
    """
    Transformer Particle Swarm Optimization.

    Parameters
    ----------
    model_constructor : function
        Function to create a new model instance.
    model_args : tuple
        Arguments for the model constructor.
    device : str
        'cuda' or 'cpu'.
    criterion : nn.Module
        Loss function.
    data_loader : torch.utils.data.DataLoader
        Data loader.
    n_particles : int
        Number of particles.
    inertia : float
        Inertia weight.
    personal_best_weight : float
        Personal best weight.
    global_best_weight : float
        Global best weight.

    Usage:
    >>> pso = TransformerParticleSwarmOptimization(
    ...     SimpleTransformer,
    ...     (1000, 512, 8, 6, 10),
    ...     device="cuda",
    ...     criterion=nn.CrossEntropyLoss(),
    ...     data_loader=your_dataloader
    ... )

    """

    def __init__(
        self,
        model_constructor,  # Function to create a new model instance
        model_args,  # Arguments for the model constructor
        device,  # 'cuda' or 'cpu'
        criterion,
        data_loader,
        n_particles=10,
        inertia=0.5,
        personal_best_weight=1.5,
        global_best_weight=1.5,
    ):
        super(TransformerParticleSwarmOptimization, self).__init__()
        self.model_constructor = model_constructor
        self.model_args = model_args
        self.criterion = criterion
        self.data_loader = data_loader
        self.device = device

        self.n_particles = n_particles
        self.inertia = inertia
        self.personal_best_weight = personal_best_weight
        self.global_best_weight = global_best_weight

        # Representing particles using model parameters
        param_size = sum(
            p.numel() for p in model_constructor(*model_args).parameters()
        )
        self.particles = [
            self.model_constructor(*model_args).to(device)
            for _ in range(n_particles)
        ]
        self.velocities = [
            torch.zeros((param_size,)).to(device) for _ in range(n_particles)
        ]
        self.personal_best = [deepcopy(p.state_dict()) for p in self.particles]
        self.global_best = deepcopy(self.particles[0].state_dict())

    def compute_fitness(self, model_state):
        """
        Compute the fitness of a model.
        """
        model = self.model_constructor(*self.model_args).to(self.device)
        model.load_state_dict(model_state)
        model.eval()

        total_loss = 0.0
        with torch.no_grad():
            for inputs, targets in self.data_loader:
                outputs = model(inputs.to(self.device))
                loss = self.criterion(outputs, targets.to(self.device))
                total_loss += loss.item()
        return 1.0 / (1.0 + total_loss)

    def update(self):
        """
        Update particles.
        """
        # Update particles
        for idx, particle in enumerate(self.particles):
            fitness = self.compute_fitness(particle.state_dict())

            # Update personal best
            if fitness > self.compute_fitness(self.personal_best[idx]):
                self.personal_best[idx] = deepcopy(particle.state_dict())

            # Update global best
            if fitness > self.compute_fitness(self.global_best):
                self.global_best = deepcopy(particle.state_dict())

            # Update velocities and positions
            for name, param in particle.named_parameters():
                delta = self.personal_best_weight * torch.rand_like(param) * (
                    self.personal_best[idx][name].to(self.device) - param.data
                ) + self.global_best_weight * torch.rand_like(param) * (
                    self.global_best[name].to(self.device) - param.data
                )
                self.velocities[idx] += (
                    self.inertia * self.velocities[idx] + delta
                )
                param.data += self.velocities[idx]

    def optimize(self, iterations=1000):
        """Optimize the model."""
        for _ in range(iterations):
            self.update()
            best_particle_score = self.compute_fitness(self.global_best)
            print(
                f"Iteration {_ + 1}/{iterations} - Best Particle Fitness:"
                f" {best_particle_score}"
            )

    def get_best_model(self):
        """Get the best model."""
        best_model = self.model_constructor(*self.model_args).to(self.device)
        best_model.load_state_dict(self.global_best)
        return best_model


# # Define model and optimization parameters
# input_dim = 1000
# d_model = 512
# nhead = 8
# num_layers = 3
# output_dim = 10

# batch_size = 32
# sequence_length = 50

# # Instantiate the optimizer
# pso = ParticleSwarmOptimization(
#     SimpleTransformer,
#     (input_dim, d_model, nhead, num_layers, output_dim),
#     device="cuda",  # or 'cpu'
#     criterion=nn.CrossEntropyLoss(),
#     # data_loader=your_dataloader  # replace with your dataloader
# )

# # Run optimization
# pso.optimize(iterations=100)

# # Get the best model
# best_model = pso.get_best_model()

# # Generate a random input tensor
# x = torch.randint(0, input_dim, (batch_size, sequence_length)).to(
#     "cuda"
# )  # ensure it's on the same device as your model

# # Pass the tensor through the model
# output = best_model(x)
# print(output.shape)  # should be [batch_size, output_dim]
