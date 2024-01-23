import torch
from torch import nn, Tensor
from dataclasses import dataclass
from zeta.nn import FeedForward


def estimate_neuron_score(act):
    reduce_axes = list(range(act.dim() - 1))
    score = torch.mean(torch.abs(act), dim=reduce_axes)
    return score


@dataclass
class SwarmNeighborhoodEncoder(nn.Module):
    """
    A class representing the encoder for swarm neighborhood observations.

    Args:
        self_obs_dim (int): The dimension of the self-observation.
        neighbor_obs_dim (int): The dimension of the neighbor observations.
        neighbor_hidden_size (int): The hidden size of the neighbor encoder.
        num_use_neighbor_obs (int): The number of neighbor observations to use.
    """

    self_obs_dim: int
    neighbor_obs_dim: int
    neighbor_hidden_size: int
    num_use_neighbor_obs: int


@dataclass
class SwarmNeighborhoodEncoderDeepsets(SwarmNeighborhoodEncoder):
    neighbor_obs_dim: int
    neighbor_hidden_size: int
    self_obs_dim: int
    num_use_neighbor_obs: int
    mult: int = 4
    args: dict = None

    def __post_init__(self):
        self.ffn = FeedForward(
            self.neighbor_obs_dim,
            self.neighbor_hidden_size,
            self.mult,
            self.args,
        )

    def forward(
        self,
        self_obs: Tensor,
        obs: Tensor,
        all_neighbor_obs_size: int,
        batch: int,
    ) -> Tensor:
        """
        Forward pass of the SwarmNeighborhoodEncoder.

        Args:
            self_obs (Tensor): Self observation tensor.
            obs (Tensor): Observation tensor.
            all_neighbor_obs_size (int): Size of all neighbor observations.
            batch (int): Batch size.

        Returns:
            Tensor: Mean embedding tensor.
        """
        obs_neighbors = obs[
            :, self.self_obs_dim : self.self_obs_dim + all_neighbor_obs_size
        ]
        obs_neighbors = obs_neighbors.reshape(-1, self.neighbor_obs_dim)
        neighbor_embeds = self.embedding_mlp(obs_neighbors)
        neighbor_embeds = neighbor_embeds.reshape(
            batch, -1, self.neighbor_hidden_size
        )
        mean_embed = torch.mean(neighbor_embeds, dim=1)
        return mean_embed


@dataclass
class SwarmNeighborhoodEncoderAttention(SwarmNeighborhoodEncoder):
    """
    A class that represents a swarm neighborhood encoder with attention mechanism.

    Args:
        neighbor_obs_dim (int): The dimension of the neighbor observations.
        neighbor_hidden_size (int): The hidden size of the neighbor encoder.
        self_obs_dim (int): The dimension of the self observations.
        num_use_neighbor_obs (int): The number of neighbor observations to use.
        mult (int, optional): The multiplier for the hidden size in the MLPs. Defaults to 4.
        args (dict, optional): Additional arguments for the MLPs. Defaults to None.
    """

    neighbor_obs_dim: int
    neighbor_hidden_size: int
    self_obs_dim: int
    num_use_neighbor_obs: int
    mult: int = 4
    args: dict = None

    def __post_init__(self):
        self.embedding_mlp = FeedForward(
            self.self_obs_dim + self.neighbor_obs_dim,
            self.neighbor_hidden_size,
            self.mult,
            self.args,
        )

        self.neighbor_value_mlp = FeedForward(
            self.neighbor_hidden_size,
            self.neighbor_hidden_size,
            self.mult,
            self.args,
        )

        # Outputs scalar score alpha_i for each neighbor
        self.attention_mlp = FeedForward(
            self.neighbor_hidden_size * 2,
            self.neighbor_hidden_size,
            self.mult,
            self.args,
        )

    def forward(
        self,
        self_obs: Tensor,
        obs: Tensor,
        all_neighbor_obs_size: int,
        batch_size: int,
    ) -> Tensor:
        obs_neighbors = obs[
            :, self.self_obs_dim : self.self_obs_dim + all_neighbor_obs_size
        ]
        obs_neighbors = obs_neighbors.reshape(-1, self.neighbor_obs_dim)

        # Concat self observation with neighbor observation
        self_obs_repeat = self_obs.repeat(self.num_use_neighbor_obs, 1)
        mlp_input = torch.cat((self_obs_repeat, obs_neighbors), dim=1)
        neighbor_embeddings = self.embedding_mlp(mlp_input)
        neighbor_values = self.neighbor_value_mlp(neighbor_embeddings)
        neighbor_embeddings_mean_input = neighbor_embeddings.reshape(
            batch_size, -1, self.neighbor_hidden_size
        )
        neighbor_embeddings_mean = torch.mean(
            neighbor_embeddings_mean_input, dim=1
        )
        neighbor_embeddings_mean_repeat = neighbor_embeddings_mean.repeat(
            self.num_use_neighbor_obs, 1
        )
        attention_mlp_input = torch.cat(
            (neighbor_embeddings, neighbor_embeddings_mean_repeat), dim=1
        )
        attention_weights = self.attention_mlp(attention_mlp_input).view(
            batch_size, -1
        )
        attention_weights_softmax = torch.nn.functional.softmax(
            attention_weights, dim=1
        )
        attention_weights_softmax = attention_weights_softmax.view(-1, 1)

        final_neighbor_embedding = attention_weights_softmax * neighbor_values
        final_neighbor_embedding = final_neighbor_embedding.view(
            batch_size, -1, self.neighbor_hidden_size
        )
        final_neighbor_embedding = torch.sum(final_neighbor_embedding, dim=1)

        return final_neighbor_embedding


@dataclass
class SwarmNeighborEncoderMLP(SwarmNeighborhoodEncoder):
    """
    A class representing a multi-layer perceptron (MLP) encoder for swarm neighbor observations.

    Args:
        neighbor_obs_dim (int): The dimension of each neighbor observation.
        neighbor_hidden_size (int): The size of the hidden layer in the MLP.
        self_obs_dim (int): The dimension of the self observation.
        num_use_neighbor_obs (int): The number of neighbor observations to use.
        mult (int, optional): The multiplier for the hidden layer size. Defaults to 4.
        args (dict, optional): Additional arguments for the MLP. Defaults to None.
    """

    neighbor_obs_dim: int
    neighbor_hidden_size: int
    self_obs_dim: int
    num_use_neighbor_obs: int
    mult: int = 4
    args: dict = None

    def __post_init__(self):
        """
        Initialize the MLP encoder.

        This method creates an MLP with the specified dimensions and parameters.
        """
        self.neighbor_mlp = FeedForward(
            self.neighbor_obs_dim * self.num_use_neighbor_obs,
            self.neighbor_hidden_size,
            self.mult,
            self.args,
        )

    def forward(
        self,
        self_obs: Tensor,
        obs: Tensor,
        all_neighbor_obs_size: int,
        batch_size: int,
    ) -> Tensor:
        """
        Perform a forward pass through the MLP encoder.

        Args:
            self_obs (Tensor): The self observation tensor.
            obs (Tensor): The observation tensor.
            all_neighbor_obs_size (int): The size of all neighbor observations.
            batch_size (int): The size of the batch.

        Returns:
            Tensor: The final neighborhood embedding tensor.
        """
        obs_neighbors = obs[
            :, self.self_obs_dim : self.self_obs_dim + all_neighbor_obs_size
        ]
        final_neighborhood_embedding = self.neighbor_mlp(obs_neighbors)
        return final_neighborhood_embedding
