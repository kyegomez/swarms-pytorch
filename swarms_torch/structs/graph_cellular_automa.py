import torch
import torch.nn as nn


class GraphCellularAutomata(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphCellularAutomata, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class ReplicationModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ReplicationModel, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # for binary classification
        )

    def forward(self, x):
        return self.mlp(x)


class WeightUpdateModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(WeightUpdateModel, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.mlp(x)


class NDP(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(NDP, self).__init__()

        self.gc_automata = GraphCellularAutomata(
            embedding_dim, hidden_dim, embedding_dim
        )
        self.replication_model = ReplicationModel(embedding_dim, hidden_dim)
        self.weight_update_model = WeightUpdateModel(
            2 * embedding_dim, hidden_dim
        )

    def forward(self, node_embeddings, adjacency_matrix):
        # Update node embeddings using Graph Cellular Automata
        updated_embeddings = self.gc_automata(node_embeddings)

        # Check which nodes need to replicate
        replication_decisions = self.replication_model(updated_embeddings)

        # Weight update (assuming weighted network)
        num_nodes = node_embeddings.shape[0]
        edge_weights = torch.zeros((num_nodes, num_nodes))

        for i in range(num_nodes):
            for j in range(num_nodes):
                combined_embedding = torch.cat(
                    (updated_embeddings[i], updated_embeddings[j])
                )

                edge_weights[i, j] = self.weight_update_model(
                    combined_embedding
                )

        return updated_embeddings, replication_decisions, edge_weights


# # Usage examples
# embedding_dim = 16
# hidden_dim = 32
# node_embeddings = torch.rand((10, embedding_dim))  # For 10 nodes
# adjacency_matrix = torch.rand((10, 10))  # Dummy adjacency matrix for 10
# nodes

# model = NDP(embedding_dim, hidden_dim)
# updated_embeddings, replication_decisions, edge_weights = model(
#     node_embeddings, adjacency_matrix
# )

# print(updated_embeddings.shape)
# print(replication_decisions.shape)
# print(edge_weights.shape)
