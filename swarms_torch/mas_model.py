from typing import List, Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger

# Set up logger
logger.add("masi_log.log", rotation="500 MB")

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


# Agent Base Class
class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    # def backward(self, loss: torch.Tensor) -> None:
    #     loss.backward()

    def update_parameters(
        self, shared_gradients: Dict[str, torch.Tensor]
    ) -> None:
        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.grad is not None:
                    param.grad = shared_gradients[name]
        self.optimizer.step()
        self.optimizer.zero_grad()


# MLP Agent
class MLPAgent(Agent):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(MLPAgent, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),  # Add this line to flatten the input
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        self.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logger.debug(f"MLPAgent input shape: {x.shape}")
        output = self.model(x)
        logger.debug(f"MLPAgent output shape: {output.shape}")
        return output


# CNN Agent
class CNNAgent(Agent):
    def __init__(self, input_channels: int, num_classes: int):
        super(CNNAgent, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 28 * 28, num_classes),
        )
        self.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logger.debug(f"CNNAgent input shape: {x.shape}")
        output = self.model(x)
        logger.debug(f"CNNAgent output shape: {output.shape}")
        return output


# LSTM Agent
class LSTMAgent(Agent):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(LSTMAgent, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logger.debug(f"LSTMAgent input shape: {x.shape}")
        # Reshape input: (batch, channels, height, width) -> (batch, height, width * channels)
        x = x.view(x.size(0), x.size(2), -1)
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        logger.debug(f"LSTMAgent output shape: {output.shape}")
        return output


# Transformer Agent
class TransformerAgent(Agent):
    def __init__(
        self, input_size: int, num_heads: int, num_layers: int, output_size: int
    ):
        super(TransformerAgent, self).__init__()
        self.embedding = nn.Linear(input_size, 128)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.fc = nn.Linear(128, output_size)
        self.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logger.debug(f"TransformerAgent input shape: {x.shape}")
        # Reshape input: (batch, channels, height, width) -> (batch, height, width * channels)
        x = x.view(x.size(0), x.size(2), -1)
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # (sequence_length, batch_size, embedding_dim)
        transformer_out = self.transformer_encoder(x)
        transformer_out = transformer_out.permute(
            1, 0, 2
        )  # Back to (batch_size, sequence_length, embedding_dim)
        output = self.fc(transformer_out[:, -1, :])
        logger.debug(f"TransformerAgent output shape: {output.shape}")
        return output


# Multi-Architecture Swarm Intelligence (MASI) class
class MultiArchitectureSwarm(nn.Module):
    def __init__(
        self,
        num_mlp_agents: int,
        num_cnn_agents: int,
        num_lstm_agents: int,
        num_transformer_agents: int,
        input_sizes: Dict[str, Any],
        output_size: int,
    ):
        super(MultiArchitectureSwarm, self).__init__()

        self.agents: List[Agent] = []

        # Initialize MLP Agents
        for _ in range(num_mlp_agents):
            agent = MLPAgent(
                input_size=input_sizes["mlp"]["input_size"],
                hidden_size=input_sizes["mlp"]["hidden_size"],
                output_size=output_size,
            )
            self.agents.append(agent)

        # Initialize CNN Agents
        for _ in range(num_cnn_agents):
            agent = CNNAgent(
                input_channels=input_sizes["cnn"]["input_channels"],
                num_classes=output_size,
            )
            self.agents.append(agent)

        # Initialize LSTM Agents
        for _ in range(num_lstm_agents):
            agent = LSTMAgent(
                input_size=input_sizes["lstm"]["input_size"],
                hidden_size=input_sizes["lstm"]["hidden_size"],
                output_size=output_size,
            )
            self.agents.append(agent)

        # Initialize Transformer Agents
        for _ in range(num_transformer_agents):
            agent = TransformerAgent(
                input_size=input_sizes["transformer"]["input_size"],
                num_heads=input_sizes["transformer"]["num_heads"],
                num_layers=input_sizes["transformer"]["num_layers"],
                output_size=output_size,
            )
            self.agents.append(agent)

        logger.info(f"Initialized {len(self.agents)} agents.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        agent_outputs = []

        for agent in self.agents:
            agent_output = agent(x)
            agent_outputs.append(agent_output)

        # Aggregate outputs (Simple averaging for now)
        global_output = self.aggregate_agent_outputs(agent_outputs)

        return global_output

    def aggregate_agent_outputs(
        self, agent_outputs: List[torch.Tensor]
    ) -> torch.Tensor:
        # Stack outputs and calculate mean
        logger.debug(f"Aggregating outputs from {len(agent_outputs)} agents.")
        stacked_outputs = torch.stack(agent_outputs)
        logger.debug(f"Stacked outputs shape: {stacked_outputs.shape}")
        global_output = torch.mean(stacked_outputs, dim=0)
        logger.debug(f"Global output shape: {global_output.shape}")
        return global_output
