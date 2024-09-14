from typing import List, Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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


# Initialize Agents
def initialize_agents(
    num_mlp_agents: int,
    num_cnn_agents: int,
    num_lstm_agents: int,
    num_transformer_agents: int,
    input_sizes: Dict[str, Any],
    output_size: int,
) -> List[Agent]:
    agents: List[Agent] = []

    # MLP Agents
    for _ in range(num_mlp_agents):
        agent = MLPAgent(
            input_size=input_sizes["mlp"]["input_size"],
            hidden_size=input_sizes["mlp"]["hidden_size"],
            output_size=output_size,
        )
        agents.append(agent)

    # CNN Agents
    for _ in range(num_cnn_agents):
        agent = CNNAgent(
            input_channels=input_sizes["cnn"]["input_channels"],
            num_classes=output_size,
        )
        agents.append(agent)

    # LSTM Agents
    for _ in range(num_lstm_agents):
        agent = LSTMAgent(
            input_size=input_sizes["lstm"]["input_size"],
            hidden_size=input_sizes["lstm"]["hidden_size"],
            output_size=output_size,
        )
        agents.append(agent)

    # Transformer Agents
    for _ in range(num_transformer_agents):
        agent = TransformerAgent(
            input_size=input_sizes["transformer"]["input_size"],
            num_heads=input_sizes["transformer"]["num_heads"],
            num_layers=input_sizes["transformer"]["num_layers"],
            output_size=output_size,
        )
        agents.append(agent)

    logger.info(f"Initialized {len(agents)} agents.")
    return agents


# Aggregate Outputs
def aggregate_agent_outputs(agent_outputs: List[torch.Tensor]) -> torch.Tensor:
    # Simple average of outputs
    logger.debug(f"Aggregating outputs from {len(agent_outputs)} agents.")
    stacked_outputs = torch.stack(agent_outputs)
    logger.debug(f"Stacked outputs shape: {stacked_outputs.shape}")
    global_output = torch.mean(stacked_outputs, dim=0)
    logger.debug(f"Global output shape: {global_output.shape}")
    return global_output


# Compute Loss
def compute_loss(
    global_output: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    criterion = nn.CrossEntropyLoss()
    loss = criterion(global_output, targets)
    logger.debug(f"Computed loss: {loss.item()}")
    return loss


# Compute Agent-specific Loss (Optional)
def compute_agent_loss(agent: Agent, loss: torch.Tensor) -> torch.Tensor:
    # For simplicity, all agents share the same loss
    return loss


# Aggregate Gradients
def aggregate_gradients(agents: List[Agent]) -> Dict[str, torch.Tensor]:
    # Average gradients across all agents
    shared_gradients: Dict[str, torch.Tensor] = {}
    num_agents = len(agents)
    for name, param in agents[0].named_parameters():
        if param.grad is not None:
            shared_gradients[name] = param.grad.clone() / num_agents
            for other_agent in agents[1:]:
                shared_gradients[name] += (
                    other_agent._parameters[name].grad.clone() / num_agents
                )
    logger.debug("Aggregated gradients.")
    return shared_gradients


# Evaluate Performance
def evaluate_swarm_performance(
    agents: List[Agent], validation_loader: DataLoader
) -> None:
    correct = 0
    total = 0
    with torch.no_grad():
        for data in validation_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            agent_outputs = []
            for agent in agents:
                output = agent(inputs)
                agent_outputs.append(output)
            global_output = aggregate_agent_outputs(agent_outputs)
            _, predicted = torch.max(global_output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    logger.info(f"Validation Accuracy: {accuracy:.2f}%")


# Main Training Loop
def train_swarm(
    agents: List[Agent],
    train_loader: DataLoader,
    validation_loader: DataLoader,
    num_epochs: int,
    evaluation_interval: int,
) -> None:
    for epoch in range(num_epochs):
        for i, data in enumerate(train_loader, 0):
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)

            agent_outputs = []
            total_loss = 0

            # Each agent processes the data
            for agent in agents:
                agent.optimizer.zero_grad()
                agent_output = agent(inputs)
                agent_outputs.append(agent_output)

                # Compute individual agent loss
                agent_loss = compute_loss(agent_output, targets)
                total_loss += agent_loss.item()

                # Backward pass and update for each agent
                agent_loss.backward()
                agent.optimizer.step()

            # Aggregate outputs (for logging purposes)
            global_output = aggregate_agent_outputs(agent_outputs)

            # Log the average loss
            avg_loss = total_loss / len(agents)
            logger.debug(f"Batch [{i}] Average loss: {avg_loss:.4f}")

        # Evaluate performance
        if (epoch + 1) % evaluation_interval == 0:
            logger.info(f"Epoch [{epoch + 1}/{num_epochs}]")
            evaluate_swarm_performance(agents, validation_loader)


# Example Usage
if __name__ == "__main__":
    # Hyperparameters
    num_mlp_agents = 2
    num_cnn_agents = 2
    num_lstm_agents = 2
    num_transformer_agents = 2
    num_epochs = 10
    evaluation_interval = 1
    batch_size = 64
    output_size = 10  # For example, number of classes in classification

    # Input sizes for different agents
    input_sizes = {
        "mlp": {"input_size": 784, "hidden_size": 128},  # Example for MNIST
        "cnn": {"input_channels": 1},
        "lstm": {
            "input_size": 28,
            "hidden_size": 128,
        },  # Sequence length for MNIST rows
        "transformer": {"input_size": 28, "num_heads": 4, "num_layers": 2},
    }

    # Initialize agents
    agents = initialize_agents(
        num_mlp_agents,
        num_cnn_agents,
        num_lstm_agents,
        num_transformer_agents,
        input_sizes,
        output_size,
    )

    # Load and preprocess data
    from torchvision import datasets, transforms

    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    validation_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False
    )

    # Train swarm
    train_swarm(
        agents, train_loader, validation_loader, num_epochs, evaluation_interval
    )
