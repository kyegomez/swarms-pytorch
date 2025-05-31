import torch
import torch.nn as nn
import torch.optim as optim
import gym


class MAgent:
    class Agent(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.policy = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, output_dim),
                nn.Softmax(dim=-1),
            )

        def forward(self, state):
            return self.policy(state)

    class MultiGymEnvironment:
        def __init__(self, env_name, num_agents):
            self.envs = [gym.make(env_name) for _ in range(num_agents)]
            self.agents = [
                MAgent.Agent(
                    self.envs[0].observation_space.shape[0],
                    self.envs[0].action_space.n,
                )
                for _ in range(num_agents)
            ]
            self.optimizers = [
                optim.Adam(agent.parameters()) for agent in self.agents
            ]

        def step(self, agent_actions):
            rewards = []
            for env, action in zip(self.envs, agent_actions):
                _, reward, _, _ = env.step(action)
                rewards.append(reward)
            return rewards

        def get_states(self):
            states = [env.reset() for env in self.envs]
            return states

        def train(self, epochs=1000):
            for epoch in range(epochs):
                states = self.get_states()
                actions = [
                    torch.argmax(agent(torch.FloatTensor(state))).item()
                    for agent, state in zip(self.agents, states)
                ]
                rewards = self.step(actions)

                for agent, optimizer, reward in zip(
                    self.agents, self.optimizers, rewards
                ):
                    loss = (
                        -torch.log(agent(torch.FloatTensor(states))) * reward
                    )  # Example loss function
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
