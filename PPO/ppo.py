import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, MultivariateNormal
import numpy as np


class Memory:
    """Memory buffer to store trajectories."""
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def store(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def clear(self):
        """Clears memory after an update."""
        self.states, self.actions, self.log_probs = [], [], []
        self.rewards, self.dones, self.values = [], [], []


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Computes Generalized Advantage Estimation (GAE)."""
    advantages = []
    gae = 0
    next_value = 0  # Bootstrap value for terminal states

    for step in reversed(range(len(rewards))):
        delta = rewards[step] + (1 - dones[step]) * gamma * next_value - values[step]
        gae = delta + (1 - dones[step]) * gamma * lam * gae
        advantages.insert(0, gae)
        next_value = values[step]

    returns = np.array(advantages) + np.array(values)
    return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)


class PPO:
    """PPO agent for training and updating the policy."""
    def __init__(self, policy_net, value_net, lr=3e-4, gamma=0.99, lam=0.95, eps_clip=0.2, epochs=10, batch_size=64):
        self.policy_net = policy_net
        self.value_net = value_net
        self.optimizer_policy = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.optimizer_value = optim.Adam(self.value_net.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.eps_clip = eps_clip
        self.epochs = epochs
        self.batch_size = batch_size

    def update(self, memory):
        """Performs the PPO update."""
        states = torch.tensor(np.array(memory.states), dtype=torch.float32)
        actions = torch.tensor(np.array(memory.actions), dtype=torch.float32)
        log_probs_old = torch.tensor(np.array(memory.log_probs), dtype=torch.float32)
        values = torch.tensor(np.array(memory.values), dtype=torch.float32)
        rewards = np.array(memory.rewards)
        dones = np.array(memory.dones)

        # Compute advantages and returns
        advantages, returns = compute_gae(rewards, values, dones, self.gamma, self.lam)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # Normalize

        for _ in range(self.epochs):
            # Get new action probabilities
            distb = self.policy_net(states)
            log_probs_new = distb.log_prob(actions).sum(dim=-1)  # Sum over action dimensions if continuous

            # Compute ratio for PPO clipping
            ratio = torch.exp(log_probs_new - log_probs_old)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value function loss
            values_pred = self.value_net(states)
            value_loss = nn.MSELoss()(values_pred, returns)

            # Optimize policy network
            self.optimizer_policy.zero_grad()
            policy_loss.backward()
            self.optimizer_policy.step()

            # Optimize value network
            self.optimizer_value.zero_grad()
            value_loss.backward()
            self.optimizer_value.step()

        # Clear memory after updating
        memory.clear()
