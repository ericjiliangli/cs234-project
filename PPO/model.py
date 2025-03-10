import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, MultivariateNormal


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, discrete) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, action_dim),
        )

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete

        if not self.discrete:
            self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, states):
        if self.discrete:
            probs = torch.softmax(self.net(states), dim=-1)
            distb = Categorical(probs)
        else:
            mean = self.net(states)
            std = torch.exp(self.log_std)
            cov_mtx = torch.diag(std**2)
            distb = MultivariateNormal(mean, covariance_matrix=cov_mtx)

        return distb


class ValueNetwork(nn.Module):
    def __init__(self, state_dim) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1),
        )

    def forward(self, states):
        return self.net(states).squeeze(-1)


def init_weights(m):
    """Weight initialization function"""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def create_networks(state_dim, action_dim, discrete):
    """Creates and initializes policy and value networks"""
    policy_net = PolicyNetwork(state_dim, action_dim, discrete)
    value_net = ValueNetwork(state_dim)
    
    policy_net.apply(init_weights)
    value_net.apply(init_weights)

    return policy_net, value_net
