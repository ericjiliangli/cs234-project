import torch
import torch.nn as nn
from torch.distributions import Categorical, MultivariateNormal, Normal
    
#Policy Network
class PolicyNet(nn.Module):
    #Constructor
    def __init__(self, state_dim, action_dim, discrete=False):
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
        # self.dist = DiagGaussian(action_dim, action_dim)
        
    #Forward pass
    def forward(self, state, deterministic=False):
        # state = state
        # feature = self.net(state)
        # dist = self.dist(feature)
        
        mean = self.net(state)

        std = torch.exp(self.log_std)
        cov_mtx = torch.eye(self.action_dim) * (std ** 2)

        dist = MultivariateNormal(mean, cov_mtx)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        
        return action, dist.log_prob(action)
    
    #Choose an action (stochastically or deterministically)
    def choose_action(self, state, deterministic=False):
        mean = self.net(state)

        std = torch.exp(self.log_std)
        cov_mtx = torch.eye(self.action_dim) * (std ** 2)

        dist = MultivariateNormal(mean, cov_mtx)

        if deterministic:
            return dist.mode()

        return dist.sample()
    
    #Evaluate a state-action pair (output log-prob. & entropy)
    def evaluate(self, state, action):
        mean = self.net(state)

        std = torch.exp(self.log_std)
        cov_mtx = torch.eye(self.action_dim) * (std ** 2)

        dist = MultivariateNormal(mean, cov_mtx)
        
        return dist.log_prob(action), dist.entropy()

#Value Network
class ValueNet(nn.Module):
    #Constructor
    def __init__(self, s_dim):
        super(ValueNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1),
        )
    
    #Forward pass
    def forward(self, state):
        return self.net(state)