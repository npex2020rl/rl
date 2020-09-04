import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent
from torch.distributions.normal import Normal


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden1, hidden2):
        # actor f_\phi(s)
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)

        self.fc3 = nn.Linear(hidden2, act_dim)  # for \mu
        self.fc4 = nn.Linear(hidden2, act_dim)  # for \sigma

    def forward(self, obs):
        x = torch.tanh(self.fc1(obs))
        x = torch.tanh(self.fc2(x))

        mu = self.fc3(x)
        log_sigma = self.fc4(x)

        sigma = torch.exp(log_sigma)

        return mu, sigma

    def log_prob(self, obs, act):
        mu, sigma = self.forward(obs)
        act_distribution = Independent(Normal(mu, sigma), 1)
        log_prob = act_distribution.log_prob(act)
        return log_prob

class Critic(nn.Module):
    # critic V(s ; \theta)
    def __init__(self, obs_dim, hidden1, hidden2):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)

    def forward(self, obs):
        x = torch.tanh(self.fc1(obs))
        x = torch.tanh(self.fc2(x))

        return self.fc3(x)
