import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """
    implementation of actor network mu(s)
    """

    def __init__(self, state_dim, action_dim, hidden_size1, hidden_size2):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, action_dim)

    def forward(self, state):

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))    # each entry of the action lies in (-1, 1)

        return x


class Critic(nn.Module):
    """
    implementation of critic network Q(s, a)
    """
    def __init__(self, state_dim, action_dim, hidden_size1, hidden_size2):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
