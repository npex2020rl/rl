import torch
import torch.nn as nn
import torch.nn.functional as F

class TransitionModel(nn.Module):
    def __init__(self, state_dim, act_dim, hidden1, hidden2):
        super(TransitionModel, self).__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.fc1 = nn.Linear(state_dim + act_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, state_dim)

    def forward(self, state, act):
        x = torch.cat([state, act], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        delta = self.fc3(x)

        next_state = state + delta  # \hat{s}_{t+1} = s_t + f(s_t, a_t ; \theta)
        
        return next_state