import copy
import torch
import numpy as np
from torch.optim import Adam
from torch.nn import MSELoss
import gym
from buffer import ReplayBuffer
from dqn_model import Critic


class DQNAgent:
    def __init__(self,
                 dimS,
                 nA,
                 gamma=0.99,
                 hidden1=64,
                 hidden2=64,
                 lr=1e-3,
                 tau=1e-3,
                 buffer_size=100000,
                 batch_size=64,
                 render=False):

        args = locals()
        print('agent spec')
        print('-' * 80)
        print(args)
        print('-' * 80)

        self.dimS = dimS
        self.nA = nA

        # set networks
        self.Q = Critic(dimS, nA, hidden_size1=hidden1, hidden_size2=hidden2)
        self.target_Q = copy.deepcopy(self.Q)

        # freeze the target network
        for p in self.target_Q.parameters():
            p.requires_grad_(False)

        self.optimizer = Adam(self.Q.parameters(), lr=lr)

        self.gamma = gamma
        self.tau = tau

        self.buffer = ReplayBuffer(dimS, buffer_size)
        self.batch_size = batch_size

        self.render = render

        return

    def hard_target_update(self):
        # hard target update
        # this will not be used in our implementation
        self.target_Q.load_state_dict(self.Q.state_dict())
        return

    def target_update(self):
        # soft target update
        # when \tau = 1, this is equivalent to hard target update
        for p, target_p in zip(self.Q.parameters(), self.target_Q.parameters()):
            target_p.data.copy_(self.tau * p.data + (1.0 - self.tau) * target_p.data)
        return

    def get_action(self, state, eps):

        self.Q.eval()
        dimS = self.dimS
        nA = self.nA

        s = torch.tensor(state, dtype=torch.float).view(1, dimS)

        q = self.Q(s)

        # simple implementation of \epsilon-greedy method
        if np.random.rand() < eps:
            a = np.random.randint(nA)
        else:
            # greedy selection
            a = np.argmax(q.cpu().data.numpy())

        return a

    def train(self):

        self.Q.train()
        gamma = self.gamma
        batch = self.buffer.sample_batch(self.batch_size)

        # unroll batch
        with torch.no_grad():
            observations = torch.tensor(batch['state'], dtype=torch.float)
            actions = torch.tensor(batch['action'], dtype=torch.long)
            rewards = torch.tensor(batch['reward'], dtype=torch.float)
            next_observations = torch.tensor(batch['next_state'], dtype=torch.float)
            terminals = torch.tensor(batch['done'], dtype=torch.float)

            mask = 1.0 - terminals

            next_q = torch.unsqueeze(self.target_Q(next_observations).max(1)[0], 1)
            target = rewards + gamma * mask * next_q

        out = self.Q(observations).gather(1, actions)

        loss_ftn = MSELoss()
        loss = loss_ftn(out, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.target_update()

        return

    def save_model(self, path):
        checkpoint_path = path + 'model.pth.tar'
        torch.save(
            {
                'critic': self.Q.state_dict(),
                'target_critic': self.target_Q.state_dict(),
                'optimizer': self.optimizer.state_dict()
            },
            checkpoint_path

        )
        return

    def load_model(self, path):
        checkpoint = torch.load(path)

        self.Q.load_state_dict(checkpoint['critic'])
        self.target_Q.load_state_dict(checkpoint['target_critic'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        return
