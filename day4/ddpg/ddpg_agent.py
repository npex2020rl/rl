import torch.optim
from torch.nn import MSELoss
import numpy as np
import copy
from buffer import ReplayBuffer
from model import Actor, Critic


class DDPGAgent:
    def __init__(self,
                 dimS,
                 dimA,
                 gamma=0.99,
                 actor_lr=1e-4,
                 critic_lr=1e-3,
                 tau=1e-3,
                 sigma=0.1,
                 hidden_size1=400,
                 hidden_size2=300,
                 buffer_size=int(1e6),
                 batch_size=128,
                 render=False):

        self.dimS = dimS
        self.dimA = dimA

        self.gamma = gamma
        self.pi_lr = actor_lr
        self.q_lr = critic_lr
        self.tau = tau
        self.sigma = sigma

        self.batch_size = batch_size
        # networks definition
        # pi : actor network, Q : critic network
        self.pi = Actor(dimS, dimA, hidden_size1, hidden_size2)
        self.Q = Critic(dimS, dimA, hidden_size1, hidden_size2)

        # target networks
        self.targ_pi = copy.deepcopy(self.pi)
        self.targ_Q = copy.deepcopy(self.Q)

        self.buffer = ReplayBuffer(dimS, dimA, limit=buffer_size)

        self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), lr=self.q_lr)
        self.pi_optimizer = torch.optim.Adam(self.pi.parameters(), lr=self.pi_lr)

        self.render = render

    def target_update(self):
        # soft-update for both actors and critics
        # \theta^\prime = \tau * \theta + (1 - \tau) * \theta^\prime
        for th, targ_th in zip(self.pi.parameters(), self.targ_pi.parameters()):        # th : theta
            targ_th.data.copy_(self.tau * th.data + (1.0 - self.tau) * targ_th.data)

        for th, targ_th in zip(self.Q.parameters(), self.targ_Q.parameters()):
            targ_th.data.copy_(self.tau * th.data + (1.0 - self.tau) * targ_th.data)

    def get_action(self, state, eval=False):

        state = torch.tensor(state, dtype=torch.float)

        with torch.no_grad():
            action = self.pi(state)
            action = action.numpy()
        if not eval:
            # for exploration, we use a behavioral policy of the form
            # \beta(s) = \pi(s) + N(0, \sigma^2)
            noise = self.sigma * np.random.randn(self.dimA)
            return action + noise
        else:
            return action

    def train(self):
        """
        train actor-critic network using DDPG
        """

        batch = self.buffer.sample_batch(batch_size=self.batch_size)

        # unroll batch
        observations = torch.tensor(batch['state'], dtype=torch.float)
        actions = torch.tensor(batch['action'], dtype=torch.float)
        rewards = torch.tensor(batch['reward'], dtype=torch.float)
        next_observations = torch.tensor(batch['next_state'], dtype=torch.float)
        terminal_flags = torch.tensor(batch['done'], dtype=torch.float)

        mask = torch.tensor([1.]) - terminal_flags

        # compute TD targets based on target networks
        # if done, set target value to reward

        target = rewards + self.gamma * mask * self.targ_Q(next_observations, self.targ_pi(next_observations))

        out = self.Q(observations, actions)
        loss_ftn = MSELoss()
        loss = loss_ftn(out, target)
        self.Q_optimizer.zero_grad()
        loss.backward()
        self.Q_optimizer.step()

        pi_loss = - torch.mean(self.Q(observations, self.pi(observations)))
        self.pi_optimizer.zero_grad()
        pi_loss.backward()
        self.pi_optimizer.step()

        self.target_update()

    def save_model(self, path):
        checkpoint_path = path + 'model.pth.tar'
        torch.save(
                    {'actor': self.pi.state_dict(),
                     'critic': self.Q.state_dict(),
                     'target_actor': self.targ_pi.state_dict(),
                     'target_critic': self.targ_Q.state_dict(),
                     'actor_optimizer': self.pi_optimizer.state_dict(),
                     'critic_optimizer': self.Q_optimizer.state_dict()
                    },
                    checkpoint_path)

        return

    def load_model(self, path):
        checkpoint = torch.load(path)

        self.pi.load_state_dict(checkpoint['actor'])
        self.Q.load_state_dict(checkpoint['critic'])
        self.targ_pi.load_state_dict(checkpoint['target_actor'])
        self.targ_Q.load_state_dict(checkpoint['target_critic'])
        self.pi_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.Q_optimizer.load_state_dict(checkpoint['critic_optimizer'])

        return


if __name__ == '__main__':
    agent = DDPGAgent(3, 2, 1)
    print(agent.pi.state_dict())