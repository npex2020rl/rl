import torch
import numpy as np
from torch.optim import Adam

from torch.distributions import Independent
from torch.distributions.normal import Normal
import gym
from models import Actor, Critic
from memory import OnPolicyMemory
from itertools import chain
from utils import cg, fisher_vector_product, backtracking_line_search, update_model, flat_params


class TRPOAgent:
    def __init__(
                 self,
                 dimS,
                 dimA,
                 ctrl_range,
                 gamma=0.99,
                 lr=3e-4,
                 lam=0.95,
                 epsilon=0.2,
                 hidden1=64,
                 hidden2=32,
                 mem_size=4000,
                 device='cpu',
                 render=False
                 ):

        self.dimS = dimS
        self.dimA = dimA
        self.ctrl_range = ctrl_range

        self.gamma = gamma

        self.delta = 1e-3

        self.lam = lam          # GAE constant
        self.epsilon = epsilon  # clipping constant

        self.hidden1 = hidden1
        self.hidden2 = hidden2


        self.pi = Actor(dimS, dimA, hidden1, hidden2).to(device)
        self.V = Critic(dimS, hidden1, hidden2).to(device)

        params = chain(self.pi.parameters(), self.V.parameters())

        self.optimizer = Adam(params, lr=lr)
        self.critic_optim = Adam(self.V.parameters())
        self.Memory = OnPolicyMemory(dimS, dimA, gamma, lam, lim=mem_size)

        self.device = device
        self.render = render

    def get_action(self, obs):
        obs = torch.tensor(obs, dtype=torch.float).to(self.device)
        with torch.no_grad():
            mu, sigma = self.pi(obs)
            act_distribution = Independent(Normal(mu, sigma), 1)
            action = act_distribution.sample()

            log_prob = act_distribution.log_prob(action)
            val = self.V(obs)

        action = action.cpu().numpy()
        log_prob = log_prob.cpu().numpy()
        val = val.cpu().numpy()

        return action, log_prob, val

    def train(self, num_iter):
        print('training...')
        batch = self.Memory.load()

        states = torch.tensor(batch['state'], dtype=torch.float).to(self.device)
        actions = torch.tensor(batch['action'], dtype=torch.float).to(self.device)
        target_v = torch.tensor(batch['val'], dtype=torch.float).to(self.device)
        A = torch.tensor(batch['A'], dtype=torch.float).to(self.device)
        old_log_probs = torch.tensor(batch['log_prob'], dtype=torch.float).to(self.device)

        # train policy network
        for i in range(num_iter + 1):

            ################
            # train critic #
            ################
            out = self.V(states)
            critic_loss = torch.mean((out - target_v)**2)

            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            ###############
            # train actor #
            ###############

            log_probs = self.pi.log_prob(states, actions)

            # \pi(a_t | s_t ; \phi) / \pi(a_t | s_t ; \phi_old)
            prob_ratio = torch.exp(log_probs - old_log_probs)

            actor_loss = torch.mean(prob_ratio * A)
            loss_grad = torch.autograd.grad(actor_loss, self.pi.parameters())
            # flatten gradients of params
            g = torch.cat([grad.view(-1) for grad in loss_grad]).data

            s = cg(fisher_vector_product, g, self.pi, states)

            sAs = torch.sum(fisher_vector_product(s, self.pi, states) * s, dim=0, keepdim=True)
            step_size = torch.sqrt(2 * self.delta / sAs)[0]
            step = step_size * s


            old_actor = Actor(self.dimS, self.dimA, self.hidden1, self.hidden2)

            old_actor.load_state_dict(self.pi.state_dict())

            params = flat_params(self.pi)

            backtracking_line_search(old_actor,
                                     self.pi,
                                     actor_loss,
                                     g,
                                     old_log_probs,
                                     params,
                                     step,
                                     self.delta,
                                     A,
                                     states,
                                     actions)

        return

    def eval(self, env_id, t, eval_num=5):
        """
        evaluation of agent
        """
        env = gym.make(env_id)
        log = []
        for ep in range(eval_num):
            state = env.reset()
            step_count = 0
            ep_reward = 0
            done = False

            while not done:
                if self.render and ep == 0:
                    env.render()

                action, _, _ = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                step_count += 1
                state = next_state
                ep_reward += reward

            if self.render and ep == 0:
                env.close()

            log.append(ep_reward)
        avg = sum(log) / eval_num

        print('step {} : {:.4f}'.format(t, avg))

        return [t, avg]