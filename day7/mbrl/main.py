import argparse
import gym
from mb_agent import ModelBasedAgent
from reward_ftns import pendulum_reward
from utils import collect_rand_trajectories
import gym_mbrl




parser = argparse.ArgumentParser()
parser.add_argument('--env', required=True)
parser.add_argument('--max_iter', required=False, default=50, type=int)
parser.add_argument('--K', required=False, default=1000, type=int)
parser.add_argument('--init_traj', required=False, default=100, type=int)
parser.add_argument('--H', required=False, default=80, type=int)
parser.add_argument('--hidden', required=False, default=64, type=int)

args = parser.parse_args()
hidden = args.hidden
max_iter = int(args.max_iter)   # number of trajectories we will use in total
env_id = args.env
K = args.K
H = args.H

# main
# get random trajectories first
# prepare initialized model
env = gym.make(env_id)
state_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
ctrl_range = env.action_space.high[0]

agent = ModelBasedAgent(state_dim, act_dim, ctrl_range, hidden1=hidden, hidden2=hidden)

collect_rand_trajectories(env_id, agent.memory, args.init_traj)

batch_size = 512

# pre-train the network using randomly collected data
num_epochs = 10
epoch_size = len(agent.memory) // batch_size

for i in range(max_iter + num_epochs):
    # training loop
    # train the model
    if i == 0:
        print('pre-training...')

    if i == num_epochs:
        print('start MPC control...')

    if i < num_epochs:
        # first train the network only using randomly collected data
        for epoch in range(epoch_size):
            loss = agent.train(batch_size=batch_size)
        print('[iter {}] loss val : {:.4f}'.format(i, loss))

    else:
        loss = agent.train(batch_size=batch_size)
        print('[iter {}] loss val : {:.4f} /'.format(i, loss), end=' ')
        state = env.reset()

        done = False
        score = 0.

        while not done:
            if i % 5 == 0:
                env.render()
            # environment roll-out
            # at each step, select an action using MPC (on-policy data)
            # use mpi for random-sampling shooting
            action = agent.execute_action(state, pendulum_reward, K, H)
            next_state, rew, done, _ = env.step(action)
            agent.memory.append(state, action, next_state)
            score += rew
            state = next_state

        env.close()
        print('score = {:4f}'.format(score))
