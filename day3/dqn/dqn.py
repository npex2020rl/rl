import csv
import os
import time
import argparse
import gym
from dqn_agent import DQNAgent


def run_dqn(env_id,
            gamma=0.99,
            lr=1e-3,
            polyak=1e-3,
            hidden1=256,
            hidden2=256,
            max_iter=1e6,
            buffer_size=1e6,
            fill_buffer=20000,
            batch_size=128,
            train_interval=50,
            start_train=10000,
            eval_interval=2000,
            num_checkpoints=5,
            render=False):

    params = locals()

    max_iter = int(max_iter)
    buffer_size = int(buffer_size)

    checkpoint_interval = max_iter // (num_checkpoints - 1)

    env = gym.make(env_id)

    dimS = env.observation_space.shape[0]
    nA = env.action_space.n
    max_ep_len = env._max_episode_steps


    agent = DQNAgent(dimS, nA,
                     gamma, hidden1,
                     hidden2, lr, polyak,
                     buffer_size, batch_size,
                     render=render)

    # logger set-up

    set_log_dir(env_id)
    current_time = time.strftime("%m%d-%H%M%S")
    train_log = open('./train_log/' + env_id + '/HJDQN_' + current_time + '.csv',
                     'w',
                     encoding='utf-8',
                     newline='')

    eval_log = open('./eval_log/' + env_id + '/HJDQN_' + current_time + '.csv',
                    'w',
                    encoding='utf-8',
                    newline='')

    train_logger = csv.writer(train_log)
    eval_logger = csv.writer(eval_log)

    with open('./eval_log/' + env_id + '/HJDQN_' + current_time + '.txt', 'w') as f:
        for key, val in params.items():
            print(key, '=', val, file=f)

    # start environment roll-out
    state = env.reset()
    step_count = 0
    ep_reward = 0.0

    # set params used for \epsilon-greedy method
    eps = 1.0
    eps_decay = 0.995
    min_eps = 0.01

    # main loop
    for t in range(max_iter + 1):

        if t < fill_buffer:
            action = env.action_space.sample()
        else:
            action = agent.get_action(state, eps)
        eps = max(eps * eps_decay, min_eps)


        next_state, reward, done, _ = env.step(action)
        step_count += 1

        if step_count == max_ep_len:
            done = False

        agent.buffer.append(state, action, reward, next_state, done)

        ep_reward += reward
        state = next_state

        if done or (step_count == max_ep_len):
            train_logger.writerow([t, ep_reward])

            state = env.reset()
            step_count = 0
            ep_reward = 0

        # Start training after sufficient number of transition samples are gathered
        if (t >= start_train) and (t % train_interval == 0):
            for _ in range(train_interval):
                agent.train()

        if t % eval_interval == 0:
            eval_score = eval_agent(agent, env_id, render=False)
            log = [t, eval_score]
            print('step {} : {:.4f}'.format(t, eval_score))
            eval_logger.writerow(log)

        if t % (10 * eval_interval) == 0:
            if render:
                render_agent(agent, env_id)

        if t % checkpoint_interval == 0:
            agent.save_model('./checkpoints/' + env_id + '/DQN(iter={})'.format(t))

    train_log.close()
    eval_log.close()

    return

def render_agent(agent, env_id):
    eval_agent(agent, env_id, eval_num=1, render=True)


def eval_agent(agent, env_id, eval_num=5, render=False):
    log = []
    for ep in range(eval_num):
        env = gym.make(env_id)

        state = env.reset()
        step_count = 0
        ep_reward = 0
        done = False

        while not done:
            if render and ep == 0:
                env.render()

            action = agent.get_action(state, 0.0)
            next_state, reward, done, _ = env.step(action)
            step_count += 1
            state = next_state
            ep_reward += reward

        if render and ep == 0:
            env.close()
        log.append(ep_reward)

    avg = sum(log) / eval_num

    return avg


def set_log_dir(env_id):
    if not os.path.exists('./train_log/'):
        os.mkdir('./train_log/')
    if not os.path.exists('./eval_log/'):
        os.mkdir('./eval_log/')

    if not os.path.exists('./train_log/' + env_id + '/'):
        os.mkdir('./train_log/' + env_id + '/')
    if not os.path.exists('./eval_log/' + env_id + '/'):
        os.mkdir('./eval_log/' + env_id + '/')

    if not os.path.exists('./checkpoints/'):
        os.mkdir('./checkpoints')

    if not os.path.exists('./checkpoints/' + env_id + '/'):
        os.mkdir('./checkpoints/' + env_id + '/')

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', required=True)
    parser.add_argument('--max_iter', required=False, default=1e5, type=float)
    parser.add_argument('--eval_interval', required=False, default=1000, type=int)
    parser.add_argument('--render', required=False, default=False, type=bool)
    parser.add_argument('--tau', required=False, default=1e-3, type=float)
    parser.add_argument('--q_lr', required=False, default=1e-3, type=float)
    parser.add_argument('--hidden1', required=False, default=128, type=int)
    parser.add_argument('--hidden2', required=False, default=128, type=int)
    parser.add_argument('--train_interval', required=False, default=50, type=int)
    parser.add_argument('--start_train', required=False, default=1000, type=int)
    parser.add_argument('--fill_buffer', required=False, default=1000, type=int)
    parser.add_argument('--num_checkpoints', required=False, default=5, type=int)
    parser.add_argument('--batch_size', required=False, default=64, type=int)

    args = parser.parse_args()


    run_dqn(args.env,
            gamma=0.99,
            lr=args.q_lr,
            polyak=args.tau,
            hidden1=args.hidden1,
            hidden2=args.hidden2,
            max_iter=args.max_iter,
            buffer_size=1e6,
            fill_buffer=args.fill_buffer,
            batch_size=args.batch_size,
            train_interval=args.train_interval,
            start_train=args.start_train,
            eval_interval=args.eval_interval,
            num_checkpoints=args.num_checkpoints,
            render=args.render)
