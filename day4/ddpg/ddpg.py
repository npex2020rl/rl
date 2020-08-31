import time
import csv
import gym
import argparse
from ddpg_agent import DDPGAgent
from utils import get_env_spec, set_log_dir


def run_ddpg(env_id,
             gamma=0.99,
             actor_lr=1e-4,
             critic_lr=1e-3,
             tau=1e-3,
             sigma=0.1,
             hidden_size1=64,
             hidden_size2=64,
             max_iter=1e5,
             eval_interval=1000,
             start_train=10000,
             train_interval=50,
             buffer_size=1e5,
             fill_buffer=20000,
             batch_size=64,
             num_checkpoints = 5,
             render=False,
             ):

    args = locals()

    max_iter = int(max_iter)
    buffer_size = int(buffer_size)
    checkpoint_interval = max_iter // (num_checkpoints - 1)
    env = gym.make(env_id)

    dimS, dimA, ctrl_range, max_ep_len = get_env_spec(env)

    agent = DDPGAgent(dimS,
                      dimA,
                      gamma=gamma,
                      actor_lr=actor_lr,
                      critic_lr=critic_lr,
                      tau=tau,
                      sigma=sigma,
                      hidden_size1=hidden_size1,
                      hidden_size2=hidden_size2,
                      buffer_size=buffer_size,
                      batch_size=batch_size,
                      render=render)

    set_log_dir(env_id)
    current_time = time.strftime("%m%d-%H%M%S")
    train_log = open('./train_log/' + env_id + '/DDPG_' + current_time + '.csv',
                     'w',
                     encoding='utf-8',
                     newline='')
    eval_log = open('./eval_log/' + env_id + '/DDPG_' + current_time + '.csv',
                    'w',
                    encoding='utf-8',
                    newline='')

    train_logger = csv.writer(train_log)
    eval_logger = csv.writer(eval_log)
    with open('./eval_log/' + env_id + '/DDPG_' + current_time + '.txt', 'w') as f:
        for key, val in args.items():
            print(key, '=', val, file=f)


    state = env.reset()
    step_count = 0
    ep_reward = 0

    # main loop
    for t in range(max_iter + 1):
        if t < fill_buffer:
            action = env.action_space.sample()
        else:
            action = agent.get_action(state)

        next_state, reward, done, _ = env.step(action)
        step_count += 1

        if step_count == max_ep_len:
            done = False

        agent.buffer.append(state, action, reward, next_state, done)

        state = next_state
        ep_reward += reward

        if done or (step_count == max_ep_len):
            train_logger.writerow([t, ep_reward])
            state = env.reset()
            step_count = 0
            ep_reward = 0

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
            agent.save_model('./checkpoints/' + env_id + '/DDPG(iter={})'.format(t))

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

            action = agent.get_action(state, eval=True)
            next_state, reward, done, _ = env.step(action)
            step_count += 1
            state = next_state
            ep_reward += reward

        if render and ep == 0:
            env.close()
        log.append(ep_reward)

    avg = sum(log) / eval_num

    return avg



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', required=True)
    parser.add_argument('--max_iter', required=False, default=1e5, type=float)
    parser.add_argument('--eval_interval', required=False, default=1000, type=int)
    parser.add_argument('--render', required=False, default=False, type=bool)
    parser.add_argument('--tau', required=False, default=1e-3, type=float)
    parser.add_argument('--actor_lr', required=False, default=1e-4, type=float)
    parser.add_argument('--critic_lr', required=False, default=1e-3, type=float)
    parser.add_argument('--sigma', required=False, default=0.1, type=float)
    parser.add_argument('--hidden1', required=False, default=64, type=int)
    parser.add_argument('--hidden2', required=False, default=64, type=int)
    parser.add_argument('--train_interval', required=False, default=50, type=int)
    parser.add_argument('--start_train', required=False, default=1000, type=int)
    parser.add_argument('--fill_buffer', required=False, default=1000, type=int)
    parser.add_argument('--batch_size', required=False, default=64, type=int)

    args = parser.parse_args()

    run_ddpg(args.env, gamma=0.99,
             actor_lr=args.actor_lr,
             critic_lr=args.critic_lr,
             tau=args.tau,
             sigma=args.sigma,
             hidden_size1=args.hidden1,
             hidden_size2=args.hidden2,
             max_iter=args.max_iter,
             eval_interval=args.eval_interval,
             start_train=args.start_train,
             train_interval=args.train_interval,
             buffer_size=1e6,
             fill_buffer=args.fill_buffer,
             render=args.render)
