import time
import csv
import gym
import mujoco_py
import rl_algos.envs.toy_env
from rl_algos.sac.sac_agent import SACAgent
from rl_algos.utils import get_env_spec, set_log_dir


def run_sac(
            env_id,
            max_iter=1e6,
            eval_interval=2000,
            start_train=10000,
            train_interval=50,
            buffer_size=1e6,
            fill_buffer=20000,
            gamma=0.99,
            pi_lr=3e-4,
            q_lr=3e-4,
            polyak=5e-3,
            alpha=0.2,
            hidden1=256,
            hidden2=256,
            batch_size=128,
            device='cpu',
            render='False'
            ):

    args = locals()

    max_iter = int(max_iter)
    buffer_size = int(buffer_size)
    env = gym.make(env_id)

    dimS, dimA, _, ctrl_range, max_ep_len = get_env_spec(env)

    agent = SACAgent(
                     dimS,
                     dimA,
                     ctrl_range,
                     gamma=gamma,
                     pi_lr=pi_lr,
                     q_lr=q_lr,
                     polyak=polyak,
                     alpha=alpha,
                     hidden1=hidden1,
                     hidden2=hidden2,
                     buffer_size=buffer_size,
                     batch_size=batch_size,
                     device=device,
                     render=render
                     )

    set_log_dir(env_id)

    num_checkpoints = 5
    checkpoint_interval = max_iter // (num_checkpoints - 1)
    current_time = time.strftime("%m%d-%H%M%S")
    train_log = open('./train_log/' + env_id + '/SAC_' + current_time + '.csv',
                     'w', encoding='utf-8', newline='')

    path = './eval_log/' + env_id + '/SAC_' + current_time
    eval_log = open(path + '.csv', 'w', encoding='utf-8', newline='')

    train_logger = csv.writer(train_log)
    eval_logger = csv.writer(eval_log)

    with open(path + '.txt', 'w') as f:
        for key, val in args.items():
            print(key, '=', val, file=f)

    state = env.reset()
    step_count = 0
    ep_reward = 0

    # main loop
    start = time.time()
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
            log = agent.eval(env_id, t)
            print('total duration : {:.4f} sec'.format(time.time() - start))
            eval_logger.writerow(log)

        if t % checkpoint_interval == 0:
            agent.save_model('./checkpoints/' + env_id + '/sac_{}th_iter_'.format(t))


    train_log.close()
    eval_log.close()

    return
