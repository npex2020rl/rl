import time
import csv
import torch
import gym
from agent import TRPOAgent
from utils import get_env_spec, set_log_dir
import argparse

def run_trpo(
            env_id,
            max_iter=1e6,
            gamma=0.99,
            lr=3e-4,
            lam=0.95,
            epsilon=0.2,
            hidden1=64,
            hidden2=64,
            eval_interval=2000,
            steps_per_epoch=4000,
            device='cpu',
            render='False'
            ):

    max_iter = int(max_iter)

    env = gym.make(env_id)

    dimS, dimA, ctrl_range, max_ep_len = get_env_spec(env)

    agent = TRPOAgent(
                        dimS,
                        dimA,
                        ctrl_range,
                        gamma=gamma,
                        lr=lr,
                        lam=lam,
                        epsilon=epsilon,
                        hidden1=hidden1,
                        hidden2=hidden2,
                        mem_size=steps_per_epoch,
                        device=device,
                        render=render
                     )

    set_log_dir(env_id)
    current_time = time.strftime("%m%d-%H%M%S")
    train_log = open('./train_log/' + env_id + '/PPO_' + current_time + '.csv',
                     'w', encoding='utf-8', newline='')
    eval_log = open('./eval_log/' + env_id + '/PPO_' + current_time + '.csv',
                    'w', encoding='utf-8', newline='')

    train_logger = csv.writer(train_log)
    eval_logger = csv.writer(eval_log)

    num_epochs = max_iter // steps_per_epoch

    total_t = 0
    for epoch in range(num_epochs):
        # start agent-env interaction
        state = env.reset()
        step_count = 0
        ep_reward = 0

        for t in range(steps_per_epoch):
            # collect transition samples by executing the policy
            action, log_prob, v = agent.get_action(state)

            next_state, reward, done, _ = env.step(action)
            agent.Memory.append(state, action, reward, v, log_prob)

            ep_reward += reward

            step_count += 1

            if (step_count == max_ep_len) or (t == steps_per_epoch - 1):
                # termination of env by env wrapper, or by truncation due to memory size
                s_last = torch.tensor(next_state, dtype=torch.float).to(device)
                v_last = agent.V(s_last).item()
                agent.Memory.compute_values(v_last)
            elif done:
                # episode done as the agent reach a terminal state
                v_last = 0.0
                agent.Memory.compute_values(v_last)

            state = next_state

            if done:
                train_logger.writerow([total_t, ep_reward])
                state = env.reset()
                step_count = 0
                ep_reward = 0

            if total_t % eval_interval == 0:
                log = agent.eval(env_id, total_t)
                eval_logger.writerow(log)

            total_t += 1

        # train agent at the end of each epoch
        agent.train(num_iter=1)

    train_log.close()
    eval_log.close()

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', required=True)
    parser.add_argument('--max_iter', required=False, default=1e5, type=float)
    parser.add_argument('--render', required=False, default=False, type=bool)


    args = parser.parse_args()

    run_trpo(args.env,
             args.max_iter,
             render=args.render
             )
