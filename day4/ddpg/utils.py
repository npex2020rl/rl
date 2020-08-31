import os
import numpy as np

def get_env_spec(env):
    print('environment : ' + env.unwrapped.spec.id)
    print('obs dim : ', env.observation_space.shape, '/ ctrl dim : ', env.action_space.shape)
    dimS = env.observation_space.shape[0]
    dimA = env.action_space.shape[0]
    ctrl_range = env.action_space.high[0]
    max_ep_len = env._max_episode_steps
    print('-' * 80)

    print('ctrl range : ({:.2f}, {:.2f})'.format(-ctrl_range, ctrl_range))
    print('max_ep_len : ', max_ep_len)
    print('-' * 80)

    return dimS, dimA, ctrl_range, max_ep_len


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

# some helper functions used for value plot

def get_th(x, y):
    # transform cartesian coordinate of a given point into polar coordinate
    # here we only compute \th
    if x > 0.:
        return np.arctan(y / x) - (np.pi / 2.)
    elif x == 0.:
        return 0. if y > 0. else -np.pi
    else:
        return np.arctan(y / x) + (np.pi / 2.)


def obs2state(obs):
    # transform an observation (\cos(\th), \sin(th), \dot\th) to the corresponding state (\th, \thdot)
    # observation : a point on the unit circle
    th = get_th(obs[0], obs[1])
    th_dot = obs[2]
    state = (th, th_dot)
    return state