import gym


def collect_rand_trajectories(env_id, transition_memory, num_trajectories):
    print('collecting random trajectories...')
    env = gym.make(env_id)

    for i in range(num_trajectories):
        # collect random trajectories
        # able to be accelerated with mpi
        state = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_state, _, done, _ = env.step(action)
            transition_memory.append(state, action, next_state)
            state = next_state

        if i % 10 == 9:
            print('{} trajectories collected'.format(i + 1))
    print('done')
    return
