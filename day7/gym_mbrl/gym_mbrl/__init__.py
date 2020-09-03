from gym.envs.registration import register


register(
    id='MyPendulum-v0',
    entry_point='gym_mbrl.envs:MyPendulumEnv',
    max_episode_steps=200,
)
