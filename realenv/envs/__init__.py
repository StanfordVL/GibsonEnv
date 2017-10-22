from gym.envs.registration import registry, register, make, spec


register(
    id='HumanoidWalking-v0',
    entry_point='realenv.envs.simple_env:SimpleEnv'
)