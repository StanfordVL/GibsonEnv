from gym.envs.registration import registry, register, make, spec


register(
    id='SimpleHumanoidEnv-v0',
    entry_point='realenv.core.physics.simple_humanoid_env:SimpleHumanoidEnv'
)

register(
    id='PhysicsEnv-v0',
    entry_point='realenv.core.physics.physics_env:PhysicsEnv'
)

register(
    id='HumanoidWalkingEnv-v0',
    entry_point='realenv.core.physics.physics_env:HumanoidWalkingEnv'
)


register(
    id='AntWalkingEnv-v0',
    entry_point='realenv.core.physics.physics_env:AntWalkingEnv'
)

register(
    id='HuskyWalkingEnv-v0',
    entry_point='realenv.core.physics.physics_env:HuskyWalkingEnv'
)
