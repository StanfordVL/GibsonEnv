#from realenv.client.client_actions import client_actions as actions
#from realenv.client.vnc_client import VNCClient as VNCClient


from gym.envs.registration import registry, register, make, spec

#===================== Full Environments =====================#
register(
    id='HumanoidWalking-v0',
    entry_point='realenv.envs.simple_env:HumanoidWalkingEnv'
)

register(
    id='AntWalkingEnv-v0',
    entry_point='realenv.envs.simple_env:AntWalkingEnv'
)


register(
    id='HuskyWalkingEnv-v0',
    entry_point='realenv.envs.simple_env:HuskyWalkingEnv'
)




#==================== Physics Environments ====================#

register(
    id='PhysicsSimpleHumanoidEnv-v0',
    entry_point='realenv.core.physics.simple_humanoid_env:SimpleHumanoidEnv'
)


register(
    id='PhysicsHumanoidWalkingEnv-v0',
    entry_point='realenv.core.physics.physics_env:HumanoidWalkingEnv'
)


register(
    id='PhysicsAntWalkingEnv-v0',
    entry_point='realenv.core.physics.physics_env:AntWalkingEnv'
)

register(
    id='PhysicsHuskyWalkingEnv-v0',
    entry_point='realenv.core.physics.physics_env:HuskyWalkingEnv'
)


'''
register(
    id='PhysicsEnv-v0',
    entry_point='realenv.core.physics.physics_env:PhysicsEnv'
)
'''