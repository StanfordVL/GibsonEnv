#from realenv.client.client_actions import client_actions as actions
#from realenv.client.vnc_client import VNCClient as VNCClient


from gym.envs.registration import registry, register, make, spec

#===================== Full Environments =====================#
## Eventually we will package all environments for encapsulation
#  An environment can be invoked by:
#   env = gym.make('HumanoidCamera-v0')        ## unconfigurable
##  env = HumanoidCamera(human=True)           ## configurable 


register(
    id='HumanoidCamera-v0',
    entry_point='realenv.envs.humanoid_env:HumanoidCameraEnv'
)

register(
    id='HumanoidSensor-v0',
    entry_point='realenv.envs.humanoid_env:HumanoidSensorEnv'
)


register(
    id='AntWalkingEnv-v0',
    entry_point='realenv.envs.simple_env:AntWalkingEnv'
)


register(
    id='HuskyWalkingEnv-v0',
    entry_point='realenv.envs.simple_env:HuskyWalkingEnv'
)
