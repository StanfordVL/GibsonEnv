from gibson.envs.humanoid_env import HumanoidNavigateEnv
from gibson.utils.play import play
import os

timestep = 1.0/(4 * 22)
frame_skip = 4
config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'configs', 'humanoid_navigate.yaml')
print(config_file)


if __name__ == '__main__':
    #env = HuskyNavigateEnv(human=True, timestep=timestep, frame_skip=frame_skip, mode="SENSOR", is_discrete = True, resolution="MID")
    env = HumanoidNavigateEnv(human=True, is_discrete = True, config=config_file)
    play(env, zoom=4, fps=int( 1.0/(timestep * frame_skip)))