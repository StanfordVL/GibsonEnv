from gibson.envs.humanoid_env import HumanoidNavigateEnv
from gibson.utils.play import play
import os

config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'configs', 'play', 'play_humanoid_nonviz.yaml')
print(config_file)


if __name__ == '__main__':
    #env = HuskyNavigateEnv(human=True, timestep=timestep, frame_skip=frame_skip, mode="SENSOR", is_discrete = True, resolution="MID")
    env = HumanoidNavigateEnv(config=config_file)
    play(env, zoom=4)