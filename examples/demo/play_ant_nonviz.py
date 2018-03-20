from gibson.envs.ant_env import AntNavigateEnv, AntClimbEnv
from gibson.utils.play import play
import os

config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'configs', 'play', 'play_ant_nonviz.yaml')
print(config_file)


if __name__ == '__main__':
    env = AntNavigateEnv(config=config_file)
    play(env, zoom=4)