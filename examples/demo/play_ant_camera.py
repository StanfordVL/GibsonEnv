from realenv.envs.ant_env import AntNavigateEnv, AntClimbEnv
from realenv.utils.play import play

timestep = 1.0/(4 * 22)
frame_skip = 4

if __name__ == '__main__':
    env = AntClimbEnv(human=True, timestep=timestep, frame_skip=frame_skip, mode="RGBD", is_discrete = True, resolution="LARGE")
    play(env, zoom=4, fps=int( 1.0/(timestep * frame_skip)))