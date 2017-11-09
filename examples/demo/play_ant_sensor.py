from realenv.envs.ant_env import AntNavigateEnv
from realenv.utils.play import play

timestep = 1.0/(4 * 22)
frame_skip = 4

if __name__ == '__main__':
    env = AntNavigateEnv(human=True, timestep=timestep, frame_skip=frame_skip, mode="SENSOR", is_discrete = True, resolution="MID")
    play(env, zoom=4, fps=int( 1.0/(timestep * frame_skip)))