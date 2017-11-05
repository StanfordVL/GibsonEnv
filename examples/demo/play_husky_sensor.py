from realenv.envs.husky_env import HuskySensorEnv
from realenv.utils.play import play

timestep = 1.0/(4 * 22)
frame_skip = 4

if __name__ == '__main__':
    env = HuskySensorEnv(human=True, timestep=timestep, frame_skip=frame_skip, enable_sensors=False, is_discrete = True)
    play(env, zoom=4, fps=int( 1.0/(timestep * frame_skip)))