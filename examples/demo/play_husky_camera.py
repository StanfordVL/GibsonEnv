from realenv.envs.husky_env import HuskyCameraEnv
from realenv.utils.play import play

timestep = 1.0/(4 * 18)
frame_skip = 4

if __name__ == '__main__':
    env = HuskyCameraEnv(human=True, timestep=timestep, frame_skip=frame_skip, enable_sensors=False, is_discrete = True)
    play(env, zoom=4, fps=int( 1.0/(timestep * frame_skip)))