from gibson.envs.humanoid_env import HumanoidNavigateEnv
from gibson.utils.play import play

timestep = 1.0/(4 * 22)
frame_skip = 4

if __name__ == '__main__':
    #env = HuskyNavigateEnv(human=True, timestep=timestep, frame_skip=frame_skip, mode="SENSOR", is_discrete = True, resolution="MID")
    env = HumanoidNavigateEnv(human=True, timestep=timestep, frame_skip=frame_skip, mode="SENSOR", is_discrete = True, resolution="MID")
    play(env, zoom=4, fps=int( 1.0/(timestep * frame_skip)))