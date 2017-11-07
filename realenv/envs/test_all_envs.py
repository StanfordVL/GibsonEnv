from realenv.envs import *

    
env = HuskyNavigateEnv(human=True, is_discrete=True, mode="RGB", gpu_count=1, use_filler=True, resolution="MID")
