from gibson.envs.camera_env import VirtualCameraEnv
from gibson.utils.play import play
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'configs', 'play', 'test_camera.yaml')
print(config_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=config_file)
    args = parser.parse_args()

    env = VirtualCameraEnv(config = args.config)
    obs = env.reset()

    obs, _, _, _ = env.step(np.array([-15.3, 5, 1.2, 0, 0, 1, 0])) # x y z quat
    plt.imshow(obs["rgb_filled"])
    plt.show()

    obs, _, _, _ = env.step(np.array([-14.3, 5, 1.2, 0, 0, 1, 0]))
    plt.imshow(obs["rgb_filled"])
    plt.show()

    obs, _, _, _ = env.step(np.array([-13.3, 5, 1.2, 0, 0, 1, 0]))
    plt.imshow(obs["rgb_filled"])
    plt.show()
