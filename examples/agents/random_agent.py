import go_vncdriver
import time
import numpy as np
import realenv
from realenv import actions
from realenv import VNCClient

## For visualization
from PIL import Image
from gym.envs.classic_control import rendering


class RandomAgent(object):
    """The world's simplest agent"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward=None):
        return self.action_space[np.random.randint(0, len(self.action_space))]


client = VNCClient()
client.connect()

viewer = rendering.SimpleImageViewer()

if __name__ == '__main__':
	agent = RandomAgent(actions)
	ob = None

	for i in range(10000):
		## Hardcoded connection time
		if (i == 15):
			## Relocate the view to a new position
			observation, infos, err = client.reset()
		else:
			action = agent.act(ob)
			observation, infos, err = client.step(action)

			## display
			if any(infos[i]['stats.vnc.updates.n'] for i in infos.keys()):
				# TODO: is network causing bottleneck here?
				for ob in observation.keys():
					viewer.imshow(observation[ob])
		time.sleep(0.2)
