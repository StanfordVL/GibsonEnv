import go_vncdriver
import time
import realenv
from PIL import Image
from gym.envs.classic_control import rendering
from realenv.client.client_actions import client_actions

from realenv.client.vnc_client import VNCClient


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
	agent = RandomAgent(client_actions)
	ob = None

	for i in range(10000):
		if (i == 15):
			## Relocate the view to a new position
			observation, infos, err = client.refresh()
		else:
			action = agent.act(ob)
			print('action', action)
			observation, infos, err = client.step(action)

			# display
			if any(infos[i]['stats.vnc.updates.n'] for i in infos.keys()):
				for ob in observation.keys():
					# print(observation[ob])
					viewer.imshow(observation[ob])
		time.sleep(0.2)
