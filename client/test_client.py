import gc
import go_vncdriver
import time
from PIL import Image
from vnc_client import VNCClient

from gym.envs.classic_control import rendering

client = VNCClient()
client.connect()

viewer = rendering.SimpleImageViewer()



for i in range(10000):
	if (i == 3):
		observation, infos, err = client.stepN()
	else:
		observation, infos, err = client.step()
		# print(observation)
		# print([infos[i]['stats.vnc.updates.n'] for i in infos.keys()])
		if any(infos[i]['stats.vnc.updates.n'] for i in infos.keys()):
			for ob in observation.keys():
				print(observation[ob])
				viewer.imshow(observation[ob])
	time.sleep(1)
