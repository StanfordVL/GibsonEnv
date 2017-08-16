import go_vncdriver
import time

from PIL import Image

from gym.envs.classic_control import rendering

# res = go_vncdriver.example(1, 2)
# res = go_vncdriver.connect(["hi", "tester"])
# import ipdb;ipdb.set_trace()

# go_vncdriver.setup()
h = go_vncdriver.VNCSession({'address': 
			'capri19.stanford.edu:5901',
			'password': 'qwertyui',
			'name': 'conn',
		}, None)
print(h.remotes)

viewer = rendering.SimpleImageViewer()
while True:
    observation, info = h.flip()
    if any(i['vnc.updates.n'] for i in info):
        for ob in observation:
            viewer.imshow(ob)
            # Image.fromarray(ob).show()
            # print(ob.shape)
    time.sleep(1/10)

h.close()