import go_vncdriver
import time
import yaml
import os

from PIL import Image

from gym.envs.classic_control import rendering

remote_file = os.path.join(os.path.dirname(__file__), '/remote.yml')
with open('./remote.yml') as f:
    spec = yaml.load(f)

print('{}:{}'.format(spec['capri']['addr'], spec['capri']['port']))

# res = go_vncdriver.example(1, 2)
# res = go_vncdriver.connect(["hi", "tester"])
# import ipdb;ipdb.set_trace()

# go_vncdriver.setup()
h = go_vncdriver.VNCSession({'address': '{}:{}'.format(spec['capri']['addr'], spec['capri']['port']),
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