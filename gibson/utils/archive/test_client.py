import go_vncdriver
import time
from PIL import Image
from vnc_client import VNCClient
from gym.envs.classic_control import rendering
from realenv.agents.random_agent import RandomAgent
from realenv.client.client_actions import client_actions

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
