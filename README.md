# Real Environment for Training Real World AI
You shouldn't play video games all day, so shouldn't your AI. In this project we build a virtual environment that offers real world experience. You can think of it like [The Matrix](https://www.youtube.com/watch?v=3Ep_rnYweaI).

## Note
This is a 0.0.1 alpha release, for use in Stanford SVL only. 

### Installation
We currently support Linux and OSX running Python 2.7.
```shell
git clone https://github.com/fxia22/realenv.git
cd realenv
pip install -e .
```
If this errors out, you may be missing some required packages. Here's the list of required packages we know about so far (please let us know if you had to install any others).

On Ubuntu 16.04
```shell
pip install numpy
sudo apt-get install golang libjpeg-turbo8-dev make
```

On OSX, El Capitan or newer:
```shell
pip install numpy
brew install golang libjpeg-turbo
```

## Demo

Here is a demo of a human controlled agent navigating through a virtual environment. 
![demo](https://github.com/fxia22/realenv/blob/full_environment2/misc/example.gif)

Here is a demo of a random agent trying to explore the space:
![demo](https://github.com/fxia22/realenv/blob/full_environment2/misc/example2.gif)


## Run Your First Agent

This example shows how you can start training with virtually zero set up. To see it with visualization,
```shell
cd realenv/agents/
python random_agent.py
``` 

We recommend following this example because it connects to a virtual environment running remotely in `capri19.stanford.edu` (you need to be in Stanford network to access this address), and does not require you to manually set up.

```python

import go_vncdriver
import time
import numpy as np
import realenv
from realenv import actions
from realenv import VNCClient

class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward=None):
        return self.action_space[np.random.randint(0, len(self.action_space))]


client = VNCClient()
client.connect()

if __name__ == '__main__':
  agent = RandomAgent(actions)
  ob = None

  for i in range(10000):
      action = agent.act(ob)
      observation, infos, err = client.step(action)
      time.sleep(0.2)


```

The example creates an `VNCClient`, connects to the virtual environment on `capri19.stanford.edu` and performs random exploration. We will soon release pre-built docker image with virtual environment, so that you can deploy it on your own server, instead of Stanford machines.


## Setup 

In addition to the above example where you train your AI on a scalable remote environment, you can also do it locally for debugging usage. This requires some set up (5~10 mins). We will soon provide you with Docker based toolkits to do this in no time. For the current being, here are the steps you need to take:

### Deploying Locally
- You need to have `OpenCV-Python` installed on your machine. We recommend setting up a `conda environment` before you start. To install OpenCV, `conda install -c menpo opencv3 -y` does the job.
- You will need a pytorch model file and a dataset to render the views, download [here](https://drive.google.com/file/d/0B93GhAQhsnjBX2RCZkEwRlBORlU/view?usp=sharing). Replace the path in `init.sh` with path to the model and the data.
- Build renderer with `./build.sh`
- Run `init.sh`, this will run the rendering engine and vncserver.
- If your purpose is to debug your agent locally, this is all you need to do.

### Turn Your Machine into a Remote Server
As a demo, a server is running at `capri19.stanford.edu:5901`, contact feixia@stanford.edu to obtain the password. 
- Server uses XVnc4 as vnc server. In order to use, first `git clone` this repository and go into root directory, then create a password first with `vncpasswd pw`.
- Connect with the client to 5901 port. This can also be configured in `init.sh`.


### Connect to Remote Server
- We implemented a custom `VNCClient` for you, based on `go-vncdriver` by OpenAI.
- By doing the following, your agent will be able to talk to virtual environment running anywhere else (currently you are limited to `capri19.stanford.edu:5901`)
```python
import realenv
from realenv import VNCClient
client = VNCClient()
client.connect()
```
`client.step(action)` tells the remote environment to execute an action, `client.reset()` sends your agent to the nearest starting point.

