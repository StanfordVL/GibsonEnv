# Gibson Environment for Training Real World AI
You shouldn't play video games all day, so shouldn't your AI. In this project we build a virtual environment that offers real world experience. You can think of it like [The Matrix](https://www.youtube.com/watch?v=3Ep_rnYweaI).

![gibson](misc/ui.gif)


### Note
This is a 0.1.0 beta release, bug reports are welcome. 

Table of contents
=================

   * [Installation](#installation)
        * [Quick Installation (docker)](#quick-installation)
        * [Building from source](#building-from-source)
        * [Uninstalling](#uninstalling)
   * [Quick Start](#quick-start)
   * [Coding your RL agent](#coding-your-rl-agent)
   * [Environment Configuration](#environment-configuration)

Installation
=================

The minimal system requirements are the following:

- Ubuntu 16.04
- Nvidia GPU with VRAM > 6.0GB
- Nvidia driver >= 384
- CUDA >= 9.0, CuDNN >= v7

#### Download data

Download data from [here](https://drive.google.com/open?id=1jV-UN4ePwsE9XYv8m4YbNiGxRI_WWpW0) and put `assets.tar.gz` it in `gibson/assets` folder.

Quick installation
-----


We use docker to distribute our software, you need to install [docker](https://docs.docker.com/engine/installation/) and [nvidia-docker2.0](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)) first. 

Run `docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi` to verify your installation. 


1. Build your own docker image (recommended)
```bash
git clone -b dev https://github.com/fxia22/gibson.git 
cd gibson
#download data file from https://drive.google.com/open?id=1jV-UN4ePwsE9XYv8m4YbNiGxRI_WWpW0 and put it into gibson/assets folder
./build.sh decompress_data ### Download data outside docker, in case docker images need to be rebuilt
docker build . -t gibson
```
If the installation is successful, you should be able to run `docker run --runtime=nvidia -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix gibson` to create a container.


2. Or pull from our docker image
```bash
docker pull xf1280/gibson:0.1
```

Building from source
-----

First, make sure you have Nvidia driver and CUDA installed. If you install from source, CUDA 9 is not necessary, as that is for nvidia-docker 2.0. Then, let's install some dependencies:

```bash
apt-get update 
apt-get install libglew-dev libglm-dev libassimp-dev xorg-dev libglu1-mesa-dev libboost-dev \
		mesa-common-dev freeglut3-dev libopenmpi-dev cmake golang libjpeg-turbo8-dev wmctrl \ 
		xdotool libzmq3-dev zlib1g-dev\
```	

Install required deep learning libraries: Using python3.5 is recommended. You can create a python3.5 environment first. 

```bash
pip install http://download.pytorch.org/whl/cu90/torch-0.3.0.post4-cp35-cp35m-linux_x86_64.whl 
pip install torchvision
pip install tensorflow==1.3
```
Clone the repository, download data and build
```bash
git clone -b dev https://github.com/fxia22/gibson.git 
cd gibson
#download data from google drive
./build.sh decompress_data ### decompress data 
./build.sh build_local ### build C++ and CUDA files
pip install -e . ### Install python libraries
```

Install OpenAI baselines if you need to run training demo.

```bash
git clone https://github.com/fxia22/baselines.git
pip install -e baselines
```

Uninstalling
----

Uninstall gibson is easy, if you installed with docker, just run `docker images -a | grep "gibson" | awk '{print $3}' | xargs docker rmi` to clean up the image. If you installed from source, uninstall with `pip uninstall gibson`


Quick Start
=================

After getting into the docker container, you can run a few demos. You might need to run `xhost +local:root` to enable display. If you installed from source, you can run those directly. 

```bash
python examples/demo/play_husky_sensor.py ### Use ASWD to control a car to navigate around gates
```
![husky_nonviz](misc/husky_nonviz.png)
You are able to use ASWD to control a car to navigate around gates. You will not see camera output. 

```bash
python examples/demo/play_husky_camera.py ### Use ASWD to control a car to navigate around gates, with camera output
```
![husky_nonviz](misc/husky_camera.png)
You are able to use ASWD to control a car to navigate around gates. You will also be able to see camera output. 

```bash
python examples/train/train_husky_navigate_ppo2.py ### Use PPO2 to train a car to navigate down the hall way in gates based on visual input
```

![husky_train](misc/husky_train.png)
Running this command you will start training a husky robot to navigate in gates and go down the corridor. You will see some RL related statistics in terminal after each episode.


```bash
python examples/train/train_ant_navigate_ppo1.py ### Use PPO2 to train an ant to navigate down the hall way in gates based on visual input
```

![ant_train](misc/ant_train.png)
Running this command you will start training an ant to navigate in gates and go down the corridor. You will see some RL related statistics in terminal after each episode.


When running Gibson, you can start a web user interface with `python gibson/utils/web_ui.py`. This is helpful when you cannot physically access the machine running gibson or you are running on a headless cloud environment.

![web_ui](misc/web_ui.png)


More examples can be found in `examples/demo` and `examples/train` folder.


Coding Your RL Agent
====
You can code your RL agent following our convention. The interface with our environment is very simple.

First, you can create an environment by creating an instance of classes in `gibson/core/envs` folder. 


```python
env = AntNavigateEnv(is_discrete=False, config = config_file)
```

Then do one step of the simulation with `env.step`. And reset with `env.reset()`
```python
obs, rew, env_done, info = env.step(action)
```
`obs` gives the observation of the robot. `rew` is the defined reward. `env_done` marks the end of one episode, for example, when the robot dies. 
`info` gives some additional information of this step, sometimes we use this to pass additional non-visual sensor values.

We mostly followed [OpenAI gym](https://github.com/openai/gym) convention when designing the interface of RL algorithms and the environment. In order to help users start with the environment quicker, we
provide some examples at [examples/train](examples/train). The RL algorithms that we use are from [openAI baselines](https://github.com/openai/baselines) with some adaptation to work with hybrid visual and non-visual sensory data.
In particular, we used [PPO](https://github.com/openai/baselines/tree/master/baselines/ppo1) and a speed optimized version of [PPO](https://github.com/openai/baselines/tree/master/baselines/ppo2).


Environment Configuration
=================
Each environment is configured with a `yaml` file. Examples of `yaml` files can be found in `examples/configs` folder. Parameters for the file is explained below:

| Argument name        | Example value           | Explanation  |
|:-------------:|:-------------:| :-----|
| envname      | AntClimbEnv | Environment name, make sure it is the same as the class name of the environment |
| model_id      | space7      |   Scene id, in beta release, choose from space1-space8 |
| target_orn | [0, 0, 3.14]      |   Eulerian angle target orientation for navigating, the reference frame is world frame |
|target_pos | [-7, 2.6, -1.5] | target position for navigating, the reference frame is world frame |
|initial_orn | [0, 0, 3.14] | initial orientation for navigating |
|initial_pos | [-7, 2.6, 0.5] | initial position for navigating |
|fov | 1.57  | field of view for the camera, in rad |
| use_filler | true  | use neural network filler or not, it is recommended to leave this argument true |
|display_ui | true  | show pygame ui or not, if in a production environment (training), you need to turn this off |
|show_dignostic | true  | show dignostics overlaying on the RGB image |
|ui_num |2  | how many ui components to show |
| ui_components | [RGB_FILLED, DEPTH]  | which are the ui components, choose from [RGB_FILLED, DEPTH, NORMAL, SEMANTICS, RGB_PREFILLED] |
|output | [nonviz_sensor, rgb_filled, depth]  | output of the environment to the robot |
|resolution | 512 | resolution of rgb/depth image |
|speed : timestep | 0.01 | timestep of simulation in seconds |
|speed : frameskip | 1 | how many frames to run simulation for one action |
|mode | gui  | gui or headless, if in a production environment (training), you need to turn this to headless |
|verbose |false  | show dignostics in terminal |


