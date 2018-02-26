# GIBSON ENVIRONMENT for Embodied Active Agents with Real-World Perception 

You shouldn't play video games all day, so shouldn't your AI! We built a virtual environment that offers real world experience for learning perception. 

<img src=misc/ui.gif width="600">
 
**Summary**: Perception and being active (i.e. having a certain level of motion freedom) are closely tied. Learning active perception and sensorimotor control in the physical world is cumbersome as existing algorithms are too slow to efficiently learn in real-time and robots are fragile and costly. This has given a fruitful rise to learning in simulation which consequently casts a question on transferring to real-world. We developed Gibson environment with the following primary characteristics:  

**I.** being from the real-world and reflecting its semantic complexity through virtualizing real spaces,  
**II.** having a baked-in mechanism for transferring to real-world (Goggles function), and  
**III.** embodiment of the agent and making it subject to constraints of space and physics via integrating a physics engine ([Bulletphysics](http://bulletphysics.org/wordpress/)).  

**Naming**: Gibson environment is named after *James J. Gibson*, the author of "Ecological Approach to Visual Perception", 1979. “We must perceive in order to move, but we must also move in order to perceive” – JJ Gibson

Please see the [website](http://gibson.vision/) (http://env.gibson.vision/) for more technical details. This repository is intended for distribution of the environment and installation/running instructions.

#### Paper
**["Embodied Real-World Active Perception"](http://gibson.vision/)**, in **CVPR 2018**.


[![Gibson summary video](misc/vid_thumbnail_600.png)](https://youtu.be/KdxuZjemyjc "Click to watch the video summarizing Gibson environment!")



Beta
=================
**This is a 0.1.0 beta release, bug reports and suggestions for improvement are appreciated.** 

**Dataset**: To make the beta release lighter for the users, we are including a small subset (9) of the spaces in it. 
The full dataset includes hundreds of spaces which will be made available if we dont get a major bug report during the brief beta release. 

Table of contents
=================

   * [Installation](#installation)
        * [Quick Installation (docker)](#a-quick-installation-docker)
        * [Building from source](#b-building-from-source)
        * [Uninstalling](#uninstalling)
   * [Quick Start](#quick-start)
        * [Web User Interface](#web-user-interface)
        * [Rendering Semantics](#rendering-semantics)
   * [Coding your RL agent](#coding-your-rl-agent)
   * [Environment Configuration](#environment-configuration)
   * [Goggles: transferring the agent to real-world](#goggles-transferring-the-agent-to-real-world)


Installation
=================

#### Installation Method

There are two ways to instal gibson, A. using our docker image (recommended) and B. building from srouce. 

#### System requirements

The minimum system requirements are the following:

For docker installation (A): 
- Ubuntu 16.04
- Nvidia GPU with VRAM > 6.0GB
- Nvidia driver >= 384
- CUDA >= 9.0, CuDNN >= v7

For building from the source(B):
- Ubuntu >= 14.04
- Nvidia GPU with VRAM > 6.0GB
- Nvidia driver >= 375
- CUDA >= 8.0, CuDNN >= v5

#### Download data

First, our environment assets data are available [here](https://storage.googleapis.com/gibsonassets/assets.tar.gz). You can follow the installation guide below to download and set up them properly. `gibson/assets` folder stores necessary data (agent models, environments, etc) to run gibson environment. Users can add more environments files into `gibson/assets/dataset` to run gibson on more environments.
<!-- 
Make a folder `gibson/assets` and put the downloaded `assets.tar.gz` file in it.
-->
<!--
To download the file from the command line, run the following from the main directory of this repo:
```bash
filename="gibson/assets/assets.tar.gz"
fileid="1jV-UN4ePwsE9XYv8m4YbNiGxRI_WWpW0"
sudo apt-get install golang-go
export GOPATH=$HOME/go
go get github.com/ericchiang/pup
query=`curl -c ./cookie.txt -s -L "https://drive.google.com/uc?export=download&id=${fileid}" | pup 'a#uc-download-link attr{href}' | sed -e 's/amp;//g'`
curl -b ./cookie.txt -L -o ${filename} "https://drive.google.com${query}"
```
-->

A. Quick installation (docker)
-----

We use docker to distribute our software, you need to install [docker](https://docs.docker.com/engine/installation/) and [nvidia-docker2.0](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)) first. 

Run `docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi` to verify your installation. 

You can either 1. build your own docker image or 2. pull from our docker image. 1 is recommended because you have the freedom to include more or less enviroments into your docker image. For 2, we include a fixed number of 8 environments (space1-space8).

1. Build your own docker image (recommended)
```bash
git clone https://github.com/StanfordVL/GibsonEnv.git
cd gibson
wget https://storage.googleapis.com/gibsonassets/assets.tar.gz -P gibson 
./build.sh decompress_data
### the commands above downloads assets data file and decpmpress it into gibson/assets folder
docker build . -t gibson ### finish building inside docker
```
If the installation is successful, you should be able to run `docker run --runtime=nvidia -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix gibson` to create a container.


2. Or pull from our docker image
```bash
docker pull xf1280/gibson:0.1
```

B. Building from source
-----
If you don't want to use our docker image, you can also install gibson locally. This will require some dependencies to be installed. 

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
git clone https://github.com/StanfordVL/GibsonEnv.git
cd gibson
wget https://storage.googleapis.com/gibsonassets/assets.tar.gz -P gibson
./build.sh decompress_data ### decompress data 
#the commands above downloads assets data file and decpmpress it into gibson/assets folder
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

Uninstall gibson is easy. If you installed with docker, just run `docker images -a | grep "gibson" | awk '{print $3}' | xargs docker rmi` to clean up the image. If you installed from source, uninstall with `pip uninstall gibson`


Quick Start
=================

First run `xhost +local:root` on your host machine to enable display. You may need to run `export DISPLAY=:0.0` first. After getting into the docker container with `docker run --runtime=nvidia -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix gibson`, you will get an interactive shell. Now you can run a few demos. 

If you installed from source, you can run those directly using the following commands without using docker. 


```bash
python examples/demo/play_husky_sensor.py ### Use ASWD keys on your keyboard to control a car to navigate around Gates building
```

<img src=misc/husky_nonviz.png width="600">

You will be able to use ASWD keys on your keyboard to control a car to navigate around Gates building. A camera output will not be shown in this particular demo. 

```bash
python examples/demo/play_husky_camera.py ### Use ASWD keys on your keyboard to control a car to navigate around Gates building, while RGB and depth camera outputs are also shown.
```
<img src=misc/husky_camera.png width="600">

You will able to use ASWD keys on your keyboard to control a car to navigate around Gates building. You will also be able to see the RGB and depth camera outputs. 

```bash
python examples/train/train_husky_navigate_ppo2.py ### Use PPO2 to train a car to navigate down the hall way in Gates building, using visual input from the camera.
```

<img src=misc/husky_train.png width="800">
By running this command you will start training a husky robot  to navigate in Gates building and go down the corridor with RGBD input. You will see some RL related statistics in the terminal after each episode.


```bash
python examples/train/train_ant_navigate_ppo1.py ### Use PPO1 to train an ant to navigate down the hall way in Gates building, using visual input from the camera.
```

<img src=misc/ant_train.png width="800">
By running this command you will start training an ant to navigate in Gates building and go down the corridor with RGBD input. You will see some RL related statistics in the terminal after each episode.


Web User Interface
----
When running Gibson, you can start a web user interface with `python gibson/utils/web_ui.py`. This is helpful when you cannot physically access the machine running gibson or you are running on a headless cloud environment.


<img src=misc/web_ui.png width="600">

Rendering Semantics
----
<img src=misc/instance_colorcoding_semantics.png width="600">

Gibson can provide pixel-wise frame-by-frame semantic masks when the model is semantically annotated. As of now we have incorporated models from [Matterport 3D](https://niessner.github.io/Matterport/) and [Matterport 3D](https://niessner.github.io/Matterport/) for this purpose, and we refer you to the original dataset's reference for the list of their semantic classes and annotations. 

For detailed instructions of rendering semantics in Gibson, see [semantic instructions](gibson/utils/semantics.md). In the light beta release, the space `17DRP5sb8fy` includes Matterport 3D style semantic annotation and `space7` includes Stanford 2D3DS style annotation. 

**Agreement**: If you choose to use the models from [Stanford 2D3DS](http://buildingparser.stanford.edu/) or [Matterport 3D](https://niessner.github.io/Matterport/) for rendering semantics, we ask you to agree to and sign their respective agreements. See [here](https://niessner.github.io/Matterport/) for Matterport3D and [here](https://github.com/alexsax/2D-3D-Semantics) for Stanford 2D3DS.


Starter Code
----

More examples can be found in `examples/demo` and `examples/train` folder. A short introduction for each demo is shown below.

| Example        | Explanation          |
|:-------------: |:-------------| 
|`demo/play_ant_camera.py`|Use 1234567890qwertyui keys on your keyboard to control an ant to navigate around Gates building, while RGB and depth camera outputs are also shown. |
|`demo/play_ant_nonviz.py`| Use 1234567890qwertyui keys on your keyboard to control an ant to navigate around Gates building.|
|`demo/play_drone_camera.py`| Use ASWDZX keys on your keyboard to control a drone to navigate around Gates building, while RGB and depth camera outputs are also shown.|
|`demo/play_drone_nonviz.py`| Use ASWDZX keys on your keyboard to control a drone to navigate around Gates building|
|`demo/play_humanoid_camera.py`| Use 1234567890qwertyui keys on your keyboard to control a humanoid to navigate around Gates building. Just kidding, controlling humaniod with keyboard is too difficult, you can only watch it fall. Press R to reset. RGB and depth camera outputs are also shown. |
|`demo/play_humanoid_nonviz.py`| Watch a humanoid fall. Press R to reset.|
|`demo/play_husky_camera.py`| Use ASWD keys on your keyboard to control a car to navigate around Gates building, while RGB and depth camera outputs are also shown.|
|`demo/play_husky_nonviz.py`| Use ASWD keys on your keyboard to control a car to navigate around Gates building|
|`train/train_husky_navigate_ppo2.py`|   Use PPO2 to train a car to navigate down the hall way in Gates building, using RGBD input from the camera.|
|`train/train_husky_navigate_ppo1.py`|   Use PPO1 to train a car to navigate down the hall way in Gates building, using RGBD input from the camera.|
|`train/train_ant_navigate_ppo1.py`| Use PPO1 to train an ant to navigate down the hall way in Gates building, using visual input from the camera. |
|`train/train_ant_climb_ppo1.py`| Use PPO1 to train an ant to climb down the stairs in Gates building, using visual input from the camera.  |
|`train_ant_gibson_flagrun_ppo1.py`| Use PPO1 to train an ant to chase a target (a red cube) in Gates building. Everytime the ant gets to target(or time out), the target will change position.|
|`train_husky_gibson_flagrun_ppo1.py`|Use PPO1 to train a car to chase a target (a red cube) in Gates building. Everytime the car gets to target(or time out), the target will change position. |



Coding Your RL Agent
====
You can code your RL agent following our convention. The interface with our environment is very simple (see some examples in the end of this section).

First, you can create an environment by creating an instance of classes in `gibson/core/envs` folder. 


```python
env = AntNavigateEnv(is_discrete=False, config = config_file)
```

Then do one step of the simulation with `env.step`. And reset with `env.reset()`
```python
obs, rew, env_done, info = env.step(action)
```
`obs` gives the observation of the robot. `rew` is the defined reward. `env_done` marks the end of one episode, for example, when the robot dies. 
`info` gives some additional information of this step; sometimes we use this to pass additional non-visual sensor values.

We mostly followed [OpenAI gym](https://github.com/openai/gym) convention when designing the interface of RL algorithms and the environment. In order to help users start with the environment quicker, we
provide some examples at [examples/train](examples/train). The RL algorithms that we use are from [openAI baselines](https://github.com/openai/baselines) with some adaptation to work with hybrid visual and non-visual sensory data.
In particular, we used [PPO](https://github.com/openai/baselines/tree/master/baselines/ppo1) and a speed optimized version of [PPO](https://github.com/openai/baselines/tree/master/baselines/ppo2).


Environment Configuration
=================
Each environment is configured with a `yaml` file. Examples of `yaml` files can be found in `examples/configs` folder. Parameters for the file is explained below:

| Argument name        | Example value           | Explanation  |
|:-------------:|:-------------:| :-----|
| envname      | AntClimbEnv | Environment name, make sure it is the same as the class name of the environment |
| model_id      | space1-space8      |   Scene id, in beta release, choose from space1-space8 |
| target_orn | [0, 0, 3.14]      |   Eulerian angle (in radian) target orientation for navigating, the reference frame is world frame. For non-navigation tasks, this parameter is ignored. |
|target_pos | [-7, 2.6, -1.5] | target position (in meter) for navigating, the reference frame is world frame. For non-navigation tasks, this parameter is ignored. |
|initial_orn | [0, 0, 3.14] | initial orientation (in radian) for navigating, the reference frame is world frame |
|initial_pos | [-7, 2.6, 0.5] | initial position (in meter) for navigating, the reference frame is world frame|
|fov | 1.57  | field of view for the camera, in radian |
| use_filler | true/false  | use neural network filler or not. It is recommended to leave this argument true. See [Gibson Environment website](http://gibson.vision/) for more information. |
|display_ui | true/false  | Gibson has two ways of showing visual output, either in multiple windows, or aggregate them into a single pygame window. This argument determiens whether to show pygame ui or not, if in a production environment (training), you need to turn this off |
|show_dignostic | true/false  | show dignostics(including fps, robot position and orientation, accumulated rewards) overlaying on the RGB image |
|ui_num |2  | how many ui components to show, this should be length of ui_components. |
| ui_components | [RGB_FILLED, DEPTH]  | which are the ui components, choose from [RGB_FILLED, DEPTH, NORMAL, SEMANTICS, RGB_PREFILLED] |
|output | [nonviz_sensor, rgb_filled, depth]  | output of the environment to the robot, choose from  [nonviz_sensor, rgb_filled, depth]. These values are independent of `ui_components`, as `ui_components` determines what to show and `output` determines what the robot receives. |
|resolution | 512 | choose from [128, 256, 512] resolution of rgb/depth image |
|initial_orn | [0, 0, 3.14] | initial orientation (in radian) for navigating, the reference frame is world frame |
|speed : timestep | 0.01 | timestep of simulation in seconds. For example, if timestep=0.01 and the simulation is running at 100fps, it will be real time, if timestep=0.1 and the simulation is running at 100fps, it will be 10x real time|
|speed : frameskip | 1 | How many frames to run simulation for one action. For tasks that does not require high frequency control, you can set frameskip to larger value to gain further speed up. |
|mode | gui/headless  | gui or headless, if in a production environment (training), you need to turn this to headless. In gui mode, there will be visual output; in headless mode, there will be no visual output. |
|verbose |true/false  | show dignostics in terminal |


Goggles: transferring the agent to real-world
=================
Gibson includes a baked-in domain adaptation mechanism, named Goggles, for when an agent trained in Gibson is going to be deployed in real-world (i.e. operate based on images coming from an onboard camera). The mechanims is essentially a learned inverse function that alters the frames coming from a real camera to what they would look like if they were rendered via Gibson, and hence, disolve the domain gap. 

<img src=http://gibson.vision/static/img/figure4.jpg width="600">


**More details:** With all the imperfections in point cloud rendering, it has been proven difficult to get completely photo-realistic rendering with neural network fixes. The remaining issues make a domain gap between the synthesized and real images. Therefore, we formulate the rendering problem as forming a joint space ensuring a correspondence between rendered and real images, rather than trying to (unsuccessfuly) render images that are identical to real ones. This provides a deterministic pathway for traversing across these domains and hence undoing the gap. We add another network "u" for target image (I_t) and define the rendering loss to minimize the distance between f(I_s) and u(I_t), where "f" and "I_s" represent the filler neural network and point cloud rendering output, respectively (see the loss in above figure). We use the same network structure for f and u. The function u(I) is trained to alter the observation in real-world, I_t, to look like the corresponding I_s and consequently dissolve the gap. We named the u network goggles, as it resembles corrective lenses for the anget for deploymen in real world. Detailed formulation and discussion of the mechanism can be found in the paper. You can download the function u and apply it when you deploy your trained agent in real-world.

