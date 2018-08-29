# GIBSON ENVIRONMENT for Embodied Active Agents with Real-World Perception 

You shouldn't play video games all day, so shouldn't your AI! We built a virtual environment simulator, Gibson, that offers real-world experience for learning perception.  

<img src=misc/ui.gif width="600">
 
**Summary**: Perception and being active (i.e. having a certain level of motion freedom) are closely tied. Learning active perception and sensorimotor control in the physical world is cumbersome as existing algorithms are too slow to efficiently learn in real-time and robots are fragile and costly. This has given a fruitful rise to learning in simulation which consequently casts a question on transferring to real-world. We developed Gibson environment with the following primary characteristics:  

**I.** being from the real-world and reflecting its semantic complexity through virtualizing real spaces,  
**II.** having a baked-in mechanism for transferring to real-world (Goggles function), and  
**III.** embodiment of the agent and making it subject to constraints of space and physics via integrating a physics engine ([Bulletphysics](http://bulletphysics.org/wordpress/)).  

**Naming**: Gibson environment is named after *James J. Gibson*, the author of "Ecological Approach to Visual Perception", 1979. “We must perceive in order to move, but we must also move in order to perceive” – JJ Gibson

Please see the [website](http://gibson.vision/) (http://gibsonenv.stanford.edu/) for more technical details. This repository is intended for distribution of the environment and installation/running instructions.

#### Paper
**["Gibson Env: Real-World Perception for Embodied Agents"](http://gibson.vision/)**, in **CVPR 2018 [Spotlight Oral]**.


[![Gibson summary video](misc/vid_thumbnail_600.png)](https://youtu.be/KdxuZjemyjc "Click to watch the video summarizing Gibson environment!")



Release
=================
**This is the 0.3.1 release. Bug reports and suggestions for improvement are appreciated.** [change log file](misc/CHANGELOG.md).  

**Database**: To make the download package lighter for the users, we are including a small subset (9) of the spaces in the core assets. 
The [full database](gibson/data/README.md) includes 572 spaces and 1440 floors. Users can download the rest of the spaces and add them to the assets folder. A diverse set of visualization of all spaces in Gibson can be seen [here](http://gibsonenv.stanford.edu/database/).

Table of contents
=================

   * [Installation](#installation)
        * [Quick Installation (docker)](#a-quick-installation-docker)
        * [Building from source](#b-building-from-source)
        * [Uninstalling](#uninstalling)
   * [Quick Start](#quick-start)
        * [Gibson FPS](#gibson-framerate)
        * [Web User Interface](#web-user-interface)
        * [Rendering Semantics](#rendering-semantics)
        * [Robotic Agents](#robotic-agents)
        * [ROS Configuration](#ros-configuration)
   * [Coding your RL agent](#coding-your-rl-agent)
   * [Environment Configuration](#environment-configuration)
   * [Goggles: transferring the agent to real-world](#goggles-transferring-the-agent-to-real-world)
   * [Citation](#citation)



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

First, our environment core assets data are available [here](https://storage.googleapis.com/gibsonassets/assets_core_v2.tar.gz). You can follow the installation guide below to download and set up them properly. `gibson/assets` folder stores necessary data (agent models, environments, etc) to run gibson environment. Users can add more environments files into `gibson/assets/dataset` to run gibson on more environments. Visit the [database readme](gibson/data/README.md) for downloading more spaces. Please sign the [license agreement](gibson/data/README.md#download) before using Gibson's database.


A. Quick installation (docker)
-----

We use docker to distribute our software, you need to install [docker](https://docs.docker.com/engine/installation/) and [nvidia-docker2.0](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)) first. 

Run `docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi` to verify your installation. 

You can either 1. build your own docker image or 2. pull from our docker image. 1 is recommended because you have the freedom to include more or less enviroments into your docker image. For 2, we include a fixed number of 8 environments (space1-space8).

1. Build your own docker image (recommended)
```bash
git clone https://github.com/StanfordVL/GibsonEnv.git
cd GibsonEnv/gibson
wget https://storage.googleapis.com/gibsonassets/assets_core_v2.tar.gz
tar -zxf assets_core_v2.tar.gz && rm assets_core_v2.tar.gz
cd assets
wget https://storage.googleapis.com/gibsonassets/dataset.tar.gz
tar -zxf dataset.tar.gz && rm dataset.tar.gz
### the commands above downloads assets data file and decpmpress it into gibson/assets folder
cd ../.. # back to GibsonEnv dir
docker build . -t gibson ### finish building inside docker
```
If the installation is successful, you should be able to run `docker run --runtime=nvidia -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v <host path to dataset folder>:/root/mount/gibson/gibson/assets/dataset gibson` to create a container. Note that we don't include
dataset files in docker image to keep our image slim, so you will need to mount it to the container when you start a container.


2. Or pull from our docker image
```bash
docker pull xf1280/gibson:0.3.1
```
#### Notes on deployment on a headless server

We have another docker file that supports deployment on a headless server and remote access with TurboVNC+virtualGL. 
You can build your own docker image with the docker file `Dockerfile_server`.
Instructions to run gibson on a headless server (requires X server running):

1. Install nvidia-docker2 dependencies following the starter guide.
2. Use `openssl req -new -x509 -days 365 -nodes -out self.pem -keyout self.pem` create `self.pem` file
3. `docker build -f Dockerfile_server -t gibson_server .` use the `Dockerfile_server` to build a new docker image that support virtualgl and turbovnc
4. `docker run --runtime=nvidia -ti --rm -e DISPLAY -v /tmp/.X11-unix/X0:/tmp/.X11-unix/X0 -v <host path to dataset folder>:/root/mount/gibson/gibson/assets/dataset -p 5901:5901 gibson_server`
in docker terminal, start `/opt/websockify/run 5901 --web=/opt/noVNC --wrap-mode=ignore -- vncserver :1 -securitytypes otp -otp -noxstartup` in background, potentially with `tmux`
5. Run gibson with `DISPLAY=:1 vglrun python <gibson example or training>`
6. Visit your `host:5901` and type in one time password to see the GUI.

If you don't have X server running, you can still run gibson, see [this guide](https://github.com/StanfordVL/GibsonEnv/wiki/Running-GibsonEnv-on-headless-server) for more details.


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
conda create -n py35 python=3.5 anaconda 
source activate py35 # the rest of the steps needs to be performed in the conda environment
conda install -c conda-forge opencv
pip install http://download.pytorch.org/whl/cu90/torch-0.3.1-cp35-cp35m-linux_x86_64.whl 
pip install torchvision==0.2.0
pip install tensorflow==1.3
```
Clone the repository, download data and build
```bash
git clone https://github.com/StanfordVL/GibsonEnv.git
cd GibsonEnv/gibson
wget https://storage.googleapis.com/gibsonassets/assets_core_v2.tar.gz
tar -zxf assets_core_v2.tar.gz
cd assets
wget https://storage.googleapis.com/gibsonassets/dataset.tar.gz
tar -zxf dataset.tar.gz
#### the commands above downloads assets data file and decpmpress it into gibson/assets folder
cd ../.. #back to GibsonEnv dir
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

First run `xhost +local:root` on your host machine to enable display. You may need to run `export DISPLAY=:0.0` first. After getting into the docker container with `docker run --runtime=nvidia -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v <host path to dataset folder>:/root/mount/gibson/gibson/assets/dataset gibson`, you will get an interactive shell. Now you can run a few demos. 

If you installed from source, you can run those directly using the following commands without using docker. 


```bash
python examples/demo/play_husky_nonviz.py ### Use ASWD keys on your keyboard to control a car to navigate around Gates building
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



Gibson Framerate
----
Below is Gibson Environment's framerate benchmarked on different platforms. Please refer to [fps branch](https://github.com/StanfordVL/GibsonEnv/tree/fps) for code to reproduce the results.
<table class="table">
  <tr>
    <th scope="row">Platform</th>
    <td colspan="3">Tested on Intel E5-2697 v4 + NVIDIA Tesla V100</td>
    <td colspan="3">Intel I7 7700 + NVIDIA GeForce GTX 1070Ti</td>
    <td colspan="3">Tested on Intel I7 6580k + NVIDIA GTX 1080Ti</td>
  </tr>
  <tr>
    <th scope="col">Resolution [nxn]</th>
    <th scope="col">128</th>
    <th scope="col">256</th>
    <th scope="col">512</th>
    <th scope="col">128</th>
    <th scope="col">256</th>
    <th scope="col">512</th>
    <th scope="col">128</th>
    <th scope="col">256</th>
    <th scope="col">512</th>
 </tr>
  <tr>
    <th scope="row">RGBD, pre network<code>f</code></th>
    <td>109.1</td>
    <td>58.5</td>
    <td>26.5</td>
    <td>134.3</td>
    <td>61.9</td>
    <td>20.5</td>
    <td>108.1</td>
    <td>60.0</td>
    <td>21.9</td>
  </tr>
  <tr>
    <th scope="row">RGBD, post network<code>f</code></th>
    <td>77.7</td>
    <td>30.6</td>
    <td>14.5</td>
    <td>80.9</td>
    <td>30.2</td>
    <td>8.5</td>
    <td>75.6</td>
    <td>35.9</td>
    <td>12.0</td>
  </tr>
  <tr>
    <th scope="row">RGBD, post small network<code>f</code></th>
    <td>87.4</td>
    <td>40.5</td>
    <td>21.2</td>
    <td>128.1</td>
    <td>61.9</td>
    <td>25.1</td>
    <td>98.8</td>
    <td>63.3</td>
    <td>27.3</td>
  </tr>
  <tr>
    <th scope="row">Depth only</th>
    <td>253.0</td>
    <td>197.9</td>
    <td>124.7</td>
    <td>362.8</td>
    <td>319.1</td>
    <td>183.0</td>
    <td>209.4</td>
    <td>141.2</td>
    <td>104.3</td>
  </tr>
  <tr>
    <th scope="row">Surface Normal only</th>
    <td>207.7</td>
    <td>129.7</td>
    <td>57.2</td>
    <td>282.5</td>
    <td>186.9</td>
    <td>81.6</td>
    <td>175.0</td>
    <td>110.5</td>
    <td>57.6</td>
  </tr>
  <tr>
    <th scope="row">Semantic only</th>
    <td>190.0</td>
    <td>144.2</td>
    <td>55.6</td>
    <td>304.5</td>
    <td>194.8</td>
    <td>73.7</td>
    <td>139.3</td>
    <td>134.1</td>
    <td>63.1</td>
  </tr>
  <tr>
    <th scope="row">Non-Visual Sensory</th>
    <td>396.1</td>
    <td>396.1</td>
    <td>396.1</td>
    <td>511.3</td>
    <td>495.6</td>
    <td>540.0</td>
    <td>260.3</td>
    <td>264.8</td>
    <td>250.0</td>
  </tr>
</table>


Web User Interface
----
When running Gibson, you can start a web user interface with `python gibson/utils/web_ui.py python gibson/utils/web_ui.py 5552`. This is helpful when you cannot physically access the machine running gibson or you are running on a headless cloud environment.

<img src=misc/web_ui.png width="600">

Rendering Semantics
----
<img src=misc/instance_colorcoding_semantics.png width="600">

Gibson can provide pixel-wise frame-by-frame semantic masks when the model is semantically annotated. As of now we have incorporated models from [Stanford 2D3DS](http://buildingparser.stanford.edu/) and [Matterport 3D](https://niessner.github.io/Matterport/) for this purpose, and we refer you to the original dataset's reference for the list of their semantic classes and annotations. 

For detailed instructions of rendering semantics in Gibson, see [semantic instructions](gibson/utils/semantics.md). In the light beta release, the space `17DRP5sb8fy` includes Matterport 3D style semantic annotation and `space7` includes Stanford 2D3DS style annotation. 

**Agreement**: If you choose to use the models from [Stanford 2D3DS](http://buildingparser.stanford.edu/) or [Matterport 3D](https://niessner.github.io/Matterport/) for rendering semantics, we ask you to agree to and sign their respective agreements. See [here](https://niessner.github.io/Matterport/) for Matterport3D and [here](https://github.com/alexsax/2D-3D-Semantics) for Stanford 2D3DS.


Robotic Agents
----

Gibson provides a base set of agents. See videos of these agents and their corresponding perceptual observation [here](http://gibsonenv.stanford.edu/agents/). 
<img src=misc/agents.gif>

To enable (optionally) abstracting away low-level control and robot dynamics for high-level tasks, we also provide a set of practical and ideal controllers for each agent.

| Agent Name     | DOF | Information      | Controller |
|:-------------: | :-------------: |:-------------: |:-------------| 
| Mujoco Ant      | 8   | [OpenAI Link](https://blog.openai.com/roboschool/) | Torque |
| Mujoco Humanoid | 17  | [OpenAI Link](https://blog.openai.com/roboschool/) | Torque |
| Husky Robot     | 4   | [ROS](http://wiki.ros.org/Robots/Husky), [Manufacturer](https://www.clearpathrobotics.com/) | Torque, Velocity, Position |
| Minitaur Robot  | 8   | [Robot Page](https://www.ghostrobotics.io/copy-of-robots), [Manufacturer](https://www.ghostrobotics.io/) | Sine Controller |
| JackRabbot      | 2   | [Stanford Project Link](http://cvgl.stanford.edu/projects/jackrabbot/) | Torque, Velocity, Position |
| TurtleBot       | 2   | [ROS](http://wiki.ros.org/Robots/TurtleBot), [Manufacturer](https://www.turtlebot.com/) | Torque, Velocity, Position |
| Quadrotor         | 6   | [Paper](https://repository.upenn.edu/cgi/viewcontent.cgi?referer=https://www.google.com/&httpsredir=1&article=1705&context=edissertations) | Position |


### Starter Code 

More demonstration examples can be found in `examples/demo` folder

| Example        | Explanation          |
|:-------------: |:-------------| 
|`play_ant_camera.py`|Use 1234567890qwerty keys on your keyboard to control an ant to navigate around Gates building, while RGB and depth camera outputs are also shown. |
|`play_ant_nonviz.py`| Use 1234567890qwerty keys on your keyboard to control an ant to navigate around Gates building.|
|`play_drone_camera.py`| Use ASWDZX keys on your keyboard to control a drone to navigate around Gates building, while RGB and depth camera outputs are also shown.|
|`play_drone_nonviz.py`| Use ASWDZX keys on your keyboard to control a drone to navigate around Gates building|
|`play_humanoid_camera.py`| Use 1234567890qwertyui keys on your keyboard to control a humanoid to navigate around Gates building. Just kidding, controlling humaniod with keyboard is too difficult, you can only watch it fall. Press R to reset. RGB and depth camera outputs are also shown. |
|`play_humanoid_nonviz.py`| Watch a humanoid fall. Press R to reset.|
|`play_husky_camera.py`| Use ASWD keys on your keyboard to control a car to navigate around Gates building, while RGB and depth camera outputs are also shown.|
|`play_husky_nonviz.py`| Use ASWD keys on your keyboard to control a car to navigate around Gates building|

More training code can be found in `examples/train` folder.

| Example        | Explanation          |
|:-------------: |:-------------| 
|`train_husky_navigate_ppo2.py`|   Use PPO2 to train a car to navigate down the hall way in Gates building, using RGBD input from the camera.|
|`train_husky_navigate_ppo1.py`|   Use PPO1 to train a car to navigate down the hall way in Gates building, using RGBD input from the camera.|
|`train_ant_navigate_ppo1.py`| Use PPO1 to train an ant to navigate down the hall way in Gates building, using visual input from the camera. |
|`train_ant_climb_ppo1.py`| Use PPO1 to train an ant to climb down the stairs in Gates building, using visual input from the camera.  |
|`train_ant_gibson_flagrun_ppo1.py`| Use PPO1 to train an ant to chase a target (a red cube) in Gates building. Everytime the ant gets to target(or time out), the target will change position.|
|`train_husky_gibson_flagrun_ppo1.py`|Use PPO1 to train a car to chase a target (a red cube) in Gates building. Everytime the car gets to target(or time out), the target will change position. |

ROS Configuration
---------

We provide examples of configuring Gibson with ROS [here](examples/ros/gibson-ros). We use turtlebot as an example, after a policy is trained in Gibson, it requires minimal changes to deploy onto a turtlebot. See [README](examples/ros/gibson-ros) for more details.




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
`obs` gives the observation of the robot. It is a dictionary with each component as a key value pair. Its keys are specified by user inside config file. E.g. `obs['nonviz_sensor']` is proprioceptive sensor data, `obs['rgb_filled']` is rgb camera data.

`rew` is the defined reward. `env_done` marks the end of one episode, for example, when the robot dies. 
`info` gives some additional information of this step; sometimes we use this to pass additional non-visual sensor values.

We mostly followed [OpenAI gym](https://github.com/openai/gym) convention when designing the interface of RL algorithms and the environment. In order to help users start with the environment quicker, we
provide some examples at [examples/train](examples/train). The RL algorithms that we use are from [openAI baselines](https://github.com/openai/baselines) with some adaptation to work with hybrid visual and non-visual sensory data.
In particular, we used [PPO](https://github.com/openai/baselines/tree/master/baselines/ppo1) and a speed optimized version of [PPO](https://github.com/openai/baselines/tree/master/baselines/ppo2).


Environment Configuration
=================
Each environment is configured with a `yaml` file. Examples of `yaml` files can be found in `examples/configs` folder. Parameters for the file is explained below. For more informat specific to Bullet Physics engine, you can see the documentation [here](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit).

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
|show_diagnostics | true/false  | show dignostics(including fps, robot position and orientation, accumulated rewards) overlaying on the RGB image |
|ui_num |2  | how many ui components to show, this should be length of ui_components. |
| ui_components | [RGB_FILLED, DEPTH]  | which are the ui components, choose from [RGB_FILLED, DEPTH, NORMAL, SEMANTICS, RGB_PREFILLED] |
|output | [nonviz_sensor, rgb_filled, depth]  | output of the environment to the robot, choose from  [nonviz_sensor, rgb_filled, depth]. These values are independent of `ui_components`, as `ui_components` determines what to show and `output` determines what the robot receives. |
|resolution | 512 | choose from [128, 256, 512] resolution of rgb/depth image |
|initial_orn | [0, 0, 3.14] | initial orientation (in radian) for navigating, the reference frame is world frame |
|speed : timestep | 0.01 | length of one physics simulation step in seconds(as defined in [Bullet](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit)). For example, if timestep=0.01 sec, frameskip=10, and the environment is running at 100fps, it will be 10x real time. Note: setting timestep above 0.1 can cause instability in current version of Bullet simulator since an object should not travel faster than its own radius within one timestep. You can keep timestep at a low value but increase frameskip to siumate at a faster speed. See [Bullet guide](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit) under "discrete collision detection" for more info.|
|speed : frameskip | 10 | how many timestep to skip when rendering frames. See above row for an example. For tasks that does not require high frequency control, you can set frameskip to larger value to gain further speed up. |
|mode | gui/headless  | gui or headless, if in a production environment (training), you need to turn this to headless. In gui mode, there will be visual output; in headless mode, there will be no visual output. |
|verbose |true/false  | show dignostics in terminal |
|fast_lq_render| true/false| if there is fast_lq_render in yaml file, Gibson will use a smaller filler network, this will render faster but generate slightly lower quality camera output. This option is useful for training RL agents fast. |

#### Making Your Customized Environment
Gibson provides a set of methods for you to define your own environments. You can follow the existing environments inside `gibson/core/envs`.

| Method name        | Usage           |
|:------------------:|:---------------------------|
| robot.render_observation(pose) | Render new observations based on pose, returns a dictionary. |
| robot.get_observation() | Get observation at current pose. Needs to be called after robot.render_observation(pose). This does not induce extra computation. |
| robot.get_position() | Get current robot position. |
| robot.get_orientation() | Get current robot orientation. |
| robot.eyes.get_position() | Get current robot perceptive camera position. |
| robot.eyes.get_orientation() | Get current robot perceptive camera orientation. | 
| robot.get_target_position() | Get robot target position. |
| robot.apply_action(action) | Apply action to robot. |  
| robot.reset_new_pose(pos, orn) | Reset the robot to any pose. |
| robot.dist_to_target() | Get current distance from robot to target. |

Goggles: transferring the agent to real-world
=================
Gibson includes a baked-in domain adaptation mechanism, named Goggles, for when an agent trained in Gibson is going to be deployed in real-world (i.e. operate based on images coming from an onboard camera). The mechanims is essentially a learned inverse function that alters the frames coming from a real camera to what they would look like if they were rendered via Gibson, and hence, disolve the domain gap. 

<img src=http://gibson.vision/public/img/figure4.jpg width="600">


**More details:** With all the imperfections in point cloud rendering, it has been proven difficult to get completely photo-realistic rendering with neural network fixes. The remaining issues make a domain gap between the synthesized and real images. Therefore, we formulate the rendering problem as forming a joint space ensuring a correspondence between rendered and real images, rather than trying to (unsuccessfuly) render images that are identical to real ones. This provides a deterministic pathway for traversing across these domains and hence undoing the gap. We add another network "u" for target image (I_t) and define the rendering loss to minimize the distance between f(I_s) and u(I_t), where "f" and "I_s" represent the filler neural network and point cloud rendering output, respectively (see the loss in above figure). We use the same network structure for f and u. The function u(I) is trained to alter the observation in real-world, I_t, to look like the corresponding I_s and consequently dissolve the gap. We named the u network goggles, as it resembles corrective lenses for the anget for deploymen in real-world. Detailed formulation and discussion of the mechanism can be found in the paper. You can download the function u and apply it when you deploy your trained agent in real-world.

In order to use goggle, you will need preferably a camera with depth sensor, we provide an example [here](examples/ros/gibson-ros/goggle.py) for Kinect. The trained goggle functions are stored in `assets/unfiller_{resolution}.pth`, and each one is paired with one filler function. You need to use the correct one depending on which filler function is used. If you don't have a camera with depth sensor, we also provide an example for RGB only [here](examples/demo/goggle_video.py).


Citation
=================

If you use Gibson Environment's software or database, please cite:
```
@inproceedings{xiazamirhe2018gibsonenv,
  title={Gibson {Env}: real-world perception for embodied agents},
  author={Xia, Fei and R. Zamir, Amir and He, Zhi-Yang and Sax, Alexander and Malik, Jitendra and Savarese, Silvio},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2018 IEEE Conference on},
  year={2018},
  organization={IEEE}
}
```
