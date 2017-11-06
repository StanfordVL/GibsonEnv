## Cambria RL Training Starter Code
Entry point for RL training with different algorithms


### How to Use
Every training code support different input modes: RGB|RGBD|SENSOR|DEPTH, FILLED|UNFILLED
Use flag `--mode` to enable different modes; use `--disable_filler` to disable filler (default on); use `--human` to turn on visualzation (not supported on aws)

To run the code on aws
```shell
source activate universe3                                            ## Activate python3 on aws
python train_husky_navigate_ppo1.py --mode RGBD --disable_filler     ## Modality: RGBD + SENSOR
python train_husky_navigate_ppo1.py --mode SENSOR --disable_filler   ## Modality: RGBD + SENSOR
```

Choosing different mode significantly affects the time. Speed Profiling (resolution=256x256):

|Mode       |use_filler =True   | use_filler=False  |
|---        |---                |---                |
|RGB/RGBD   |58 fps             |100 fps            |
|DEPTH      |-                  |320 fps            |
|SENSOR     |-                  |320 fps            |

### Environment Modality
Because Cambira environment outputs both sensor and camera data at the same time, we are forced to break from openAI gym interface, by returning camera data as default observation, and sqeezing sensor data inside meta.

By doing this, we can fix the observation dimensions from start to end, and you don't need to change network architecture when using different modalities. Please tweak the learning code accordingly to accomodate the changes. Here's an example.
```python
env = HuskyNavigateEnv(mode="RGBD")
obs, done, reward, meta = env.step(action)
obs                                         ## RGBD output
meta['sensor']                              ## sensor output

## Learning code, e.g. DQN
camera_replay_buffer.add(obs)
sensor_replay_buffer.add(meta['sensor'])

```

Note that based on environment mode, Cambria selectly fills up obs, to achieve the best performance. Certain observations (e.g. `obs[:3]` in `DEPTH` mode) is set 0. You are responsible for handling these cases.

|Env  Mode  |obs[0] |obs[1] |obs[2] |obs[3] |meta['sensor'] |
|---        |---    |---    |---    |---    |---            |
|RGB        |R      |G      |B      |D      |Sensor Output  |
|RGBD       |R      |G      |B      |D      |Sensor Output  |
|DEPTH      | -     |-      |-      |D      |Sensor Output  |
|SENSOR     | -     |-      |-      |-      |Sensor Output  |

In OpenAI gym, you can get environment's observation dimension by running `env.observation_space.shape`. Here we separate camera output from sensor output. Their dimensions can be used for building NN architecture:
```python
shape = env.sensor_space.shape              ## Sensor dimension 
x_sensor = tf.placeholder(tf.float32, [None] + list(shape))

shape = env.observation_space.shape         ## Observation dimension
x_camera = tf.placeholder(tf.float32, [None] + list(shape))
```


### Ready-to-use Sample Code
The following training schemes are currently supported.

PPO1

* train\_husky\_navigate\_ppo1 (mode: <kbd>RGB</kbd>/<kbd>RGBD</kbd>/<kbd>DEPTH</kbd>/<kbd>GREY</kbd>/<kbd>SENSOR</kbd>)

DQN

* train\_husky\_flagrun\_dqn (mode: only <kbd>SENSOR</kbd>)
* train\_husky\_navigate\_dqn (mode: <kbd>RGB</kbd>/<kbd>RGBD</kbd>/<kbd>DEPTH</kbd>/<kbd>GREY</kbd>/<kbd>SENSOR</kbd>)


### Under Development

A2C

* train\_ant\_camera\_ppo1 (cnn\_policy)
* train\_ant\_sensor\_ppo
* train\_ant\_sensor
* train\_husky\_camera\_a2c
* train\_husky\_sensor\_a2c
* train\_humanoid\_sensor


### What are you actually training the agents for?
**Husky Navigate** : train the husky to navigate 20 meters along Gates 1F hallway. Sensor/Camera input. Starting location: end of hallway near copy room, target location near Silvio's office. Rich reward: negative delta distance. 
Goal: navigate and avoid collision using only RGB input. In the future we want to add collision penalty and sparse reward.

**Husky Flagrun**: train the husky to chase randomized red flags. Sensor input only. Rich reward: negative delta distance. Goal: find randomly distributed red flags using only RGB input.

(to be continued)



### Tuning Environment
Beside tuning algorithm, you might want to also fine tune the environment.

THis is because reward function, state function, and done condition of the environment might not suit the specific task that you're training. For example, you might want to tune collision penalty score to make husky better learn to navigate, or you might not want to immediately terminate the episode if ant's body scratches the floor while climbing stairs. To do these environment tuning efficiently, contact hzyjerry or fxia.



### Modified from Baseline
`tf_utils.get_session()` -> `utils.make_gpu_session()`: multi gpu support


`tf_utils.deepq` -> `deepq: multi gpu support`, mode for rgbd vs sensor only.
    
    
    
## Remote Training on AWS
 
### Part 1 Running Environment on AWS

Adapted from https://stackoverflow.com/questions/19856192/run-opengl-on-aws-gpu-instances-with-centos

### Set up (only needs to do once)
sudo apt install xinit
start x
sudo apt-get install mesa-utils xserver-xorg libglu1-mesa-dev freeglut3-dev mesa-common-dev libxmu-dev libxi-dev -y
sudo nvidia-xconfig -a --use-display-device=None --virtual=1280x1024

### Start X session

Execute the follow code
```bash
sudo /usr/bin/X :0
```

If you get something like this, that means the X server is running
```bash
Using X configuration file: "/etc/X11/xorg.conf".
Backed up file '/etc/X11/xorg.conf' as '/etc/X11/xorg.conf.backup'
New X configuration file written to '/etc/X11/xorg.conf'

### Or this
X.Org X Server 1.18.4
Release Date: 2016-07-19
X Protocol Version 11, Revision 0
...
```

Other wise, DISPLAY=:0 is occupied
```bash
sudo pkill Xorg    ## be careful because if there's Xorg process running on other GPU, this will kill them all
sudo /usr/bin/X :0

## sudo /usr/bin/X :1  ## If you want to use DISPLAY=:1 instead of 0
```

More sanity checks to make sure X server is started
```bash
DISPLAY=:0 glxinfo
nvidia-smi
DISPLAY=:0 glxgears
```

Now you can start a separate ssh session, log in to aws, and run learning code:
```bash
DISPLAY=:0 CUDA_VISIBLE_DEVICES=0 python examples/train/train_husky_navigate_ppo1.py
```


Note: expected logs when running environment (you're fine)
```bash
X11: RandR gamma ramp support seems broken
10008
Error callbackg Environment ] |                                         | (ETA:  --:--:--) 
X11: RandR monitor support seems broken
10008
Compiling shader : ./StandardShadingRTT.vertexshader
Compiling shader : ./StandardShadingRTT.fragmentshader
...
```

Note: abnormal logs when running environment (something's wrong)
```
GPUAssert Error:.....
```

### Part 2 Visualizing Environment on AWS
```bash
## Open one terminal
ssh -i universe.pem ubuntu@ip-address
sudo xinit

## Open a second terminal
ssh -i universe.pem ubuntu@ip-address
/opt/TurboVNC/bin/vncserver
## This will tell you 'TurboVNC: ip-xxx:ID (ubuntu)' started on display ip-xxx:ID
## Make note of the ID
DISPLAY=:1 /opt/VirtualGL/bin/vglrun glxgears                   ## If ID is 1
# DISPLAY=:1 /opt/VirtualGL/bin/vglrun python ......

## Open a third ssh session by command:
ssh -L 5901:localhost:5901 -i universe.pem ubuntu@ip-addresss   ## If ID is 1
## Open local VNC Viewer: localhost:5901                        ## If ID is 1
## password: 123456
```
