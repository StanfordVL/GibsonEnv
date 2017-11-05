## Cambria RL Training Starter Code
Entry point for RL training with different algorithms


### How to Use
Every training code support different input modes: RGB|RGBD|SENSOR|DEPTH, FILLED|UNFILLED
Use flag `--mode` to enable different modes; use `--disable_filler` to disable filler (default on); use `--human` to turn on visualzation (not supported on aws)

To run the code on aws
```shell
source activate universe3                  ## Activate python3 on aws
python train_ant_navigate_ppo.py --mode RGBD --disable_filler
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

* train\_husky\_navigate\_ppo1 (mode: RGB/RGBD/DEPTH/GREY/SENSOR)

DQN

* train\_husky\_flagrun\_dqn (mode: only SENSOR)
* train\_husky\_navigate\_dqn (mode: RGB/RGBD/DEPTH/GREY/SENSOR)


### Under Development

A2C

* train\_ant\_camera\_ppo1 (cnn\_policy)
* train\_husky\_camera\_a2c
* train\_husky\_sensor\_a2c

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
    