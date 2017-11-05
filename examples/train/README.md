## Cambria RL Training Starter Code
Entry point for RL training with different algorithms


### How to Use
Every training code support different input modes: rgb|rgbd|sensor|depth, filled|unfilled
To enable different modes when training, use flats `--mode`

For example
```python
python train_ant_camera_ppo.py --mode RGB
```

To run the code on aws, use conda 3 python environment
```shell
source activate universe3
```

### Environment Modality
Because Cambira environment outputs both sensor and camera data at the same time, we are forced to break from openAI gym interface, by returning camera data as default observation, and sqeezing sensor data inside meta.

By doing this, we can fix the observation dimensions from start to end, and you don't need to change network architecture when using different modalities. Please tweak the learning code accordingly to accomodate the changes. Here's an example.
```python
env = HuskyCameraEnv(mode="RGBD")
obs, done, reward, meta = env.step(action)
obs             ## RGBD output
meta['sensor']  ## sensor output

## Learning code, e.g. DQN
camera_replay_buffer.add(obs)
sensor_replay_buffer.add(meta['sensor'])

```

Note that based on environment mode, Cambria selectly fills up obs, to achieve the best performance. Certain observations (e.g. `obs[:3]` in `DEPTH` mode) is set 0.

|Env  Mode  |obs[0] |obs[1] |obs[2] |obs[3] |meta['sensor'] |
|---        |---    |---    |---    |---    |---            |
|RGB        |R      |G      |B      |D      |Sensor Output  |
|RGBD       |R      |G      |B      |D      |Sensor Output  |
|DEPTH      | -     |-      |-      |D      |Sensor Output  |
|SENSOR     | -     |-      |-      |D      |Sensor Output  |

In OpenAI gym, you can get environment's observation dimension by running `env.observation_space.shape`. Here we separate camera output from sensor output. Their dimensions can be queried using:
```python
shape = env.sensor_space.shape          ## Sensor dimension 
x_senso = tf.placeholder(tf.float32, [None] + list(shape))

shape = env.observation_space.shape     ## Observation dimension
x_camera = tf.placeholder(tf.float32, [None] + list(shape))
```


### Ready-to-use Sample Code
The following training schemes are currently supported.

PPO1

* train\_ant\_camera\_ppo (cnn\_policy)
* train\_husky\_camera\_ppo (cnn\_policy)

DQN

* train\_husky\_flagrun\_dqn
* train\_husky\_camera\_dqn


### Under Development

A2C

* train\_husky\_camera\_a2c
* train\_husky\_sensor\_a2c

### What are you actually training the agents for?
**Husky camera** : train the husky to navigate 20 meters along Gates 1F hallway. Starting location: end of hallway near copy room, target location near Silvio's office. Rich reward: negative delta distance. 
Goal: navigate and avoid collision using only RGB input. In the future we want to add collision penalty and sparse reward.

**Husky flatrun**: train the husky to chase randomized red flags. Rich reward: negative delta distance. Goal: find randomly distributed red flags using only RGB input.

(to be continued)



### Tuning Environment
Beside tuning algorithm, you might want to also fine tune the environment.

THis is because reward function, state function, and done condition of the environment might not suit the specific task that you're training. For example, you might want to tune collision penalty score to make husky better learn to navigate, or you might not want to immediately terminate the episode if ant's body scratches the floor while climbing stairs. To do these environment tuning efficiently, contact hzyjerry or fxia.



### Modified from Baseline
tf_utils.get_session() -> utils.make_gpu_session(): multi gpu support
tf_utils.deepq -> deepq: multi gpu support, mode for depth vs rgbd
The customized functions:
    