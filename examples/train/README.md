## Cambria RL Training Starter Code
Entry point for RL training with different algorithms

### Ready-to-use Sample Code
The following training schemes are currently supported.

PPO1

* train\_ant\_camera\_ppo (cnn\_policy)
* train\_husky\_camera\_ppo (cnn\_policy)

DQN

* train\_husky\_flagrun (sensor input)

### Under Development
DQN
	
* train\_husky\_camera\_dqn

A2C

* train\_husky\_camera\_a2c
* train\_husky\_sensor\_a2c

### How to Use
Every training code support different input modes: rgb|rgbd|sensor|depth, filled|unfilled
To enable different modes when training, use flats `--mode`

For example
```python
python train_ant_camera_ppo.py --mode rgb
```

To run the code on aws, use conda 3 python environment
```shell
source activate universe3
```


### What are you actually training the agents for?
**Husky camera** : train the husky to navigate 20 meters along Gates 1F hallway. Starting location: end of hallway near copy room, target location near Silvio's office. Rich reward: negative delta distance. 
Goal: navigate and avoid collision using only RGB input. In the future we want to add collision penalty and sparse reward.

**Husky flatrun**: train the husky to chase randomized red flags. Rich reward: negative delta distance. Goal: find randomly distributed red flags using only RGB input.

(to be continued)


### Tuning Environment
Beside tuning algorithm, you might want to also fine tune the environment.

THis is because reward function, state function, and done condition of the environment might not suit the specific task that you're training. For example, you might want to tune collision penalty score to make husky better learn to navigate, or you might not want to immediately terminate the episode if ant's body scratches the floor while climbing stairs. To do these environment tuning efficiently, contact hzyjerry or fxia.

The original openAI baselines do not have very good GPU support. Here we include customized cnn_policy.py and tf_util.py to support single GPU support, you can overrite these files with your own configurations.

The customized functions:
	tf_utils.make_gpu_session()
