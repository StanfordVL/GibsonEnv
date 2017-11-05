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

### Environment Modality

obs, done, reward, meta = env.step(action)

|Env  Mode|Obs[0] Obs[1] Obs[2] Obs[3] Meta['sensor']
|RGB  |R 	G 	 	B 		D 		SENSOR
|RGBD |R 	G 		B 		D 		SENSOR
|DEPTH| - 	- 	 	- 		D  		SENSOR
|SENSOR| - 	- 		- 		D 		SENSOR

Sensor dimension: env.sensor_space.shape
Observation dimension: env.observation_space.shape


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
	