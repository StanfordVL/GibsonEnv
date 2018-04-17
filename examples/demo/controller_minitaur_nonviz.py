from gibson.envs.minitaur_env import MinitaurNavigateEnv
from gibson.utils.play import play
import argparse
import math
import os

config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'configs', 'minitaur_navigate_nonviz.yaml')

class SineStandController:
  """An example of minitaur standing and squatting on the floor.
  """
  def __init__(self):
    self.sum_reward = 0
    self.steps = 5000
    self.speed = 3
    self.amplitude = 0.5
    self.step_count = 0
    self.time_step = 0.001
    
  def act(self, env):
    if self.step_count < self.steps:
      t = self.step_count * self.time_step
      action = [math.sin(self.speed * t) * self.amplitude + math.pi / 2] * 8
      observation, _, _, _ = env.step(action)
      self.step_count += 1
    else:
      env.reset()
      self.step_count = 0


class SinePolicyController:
  """An example of minitaur walking with a sine gait."""
  # Leg model enabled
  # Accurate model enabled
  # pd control enabled
  def __init__(self):
    self.sum_reward = 0
    self.steps = 20000
    self.amplitude_1_bound = 0.7
    self.amplitude_2_bound = 0.7
    self.speed = 40

    self.step_count = 0
    self.time_step = 0.001

  def act(self, env):
    if self.step_count < self.steps:
      t = self.step_count * self.time_step

      amplitude1 = self.amplitude_1_bound
      amplitude2 = self.amplitude_2_bound
      steering_amplitude = 0
      if t < 10:
        steering_amplitude = 0.1
      elif t < 20:
        steering_amplitude = -0.1
      else:
        steering_amplitude = 0

      # Applying asymmetrical sine gaits to different legs can steer the minitaur.
      a1 = math.sin(t * self.speed) * (amplitude1 + steering_amplitude)
      a2 = math.sin(t * self.speed + math.pi) * (amplitude1 - steering_amplitude)
      a3 = math.sin(t * self.speed) * amplitude2
      a4 = math.sin(t * self.speed + math.pi) * amplitude2
      #action = [0] * 8
      action = [a1, a2, a2, a1, a3, a4, a4, a3] 
      _, reward, done, _ = env.step(action)
      self.sum_reward += reward
      if done:
        env.reset()
        self.step_count = 0
      self.step_count += 1
    else:
      self.step_count = 0
      env.reset()

class SinePolicyLeftController:
  """An example of minitaur walking with a sine gait."""
  # Leg model enabled
  # Accurate model enabled
  # pd control enabled
  def __init__(self):
    self.sum_reward = 0
    self.steps = 20000
    self.amplitude_1_bound = 0.7
    self.amplitude_2_bound = 0.7
    self.speed = 40

    self.step_count = 0
    self.time_step = 0.001

  def act(self, env):
    if self.step_count < self.steps:
      t = self.step_count * self.time_step

      amplitude1 = self.amplitude_1_bound
      amplitude2 = self.amplitude_2_bound
      steering_amplitude = 0
      if t < 10:
        steering_amplitude = 0.1
      elif t < 20:
        steering_amplitude = -0.1
      else:
        steering_amplitude = 0

      # Applying asymmetrical sine gaits to different legs can steer the minitaur.
      a1 = math.sin(t * self.speed) * (amplitude1 + steering_amplitude)
      a2 = math.sin(t * self.speed + math.pi) * (amplitude1 - steering_amplitude)
      a3 = math.sin(t * self.speed) * amplitude2
      a4 = math.sin(t * self.speed + math.pi) * amplitude2
      a12 = math.sin(t * self.speed) * (amplitude1 + steering_amplitude)
      a32 = math.sin(t * self.speed) * amplitude2
      
      #action = [0] * 8
      #action = [a1, a2, a3, a4, a2, a1, a4, a3]   #  
      #action = [a1, -a2, a2, -a1, -a3, a4, -a4, a3] # back walking
      action = [a1, a2, a2, a1, a3, a4, a4, a3] # back walking
      _, reward, done, _ = env.step(action)
      self.sum_reward += reward
      if done:
        env.reset()
        self.step_count = 0
      self.step_count += 1
    else:
      self.step_count = 0
      env.reset()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=config_file)
    args = parser.parse_args()

    env = MinitaurNavigateEnv(config = args.config)
    controller = SinePolicyLeftController()

    env.reset()
    while True:
      controller.act(env)