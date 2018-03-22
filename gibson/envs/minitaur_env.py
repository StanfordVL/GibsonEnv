'''
Customized Minitaur Environment for Cambria
Author: Zhiyang He, Stanford University
Original: Pybullet

Note that in the original pybullet environment, major difference exist in simulation
accuracy. 
Original:
    Solver iterations: 300 (Major difference)
    Time step: 1/100.0
    Action repeat: 1
Original Accurate:
    Solver iterations: 60
    Time step: 1/500.0
    Action repeat: 5
Current:
    Solver iterations: 5
    Time step: 1/88.0
    Action repeat: 4
'''


from gibson.envs.env_modalities import CameraRobotEnv, BaseRobotEnv
from gibson.envs.env_bases import *
from gibson.core.physics.drivers.minitaur import Minitaur
import os, inspect
import math
import time
import gym
from gym import spaces
from gym.utils import seeding
import gibson
import numpy as np
import pybullet

MINITAUR_TIMESTEP  = 1.0/(4 * 22)
MINITAUR_FRAMESKIP = 4

ACTION_EPS = 0.01


tracking_camera = {
    'yaw': 40,
    'z_offset': 0.5,
    'distance': 1,
    'pitch': -20
}


class MinitaurNavigateEnv(CameraRobotEnv):
    """The gym environment for the minitaur.

    It simulates the locomotion of a minitaur, a quadruped robot. The state space
    include the angles, velocities and torques for all the motors and the action
    space is the desired motor angle for each motor. The reward function is based
    on how far the minitaur walks in 1000 steps and penalizes the energy
    expenditure.
    """
    distance_weight = 1.0
    energy_weight = 0.005
    shake_weight = 0.0
    drift_weight = 0.0
    distance_limit = float("inf")
    observation_noise_stdev = 0.0
    leg_model_enabled = False
    action_bound = 1
    env_randomizer = None
    human = True 
    is_discrete = False 
    mode = "RGBD" 
    use_filler = True
    hard_reset = False

    def __init__(self, config, gpu_count=0):
        """Initialize the minitaur gym environment.
        Args:
            distance_weight: The weight of the distance term in the reward.
            energy_weight: The weight of the energy term in the reward.
            shake_weight: The weight of the vertical shakiness term in the reward.
            drift_weight: The weight of the sideways drift term in the reward.
            distance_limit: The maximum distance to terminate the episode.
            observation_noise_stdev: The standard deviation of observation noise.
            leg_model_enabled: Whether to use a leg motor to reparameterize the action
                space.
            hard_reset: Whether to wipe the simulation and load everything when reset
                is called. If set to false, reset just place the minitaur back to start
                position and set its pose to initial configuration.
            env_randomizer: An EnvRandomizer to randomize the physical properties
                during reset().
        """
    
        self.config = self.parse_config(config)
        assert(self.config["envname"] == self.__class__.__name__ or self.config["envname"] == "TestEnv")

        CameraRobotEnv.__init__(self, self.config, gpu_count, 
                                scene_type="building",
                                tracking_camera=tracking_camera)

        self.robot_introduce(Minitaur(self.config, env=self))
        self.scene_introduce()
        self.gui = self.config["mode"] == "gui"
        self.total_reward = 0
        self.total_frame = 0


        self._observation = []
        self._last_base_position = [0, 0, 0]
        self._action_bound = self.action_bound
        
        self._env_randomizer = self.env_randomizer        
        if self._env_randomizer is not None:
            self._env_randomizer.randomize_env(self)

        self._objectives = []        
        self.viewer = None
        

    def set_env_randomizer(self, env_randomizer):
        self._env_randomizer = env_randomizer

    def configure(self, args):
        self._args = args

    #def _reset(self):
        #if self._env_randomizer is not None:
        #    self._env_randomizer.randomize_env(self)

        #self._last_base_position = [0, 0, 0]
        #self._objectives = []
        
        #if not self._torque_control_enabled:
        #  for _ in range(1 / self.timestep):
        #    if self._pd_control_enabled or self._accurate_motor_model_enabled:
        #    self.robot.ApplyAction([math.pi / 2] * 8)
        #    pybullet.stepSimulation()
        #return self._noisy_observation()


    def _transform_action_to_motor_command(self, action):
        if self.leg_model_enabled:
            for i, action_component in enumerate(action):
                if not (-self._action_bound - ACTION_EPS <= action_component <= self._action_bound + ACTION_EPS):
                    raise ValueError("{}th action {} out of bounds.".format(i, action_component))
            action = self.robot.ConvertFromLegModel(action)
        return action
    
    def _step(self, action):
        """Step forward the simulation, given the action.

        Args:
          action: A list of desired motor angles for eight motors.

        Returns:
          observations: The angles, velocities and torques of all motors.
          reward: The reward for the current state-action pair.
          done: Whether the episode has ended.
          info: A dictionary that stores diagnostic information.

        Raises:
          ValueError: The action dimension is not the same as the number of motors.
          ValueError: The magnitude of actions is out of bounds.
        """

        action = self._transform_action_to_motor_command(action)
        #for _ in range(self._action_repeat):
        #  self.robot.ApplyAction(action)
        #  pybullet.stepSimulation()
        return CameraRobotEnv._step(self, action)


    def calc_rewards_and_done(self, action, state):
        ## TODO (hzyjerry): make use of action, state
        done = self._termination(state)
        rewards = self._rewards(a)
        #return reward, False
        return rewards, done


    def get_minitaur_motor_angles(self):
        """Get the minitaur's motor angles.

        Returns:
          A numpy array of motor angles.
        """
        return self.robot.GetMotorAngles()
    
    def get_minitaur_motor_velocities(self):
        """Get the minitaur's motor velocities.

        Returns:
          A numpy array of motor velocities.
        """
        return self.robot.GetMotorVelocities()

    def get_minitaur_motor_torques(self):
        """Get the minitaur's motor torques.

        Returns:
          A numpy array of motor torques.
        """
        return self.robot.GetMotorTorques()

    def get_minitaur_base_orientation(self):
        """Get the minitaur's base orientation, represented by a quaternion.

        Returns:
          A numpy array of minitaur's orientation.
        """
        return self.robot.GetBaseOrientation()

    def is_fallen(self):
        """Decide whether the minitaur has fallen.

        If the up directions between the base and the world is larger (the dot
        product is smaller than 0.85) or the base is very low on the ground
        (the height is smaller than 0.13 meter), the minitaur is considered fallen.

        Returns:
          Boolean value that indicates whether the minitaur has fallen.
        """
        orientation = self.robot.GetBaseOrientation()
        rot_mat = pybullet.getMatrixFromQuaternion(orientation)
        local_up = rot_mat[6:]
        pos = self.robot.GetBasePosition()
        return (np.dot(np.asarray([0, 0, 1]), np.asarray(local_up)) < 0.85 or
                pos[2] < 0.13)

    def _termination(self, state=None, debugmode=False):
        position = self.robot.GetBasePosition()
        distance = math.sqrt(position[0]**2 + position[1]**2)
        return self.is_fallen() or distance > self.distance_limit

    def _rewards(self, action=None, debugmode=False):
        a = action
        current_base_position = self.robot.GetBasePosition()
        forward_reward = current_base_position[0] - self._last_base_position[0]
        drift_reward = -abs(current_base_position[1] - self._last_base_position[1])
        shake_reward = -abs(current_base_position[2] - self._last_base_position[2])
        self._last_base_position = current_base_position
        energy_reward = np.abs(
            np.dot(self.robot.GetMotorTorques(),
                   self.robot.GetMotorVelocities())) * self.timestep
        reward = (
            self.distance_weight * forward_reward -
            self.energy_weight * energy_reward + self.drift_weight * drift_reward
            + self.shake_weight * shake_reward)
        self._objectives.append(
            [forward_reward, energy_reward, drift_reward, shake_reward])
        return [reward, ]

    def get_objectives(self):
        return self._objectives

    def _get_observation(self):
        self._observation = self.robot.GetObservation()
        return self._observation

    def _noisy_observation(self):
        self._get_observation()
        observation = np.array(self._observation)
        if self.observation_noise_stdev > 0:
          observation += (np.random.normal(
              scale=self.observation_noise_stdev, size=observation.shape) *
                          self.robot.GetObservationUpperBound())
        return observation
