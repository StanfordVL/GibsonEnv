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


from gibson.envs.env_modalities import CameraRobotEnv, SensorRobotEnv
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
from gibson import configs

MINITAUR_TIMESTEP  = 1.0/(4 * 22)
MINITAUR_FRAMESKIP = 4

ACTION_EPS = 0.01


tracking_camera = {
    'yaw': 20,  # demo: living room, stairs
    #'yaw'; 30,   # demo: kitchen
    'z_offset': 0.5,
    'distance': 1,
    'pitch': -20
    # 'pitch': -24  # demo: stairs
}

tracking_camera_top = {
    'yaw': 20,  # demo: living room, stairs
    #'yaw'; 30,   # demo: kitchen
    'z_offset': 0.5,
    'distance': 1,
    'pitch': -20
    # 'pitch': -24  # demo: stairs
}

class MinitaurNavigateEnv(CameraRobotEnv):
    """The gym environment for the minitaur.

    It simulates the locomotion of a minitaur, a quadruped robot. The state space
    include the angles, velocities and torques for all the motors and the action
    space is the desired motor angle for each motor. The reward function is based
    on how far the minitaur walks in 1000 steps and penalizes the energy
    expenditure.

    Attribute:
        self.human
        self.tracking_camera

    Interface:
        self.calc_rewards_and_done():   missing
    """
    metadata = {
        "render.modes": ["human", "rgb_array"],
        "video.frames_per_second": 50
    }

    def __init__(self,
                 distance_weight=1.0,
                 energy_weight=0.005,
                 shake_weight=0.0,
                 drift_weight=0.0,
                 distance_limit=float("inf"),
                 observation_noise_stdev=0.0,
                 leg_model_enabled=False,
                 hard_reset=False,
                 render=False,
                 env_randomizer=None,

                 ## Cambria specific
                 human=True, 
                 timestep=MINITAUR_TIMESTEP, 
                 frame_skip=MINITAUR_FRAMESKIP, 
                 is_discrete=False, 
                 mode="RGBD", 
                 use_filler=True, 
                 gpu_count=0, 
                 resolution="NORMAL"):
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
            render: Whether to render the simulation.
            env_randomizer: An EnvRandomizer to randomize the physical properties
                during reset().
        """
        self._time_step = 0.01
        self._observation = []

        #self._is_render = render
        self._last_base_position = [0, 0, 0]
        self._distance_weight = distance_weight
        self._energy_weight = energy_weight
        self._drift_weight = drift_weight
        self._shake_weight = shake_weight
        self._distance_limit = distance_limit
        self._observation_noise_stdev = observation_noise_stdev
        self._action_bound = 1
        self._leg_model_enabled = leg_model_enabled
        
        self._env_randomizer = env_randomizer
        self._time_step = timestep

        target_orn, target_pos   = configs.TASK_POSE[configs.NAVIGATE_MODEL_ID]["navigate"][-1]
        initial_orn, initial_pos = configs.TASK_POSE[configs.NAVIGATE_MODEL_ID]["navigate"][0]        
        self.robot = Minitaur(time_step=self._time_step, 
                              initial_pos=initial_pos,
                              initial_orn=initial_orn)

        if self._env_randomizer is not None:
            self._env_randomizer.randomize_env(self)

        self._last_base_position = [0, 0, 0]
        self._objectives = []        
        self.viewer = None
        self._hard_reset = hard_reset  # This assignment need to be after reset()


        self.human = human
        self.model_id = configs.NAVIGATE_MODEL_ID
        self.timestep = timestep
        self.frame_skip = frame_skip
        self.resolution = resolution
        self.tracking_camera = tracking_camera

        CameraRobotEnv.__init__(
            self, 
            mode, 
            gpu_count, 
            scene_type="building", 
            use_filler=use_filler)
        self.total_reward = 0
        self.total_frame = 0


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
        if self._leg_model_enabled:
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
        return self.is_fallen() or distance > self._distance_limit

    def _reward(self, action=None, debugmode=False):
        current_base_position = self.robot.GetBasePosition()
        forward_reward = current_base_position[0] - self._last_base_position[0]
        drift_reward = -abs(current_base_position[1] - self._last_base_position[1])
        shake_reward = -abs(current_base_position[2] - self._last_base_position[2])
        self._last_base_position = current_base_position
        energy_reward = np.abs(
            np.dot(self.robot.GetMotorTorques(),
                   self.robot.GetMotorVelocities())) * self._time_step
        reward = (
            self._distance_weight * forward_reward -
            self._energy_weight * energy_reward + self._drift_weight * drift_reward
            + self._shake_weight * shake_reward)
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
        if self._observation_noise_stdev > 0:
          observation += (np.random.normal(
              scale=self._observation_noise_stdev, size=observation.shape) *
                          self.robot.GetObservationUpperBound())
        return observation
