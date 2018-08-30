from gibson.data.datasets import ViewDataSet3D, get_model_path
from gibson.envs.env_bases import BaseEnv
from gibson.envs.env_modalities import *
from gibson.envs.env_ui import *
import gibson
from gym import error
from gym.utils import seeding
from transforms3d import quaternions
import pybullet as p
import pybullet_data
from tqdm import *
import subprocess, os, signal
import numpy as np
import sys
import zmq
import pygame
from pygame import surfarray
import socket
import shlex
import gym
import cv2
import os.path as osp
import os
from PIL import Image

from transforms3d.euler import euler2quat, euler2mat
from transforms3d.quaternions import quat2mat, qmult
import transforms3d.quaternions as quat


class Agent(Object):
    """Wrapper for robot
    Attributes:
        @robot
        @scale_up
        @windowsz
        @_render_width
        @_render_height
    Functionalities:
        (1) Register individual robot
        (2) Start individual UI
        (3) Configure individual rendering channel
    """
    def __init__(self, robot, env):
        self.robot = robot
        self.env = env
        self.robot.env = env
        if self.robot.resolution == 64:
            self.windowsz = 64
            self.scale_up = 4
        elif self.robot.resolution == 128:
            self.windowsz = 128
            self.scale_up = 4
        elif self.robot.resolution == 256:
            self.windowsz = 256
            self.scale_up = 2
        else:
            self.windowsz = 512
            self.scale_up = 1
        self._render_width = self.windowsz
        self._render_height = self.windowsz
        assert (self.robot.resolution in [64,128,256,512]), "Robot resolution must be in 64/128/256/512"

        self.action_space = self.robot.action_space
        ## Robot's eye observation, in sensor mode black pixels are returned
        self.observation_space = self.robot.observation_space
        self.sensor_space = self.robot.sensor_space
        # seed for robot
        self.robot.np_random = env.np_random

        self.setup_rendering_camera()


    def setup_rendering_camera(self):
        if self.test_env or not self._require_camera_input:
            return
        self.r_camera_rgb = None     ## Rendering engine
        self.r_camera_mul = None     ## Multi channel rendering engine
        self.r_camera_dep = None
        self.r_camera_semt = None
        self.check_port_available()
        self.setup_camera_multi()
        self.setup_camera_rgb()
        if 'semantics' in self.config["output"]:
            self.setup_semantic_parser()
        
        ui_map = {
            1: OneViewUI,
            2: TwoViewUI,
            3: ThreeViewUI,
            4: FourViewUI,
        }

    def apply_action(a):
        self.robot.apply_action(a)


class MultiAgentEnv(SemanticRobotEnv):
    def __init__(self, config, gpu_count, scene_type, tracking_camera):
        SemanticRobotEnv.__init__(self, config, gpu_count, scene_type, tracking_camera)
        assert config['num_agent'] > 0    
        if self.gui:
            if config['num_agent'] > 1:
                self.screen_arr = np.zeros([config[num_agent], 612, 512, 3])    
            else:
                self.screen_arr = np.zeros([612, 512, 3])
        self.agents = [None] * config['num_agents']

    def agent_introduce(self, robots):
        """Introduce all robots as agents"""
        assert(len(robots) == config['num_agent']), "Incorrect number of robots introduced. Need {} but given{}".format(config['num_agent'], len(robots))
        for i, robot in enumerate(robots):
            self.agents[i] = Agent(robot, self)
        self._robot_introduced = True

    def step(self, actions, tag=True):
        results = []
        assert(len(actions) == len(self.agents))
        for i, act in enumerate(actions):
            self.agents[i].apply_action(act)
        self.scene.global_step()

        self.rewards = []
        self.dones = []
        for i, agent in enumerate(self.agents):
            self.rewards.append(self.agents[i]._reward(a))
            self.dones.append(self.agents[i]._termination())

        self.eps_reward += sum(self.rewards)

        if self.gui:
            pos = self.robot._get_scaled_position()
            orn = self.robot.get_orientation()
            pos = (pos[0], pos[1], pos[2] + self.tracking_camera['z_offset'])
            p.resetDebugVisualizerCamera(self.tracking_camera['distance'],self.tracking_camera['yaw'], self.tracking_camera['pitch'],pos)

        eye_pos, eye_quat = self.get_eye_pos_orientation()
        pose = [eye_pos, eye_quat]
        observations = self.render_observations(pose)

        episode = None
        if done:
            episode = {'r': self.reward,
                       'l': self.nframe}
            debugmode = 0
            if debugmode:
                print("return episode:", episode)
        #return observations, sum(self.rewards), bool(done), dict(eye_pos=eye_pos, eye_quat=eye_quat, episode=episode)


        results.append(SemanticRobotEnv.step(self, action))
        return 
