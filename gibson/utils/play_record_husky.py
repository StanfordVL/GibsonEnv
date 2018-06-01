import gym
#import pygame
import sys
import time
import matplotlib
import time
import pygame
import pybullet as p
from gibson.core.render.profiler import Profiler
from collections import deque
from threading import Thread
from tqdm import tqdm
import numpy as np
import random

def display_arr(screen, arr, video_size, transpose):
    arr_min, arr_max = arr.min(), arr.max()
    arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
    pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1) if transpose else arr)
    pyg_img = pygame.transform.scale(pyg_img, video_size)
    screen.blit(pyg_img, (0,0))

base_action_omage = np.array([0.00,0.005,0])
base_action_v = np.array([0.005, 0., 0])
TEST = False

def play(env, configs, transpose=True, zoom=None, callback=None, keys_to_action=None):
    obs_s = env.observation_space
    
    pressed_keys = []
    running = True
    env_done = True

    total_frame = 600
    skip_frame = 10

    for i in range(len(configs)):
        print("Recording {} video {}".format(configs[0]['model_id'], i))

        config = configs[i]
        print("initial", config["initial_pos"], "target", config["target_pos"])
        env.robot.config = config
        env.config = config
        env.robot.initial_pos = config["initial_pos"]
        env.robot.initial_orn = config["initial_orn"]
        env.robot.target_pos = config["target_pos"]
        obs = env.reset()
        
        print(env.robot.body_xyz)
        #env.robot.set_position(env.robot.initial_pos)
        if not TEST:
            env.UI.model_id = config['model_id']
            env.UI.point_num = config['point_num']
            if env.UI.is_recorded(config["save_dir"]): continue
            env.UI.start_record(config["save_dir"])
        pressed_keys = []
        last_keys = []              ## Prevent overacting
    
        base_action_omage = np.array([-0.05,0.05,-0.05,0.05])
        base_action_v = np.array([0.005, 0.005, 0.005, 0.005])

        control_signal = -0.5
        control_signal_omega = 0.5
        v = 0
        omega = 0
        kp = 5
        ki = 0.1
        kd = 3
        ie = 0
        de = 0
        olde = 0
        ie_omega = 0
        de_omega = 0
        olde_omage = 0
            
        #env.robot.target_pos = env.robot.body_xyz + np.array([0.3,0.3,0])
        #env.robot.target_pos = env.robot.body_xyz + np.array([20,-5,0])

        print("robot initial", env.robot.initial_pos, "target", env.robot.target_pos)
        
        
        for i in tqdm(range(total_frame)):
            diff = np.array(env.robot.target_pos) - np.array(env.robot.body_xyz)
            control_signal = min(env.robot.dist_to_target() * 0.2, 1)
            control_signal_omega = env.robot.angle_to_target
            #if (env.robot.dist_to_target() < 1):
            #    break
            #print(np.linalg.norm(diff), i)
            e = control_signal - v
            de = e - olde
            ie += e
            olde = e
            pid_v = kp * e + ki * ie + kd * de

            e_omega = control_signal_omega - omega
            de_omega = e_omega - olde_omage
            ie_omega += e_omega
            pid_omega = kp * e_omega + ki * ie_omega + kd * de_omega

            if i % skip_frame == 0:
                env.is_record = True
            else:
                env.is_record = False
            pid_v += random.uniform(-pid_v, pid_v)
            pid_omega + random.uniform(-pid_omega, pid_omega)
            obs, _, _, _ = env.step(action=pid_v * base_action_v + pid_omega * base_action_omage)
            #print(i, env.robot.angle_to_target)
            #sys.exit()
            #obs, rew, env_done, info = env.step(action)
            v = obs["nonviz_sensor"][3]
            omega = obs["nonviz_sensor"][-1]

        
        env.robot.set_position(env.robot.target_pos)
        env.step(np.zeros(4))
        time.sleep(1)
        env.robot.set_position(env.robot.initial_pos)
        env.step(np.zeros(4))
        time.sleep(1)
        
        #print(np.linalg.norm(diff), i)
        if not TEST:
            env.UI.end_record()        
