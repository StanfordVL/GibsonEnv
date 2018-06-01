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

def display_arr(screen, arr, video_size, transpose):
    arr_min, arr_max = arr.min(), arr.max()
    arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
    pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1) if transpose else arr)
    pyg_img = pygame.transform.scale(pyg_img, video_size)
    screen.blit(pyg_img, (0,0))

base_action_omage = np.array([0.00,0.005,0])
base_action_v = np.array([0.005, 0., 0])


def play(env, configs, transpose=True, zoom=None, callback=None, keys_to_action=None):
    obs_s = env.observation_space
    
    pressed_keys = []
    running = True
    env_done = True

    total_frame = 50

    for i in range(len(configs)):
        print("Recording {} video {}".format(configs[0]['model_id'], i))

        config = configs[i]
        env.robot.config = config
        env.config = config
        obs = env.reset()
        #env.robot.set_position(env.robot.initial_pos)
        env.UI.model_id = config['model_id']
        env.UI.point_num = config['point_num']
        if env.UI.is_recorded(config["save_dir"]): continue
        env.UI.start_record(config["save_dir"])
        pressed_keys = []
        last_keys = []              ## Prevent overacting
    

        control_signal = -0.5
        control_signal_omega = 0.5
        v = 0
        omega = 0
        kp = 200
        ki = 0.6
        kd = 20
        ie = 0
        de = 0
        olde = 0
        ie_omega = 0
        de_omega = 0
        olde_omage = 0
            
        #env.robot.target_pos = env.robot.body_xyz + np.array([0.3,0.3,0])

        for i in tqdm(range(total_frame)):
            diff = np.array(env.robot.target_pos) - np.array(env.robot.body_xyz)
            control_signal = min(diff[0] * 0.2, 1)
            control_signal_omega = diff[1]
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

            obs, _, _, _ = env.step(action=pid_v * base_action_v + pid_omega * base_action_omage)
            #obs, rew, env_done, info = env.step(action)

        env.robot.set_position(env.robot.target_pos)
        env.step(np.zeros(3))
        time.sleep(1)
        env.robot.set_position(env.robot.initial_pos)
        env.step(np.zeros(3))
        time.sleep(1)
        print(np.linalg.norm(diff), i)
        env.UI.end_record()        
