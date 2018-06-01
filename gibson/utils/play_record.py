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

def display_arr(screen, arr, video_size, transpose):
    arr_min, arr_max = arr.min(), arr.max()
    arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
    pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1) if transpose else arr)
    pyg_img = pygame.transform.scale(pyg_img, video_size)
    screen.blit(pyg_img, (0,0))

def play(env, configs, transpose=True, zoom=None, callback=None, keys_to_action=None):

    obs_s = env.observation_space
    if keys_to_action is None:
        if hasattr(env, 'get_keys_to_action'):
            keys_to_action = env.get_keys_to_action()
        elif hasattr(env.unwrapped, 'get_keys_to_action'):
            keys_to_action = env.unwrapped.get_keys_to_action()
    relevant_keys = set(sum(map(list, keys_to_action.keys()),[]))
    
    pressed_keys = []
    running = True
    env_done = True

    total_frame = 200

    for i in range(len(configs)):
        print("Recording {} video {}".format(configs[0]['model_id'], i))

        config = configs[i]
        env.robot.config = config
        env.config = config
        obs = env.reset()
        env.UI.model_id = config['model_id']
        env.UI.point_num = config['point_num']
        if env.UI.is_recorded("/media/Drive3/Gibson_Models/572_avi"): continue
        env.UI.start_record("/media/Drive3/Gibson_Models/572_avi")
        pressed_keys = []
        last_keys = []              ## Prevent overacting
    
        for i in tqdm(range(total_frame)):
            if len(pressed_keys) == 0:
                action = keys_to_action[()]
                start = time.time()
                obs, rew, env_done, info = env.step(action)
            for p_key in pressed_keys:
                action = keys_to_action[(p_key, )]
                prev_obs = obs
                start = time.time()
                obs, rew, env_done, info = env.step(action)
            if callback is not None:
                callback(prev_obs, obs, action, rew, env_done, info)
            key_codes = env.get_key_pressed(relevant_keys)
            pressed_keys = []

            if i >= 50:
                pressed_keys = [ord('w')]

            for key in key_codes:
                if key == ord('r') and key not in last_keys:
                    do_restart = True
                if key == ord('j') and key not in last_keys:
                    env.robot.turn_left()
                if key == ord('l') and key not in last_keys:
                    env.robot.turn_right()
                if key == ord('i') and key not in last_keys:
                    env.robot.move_forward()
                if key == ord('k') and key not in last_keys:
                    env.robot.move_backward()
                if key not in relevant_keys:
                    continue
                pressed_keys.append(key) 
                
            last_keys = key_codes

        env.UI.end_record()        
