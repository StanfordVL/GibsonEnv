from gibson.envs.env_modalities import CameraRobotEnv, BaseRobotEnv
from gibson.envs.env_bases import *
from gibson.core.physics.robot_locomotors import Quadrotor
from transforms3d import quaternions
import os
import numpy as np
import sys
import pybullet as p
from gibson.core.physics.scene_stadium import SinglePlayerStadiumScene
import pybullet_data
import cv2

CALC_OBSTACLE_PENALTY = 1

tracking_camera = {
    'yaw': 20,
    'z_offset': 0.5,
    'distance': 1,
    'pitch': -20
}

tracking_camera_top = {
    'yaw': 20,  # demo: living room, stairs
    'z_offset': 0.5,
    'distance': 1,
    'pitch': -20
}

class DroneNavigateEnv(CameraRobotEnv):
    """Specfy navigation reward
    """
    def __init__(
            self,
            config,
            is_discrete=False,
            gpu_count=0):

        self.config = self.parse_config(config)
        self.gui = self.config["mode"] == "gui"
        self.model_id = self.config["model_id"]
        self.timestep = self.config["speed"]["timestep"]
        self.frame_skip = self.config["speed"]["frameskip"]
        self.resolution = self.config["resolution"]
        self.tracking_camera = tracking_camera
        target_orn, target_pos   = self.config["target_orn"], self.config["target_pos"]
        initial_orn, initial_pos = self.config["initial_orn"], self.config["initial_pos"]
        self.total_reward = 0
        self.total_frame = 0
        
        CameraRobotEnv.__init__(
            self,
            config,
            gpu_count,
            scene_type="building", 
            use_filler=self.config["use_filler"])
        self.robot_introduce(Quadrotor(
            is_discrete=is_discrete,
            initial_pos=initial_pos,
            initial_orn=initial_orn,
            target_pos=target_pos,
            resolution=self.resolution,
            env = self
            ))
        self.scene_introduce()

        assert(self.config["envname"] == self.__class__.__name__ or self.config["envname"] == "TestEnv")


    def calc_rewards_and_done(self, a, state):
        done = self._termination(state)
        rewards = self._rewards(a)
        debugmode = 0
        if debugmode:
            print("Frame %f reward %f" % (self.nframe, sum(rewards)))

        self.total_reward = self.total_reward + sum(rewards)
        self.total_frame = self.total_frame + 1
        #print(self.total_frame, self.total_reward)
        return rewards, done

    def add_text(self, img):
        font = cv2.FONT_HERSHEY_SIMPLEX
        x,y,z = self.robot.body_xyz
        r,p,ya = self.robot.body_rpy
        cv2.putText(img, 'x:{0:.4f} y:{1:.4f} z:{2:.4f}'.format(x,y,z), (10, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, 'ro:{0:.4f} pth:{1:.4f} ya:{2:.4f}'.format(r,p,ya), (10, 40), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, 'potential:{0:.4f}'.format(self.potential), (10, 60), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, 'fps:{0:.4f}'.format(self.fps), (10, 80), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    def _rewards(self, action=None, debugmode=False):
        a = action
        potential_old = self.potential
        self.potential = self.robot.calc_potential()
        progress = float(self.potential - potential_old)

        feet_collision_cost = 0.0
        for i, f in enumerate(
                self.robot.feet):  # TODO: Maybe calculating feet contacts could be done within the robot code
            # print(f.contact_list())
            contact_ids = set((x[2], x[4]) for x in f.contact_list())
            # print("CONTACT OF '%d' WITH %d" % (contact_ids, ",".join(contact_names)) )
            if (self.ground_ids & contact_ids):
                # see Issue 63: https://github.com/openai/roboschool/issues/63
                # feet_collision_cost += self.foot_collision_cost
                self.robot.feet_contact[i] = 1.0
            else:
                self.robot.feet_contact[i] = 0.0
        # print(self.robot.feet_contact)

        electricity_cost  = self.electricity_cost  * float(np.abs(a*self.robot.joint_speeds).mean())  # let's assume we 
        electricity_cost  += self.stall_torque_cost * float(np.square(a).mean())


        debugmode = 0

        #alive = len(self.robot.parts['top_bumper_link'].contact_list())
        #if alive == 0:
        #    alive_score = 0.1
        #else:
        #    alive_score = -0.1

        wall_contact = [pt for pt in self.robot.parts['base_link'].contact_list() if pt[6][2] > 0.15]
        wall_collision_cost = self.wall_collision_cost * len(wall_contact)

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
        close_to_goal = 0
        if self.robot.is_close_to_goal():
            close_to_goal = 0.5

        obstacle_penalty = 0

        debugmode = 0

        debugmode = 0
        if (debugmode):
            print("Wall contact points", len(wall_contact))
            print("Collision cost", wall_collision_cost)
            print("electricity_cost", electricity_cost)
            print("close to goal", close_to_goal)
            #print("progress")
            #print(progress)
            #print("electricity_cost")
            #print(electricity_cost)
            #print("joints_at_limit_cost")
            #print(joints_at_limit_cost)
            #print("feet_collision_cost")
            #print(feet_collision_cost)

        rewards = [
            #alive,
            progress,
            #wall_collision_cost,
            close_to_goal,
            obstacle_penalty
            #electricity_cost,
            #joints_at_limit_cost,
            #feet_collision_cost
        ]
        return rewards

    def _termination(self, state=None, debugmode=False):

        done = self.nframe > 250 or self.robot.body_xyz[2] < 0
        #done = alive < 0
        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True
        if done:
            print("Episode reset")
        return done

    def  _reset(self):
        self.total_frame = 0
        self.total_reward = 0

        obs = CameraRobotEnv._reset(self)
        return obs
