from gibson.envs.env_modalities import CameraRobotEnv, BaseRobotEnv
from gibson.envs.env_bases import *
from gibson.core.physics.robot_locomotors import Turtlebot, JR, JR2
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

class TurtlebotNavigateEnv(CameraRobotEnv):
    """Specfy navigation reward
    """
    def __init__(self, config, gpu_idx=0, gravity=9.8, collision_enabled=True):
        self.config = self.parse_config(config)
        assert(self.config["envname"] == self.__class__.__name__ or self.config["envname"] == "TestEnv")

        CameraRobotEnv.__init__(self, self.config, gpu_idx,
                                scene_type="stadium" if self.config["model_id"]=="stadium" else "building",
                                tracking_camera=tracking_camera)

        self.robot_introduce(Turtlebot(self.config, env=self))
        self.scene_introduce(gravity=gravity, collision_enabled=collision_enabled)
        self.total_reward = 0
        self.total_frame = 0


    def get_odom(self):
        return np.array(self.robot.body_xyz) - np.array(self.config["initial_pos"]), np.array(self.robot.body_rpy)

    def add_text(self, img):
        font = cv2.FONT_HERSHEY_SIMPLEX
        x,y,z = self.robot.get_position()
        r,p,ya = self.robot.get_rpy()
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


        steering_cost = self.robot.steering_cost(a)
        debugmode = 0
        if debugmode:
            print("steering cost", steering_cost)

        wall_contact = [pt for pt in self.robot.parts['base_link'].contact_list() if pt[6][2] > 0.15]
        wall_collision_cost = self.wall_collision_cost * len(wall_contact)

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
        close_to_target = 0

        if self.robot.dist_to_target() < 2:
            close_to_target = 0.5

        angle_cost = self.robot.angle_cost()

        obstacle_penalty = 0
        if CALC_OBSTACLE_PENALTY and self._require_camera_input:
            obstacle_penalty = get_obstacle_penalty(self.robot, self.render_depth)

        debugmode = 0
        if debugmode:
            print("angle cost", angle_cost)

        debugmode = 0
        if (debugmode):
            print("Wall contact points", len(wall_contact))
            print("Collision cost", wall_collision_cost)
            print("electricity_cost", electricity_cost)
            print("close to target", close_to_target)
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
            close_to_target,
            steering_cost,
            angle_cost,
            obstacle_penalty
            #electricity_cost,
            #joints_at_limit_cost,
            #feet_collision_cost
        ]
        return rewards

    def _termination(self, debugmode=False):
        height = self.robot.get_position()[2]
        pitch = self.robot.get_rpy()[1]
        alive = float(self.robot.alive_bonus(height, pitch))

        done = not alive or self.nframe > 250 or height < 0
        #if done:
        #    print("Episode reset")
        return done

    def _flag_reposition(self):
        target_pos = self.robot.target_pos

        self.flag = None
        if self.gui and not self.config["display_ui"]:
            self.visual_flagId = p.createVisualShape(p.GEOM_MESH, fileName=os.path.join(pybullet_data.getDataPath(), 'cube.obj'), meshScale=[0.5, 0.5, 0.5], rgbaColor=[1, 0, 0, 0.7])
            self.last_flagId = p.createMultiBody(baseVisualShapeIndex=self.visual_flagId, baseCollisionShapeIndex=-1, basePosition=[target_pos[0], target_pos[1], 0.5])

    def  _reset(self):
        self.total_frame = 0
        self.total_reward = 0
        obs = CameraRobotEnv._reset(self)
        self._flag_reposition()
        return obs

def get_obstacle_penalty(robot, depth):
    screen_sz = robot.obs_dim[0]
    screen_delta = int(screen_sz / 8)
    screen_half  = int(screen_sz / 2)
    height_offset = int(screen_sz / 4)

    obstacle_dist = (np.mean(depth[screen_half  + height_offset - screen_delta : screen_half + height_offset + screen_delta, screen_half - screen_delta : screen_half + screen_delta, -1]))
    obstacle_penalty = 0
    OBSTACLE_LIMIT = 1.5
    if obstacle_dist < OBSTACLE_LIMIT:
       obstacle_penalty = (obstacle_dist - OBSTACLE_LIMIT)
    
    debugmode = 0
    if debugmode:
        #print("Obstacle screen", screen_sz, screen_delta)
        print("Obstacle distance", obstacle_dist)
        print("Obstacle penalty", obstacle_penalty)
    return obstacle_penalty


class JRNavigateEnv(CameraRobotEnv):
    """Specfy navigation reward
    """

    def __init__(self, config, gpu_idx=0):
        self.config = self.parse_config(config)
        assert (self.config["envname"] == self.__class__.__name__ or self.config["envname"] == "TestEnv")

        CameraRobotEnv.__init__(self, self.config, gpu_idx,
                                scene_type="stadium" if self.config["model_id"] == "stadium" else "building",
                                tracking_camera=tracking_camera)

        self.robot_introduce(JR(self.config, env=self))
        self.scene_introduce()
        self.total_reward = 0
        self.total_frame = 0

    def add_text(self, img):
        font = cv2.FONT_HERSHEY_SIMPLEX
        x, y, z = self.robot.get_position()
        r, p, ya = self.robot.get_rpy()
        cv2.putText(img, 'x:{0:.4f} y:{1:.4f} z:{2:.4f}'.format(x, y, z), (10, 20), font, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(img, 'ro:{0:.4f} pth:{1:.4f} ya:{2:.4f}'.format(r, p, ya), (10, 40), font, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(img, 'potential:{0:.4f}'.format(self.potential), (10, 60), font, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
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

        electricity_cost = self.electricity_cost * float(np.abs(a * self.robot.joint_speeds).mean())  # let's assume we
        electricity_cost += self.stall_torque_cost * float(np.square(a).mean())

        steering_cost = self.robot.steering_cost(a)
        debugmode = 0
        if debugmode:
            print("steering cost", steering_cost)

        wall_contact = [pt for pt in self.robot.parts['base_link'].contact_list() if pt[6][2] > 0.15]
        wall_collision_cost = self.wall_collision_cost * len(wall_contact)

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
        close_to_target = 0

        if self.robot.dist_to_target() < 2:
            close_to_target = 0.5

        angle_cost = self.robot.angle_cost()

        obstacle_penalty = 0
        if CALC_OBSTACLE_PENALTY and self._require_camera_input:
            obstacle_penalty = get_obstacle_penalty(self.robot, self.render_depth)

        debugmode = 0
        if debugmode:
            print("angle cost", angle_cost)

        debugmode = 0
        if (debugmode):
            print("Wall contact points", len(wall_contact))
            print("Collision cost", wall_collision_cost)
            print("electricity_cost", electricity_cost)
            print("close to target", close_to_target)
            # print("progress")
            # print(progress)
            # print("electricity_cost")
            # print(electricity_cost)
            # print("joints_at_limit_cost")
            # print(joints_at_limit_cost)
            # print("feet_collision_cost")
            # print(feet_collision_cost)

        rewards = [
            # alive,
            progress,
            # wall_collision_cost,
            close_to_target,
            steering_cost,
            angle_cost,
            obstacle_penalty
            # electricity_cost,
            # joints_at_limit_cost,
            # feet_collision_cost
        ]
        return rewards

    def _termination(self, debugmode=False):
        height = self.robot.get_position()[2]
        pitch = self.robot.get_rpy()[1]
        alive = float(self.robot.alive_bonus(height, pitch))

        done = not alive or self.nframe > 250 or height < 0
        # if done:
        #    print("Episode reset")
        return done

    def _flag_reposition(self):
        target_pos = self.robot.target_pos

        self.flag = None
        if self.gui and not self.config["display_ui"]:
            self.visual_flagId = p.createVisualShape(p.GEOM_MESH,
                                                     fileName=os.path.join(pybullet_data.getDataPath(), 'cube.obj'),
                                                     meshScale=[0.5, 0.5, 0.5], rgbaColor=[1, 0, 0, 0.7])
            self.last_flagId = p.createMultiBody(baseVisualShapeIndex=self.visual_flagId, baseCollisionShapeIndex=-1,
                                                 basePosition=[target_pos[0], target_pos[1], 0.5])

    def _reset(self):
        self.total_frame = 0
        self.total_reward = 0
        obs = CameraRobotEnv._reset(self)
        self._flag_reposition()
        return obs


def get_obstacle_penalty(robot, depth):
    screen_sz = robot.obs_dim[0]
    screen_delta = int(screen_sz / 8)
    screen_half = int(screen_sz / 2)
    height_offset = int(screen_sz / 4)

    obstacle_dist = (np.mean(
        depth[screen_half + height_offset - screen_delta: screen_half + height_offset + screen_delta,
        screen_half - screen_delta: screen_half + screen_delta, -1]))
    obstacle_penalty = 0
    OBSTACLE_LIMIT = 1.5
    if obstacle_dist < OBSTACLE_LIMIT:
        obstacle_penalty = (obstacle_dist - OBSTACLE_LIMIT)

    debugmode = 0
    if debugmode:
        # print("Obstacle screen", screen_sz, screen_delta)
        print("Obstacle distance", obstacle_dist)
        print("Obstacle penalty", obstacle_penalty)
    return obstacle_penalty



class TurtlebotNavigateSpeedControlEnv(TurtlebotNavigateEnv):
    """Specfy navigation reward
    """
    def __init__(self, config, gpu_idx=0):
        #assert(self.config["envname"] == self.__class__.__name__ or self.config["envname"] == "TestEnv")
        TurtlebotNavigateEnv.__init__(self, config, gpu_idx)
        self.robot.keys_to_action = {
            (ord('s'), ): [-0.1,0], ## backward
            (ord('w'), ): [0.1,0], ## forward
            (ord('d'), ): [0,0.1], ## turn right
            (ord('a'), ): [0,-0.1], ## turn left
            (): [0,0]
        }

        self.base_action_omage = np.array([-0.001, 0.001, -0.001, 0.001])
        self.base_action_v = np.array([0.001, 0.001, 0.001, 0.001])
        self.action_space = gym.spaces.Discrete(5)
        #control_signal = -0.5
        #control_signal_omega = 0.5
        self.v = 0
        self.omega = 0
        self.kp = 100
        self.ki = 0.1
        self.kd = 25
        self.ie = 0
        self.de = 0
        self.olde = 0
        self.ie_omega = 0
        self.de_omega = 0
        self.olde_omage = 0

    def step(self, action):
        real_action = [action[0]+action[1], action[0]-action[1]]

        obs, rew, env_done, info = TurtlebotNavigateEnv.step(self, real_action)

        self.v = obs["nonviz_sensor"][3]
        self.omega = obs["nonviz_sensor"][-1]

        return obs,rew,env_done,info



class JR2NavigateEnv(CameraRobotEnv):
    """Specfy navigation reward
    """

    def __init__(self, config, gpu_idx=0):
        self.config = self.parse_config(config)
        assert (self.config["envname"] == self.__class__.__name__ or self.config["envname"] == "TestEnv")

        CameraRobotEnv.__init__(self, self.config, gpu_idx,
                                scene_type="stadium" if self.config["model_id"] == "stadium" else "building",
                                tracking_camera=tracking_camera)

        self.robot_introduce(JR2(self.config, env=self))
        self.scene_introduce()
        self.total_reward = 0
        self.total_frame = 0

    def add_text(self, img):
        font = cv2.FONT_HERSHEY_SIMPLEX
        x, y, z = self.robot.get_position()
        r, p, ya = self.robot.get_rpy()
        cv2.putText(img, 'x:{0:.4f} y:{1:.4f} z:{2:.4f}'.format(x, y, z), (10, 20), font, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(img, 'ro:{0:.4f} pth:{1:.4f} ya:{2:.4f}'.format(r, p, ya), (10, 40), font, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(img, 'potential:{0:.4f}'.format(self.potential), (10, 60), font, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
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

        electricity_cost = self.electricity_cost * float(np.abs(a * self.robot.joint_speeds).mean())  # let's assume we
        electricity_cost += self.stall_torque_cost * float(np.square(a).mean())

        steering_cost = self.robot.steering_cost(a)
        debugmode = 0
        if debugmode:
            print("steering cost", steering_cost)

        wall_contact = [pt for pt in self.robot.parts['base_link'].contact_list() if pt[6][2] > 0.15]
        wall_collision_cost = self.wall_collision_cost * len(wall_contact)

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
        close_to_target = 0

        if self.robot.dist_to_target() < 2:
            close_to_target = 0.5

        angle_cost = self.robot.angle_cost()

        obstacle_penalty = 0
        if CALC_OBSTACLE_PENALTY and self._require_camera_input:
            obstacle_penalty = get_obstacle_penalty(self.robot, self.render_depth)

        debugmode = 0
        if debugmode:
            print("angle cost", angle_cost)

        debugmode = 0
        if (debugmode):
            print("Wall contact points", len(wall_contact))
            print("Collision cost", wall_collision_cost)
            print("electricity_cost", electricity_cost)
            print("close to target", close_to_target)
            # print("progress")
            # print(progress)
            # print("electricity_cost")
            # print(electricity_cost)
            # print("joints_at_limit_cost")
            # print(joints_at_limit_cost)
            # print("feet_collision_cost")
            # print(feet_collision_cost)

        rewards = [
            # alive,
            progress,
            # wall_collision_cost,
            close_to_target,
            steering_cost,
            angle_cost,
            obstacle_penalty
            # electricity_cost,
            # joints_at_limit_cost,
            # feet_collision_cost
        ]
        return rewards

    def _termination(self, debugmode=False):
        height = self.robot.get_position()[2]
        pitch = self.robot.get_rpy()[1]
        alive = float(self.robot.alive_bonus(height, pitch))

        done = not alive or self.nframe > 250 or height < 0
        # if done:
        #    print("Episode reset")
        return done

    def _flag_reposition(self):
        target_pos = self.robot.target_pos

        self.flag = None
        if self.gui and not self.config["display_ui"]:
            self.visual_flagId = p.createVisualShape(p.GEOM_MESH,
                                                     fileName=os.path.join(pybullet_data.getDataPath(), 'cube.obj'),
                                                     meshScale=[0.5, 0.5, 0.5], rgbaColor=[1, 0, 0, 0.7])
            self.last_flagId = p.createMultiBody(baseVisualShapeIndex=self.visual_flagId, baseCollisionShapeIndex=-1,
                                                 basePosition=[target_pos[0], target_pos[1], 0.5])

    def _reset(self):
        self.total_frame = 0
        self.total_reward = 0
        obs = CameraRobotEnv._reset(self)
        self._flag_reposition()
        return obs

    def get_odom(self):
        return np.array(self.robot.body_xyz) - np.array(self.config["initial_pos"]), np.array(self.robot.body_rpy)

def get_obstacle_penalty(robot, depth):
    screen_sz = robot.obs_dim[0]
    screen_delta = int(screen_sz / 8)
    screen_half = int(screen_sz / 2)
    height_offset = int(screen_sz / 4)

    obstacle_dist = (np.mean(
        depth[screen_half + height_offset - screen_delta: screen_half + height_offset + screen_delta,
        screen_half - screen_delta: screen_half + screen_delta, -1]))
    obstacle_penalty = 0
    OBSTACLE_LIMIT = 1.5
    if obstacle_dist < OBSTACLE_LIMIT:
        obstacle_penalty = (obstacle_dist - OBSTACLE_LIMIT)

    debugmode = 0
    if debugmode:
        # print("Obstacle screen", screen_sz, screen_delta)
        print("Obstacle distance", obstacle_dist)
        print("Obstacle penalty", obstacle_penalty)
    return obstacle_penalty
