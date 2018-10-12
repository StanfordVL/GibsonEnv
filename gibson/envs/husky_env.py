from gibson.envs.env_modalities import CameraRobotEnv, BaseRobotEnv, SemanticRobotEnv
from gibson.envs.env_bases import *
from gibson.core.physics.robot_locomotors import Husky
from transforms3d import quaternions
import os
import numpy as np
import sys
import pybullet as p
from gibson.core.physics.scene_stadium import SinglePlayerStadiumScene
import pybullet_data
import cv2

CALC_OBSTACLE_PENALTY = 0

tracking_camera = {
    'yaw': 110,
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

class HuskyNavigateEnv(CameraRobotEnv):
    """Specfy navigation reward
    """
    def __init__(self, config, gpu_idx=0):
        self.config = self.parse_config(config)
        assert(self.config["envname"] == self.__class__.__name__ or self.config["envname"] == "TestEnv")

        CameraRobotEnv.__init__(self, self.config, gpu_idx,
                                scene_type="stadium" if self.config["model_id"]=="stadium" else "building",
                                tracking_camera=tracking_camera)

        self.robot_introduce(Husky(self.config, env=self))
        self.scene_introduce()
        self.total_reward = 0
        self.total_frame = 0

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
            contact_ids = set((x[2], x[4]) for x in f.contact_list())
            #contact_ids = set([x[2] for x in f.contact_list()])
            if (self.ground_ids & contact_ids):
                # see Issue 63: https://github.com/openai/roboschool/issues/63
                # feet_collision_cost += self.foot_collision_cost
                self.robot.feet_contact[i] = 1.0
            else:
                self.robot.feet_contact[i] = 0.0
        
        electricity_cost  = self.electricity_cost  * float(np.abs(a*self.robot.joint_speeds).mean())
        electricity_cost  += self.stall_torque_cost * float(np.square(a).mean())


        steering_cost = self.robot.steering_cost(a)
        debugmode = 0
        if debugmode:
            print("steering cost", steering_cost)

        wall_contact = []
        
        for i, f in enumerate(self.parts):
            if self.parts[f] not in self.robot.feet:
                wall_contact += [pt for pt in self.robot.parts[f].contact_list() if pt[6][2] > 0.15]
        debugmode = 0
        if debugmode:
            print("Husky wall contact:", len(wall_contact))
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

        height = self.robot.get_position()[2]
        pitch = self.robot.get_rpy()[1]
        alive = float(self.robot.alive_bonus(height, pitch))
        
        debugmode = 0
        if (debugmode):
            #print("Wall contact points", len(wall_contact))
            print("Collision cost", wall_collision_cost)
            #print("electricity_cost", electricity_cost)
            print("close to target", close_to_target)
            print("Obstacle penalty", obstacle_penalty)
            print("Steering cost", steering_cost)
            print("progress", progress)
            #print("electricity_cost")
            #print(electricity_cost)
            #print("joints_at_limit_cost")
            #print(joints_at_limit_cost)
            #print("feet_collision_cost")
            #print(feet_collision_cost)

        rewards = [
            #alive,
            progress,
            wall_collision_cost,
            close_to_target,
            steering_cost,
            #angle_cost,
            #obstacle_penalty
            #electricity_cost,
            #joints_at_limit_cost,
            #feet_collision_cost
        ]
        return rewards

    def _termination(self, debugmode=False):
        height = self.robot.get_position()[2]
        pitch = self.robot.get_rpy()[1]
        alive = float(self.robot.alive_bonus(height, pitch)) > 0
        #alive = len(self.robot.parts['top_bumper_link'].contact_list()) == 0

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

    ## openai-gym v0.10.5 compatibility
    step  = CameraRobotEnv._step



class HuskyNavigateSpeedControlEnv(HuskyNavigateEnv):
    """Specfy navigation reward
    """
    def __init__(self, config, gpu_idx=0):
        #assert(self.config["envname"] == self.__class__.__name__ or self.config["envname"] == "TestEnv")
        HuskyNavigateEnv.__init__(self, config, gpu_idx)
        self.robot.keys_to_action = {
            (ord('s'), ): [-0.5,0], ## backward
            (ord('w'), ): [0.5,0], ## forward
            (ord('d'), ): [0,-0.5], ## turn right
            (ord('a'), ): [0,0.5], ## turn left
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


    def _step(self, action):
        control_signal, control_signal_omega = action
        self.e = control_signal - self.v
        self.de = self.e - self.olde
        self.ie += self.e
        self.olde = self.e
        pid_v = self.kp * self.e + self.ki * self.ie + self.kd * self.de

        self.e_omega = control_signal_omega - self.omega
        self.de_omega = self.e_omega - self.olde_omage
        self.ie_omega += self.e_omega
        pid_omega = self.kp * self.e_omega + self.ki * self.ie_omega + self.kd * self.de_omega

        obs, rew, env_done, info = HuskyNavigateEnv.step(self, pid_v * self.base_action_v + pid_omega * self.base_action_omage)

        self.v = obs["nonviz_sensor"][3]
        self.omega = obs["nonviz_sensor"][-1]

        return obs,rew,env_done,info

    ## openai-gym v0.10.5 compatibility
    step  = _step



class HuskyGibsonFlagRunEnv(CameraRobotEnv):
    """Specfy flagrun reward
    """
    def __init__(self, config, gpu_idx=0):
        self.config = self.parse_config(config)
        print(self.config["envname"])
        assert(self.config["envname"] == self.__class__.__name__ or self.config["envname"] == "TestEnv")
        CameraRobotEnv.__init__(self, self.config, gpu_idx,
                                scene_type="building",
                                tracking_camera=tracking_camera)

        self.robot_introduce(Husky(self.config, env=self))
        self.scene_introduce()

        self.total_reward = 0
        self.total_frame = 0
        self.flag_timeout = 1
        self.visualid = -1
        self.lastid = None
        self.gui = self.config["mode"] == "gui"
        
        if self.gui:
            self.visualid = p.createVisualShape(p.GEOM_MESH, fileName=os.path.join(pybullet_data.getDataPath(), 'cube.obj'), meshScale=[0.2, 0.2, 0.2], rgbaColor=[1, 0, 0, 0.7])
        self.colisionid = p.createCollisionShape(p.GEOM_MESH, fileName=os.path.join(pybullet_data.getDataPath(), 'cube.obj'), meshScale=[0.2, 0.2, 0.2])

        self.lastid = None
        self.obstacle_dist = 100
        
    def _reset(self):
        obs = CameraRobotEnv._reset(self)
        return obs

    def _flag_reposition(self):
        #self.walk_target_x = self.np_random.uniform(low=-self.scene.stadium_halflen,
        #                                            high=+self.scene.stadium_halflen)
        #self.walk_target_y = self.np_random.uniform(low=-self.scene.stadium_halfwidth,
        #                                            high=+self.scene.stadium_halfwidth)
        force_x = self.np_random.uniform(-300,300)
        force_y = self.np_random.uniform(-300, 300)

        more_compact = 0.5  # set to 1.0 whole football field
        #self.walk_target_x *= more_compact
        #self.walk_target_y *= more_compact

        startx, starty, _ = self.robot.get_position()


        self.flag = None
        #self.flag = self.scene.cpp_world.debug_sphere(self.walk_target_x, self.walk_target_y, 0.2, 0.2, 0xFF8080)
        self.flag_timeout = 3000 / self.scene.frame_skip
        #print('targetxy', self.flagid, self.walk_target_x, self.walk_target_y, p.getBasePositionAndOrientation(self.flagid))
        #p.resetBasePositionAndOrientation(self.flagid, posObj = [self.walk_target_x, self.walk_target_y, 0.5], ornObj = [0,0,0,0])
        if self.lastid:
            p.removeBody(self.lastid)

        self.lastid = p.createMultiBody(baseMass = 1, baseVisualShapeIndex=self.visualid, baseCollisionShapeIndex=self.colisionid, basePosition=[startx, starty, 0.5])
        p.applyExternalForce(self.lastid, -1, [force_x,force_y,50], [0,0,0], p.LINK_FRAME)

        ball_xyz, _ = p.getBasePositionAndOrientation(self.lastid)

        self.robot.walk_target_x = ball_xyz[0]
        self.robot.walk_target_y = ball_xyz[1]

    def _rewards(self, action = None, debugmode=False):
        a = action
        potential_old = self.potential
        self.potential = self.robot.calc_potential()
        if self.flag_timeout > 225:
            progress = 0
        else:
            progress = float(self.potential - potential_old)

        if not a is None:
            electricity_cost = self.electricity_cost * float(np.abs(
                a * self.robot.joint_speeds).mean())  # let's assume we have DC motor with controller, and reverse current braking
            electricity_cost += self.stall_torque_cost * float(np.square(a).mean())
        else:
            electricity_cost = 0

        alive = len(self.robot.parts['top_bumper_link'].contact_list())
        if alive == 0:
            alive_score = 0.1
        else:
            alive_score = -0.1

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
        debugmode = 0
        if (debugmode):
            print("progress")
            print(progress)

        obstacle_penalty = 0

        #print("obs dist %.3f" %self.obstacle_dist)
        if self.obstacle_dist < 0.7:
            obstacle_penalty = self.obstacle_dist - 0.7

        rewards = [
            alive_score,
            progress,
            obstacle_penalty
        ]
        return rewards

    def _termination(self, debugmode=False):
        alive = len(self.robot.parts['top_bumper_link'].contact_list())
        done = alive > 0 or self.nframe > 500
        if (debugmode):
            print("alive=")
            print(alive)
        print(len(self.robot.parts['top_bumper_link'].contact_list()), self.nframe, done)
        return done

    def _step(self, a):
        state, reward, done, meta = CameraRobotEnv._step(self, a)
        if self.flag_timeout <= 0 or (self.flag_timeout < 225 and self.robot.walk_target_dist < 0.8):
            self._flag_reposition()
        self.flag_timeout -= 1

        if "depth" in self.config["output"]:
            depth_obs = self.get_observations()["depth"]
            x_start = int(self.windowsz/2-16)
            x_end   = int(self.windowsz/2+16)
            y_start = int(self.windowsz/2-16)
            y_end   = int(self.windowsz/2+16)
            self.obstacle_dist = (np.mean(depth_obs[x_start:x_end, y_start:y_end, -1]))

        return state, reward, done, meta

    ## openai-gym v0.10.5 compatibility
    step  = _step


class HuskySemanticNavigateEnv(SemanticRobotEnv):
    """Specfy navigation reward
    """
    def __init__(self, config, gpu_idx=0):
        #assert(self.config["envname"] == self.__class__.__name__ or self.config["envname"] == "TestEnv")
        self.config = self.parse_config(config)
        SemanticRobotEnv.__init__(self, self.config, gpu_idx,
                                  scene_type="building",
                                  tracking_camera=tracking_camera)
        self.robot_introduce(Husky(self.config, env=self))
        self.scene_introduce()

        self.total_reward = 0
        self.total_frame = 0
        self.flag_timeout = 1
        self.visualid = -1
        self.lastid = None
        self.gui = self.config["mode"] == "gui"

        if self.gui:
            self.visualid = p.createVisualShape(p.GEOM_MESH, fileName=os.path.join(pybullet_data.getDataPath(), 'cube.obj'), meshScale=[0.2, 0.2, 0.2], rgbaColor=[1, 0, 0, 0.7])
        self.colisionid = p.createCollisionShape(p.GEOM_MESH, fileName=os.path.join(pybullet_data.getDataPath(), 'cube.obj'), meshScale=[0.2, 0.2, 0.2])

        self.lastid = None
        self.obstacle_dist = 100

        self.semantic_flagIds = []

        debug_semantic = 1
        if debug_semantic and self.gui:
            for i in range(self.semantic_pos.shape[0]):
                pos = self.semantic_pos[i]
                pos[2] += 0.2   # make flag slight above object 
                visualId = p.createVisualShape(p.GEOM_MESH, fileName=os.path.join(pybullet_data.getDataPath(), 'cube.obj'), meshScale=[0.1, 0.1, 0.1], rgbaColor=[1, 0, 0, 0.7])
                flagId = p.createMultiBody(baseVisualShapeIndex=visualId, baseCollisionShapeIndex=-1, basePosition=pos)
                self.semantic_flagIds.append(flagId)

    def step(self, action):
        obs, rew, env_done, info = SemanticRobotEnv.step(self,action=action)
        self.close_semantic_ids = self.get_close_semantic_pos(dist_max=1.0, orn_max=np.pi/5)
        for i in self.close_semantic_ids:
            flagId = self.semantic_flagIds[i]
            p.changeVisualShape(flagId, -1, rgbaColor=[0, 1, 0, 1])
        return obs,rew,env_done,info

    def _rewards(self, action = None, debugmode=False):
        a = action
        potential_old = self.potential
        self.potential = self.robot.calc_potential()
        if self.flag_timeout > 225:
            progress = 0
        else:
            progress = float(self.potential - potential_old)

        if not a is None:
            electricity_cost = self.electricity_cost * float(np.abs(
                a * self.robot.joint_speeds).mean())  # let's assume we have DC motor with controller, and reverse current braking
            electricity_cost += self.stall_torque_cost * float(np.square(a).mean())
        else:
            electricity_cost = 0

        alive = len(self.robot.parts['top_bumper_link'].contact_list())
        if alive == 0:
            alive_score = 0.1
        else:
            alive_score = -0.1

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
        debugmode = 0
        if (debugmode):
            print("progress")
            print(progress)

        obstacle_penalty = 0

        #print("obs dist %.3f" %self.obstacle_dist)
        if self.obstacle_dist < 0.7:
            obstacle_penalty = self.obstacle_dist - 0.7

        rewards = [
            alive_score,
            progress,
            obstacle_penalty
        ]
        return rewards

    def _termination(self, debugmode=False):
        alive = len(self.robot.parts['top_bumper_link'].contact_list())
        done = alive > 0 or self.nframe > 500
        if (debugmode):
            print("alive=")
            print(alive)
        #print(len(self.robot.parts['top_bumper_link'].contact_list()), self.nframe, done)
        return done

    def _reset(self):
        CameraRobotEnv._reset(self)
        for flagId in self.semantic_flagIds:
            p.changeVisualShape(flagId, -1, rgbaColor=[1, 0, 0, 1])


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