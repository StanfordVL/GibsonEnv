from realenv.envs.env_modalities import CameraRobotEnv, SensorRobotEnv
from realenv.envs.env_bases import *
from realenv.core.physics.robot_locomotors import Husky
from transforms3d import quaternions
from realenv.configs import *
import numpy as np
import sys
import pybullet as p
from realenv.core.physics.scene_stadium import SinglePlayerStadiumScene
import pybullet_data

HUMANOID_TIMESTEP  = 1.0/(4 * 22)
HUMANOID_FRAMESKIP = 4

import os
from realenv.configs import *

class HuskyEnv:
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }
    def __init__(self, is_discrete=False):
        self.physicsClientId=-1
        target_orn, target_pos = INITIAL_POSE["husky"][MODEL_ID][-1]
        self.robot = Husky(is_discrete, target_pos=target_pos)
        self.nframe = 0

    def get_keys_to_action(self):
        return self.robot.keys_to_action


    def calc_rewards(self, a, state):
        self.nframe += 1

        # dummy state if a is None
        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            self.robot.apply_action(a)
            self.scene.global_step()
       
        alive = float(self.robot.alive_bonus(state[0] + self.robot.initial_z, self.robot.body_rpy[
            1]))  # state[0] is body height above ground, body_rpy[1] is pitch

        done = self.nframe > 300
        #done = alive < 0
        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

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
        electricity_cost += self.stall_torque_cost * float(np.square(a).mean())

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
        debugmode = 0
        if (debugmode):
            print("alive=")
            print(alive)
            print("progress")
            print(progress)
            print("electricity_cost")
            print(electricity_cost)
            print("joints_at_limit_cost")
            print(joints_at_limit_cost)
            print("feet_collision_cost")
            print(feet_collision_cost)

        print("Frame %f reward %f" % (self.nframe, progress))
        return [
            #alive,
            progress,
            #electricity_cost,
            #joints_at_limit_cost,
            #feet_collision_cost
         ], done
        

class HuskyCameraEnv(HuskyEnv, CameraRobotEnv):
    def __init__(self, human=True, timestep=HUMANOID_TIMESTEP, 
        frame_skip=HUMANOID_FRAMESKIP, enable_sensors=False,
        is_discrete=False, mode="RGBD", use_filler=True, gpu_count=0):
        self.human = human
        self.timestep = timestep
        self.frame_skip = frame_skip
        self.enable_sensors = enable_sensors
        HuskyEnv.__init__(self, is_discrete)
        CameraRobotEnv.__init__(self, use_filler, mode, gpu_count)

        #self.tracking_camera['pitch'] = -45 ## stairs
        yaw = 90     ## demo: living room
        #yaw = 30    ## demo: kitchen
        offset = 0.5
        distance = 1.2 ## living room
        #self.tracking_camera['yaw'] = 90     ## demo: stairs

        
        self.tracking_camera['yaw'] = yaw   ## living roon
        self.tracking_camera['pitch'] = -10
        
        self.tracking_camera['distance'] = distance
        self.tracking_camera['z_offset'] = offset
    
    def  _reset(self):
        obs = CameraRobotEnv._reset(self)
        self.nframe = 0
        return obs

class HuskySensorEnv(HuskyEnv, SensorRobotEnv):
    def __init__(self, human=True, timestep=HUMANOID_TIMESTEP, 
        frame_skip=HUMANOID_FRAMESKIP, enable_sensors=False,
        is_discrete=False, scene_fn=create_single_player_building_scene):
        self.human = human
        self.timestep = timestep
        self.frame_skip = frame_skip
        HuskyEnv.__init__(self, is_discrete)
        SensorRobotEnv.__init__(self, scene_fn)
        self.nframe = 0
    def  _reset(self):
        obs = SensorRobotEnv._reset(self)
        self.nframe = 0
        return obs


class HuskyFlagRunEnv(HuskyEnv, SensorRobotEnv):
    def __init__(self, human=True, timestep=HUMANOID_TIMESTEP,
                 frame_skip=HUMANOID_FRAMESKIP, enable_sensors=False,
                 is_discrete=False):
        self.human = human
        self.timestep = timestep
        self.frame_skip = frame_skip
        HuskyEnv.__init__(self, is_discrete=is_discrete)
        SensorRobotEnv.__init__(self, scene_fn=create_single_player_stadium_scene)
        self.nframe = 0
        self.flag_timeout = 1

        if self.human:
            self.visualid = p.createVisualShape(p.GEOM_MESH, fileName=os.path.join(pybullet_data.getDataPath(), 'cube.obj'), meshScale=[0.5, 0.5, 0.5], rgbaColor=[1, 0, 0, 0.7])
        self.lastid = None

    def _reset(self):
        obs = SensorRobotEnv._reset(self)
        self.nframe = 0
        return obs

    def flag_reposition(self):

        self.walk_target_x = self.np_random.uniform(low=-self.scene.stadium_halflen,
                                                    high=+self.scene.stadium_halflen)
        self.walk_target_y = self.np_random.uniform(low=-self.scene.stadium_halfwidth,
                                                    high=+self.scene.stadium_halfwidth)

        more_compact = 0.5  # set to 1.0 whole football field
        self.walk_target_x *= more_compact
        self.walk_target_y *= more_compact

        self.flag = None
        #self.flag = self.scene.cpp_world.debug_sphere(self.walk_target_x, self.walk_target_y, 0.2, 0.2, 0xFF8080)
        self.flag_timeout = 600 / self.scene.frame_skip
        #print('targetxy', self.flagid, self.walk_target_x, self.walk_target_y, p.getBasePositionAndOrientation(self.flagid))
        #p.resetBasePositionAndOrientation(self.flagid, posObj = [self.walk_target_x, self.walk_target_y, 0.5], ornObj = [0,0,0,0])
        if self.human:
            if self.lastid:
                p.removeBody(self.lastid)

            self.lastid = p.createMultiBody(baseVisualShapeIndex=self.visualid, baseCollisionShapeIndex=-1, basePosition=[self.walk_target_x, self.walk_target_y, 0.5])

        self.robot.walk_target_x = self.walk_target_x
        self.robot.walk_target_y = self.walk_target_y

    def _step(self, a=None):

        if self.flag_timeout <= 0:
            self.flag_reposition()

        self.nframe += 1
        self.flag_timeout -= 1

        # dummy state if a is None
        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            if not a is None:
                self.robot.apply_action(a)
            self.scene.global_step()

        state = self.robot.calc_state()  # also calculates self.joints_at_limit

        alive = len(self.robot.parts['top_bumper_link'].contact_list())
        if alive == 0:
            alive_score = 0.1
        else:
            alive_score = -0.1


        done = alive > 0 or self.nframe > 500

        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        potential_old = self.potential
        self.potential = self.robot.calc_potential()
        progress = float(self.potential - potential_old)

        if not a is None:
            electricity_cost = self.electricity_cost * float(np.abs(
                a * self.robot.joint_speeds).mean())  # let's assume we have DC motor with controller, and reverse current braking
            electricity_cost += self.stall_torque_cost * float(np.square(a).mean())
        else:
            electricity_cost = 0

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
        debugmode = 0
        if (debugmode):
            print("alive=")
            print(alive)
            print("progress")
            print(progress)

        self.rewards = [
            alive_score,
            progress,
        ]

        self.reward += sum(self.rewards)

        if self.human:
            humanPos, humanOrn = p.getBasePositionAndOrientation(self.robot_tracking_id)
            humanPos = (humanPos[0], humanPos[1], humanPos[2] + self.tracking_camera['z_offset'])

            if MAKE_VIDEO:
                p.resetDebugVisualizerCamera(self.tracking_camera['distance'], self.tracking_camera['yaw'],
                                             self.tracking_camera['pitch'], humanPos)  ## demo: kitchen, living room
            # p.resetDebugVisualizerCamera(distance,yaw,-42,humanPos);        ## demo: stairs

        eye_pos = self.robot.eyes.current_position()
        x, y, z, w = self.robot.eyes.current_orientation()
        eye_quat = quaternions.qmult([w, x, y, z], self.robot.eye_offset_orn)

        return state, sum(self.rewards), bool(done), {"eye_pos": eye_pos, "eye_quat": eye_quat}
