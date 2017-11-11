from realenv.envs.env_modalities import CameraRobotEnv, SensorRobotEnv
from realenv.envs.env_bases import *
from realenv.core.physics.robot_locomotors import Humanoid
from transforms3d import quaternions
from realenv import configs
import os
import numpy as np
import sys
import pybullet as p
from realenv.core.physics.scene_stadium import SinglePlayerStadiumScene
import pybullet_data

HUMANOID_TIMESTEP  = 1.0/(4 * 22)
HUMANOID_FRAMESKIP = 4


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


class HumanoidNavigateEnv(CameraRobotEnv):
    """Specfy navigation reward
    """
    def __init__(
            self, 
            human=True, 
            timestep=HUMANOID_TIMESTEP, 
            frame_skip=HUMANOID_FRAMESKIP, 
            is_discrete=False, 
            mode="RGBD", 
            use_filler=True, 
            gpu_count=0, 
            resolution="NORMAL"):
        self.human = human
        self.model_id = configs.NAVIGATE_MODEL_ID
        self.timestep = timestep
        self.frame_skip = frame_skip
        self.resolution = resolution
        self.tracking_camera = tracking_camera
        target_orn, target_pos   = configs.INITIAL_POSE["humanoid"][configs.NAVIGATE_MODEL_ID][-1]
        initial_orn, initial_pos = configs.INITIAL_POSE["humanoid"][configs.NAVIGATE_MODEL_ID][0]
        self.robot = Humanoid(
            is_discrete=is_discrete, 
            initial_pos=initial_pos,
            initial_orn=initial_orn,
            target_pos=target_pos,
            resolution=resolution)
        CameraRobotEnv.__init__(
            self, 
            mode, 
            gpu_count, 
            scene_type="building", 
            use_filler=use_filler)
        self.total_reward = 0
        self.total_frame = 0


        
    def calc_rewards_and_done(self, a, state):
        alive = float(self.robot.alive_bonus(state[0]+self.robot.initial_z, self.robot.body_rpy[1]))   # state[0] is body height above ground, body_rpy[1] is pitch
        done = alive < 0
        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        potential_old = self.potential
        self.potential = self.robot.calc_potential()
        progress = float(self.potential - potential_old)

        feet_collision_cost = 0.0
        for i,f in enumerate(self.robot.feet): # TODO: Maybe calculating feet contacts could be done within the robot code
            #print(f.contact_list())
            contact_ids = set((x[2], x[4]) for x in f.contact_list())
            #print("CONTACT OF '%d' WITH %d" % (contact_ids, ",".join(contact_names)) )
            if (self.ground_ids & contact_ids):
                            #see Issue 63: https://github.com/openai/roboschool/issues/63
                #feet_collision_cost += self.foot_collision_cost
                self.robot.feet_contact[i] = 1.0
            else:
                self.robot.feet_contact[i] = 0.0
        #print(self.robot.feet_contact)


        electricity_cost  = self.electricity_cost  * float(np.abs(a*self.robot.joint_speeds).mean())  # let's assume we have DC motor with controller, and reverse current braking
        electricity_cost += self.stall_torque_cost * float(np.square(a).mean())
        

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
        debugmode=0
        if(debugmode):
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

        rewards =[
            alive,
            progress,
            electricity_cost,
            joints_at_limit_cost,
            feet_collision_cost
            ]
            
        print("Frame %f reward %f" % (self.nframe, sum(rewards)))

        self.total_reward = self.total_reward + sum(rewards)
        self.total_frame = self.total_frame + 1
        print(self.total_frame, self.total_reward)
        return rewards, done


    def flag_reposition(self):
        walk_target_x = self.robot.walk_target_x
        walk_target_y = self.robot.walk_target_y

        self.flag = None
        if self.human:
            self.visual_flagId = p.createVisualShape(p.GEOM_MESH, fileName=os.path.join(pybullet_data.getDataPath(), 'cube.obj'), meshScale=[0.5, 0.5, 0.5], rgbaColor=[1, 0, 0, 0.7])
            self.last_flagId = p.createMultiBody(baseVisualShapeIndex=self.visual_flagId, baseCollisionShapeIndex=-1, basePosition=[walk_target_x, walk_target_y, 0.5])
        
    def  _reset(self):
        self.total_frame = 0
        self.total_reward = 0
        obs = CameraRobotEnv._reset(self)
        self.flag_reposition()
        return obs



'''

class HumanoidCameraEnv(HumanoidEnv, CameraRobotEnv):
    def __init__(self, human=True, timestep=HUMANOID_TIMESTEP, 
        frame_skip=HUMANOID_FRAMESKIP, enable_sensors=False, 
        mode='RGBD', use_filler=True):
        self.human = human
        self.timestep = timestep
        self.frame_skip = frame_skip
        self.enable_sensors = enable_sensors
        HumanoidEnv.__init__(self, mode)
        CameraRobotEnv.__init__(self, use_filler)
        #self.tracking_camera['yaw'] = 30    ## living room
        #self.tracking_camera['distance'] = 1.5
        #self.tracking_camera['pitch'] = -45 ## stairs

        #distance=2.5 ## demo: living room ,kitchen
        self.tracking_camera['distance'] = 1.3   ## demo: stairs
        self.tracking_camera['pitch'] = -35 ## stairs

        #yaw = 0     ## demo: living room
        #yaw = 30    ## demo: kitchen
        self.tracking_camera['yaw'] = -60     ## demo: stairs


class HumanoidSensorEnv(HumanoidEnv, SensorRobotEnv):
    def __init__(self, human=True, timestep=HUMANOID_TIMESTEP, 
        frame_skip=HUMANOID_FRAMESKIP, enable_sensors=False):
        self.human = human
        self.timestep = timestep
        self.frame_skip = frame_skip
        self.enable_sensors = enable_sensors
        HumanoidEnv.__init__(self)
        SensorRobotEnv.__init__(self)
        self.tracking_camera['distance'] = 1.3   ## demo: stairs
        self.tracking_camera['pitch'] = -35 ## stairs

        #yaw = 0     ## demo: living room
        #yaw = 30    ## demo: kitchen
        self.tracking_camera['yaw'] = -60     ## demo: stairs
'''