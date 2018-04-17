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
import pdb

CALC_OBSTACLE_PENALTY = 1

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

class BoxPuttingEnv(CameraRobotEnv):
    def __init__(self, config, gpu_count=0):
        self.config = self.parse_config(config)
        assert(self.config["envname"] == self.__class__.__name__ or self.config["envname"] == "TestEnv")

        CameraRobotEnv.__init__(self, self.config, gpu_count, 
                                scene_type="stadium" if self.config["model_id"]=="stadium" else "building",
                                tracking_camera=tracking_camera)

        self.robot_introduce(Husky(self.config, env=self))
        self.scene_introduce()
        self.total_reward = 0
        self.total_frame = 0
        self.mouse_params = {'box_added' : False,
                'loc' : [100,100],
                'shape_id' : p.createVisualShape(shapeType=p.GEOM_SPHERE,
                    radius=0.1, rgbaColor=[1.,0.,0.,1.])}


    def _rewards(self, action=None, debugmode=False):
        return [0] 

    def calculate_3D_coords(self, mouse_loc):
        camW,camH,viewMat,projMat,_,_,_,_,_,_,_,_ = p.getDebugVisualizerCamera()
        # scale mouse location back to (-1, 1)
        W = 2. * mouse_loc[0] / float(camW) - 1.
        H = -(2. * mouse_loc[1] / float(camH) - 1.)

        # homogenize
        cube_front = np.asarray([W,H,-1.,1.])
        cube_back = np.asarray([W,H,1.,1.])

        # Project Back
        viewMat = np.reshape(viewMat, (4,4))
        projMat = np.reshape(projMat, (4,4))
        p_v_inv = np.linalg.inv(np.matmul(projMat.T, viewMat.T))
        w_front = np.dot(p_v_inv, cube_front) 
        w_back = np.dot(p_v_inv, cube_back)
        w_front = w_front / w_front[-1]
        w_back = w_back / w_back[-1]
        return (w_front[:-1], w_back[:-1])

    def resolve_mouse_event(self):
        events = p.getMouseEvents()
        mouse_move_events = [e for e in events if e[0] == 1]
        if len(mouse_move_events) != 0:
            mouse_curr_loc = mouse_move_events[-1][1:3] 
            self.mouse_params['loc'] = mouse_curr_loc
        else:
            mouse_curr_loc = self.mouse_params['loc']
        # Calculate Box Location based on camera matrixes
        x = self.calculate_3D_coords(mouse_curr_loc)
        obj_id, _, _, hit_pos, hit_normal = p.rayTest(x[0], x[1])[0]
        if obj_id == -1:
            return

        # if currently having a box on screen
        if self.mouse_params['box_added']:
            # remove current viability array

            # move box to curr_loc
            p.resetBasePositionAndOrientation(self.mouse_params['v_box'], posObj=hit_pos, ornObj=[1,1,1,1]) 
            # add current viability array

            for e in events:
                if e[0] == 2 and e[3] == 2 and e[-1] == 3:
                    # record action

                    # remove box
                    p.removeBody(self.mouse_params['v_box']) 
                    self.mouse_params['box_added'] = False
                    return
        else:
            # iterate through event, check if box should be added
            # adding a box is signaled by Right Mouse Click (not hold)
            for e in events:
                if e[0] == 2 and e[3] == 2 and e[-1] == 3:
                    # added box
                    self.mouse_params['v_box'] = p.createMultiBody(baseMass=1,
                            baseInertialFramePosition=[0,0,0],
                            baseVisualShapeIndex = self.mouse_params['shape_id'],
                            basePosition = hit_pos) 
                    self.mouse_params['box_added'] = True
                    return

            

    def _step(self, a):
        t = time.time()
        base_obs, sensor_reward, done, sensor_meta = CameraRobotEnv._step(self, a)
        self.resolve_mouse_event()
        return base_obs, sensor_reward, done, sensor_meta
 
    def _termination(self, debugmode=False):
        height = self.robot.get_position()[2]
        pitch = self.robot.get_rpy()[1]
        alive = float(self.robot.alive_bonus(height, pitch))
        
        alive = len(self.robot.parts['top_bumper_link'].contact_list()) == 0

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

