## Issue related to time resolution/smoothness
#  http://bulletphysics.org/mediawiki-1.5.8/index.php/Stepping_The_World

import pybullet as p
import time
import random
import zmq
import math
import argparse
import os
import json
import numpy as np
import settings
from transforms3d import euler, quaternions
from realenv.core.physics.physics_object import PhysicsObject
from realenv.core.render.profiler import Profiler


class PhysicsEnv(object):
    metadata = {
        'render.modes': ['human_eye', 'rgb_array', 'human_play'],
        'video.frames_per_second' : 20
    }

    def __init__(self, obj_path, render_mode, pose, fps = 12):
        self.debug_sliders = {}
        self.r_mode     = render_mode
        self.init_pose  = pose

        ## The visual render runs at fps, but in order to avoid wall crossing
        ## and other issues in physics renderer, we need to keep physics
        ## engine running at a minimum 100 frames per sec
        self.step_count  = int(math.ceil(100.0 / fps))
        self.time_step   = 1.0 / (fps * self.step_count)
        
        self._setup_context(obj_path)
        self._set_gravity()
        self._set_frame_skip()
        self._update_debug_panels()
        self._reset(pose)


    def _reset(self, pose=None):
        file_dir = os.path.dirname(os.path.abspath(__file__))
        self.objectUid = p.loadURDF(os.path.join(file_dir, "models/quadrotor.urdf"), globalScaling = 0.8)
        #self.objectUid = p.loadURDF(os.path.join(file_dir, "models/husky.urdf"), globalScaling = 0.8)

        self.target_pos = np.array([-4.35, -1.71, 0.8])
        v_t = 1             # 1m/s max speed
        v_r = np.pi/5       # 36 degrees/s

        if pose:
            pos, quat_xyzw = pose[0], pose[1]
        else:
            pos, quat_xyzw = pose_init[0], pose_init[1]
        self.hero = PhysicsObject(self.objectUid, p, pos, quat_xyzw, v_t, v_r)
        print("Generated cart", self.objectUid)


    def _set_gravity(self):
        """Subclass can override this method, for different modes"""
        p.setGravity(0,0,-10)


    def _set_frame_skip(self):
        p.setTimeStep(self.time_step)


    def _setup_context(self, obj_path):
        if self.r_mode == "human_eye" or self.r_mode == "human_play":
            p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
            collisionId = p.createCollisionShape(p.GEOM_MESH, fileName=obj_path, meshScale=[1, 1, 1], flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
            visualId = p.createVisualShape(p.GEOM_MESH, fileName=obj_path, meshScale=[1, 1, 1], rgbaColor = [1, 0.2, 0.2, 0.3], specularColor=[0.4, 4.0])
            boundaryUid = p.createMultiBody(baseCollisionShapeIndex = collisionId, baseVisualShapeIndex = visualId)
            p.changeVisualShape(boundaryUid, -1, rgbaColor=[1, 0.2, 0.2, 0.3], specularColor=[1, 1, 1])
        else:
            p.connect(p.DIRECT)
            print("setting up direct mode")
            collisionId = p.createCollisionShape(p.GEOM_MESH, fileName=obj_path, meshScale=[1, 1, 1], flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
            visualId = 0
            boundaryUid = p.createMultiBody(baseCollisionShapeIndex = collisionId, baseVisualShapeIndex = visualId)

        p.setRealTimeSimulation(0)


    def _render(self, action, restart=False):
        """Execute one frame"""
        if self.r_mode == "human_eye" or self.r_mode == "rgb_array":
            self.hero.parseActionAndUpdate(action)
        elif self.r_mode == "human_play":
            self.hero.getUpdateFromKeyboard(restart=restart)
        else:
            raise Exception 

        print("step count", self.step_count)
        with Profiler("Physics internal"):
            for s in range(self.step_count):
                p.stepSimulation()
        
        self._update_debug_panels()

        pos_xyz, quat_wxyz = self.hero.getViewPosAndOrientation()
        state = {
            'distance_to_target': np.sum(np.square(pos_xyz - self.target_pos))
        }
        print(pos_xyz)
        return [pos_xyz, quat_wxyz], state        

    def _update_debug_panels(self):
        if not (self.r_mode == "human_eye" or self.r_mode == "human_play"):
            return

        if not self.debug_sliders:
            cameraDistSlider  = p.addUserDebugParameter("Distance",0,15,4)
            cameraYawSlider   = p.addUserDebugParameter("Camera Yaw",-180,180,-45)
            cameraPitchSlider = p.addUserDebugParameter("Camera Pitch",-90,90,-30)
            self.debug_sliders = {
                'dist' :cameraDistSlider,
                'yaw'  : cameraYawSlider,
                'pitch': cameraPitchSlider
            }
            self.viewMatrix = p.computeViewMatrixFromYawPitchRoll([0, 0, 0], 10, 0, 90, 0, 2)
            self.projMatrix = p.computeProjectionMatrix(-0.01, 0.01, -0.01, 0.01, 0.01, 128)
            p.getCameraImage(256, 256, viewMatrix = self.viewMatrix, projectionMatrix = self.projMatrix)

        else:
            cameraDist = p.readUserDebugParameter(self.debug_sliders['dist'])
            cameraYaw  = p.readUserDebugParameter(self.debug_sliders['yaw'])
            cameraPitch = p.readUserDebugParameter(self.debug_sliders['pitch'])
            
            pos_xyz, quat_wxyz = self.hero.getViewPosAndOrientation()
            p.getCameraImage(256, 256, viewMatrix = self.viewMatrix, projectionMatrix = self.projMatrix)
            p.resetDebugVisualizerCamera(cameraDist, cameraYaw, cameraPitch, pos_xyz)


    def _camera_init_orientation(self, quat):
        to_z_facing = euler.euler2quat(np.pi/2, np.pi, 0)
        return quaternions.qmult(to_z_facing, quat_wxyz)

    def _stepNsteps(self, N, pObject):
        for _ in range(N):
            p.stepSimulation()
            pObject.parseActionAndUpdate()
        pObject.clearUpDelta()
