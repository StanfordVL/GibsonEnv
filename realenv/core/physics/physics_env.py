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
from realenv.core.physics.env_bases import MJCFBaseBulletEnv
from realenv.core.physics.robot_locomotors import Humanoid, Ant, Husky
from .scene_building import SinglePlayerBuildingScene
import gym



class PhysicsExtendedEnv(MJCFBaseBulletEnv):
    def __init__(self, robot, render=False):
        print("PhysicsExtendedEnv::__init__")
        MJCFBaseBulletEnv.__init__(self, robot, render)
        
        self.camera_x = 0
        self.walk_target_x = 1e3  # kilometer away
        self.walk_target_y = 0

    def create_single_player_scene(self):
        self.building_scene = SinglePlayerBuildingScene(gravity=9.8, timestep=0.0165/4, frame_skip=4)
        return self.building_scene

    def _reset(self):
        
        r = MJCFBaseBulletEnv._reset(self)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)

        self.parts, self.jdict, self.ordered_joints, self.robot_body = self.robot.addToScene(
            self.building_scene.building_obj)
        #print(self.parts)
        #self.ground_ids = set([(self.parts[f].bodies[self.parts[f].bodyIndex], self.parts[f].bodyPartIndex) for f in self.foot_ground_object_names])
        self.ground_ids = set([(self.building_scene.building_obj, 0)])
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
        return r

    def move_robot(self, init_x, init_y, init_z):
        "Used by multiplayer building to move sideways, to another running lane."
        self.cpp_robot.query_position()
        pose = self.cpp_robot.root_part.pose()
        pose.move_xyz(init_x, init_y, init_z)  # Works because robot loads around (0,0,0), and some robots have z != 0 that is left intact
        self.cpp_robot.set_pose(pose)

    electricity_cost     = -2.0 # cost for using motors -- this parameter should be carefully tuned against reward for making progress, other values less improtant
    stall_torque_cost   = -0.1  # cost for running electric current through a motor even at zero rotational speed, small
    foot_collision_cost  = -1.0 # touches another leg, or other objects, that cost makes robot avoid smashing feet into itself
    foot_ground_object_names = set(["buildingFloor"])  # to distinguish ground and other objects
    joints_at_limit_cost = -0.1 # discourage stuck joints

    def _step(self, a):
        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            self.robot.apply_action(a)
            self.scene.global_step()

        state = self.robot.calc_state()  # also calculates self.joints_at_limit

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

        self.rewards = [
            alive,
            progress,
            electricity_cost,
            joints_at_limit_cost,
            feet_collision_cost
            ]
        if (debugmode):
            print("rewards=")
            print(self.rewards)
            print("sum rewards")
            print(sum(self.rewards))
        self.HUD(state, a, done)
        self.reward += sum(self.rewards)

        eye_pos = self.robot.eyes.current_position()
        x, y, z ,w = self.robot.eyes.current_orientation()
        eye_quat = [w, x, y, z]

        return state, sum(self.rewards), bool(done), {"eye_pos":eye_pos, "eye_quat":eye_quat}

    def camera_adjust(self):
        x, y, z = self.body_xyz
        self.camera_x = 0.98*self.camera_x + (1-0.98)*x
        self.camera.move_and_look_at(self.camera_x, y-2.0, 1.4, x, y, 1.0)



class HumanoidWalkingEnv(PhysicsExtendedEnv):
    def __init__(self):
        self.robot = Humanoid()
        PhysicsExtendedEnv.__init__(self, self.robot)
        self.electricity_cost  = 4.25*PhysicsExtendedEnv.electricity_cost
        self.stall_torque_cost = 4.25*PhysicsExtendedEnv.stall_torque_cost

class AntWalkingEnv(PhysicsExtendedEnv):
    def __init__(self):
        self.robot = Ant()
        PhysicsExtendedEnv.__init__(self, self.robot)

class HuskyWalkingEnv(PhysicsExtendedEnv):
    def __init__(self):
        self.robot = Husky()
        PhysicsExtendedEnv.__init__(self, self.robot)






### Legacy code (hzyjerry): keeping here only for reference

class PhysicsEnv(MJCFBaseBulletEnv):
    metadata = {
        'render.modes': ['human_eye', 'rgb_array', 'human_play'],
        'video.frames_per_second' : 20
    }

    def __init__(self, obj_path, render_mode, pose, update_freq = 12):
        MJCFBaseBulletEnv.__init__(self, robot, render=False)
        self.debug_sliders = {}
        self.r_mode     = render_mode
        self.init_pose  = pose

        ## The visual render runs at update_freq, but in order to avoid wall crossing
        ## and other issues in physics renderer, we need to keep physics
        ## engine running at a minimum 100 frames per sec
        self.action_repeat  = int(math.ceil(100.0 / update_freq))
        self.time_step      = 1.0 / (update_freq * self.action_repeat)
        
        self._setup_context(obj_path)
        self._set_gravity()
        self._set_frame_skip()
        self._update_debug_panels()
        self._reset(pose)


    def _reset(self, pose=None):
        r = MJCFBaseBulletEnv._reset(self)
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
        self.robot = PhysicsObject(self.objectUid, p, pos, quat_xyzw, v_t, v_r)
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


    def _step(self, action, restart=False):
        """Execute one frame"""
        if self.r_mode == "human_eye" or self.r_mode == "rgb_array":
            self.robot.parseActionAndUpdate(action)
        elif self.r_mode == "human_play":
            self.robot.getUpdateFromKeyboard(restart=restart)
        else:
            raise Exception 

        print("action repeat", self.action_repeat)
        with Profiler("Physics internal"):
            for s in range(self.action_repeat):
                p.stepSimulation()
        
        self._update_debug_panels()

        pos_xyz, quat_wxyz = self.robot.getViewPosAndOrientation()
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
            
            pos_xyz, quat_wxyz = self.robot.getViewPosAndOrientation()
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
