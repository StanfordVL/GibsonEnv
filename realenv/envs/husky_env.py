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


tracking_camera = {
    'yaw': 90,  # demo: living room, stairs
    #'yaw'; 30,   # demo: kitchen
    'z_offset': 0.5,
    'distance': 1.2,
    'pitch': -10
    # 'pitch': -24  # demo: stairs
}

class HuskyNavigateEnv(CameraRobotEnv):
    """Specfy navigation reward
    """
    def __init__(self, human=True, timestep=HUMANOID_TIMESTEP, 
        frame_skip=HUMANOID_FRAMESKIP, is_discrete=False, mode="RGBD", 
        use_filler=True, gpu_count=0, scene_type="building"):
        
        target_orn, target_pos = INITIAL_POSE["husky"][MODEL_ID][-1]
        self.robot = Husky(is_discrete, target_pos=target_pos)
        self.human = human
        self.timestep = timestep
        self.frame_skip = frame_skip

        CameraRobotEnv.__init__(self, mode, gpu_count, scene_type, use_filler=use_filler)

        self.tracking_camera = tracking_camera

    def calc_rewards_and_done(self, a, state):
       
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
    
    def  _reset(self):
        obs = CameraRobotEnv._reset(self)
        return obs

class HuskyFlagRunEnv(CameraRobotEnv):
    """Specfy flagrun reward
    """
    def __init__(self, human=True, timestep=HUMANOID_TIMESTEP,
                 frame_skip=HUMANOID_FRAMESKIP, is_discrete=False, 
                 gpu_count=0, scene_type="stadium"):
        self.robot = Husky(is_discrete)
        self.human = human
        self.timestep = timestep
        self.frame_skip = frame_skip
        ## Mode initialized with mode=SENSOR
        CameraRobotEnv.__init__(self, "SENSOR", gpu_count, scene_type)

        self.flag_timeout = 1
        self.tracking_camera = tracking_camera

        if self.human:
            self.visualid = p.createVisualShape(p.GEOM_MESH, fileName=os.path.join(pybullet_data.getDataPath(), 'cube.obj'), meshScale=[0.5, 0.5, 0.5], rgbaColor=[1, 0, 0, 0.7])
        self.lastid = None

    def _reset(self):
        obs = CameraRobotEnv._reset(self)
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

    def calc_rewards_and_done(self, a, state):
        potential_old = self.potential
        self.potential = self.robot.calc_potential()
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


        done = alive > 0 or self.nframe > 500

        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
        debugmode = 0
        if (debugmode):
            print("alive=")
            print(alive)
            print("progress")
            print(progress)

        return [
            alive_score,
            progress,
        ], done


    def _step(self, a):
        state, reward, done, meta = CameraRobotEnv._step(self, a)
        if self.flag_timeout <= 0:
            self.flag_reposition()
        self.flag_timeout -= 1

        return state, reward, done, meta


class HuskyFetchEnv(CameraRobotEnv):
    """Specfy flagrun reward
    """
    def __init__(self, human=True, timestep=HUMANOID_TIMESTEP,
                 frame_skip=HUMANOID_FRAMESKIP, is_discrete=False,
                 gpu_count=0, scene_type="building"):
        self.robot = Husky(is_discrete)
        self.human = human
        self.timestep = timestep
        self.frame_skip = frame_skip
        ## Mode initialized with mode=SENSOR
        CameraRobotEnv.__init__(self, "SENSOR", gpu_count, scene_type)

        self.flag_timeout = 1
        self.tracking_camera = tracking_camera

        if self.human:
            self.visualid = p.createVisualShape(p.GEOM_MESH, fileName=os.path.join(pybullet_data.getDataPath(), 'cube.obj'), meshScale=[0.2, 0.2, 0.2], rgbaColor=[1, 0, 0, 0.7])
            self.colisionid = p.createCollisionShape(p.GEOM_MESH, fileName=os.path.join(pybullet_data.getDataPath(), 'cube.obj'), meshScale=[0.2, 0.5, 0.2])

        self.lastid = None

    def _reset(self):
        obs = CameraRobotEnv._reset(self)
        return obs

    def flag_reposition(self):
        #self.walk_target_x = self.np_random.uniform(low=-self.scene.stadium_halflen,
        #                                            high=+self.scene.stadium_halflen)
        #self.walk_target_y = self.np_random.uniform(low=-self.scene.stadium_halfwidth,
        #                                            high=+self.scene.stadium_halfwidth)


        force_x = self.np_random.uniform(-300,300)
        force_y = self.np_random.uniform(-300, 300)

        more_compact = 0.5  # set to 1.0 whole football field
        #self.walk_target_x *= more_compact
        #self.walk_target_y *= more_compact

        startx, starty, _ = self.robot.body_xyz


        self.flag = None
        #self.flag = self.scene.cpp_world.debug_sphere(self.walk_target_x, self.walk_target_y, 0.2, 0.2, 0xFF8080)
        self.flag_timeout = 600 / self.scene.frame_skip
        #print('targetxy', self.flagid, self.walk_target_x, self.walk_target_y, p.getBasePositionAndOrientation(self.flagid))
        #p.resetBasePositionAndOrientation(self.flagid, posObj = [self.walk_target_x, self.walk_target_y, 0.5], ornObj = [0,0,0,0])
        if self.human:
            if self.lastid:
                p.removeBody(self.lastid)

            self.lastid = p.createMultiBody(baseMass = 1, baseVisualShapeIndex=self.visualid, baseCollisionShapeIndex=self.colisionid, basePosition=[startx, starty, 0.5])
            p.applyExternalForce(self.lastid, -1, [force_x,force_y,50], [0,0,0], p.LINK_FRAME)

        ball_xyz, _ = p.getBasePositionAndOrientation(self.lastid)

        self.robot.walk_target_x = ball_xyz[0]
        self.robot.walk_target_y = ball_xyz[1]

    def calc_rewards_and_done(self, a, state):
        if self.lastid:
            ball_xyz, _ = p.getBasePositionAndOrientation(self.lastid)
            self.robot.walk_target_x = ball_xyz[0]
            self.robot.walk_target_y = ball_xyz[1]


        potential_old = self.potential
        self.potential = self.robot.calc_potential()
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


        done = alive > 0 or self.nframe > 500

        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
        debugmode = 0
        if (debugmode):
            print("alive=")
            print(alive)
            print("progress")
            print(progress)

        return [
            alive_score,
            progress,
        ], done


    def _step(self, a):
        state, reward, done, meta = CameraRobotEnv._step(self, a)
        if self.flag_timeout <= 0:
            self.flag_reposition()
        self.flag_timeout -= 1

        return state, reward, done, meta


class HuskyFetchKernelizedRewardEnv(CameraRobotEnv):
    """Specfy flagrun reward
    """
    def __init__(self, human=True, timestep=HUMANOID_TIMESTEP,
                 frame_skip=HUMANOID_FRAMESKIP, is_discrete=False,
                 gpu_count=0, scene_type="building"):
        self.robot = Husky(is_discrete)
        self.human = human
        self.timestep = timestep
        self.frame_skip = frame_skip
        ## Mode initialized with mode=SENSOR
        CameraRobotEnv.__init__(self, "SENSOR", gpu_count, scene_type)

        self.flag_timeout = 1
        self.tracking_camera = tracking_camera

        if self.human:
            self.visualid = p.createVisualShape(p.GEOM_MESH, fileName=os.path.join(pybullet_data.getDataPath(), 'cube.obj'), meshScale=[0.2, 0.2, 0.2], rgbaColor=[1, 0, 0, 0.7])
            self.colisionid = p.createCollisionShape(p.GEOM_MESH, fileName=os.path.join(pybullet_data.getDataPath(), 'cube.obj'), meshScale=[0.2, 0.5, 0.2])

        self.lastid = None

    def _reset(self):
        obs = CameraRobotEnv._reset(self)
        return obs

    def flag_reposition(self):
        #self.walk_target_x = self.np_random.uniform(low=-self.scene.stadium_halflen,
        #                                            high=+self.scene.stadium_halflen)
        #self.walk_target_y = self.np_random.uniform(low=-self.scene.stadium_halfwidth,
        #                                            high=+self.scene.stadium_halfwidth)


        force_x = self.np_random.uniform(-300,300)
        force_y = self.np_random.uniform(-300, 300)

        more_compact = 0.5  # set to 1.0 whole football field
        #self.walk_target_x *= more_compact
        #self.walk_target_y *= more_compact

        startx, starty, _ = self.robot.body_xyz


        self.flag = None
        #self.flag = self.scene.cpp_world.debug_sphere(self.walk_target_x, self.walk_target_y, 0.2, 0.2, 0xFF8080)
        self.flag_timeout = 600 / self.scene.frame_skip
        #print('targetxy', self.flagid, self.walk_target_x, self.walk_target_y, p.getBasePositionAndOrientation(self.flagid))
        #p.resetBasePositionAndOrientation(self.flagid, posObj = [self.walk_target_x, self.walk_target_y, 0.5], ornObj = [0,0,0,0])
        if self.human:
            if self.lastid:
                p.removeBody(self.lastid)

            self.lastid = p.createMultiBody(baseMass = 1, baseVisualShapeIndex=self.visualid, baseCollisionShapeIndex=self.colisionid, basePosition=[startx, starty, 0.5])
            p.applyExternalForce(self.lastid, -1, [force_x,force_y,50], [0,0,0], p.LINK_FRAME)

        ball_xyz, _ = p.getBasePositionAndOrientation(self.lastid)

        self.robot.walk_target_x = ball_xyz[0]
        self.robot.walk_target_y = ball_xyz[1]

    def calc_rewards_and_done(self, a, state):
        if self.lastid:
            ball_xyz, _ = p.getBasePositionAndOrientation(self.lastid)
            self.robot.walk_target_x = ball_xyz[0]
            self.robot.walk_target_y = ball_xyz[1]


        potential_old = self.potential
        self.potential = self.robot.calc_potential()
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


        done = alive > 0 or self.nframe > 500

        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
        debugmode = 0
        if (debugmode):
            print("alive=")
            print(alive)
            print("progress")
            print(progress)

        return [
            alive_score,
            progress,
        ], done


    def _step(self, a):
        state, reward, done, meta = CameraRobotEnv._step(self, a)
        if self.flag_timeout <= 0:
            self.flag_reposition()
        self.flag_timeout -= 1

        return state, reward, done, meta
