from gibson.envs.env_modalities import CameraRobotEnv, BaseRobotEnv
from gibson.envs.env_bases import *
from gibson.core.physics.robot_locomotors import Humanoid, Ant
from transforms3d import quaternions
import os
import numpy as np
import sys
import pybullet as p
from gibson.core.physics.scene_stadium import SinglePlayerStadiumScene
import pybullet_data


tracking_camera = {
    'yaw': 20,  # demo: living room, stairs
    #'yaw'; 30,   # demo: kitchen
    'z_offset': 0.5,
    'distance': 2,
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
            config,
            human=True,
            is_discrete=False, 
            use_filler=True,
            gpu_count=0, 
            resolution=512):
        self.config = self.parse_config(config)
        self.human = human
        self.model_id = self.config["model_id"]
        self.timestep = self.config["speed"]["timestep"]
        self.frame_skip = self.config["speed"]["frameskip"]
        self.resolution = resolution
        self.tracking_camera = tracking_camera
        target_orn, target_pos = self.config["target_orn"], self.config["target_pos"]
        initial_orn, initial_pos = self.config["initial_orn"], self.config["initial_pos"]

        CameraRobotEnv.__init__(
            self,
            config,
            gpu_count,
            scene_type="building",
            use_filler=self.config["use_filler"])


        self.robot_introduce(Humanoid(
            initial_pos,
            initial_orn,
            is_discrete=is_discrete,
            target_pos=target_pos,
            resolution=self.resolution,
            env = self))

        self.scene_introduce()

        self.total_reward = 0
        self.total_frame = 0

    def calc_rewards_and_done(self, a, state):
        done = self._termination(state)
        rewards = self._rewards(a)
        print("Frame %f reward %f" % (self.nframe, sum(rewards)))
        self.total_reward = self.total_reward + sum(rewards)
        self.total_frame = self.total_frame + 1
        return rewards, done

    def _rewards(self, action=None, debugmode=False):
        a = action
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
        
        if(debugmode):
            print("progress")
            print(progress)
            print("electricity_cost")
            print(electricity_cost)
            print("joints_at_limit_cost")
            print(joints_at_limit_cost)
            print("feet_collision_cost")
            print(feet_collision_cost)

        rewards =[
            #alive,
            progress,
            electricity_cost,
            joints_at_limit_cost,
            feet_collision_cost
            ]
        return rewards

    def _termination(self, state=None, debugmode=False):
        alive = float(self.robot.alive_bonus(state[0]+self.robot.initial_z, self.robot.body_rpy[1])) # state[0] is body height above ground, body_rpy[1] is pitch
        done = alive < 0
        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True
        if(debugmode):
            print("alive=")
            print(alive)
        return done

    def _flag_reposition(self):
        walk_target_x = self.robot.walk_target_x
        walk_target_y = self.robot.walk_target_y

        self.flag = None
        if self.human and not self.config["display_ui"]:
            self.visual_flagId = p.createVisualShape(p.GEOM_MESH, fileName=os.path.join(pybullet_data.getDataPath(), 'cube.obj'), meshScale=[0.5, 0.5, 0.5], rgbaColor=[1, 0, 0, 0])
            self.last_flagId = p.createMultiBody(baseVisualShapeIndex=self.visual_flagId, baseCollisionShapeIndex=-1, basePosition=[walk_target_x, walk_target_y, 0.5])
        
    def _reset(self):
        self.total_frame = 0
        self.total_reward = 0
        obs = CameraRobotEnv._reset(self)
        self._flag_reposition()
        return obs


class HumanoidGibsonFlagRunEnv(CameraRobotEnv):
    """Specfy flagrun reward
    """

    def __init__(
            self,
            config,
            human=True,
            is_discrete=False,
            gpu_count=0,
            scene_type="building",
            ):

        self.config = self.parse_config(config)
        self.human = human
        self.model_id = self.config["model_id"]
        self.timestep = self.config["speed"]["timestep"]
        self.frame_skip = self.config["speed"]["frameskip"]
        self.resolution = self.config["resolution"]
        self.tracking_camera = tracking_camera
        target_orn, target_pos = self.config["target_orn"], self.config["target_pos"]
        initial_orn, initial_pos = self.config["initial_orn"], self.config["initial_pos"]
        self.total_reward = 0
        self.total_frame = 0

        self.flag_timeout = 1
        self.visualid = -1
        self.lastid = None

        CameraRobotEnv.__init__(
            self,
            config,
            gpu_count,
            scene_type="building")

        self.robot_introduce(Humanoid(
            initial_pos,
            initial_orn,
            is_discrete=is_discrete,
            target_pos=target_pos,
            resolution=self.resolution,
            env = self))

        self.scene_introduce()

        if self.human:
            self.visualid = p.createVisualShape(p.GEOM_MESH,
                                                fileName=os.path.join(pybullet_data.getDataPath(), 'cube.obj'),
                                                meshScale=[0.2, 0.2, 0.2], rgbaColor=[1, 0, 0, 0.7])
        self.colisionid = p.createCollisionShape(p.GEOM_MESH,
                                                 fileName=os.path.join(pybullet_data.getDataPath(), 'cube.obj'),
                                                 meshScale=[0.2, 0.5, 0.2])
        assert (self.config["envname"] == self.__class__.__name__ or self.config["envname"] == "TestEnv")

    def _reset(self):
        obs = CameraRobotEnv._reset(self)
        return obs

    def _flag_reposition(self):
        # self.walk_target_x = self.np_random.uniform(low=-self.scene.stadium_halflen,
        #                                            high=+self.scene.stadium_halflen)
        # self.walk_target_y = self.np_random.uniform(low=-self.scene.stadium_halfwidth,
        #                                            high=+self.scene.stadium_halfwidth)
        force_x = self.np_random.uniform(-300, 300)
        force_y = self.np_random.uniform(-300, 300)

        more_compact = 0.5  # set to 1.0 whole football field
        # self.walk_target_x *= more_compact
        # self.walk_target_y *= more_compact

        startx, starty, _ = self.robot.body_xyz

        self.flag = None
        # self.flag = self.scene.cpp_world.debug_sphere(self.walk_target_x, self.walk_target_y, 0.2, 0.2, 0xFF8080)
        self.flag_timeout = 3000 / self.scene.frame_skip
        # print('targetxy', self.flagid, self.walk_target_x, self.walk_target_y, p.getBasePositionAndOrientation(self.flagid))
        # p.resetBasePositionAndOrientation(self.flagid, posObj = [self.walk_target_x, self.walk_target_y, 0.5], ornObj = [0,0,0,0])
        if self.lastid:
            p.removeBody(self.lastid)

        self.lastid = p.createMultiBody(baseMass=1, baseVisualShapeIndex=self.visualid,
                                        baseCollisionShapeIndex=self.colisionid, basePosition=[startx, starty, 0.5])
        p.applyExternalForce(self.lastid, -1, [force_x, force_y, 50], [0, 0, 0], p.LINK_FRAME)

        ball_xyz, _ = p.getBasePositionAndOrientation(self.lastid)

        self.robot.walk_target_x = ball_xyz[0]
        self.robot.walk_target_y = ball_xyz[1]

    def calc_rewards_and_done(self, a, state):
        done = self._termination(state)
        rewards = self._rewards(a)
        print("Frame %f reward %f" % (self.nframe, sum(rewards)))
        self.total_reward = self.total_reward + sum(rewards)
        self.total_frame = self.total_frame + 1

        if self.lastid:
            ball_xyz, _ = p.getBasePositionAndOrientation(self.lastid)
            self.robot.walk_target_x = ball_xyz[0]
            self.robot.walk_target_y = ball_xyz[1]

        return rewards, done

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
        electricity_cost = self.electricity_cost * float(np.abs(
            a * self.robot.joint_speeds).mean())  # let's assume we have DC motor with controller, and reverse current braking
        electricity_cost += self.stall_torque_cost * float(np.square(a).mean())
        joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)

        if (debugmode):
            print("progress")
            print(progress)
            print("electricity_cost")
            print(electricity_cost)
            print("joints_at_limit_cost")
            print(joints_at_limit_cost)
            print("feet_collision_cost")
            print(feet_collision_cost)

        rewards = [
            # alive,
            progress,
            electricity_cost,
            joints_at_limit_cost,
            feet_collision_cost
        ]
        return rewards

    def _termination(self, state=None, debugmode=False):
        alive = float(self.robot.alive_bonus(state[0] + self.robot.initial_z, self.robot.body_rpy[
            1]))  # state[0] is body height above ground, body_rpy[1] is pitch
        done = alive < 0
        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True
        if (debugmode):
            print("alive=")
            print(alive)
        return done

    def _step(self, a):
        state, reward, done, meta = CameraRobotEnv._step(self, a)
        if self.flag_timeout <= 0:
            self._flag_reposition()
        self.flag_timeout -= 1

        return state, reward, done, meta
