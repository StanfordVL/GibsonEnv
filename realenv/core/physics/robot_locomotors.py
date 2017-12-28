from realenv.core.physics.robot_bases import BaseRobot
import numpy as np
import pybullet as p
import os
import gym, gym.spaces
from transforms3d.euler import euler2quat
import transforms3d.quaternions as quat
import realenv.configs as configs
import sys

OBSERVATION_EPS = 0.01


def quatWXYZ2quatXYZW(wxyz):
    return np.concatenate((wxyz[1:], wxyz[:1]))

def quatXYZW2quatWXYZ(wxyz):
    return np.concatenate((wxyz[-1:], wxyz[:-1]))


class WalkerBase(BaseRobot):
    """ Built on top of BaseRobot
    Handles action_dim, sensor_dim, scene
      base_position, apply_action, calc_state
      reward
    """
    def __init__(self, 
        filename,           # robot file name 
        robot_name,         # robot name
        action_dim,         # action dimension
        power,
        target_pos,
        sensor_dim=None,
        scale = 1, 
        resolution="NORMAL",
        mode="RGBD"
    ):
        BaseRobot.__init__(self, filename, robot_name, scale)

        self.resolution = resolution
        self.obs_dim = None
        if resolution == "SMALL":
            self.obs_dim = [64, 64, 4]
        elif resolution == "XSMALL":
            self.obs_dim = [32, 32, 4]
        elif resolution == "MID":
            self.obs_dim = [128, 128, 4]
        elif resolution == "LARGE":
            self.obs_dim = [512, 512, 4]
        elif resolution == "XLARGE":
            self.obs_dim = [1024, 1024, 4]
        else:
            self.obs_dim = [256, 256, 4]

        if mode=="RGB":
            self.obs_dim[2] = 3
        elif mode=="DEPTH":
            self.obs_dim[2] = 1
        assert type(sensor_dim) == int, "Sensor dimension must be int, got {}".format(type(sensor_dim))
        assert type(action_dim) == int, "Action dimension must be int, got {}".format(type(action_dim))

        action_high = np.ones([action_dim])
        self.action_space = gym.spaces.Box(-action_high, action_high)
        obs_high = np.inf * np.ones(self.obs_dim) + OBSERVATION_EPS
        self.observation_space = gym.spaces.Box(-obs_high, obs_high)
        sensor_high = np.inf * np.ones([sensor_dim])
        self.sensor_space = gym.spaces.Box(-sensor_high, sensor_high)

        self.power = power
        self.camera_x = 0
        self.walk_target_x = target_pos[0]
        self.walk_target_y = target_pos[1]
        self.walk_target_z = target_pos[2]
        self.body_xyz=[0, 0, 0]
        self.eye_offset_orn = euler2quat(0, 0, 0)
        self.action_dim = action_dim
        self.scale = scale
        self.angle_to_target = 0

    def robot_specific_reset(self):
        for j in self.ordered_joints:
            j.reset_current_position(self.np_random.uniform(low=-0.1, high=0.1), 0)

        self.feet = [self.parts[f] for f in self.foot_list]
        self.feet_contact = np.array([0.0 for f in self.foot_list], dtype=np.float32)

        self.scene.actor_introduce(self)
        self.initial_z = None

    def get_position(self):
        return self.robot_body.current_position()

    def get_orientation(self):
        return self.robot_body.current_orientation()

    def reset_base_position(self, enabled, delta_orn = np.pi/3, delta_pos = 0.2):
        #print("Reset base enabled", enabled)
        if not enabled:
            return
        pos = self.robot_body.current_position()
        orn = self.robot_body.current_orientation()

        #print("collision", len(p.getContactPoints(self.robot_body.bodyIndex)))

        #while True:
        new_pos = [ pos[0] + self.np_random.uniform(low=-delta_pos, high=delta_pos),
                    pos[1] + self.np_random.uniform(low=-delta_pos, high=delta_pos),
                    pos[2] + self.np_random.uniform(low=0, high=delta_pos)]
        new_orn = quat.qmult(quat.axangle2quat([1, 0, 0], self.np_random.uniform(low=-delta_orn, high=delta_orn)), orn)
        self.robot_body.reset_orientation(new_orn)
        self.robot_body.reset_position(new_pos)
            #if (len(p.getContactPoints(self.robot_body.bodyIndex)) == 0):
            #    break
            #print("collision", p.getContactPoints(self.robot_body.bodyIndex))
        

    def apply_action(self, a):
        for n, j in enumerate(self.ordered_joints):
            j.set_motor_torque(self.power * j.power_coef * float(np.clip(a[n], -1, +1)))

    def calc_state(self):
        j = np.array([j.current_relative_position() for j in self.ordered_joints], dtype=np.float32).flatten()
        # even elements [0::2] position, scaled to -1..+1 between limits
        # odd elements  [1::2] angular speed, scaled to show -1..+1
        self.joint_speeds = j[1::2]
        self.joints_at_limit = np.count_nonzero(np.abs(j[0::2]) > 0.99)

        body_pose = self.robot_body.pose()
        parts_xyz = np.array([p.pose().xyz() for p in self.parts.values()]).flatten()
        self.body_xyz = (
        parts_xyz[0::3].mean(), parts_xyz[1::3].mean(), body_pose.xyz()[2])  # torso z is more informative than mean z
        self.body_rpy = body_pose.rpy()
        z = self.body_xyz[2]
        if self.initial_z == None:
            self.initial_z = z
        r, p, yaw = self.body_rpy
        self.walk_target_theta = np.arctan2(self.walk_target_y - self.body_xyz[1],
                                            self.walk_target_x - self.body_xyz[0])
        self.walk_target_dist = np.linalg.norm(
            [self.walk_target_y - self.body_xyz[1], self.walk_target_x - self.body_xyz[0]])
        self.walk_target_dist_xyz = np.linalg.norm(
            [self.walk_target_z - self.body_xyz[2], self.walk_target_y - self.body_xyz[1], self.walk_target_x - self.body_xyz[0]])
        angle_to_target = self.walk_target_theta - yaw
        self.angle_to_target = angle_to_target

        self.walk_height_diff = np.abs(self.walk_target_z - self.body_xyz[2])

        self.dist_to_start = np.linalg.norm(np.array(self.body_xyz) - np.array(self.initial_pos))

        debugmode= 0
        if debugmode:
            print("Robot dsebug mode: walk_height_diff", self.walk_height_diff)
            print("Robot dsebug mode: walk_target_z", self.walk_target_z)
            print("Robot dsebug mode: body_xyz", self.body_xyz[2])

        rot_speed = np.array(
            [[np.cos(-yaw), -np.sin(-yaw), 0],
             [np.sin(-yaw), np.cos(-yaw), 0],
             [        0,             0, 1]]
        )
        vx, vy, vz = np.dot(rot_speed, self.robot_body.speed())  # rotate speed back to body point of view

        debugmode=0
        if debugmode:
            print("Robot state", self.walk_target_y - self.body_xyz[1], self.walk_target_x - self.body_xyz[0])

        more = np.array([ z-self.initial_z,
            np.sin(angle_to_target), np.cos(angle_to_target),
            0.3* vx , 0.3* vy , 0.3* vz ,  # 0.3 is just scaling typical speed into -1..+1, no physical sense here
            r, p], dtype=np.float32)

        if debugmode:
            print("Robot more", more)

        if not configs.USE_SENSOR_OUTPUT:
            j.fill(0)

        if not configs.USE_GPS_OUTPUT:
            more.fill(0)
        return np.clip( np.concatenate([more] + [j] + [self.feet_contact]), -5, +5)

    def calc_potential(self):
        # progress in potential field is speed*dt, typical speed is about 2-3 meter per second, this potential will change 2-3 per frame (not per second),
        # all rewards have rew/frame units and close to 1.0 (hzyjerry) ==> make rewards similar scale
        debugmode=0
        if (debugmode):
            print("calc_potential: self.walk_target_dist x y", self.walk_target_dist)
            print("robot position", self.body_xyz, "target position", [self.walk_target_x, self.walk_target_y, self.walk_target_z])
#            print("self.scene.dt")
#            print(self.scene.dt)
#            print("self.scene.frame_skip")
#            print(self.scene.frame_skip)
            #print("self.scene.timestep", self.scene.timestep)
        return - self.walk_target_dist / self.scene.dt



    def calc_goalless_potential(self):
        return self.dist_to_start / self.scene.dt


    def is_close_to_goal(self):
        body_pose = self.robot_body.pose()
        parts_xyz = np.array([p.pose().xyz() for p in self.parts.values()]).flatten()
        self.body_xyz = (
        parts_xyz[0::3].mean(), parts_xyz[1::3].mean(), body_pose.xyz()[2])  # torso z is more informative than mean z
        dist_to_goal = np.linalg.norm([self.body_xyz[0] - self.walk_target_x, self.body_xyz[1] - self.walk_target_y])
        #print("dist to goal", dist_to_goal)
        #print(self.body_xyz[0], self.walk_target_x, self.body_xyz[1], self.walk_target_y)
        #print(self.body_xyz)
        return dist_to_goal < 2

class Hopper(WalkerBase):
    foot_list = ["foot"]

    def __init__(self):
        self.model_type = "MJCF"
        self.mjcf_scaling = 1
        WalkerBase.__init__(self, "hopper.xml", "torso", action_dim=3, sensor_dim=15, power=0.75, scale=self.mjcf_scaling)

    def alive_bonus(self, z, pitch):
        return +1 if z > 0.8 and abs(pitch) < 1.0 else -1


class Walker2D(WalkerBase):
    foot_list = ["foot", "foot_left"]

    def __init__(self):
        self.model_type = "MJCF"
        self.mjcf_scaling = 1
        WalkerBase.__init__(self, "walker2d.xml", "torso", action_dim=6, sensor_dim=22, power=0.40, scale=self.mjcf_scaling)

    def alive_bonus(self, z, pitch):
        return +1 if z > 0.8 and abs(pitch) < 1.0 else -1

    def robot_specific_reset(self):
        WalkerBase.robot_specific_reset(self)
        for n in ["foot_joint", "foot_left_joint"]:
            self.jdict[n].power_coef = 30.0


class HalfCheetah(WalkerBase):
    foot_list = ["ffoot", "fshin", "fthigh",  "bfoot", "bshin", "bthigh"]  # track these contacts with ground

    def __init__(self):
        self.model_type = "MJCF"
        self.mjcf_scaling = 1
        WalkerBase.__init__(self, "half_cheetah.xml", "torso", action_dim=6, sensor_dim=26, power=0.90, scale=self.mjcf_scaling)

    def alive_bonus(self, z, pitch):
        # Use contact other than feet to terminate episode: due to a lot of strange walks using knees
        return +1 if np.abs(pitch) < 1.0 and not self.feet_contact[1] and not self.feet_contact[2] and not self.feet_contact[4] and not self.feet_contact[5] else -1

    def robot_specific_reset(self):
        WalkerBase.robot_specific_reset(self)
        self.jdict["bthigh"].power_coef = 120.0
        self.jdict["bshin"].power_coef  = 90.0
        self.jdict["bfoot"].power_coef  = 60.0
        self.jdict["fthigh"].power_coef = 140.0
        self.jdict["fshin"].power_coef  = 60.0
        self.jdict["ffoot"].power_coef  = 30.0


class Ant(WalkerBase):
    foot_list = ['front_left_foot', 'front_right_foot', 'left_back_foot', 'right_back_foot']

    def __init__(self, initial_pos, initial_orn, is_discrete=True, target_pos=[1, 0, 0], resolution="NORMAL", mode="RGBD", power=2.5):
        ## WORKAROUND (hzyjerry): scaling building instead of agent, this is because
        ## pybullet doesn't yet support downscaling of MJCF objects
        self.model_type = "MJCF"
        self.mjcf_scaling = 0.25
        WalkerBase.__init__(self, "ant.xml", "torso", action_dim=8, sensor_dim=28, power=power, target_pos=target_pos, resolution=resolution, scale=self.mjcf_scaling, mode=mode)
        self.is_discrete = is_discrete
        self.initial_pos = initial_pos
        self.initial_orn = initial_orn
        self.eye_offset_orn = euler2quat(np.pi/2, 0, np.pi/2, axes='sxyz')
        self.is_discrete = is_discrete
        self.r_f = 0.1
        if self.is_discrete:
            self.action_space = gym.spaces.Discrete(17)
            self.torque = 10
            ## Hip_1, Ankle_1, Hip_2, Ankle_2, Hip_3, Ankle_3, Hip_4, Ankle_4 
            self.action_list = [[self.r_f * self.torque, 0, 0, 0, 0, 0, 0, 0],
                                [0, self.r_f * self.torque, 0, 0, 0, 0, 0, 0],
                                [0, 0, self.r_f * self.torque, 0, 0, 0, 0, 0],
                                [0, 0, 0, self.r_f * self.torque, 0, 0, 0, 0],
                                [0, 0, 0, 0, self.r_f * self.torque, 0, 0, 0],
                                [0, 0, 0, 0, 0, self.r_f * self.torque, 0, 0],
                                [0, 0, 0, 0, 0, 0, self.r_f * self.torque, 0],
                                [0, 0, 0, 0, 0, 0, 0, self.r_f * self.torque],
                                [-self.r_f * self.torque, 0, 0, 0, 0, 0, 0, 0],
                                [0, -self.r_f * self.torque, 0, 0, 0, 0, 0, 0],
                                [0, 0, -self.r_f * self.torque, 0, 0, 0, 0, 0],
                                [0, 0, 0, -self.r_f * self.torque, 0, 0, 0, 0],
                                [0, 0, 0, 0, -self.r_f * self.torque, 0, 0, 0],
                                [0, 0, 0, 0, 0, -self.r_f * self.torque, 0, 0],
                                [0, 0, 0, 0, 0, 0, -self.r_f * self.torque, 0],
                                [0, 0, 0, 0, 0, 0, 0, -self.r_f * self.torque],
                                [0, 0, 0, 0, 0, 0, 0, 0]]
            '''
            [[self.r_f * self.torque, 0, 0, -self.r_f * self.torque, 0, 0, 0, 0], 
                                [0, 0, self.r_f * self.torque, self.r_f * self.torque, 0, 0, 0, 0], 
                                [0, 0, 0, 0, self.r_f * self.torque, self.r_f * self.torque, 0, 0], 
                                [0, 0, 0, 0, 0, 0, self.r_f * self.torque, self.r_f * self.torque], 
                                [0, 0, 0, 0, 0, 0, 0, 0]]
            '''
            self.setup_keys_to_action()

    def apply_action(self, action):
        if self.is_discrete:
            realaction = self.action_list[action]
        else:
            realaction = action
        WalkerBase.apply_action(self, realaction)

    def robot_specific_reset(self):
        WalkerBase.robot_specific_reset(self)
        roll  = self.initial_orn[0]
        pitch = self.initial_orn[1]
        yaw   = self.initial_orn[2]
        self.robot_body.reset_orientation(quatWXYZ2quatXYZW(euler2quat(roll, pitch, yaw)))
        self.robot_body.reset_position(self.initial_pos)
        #print("Initial position", self.initial_pos)

        self.reset_base_position(configs.RANDOM_INITIAL_POSE, delta_pos = 0.25)

    def alive_bonus(self, z, pitch):
        return +1 if z > 0.26 else -1  # 0.25 is central sphere rad, die if it scrapes the ground

    def setup_keys_to_action(self):
        self.keys_to_action = {
            #(ord('s'), ): 0, ## backward
            #(ord('w'), ): 1, ## forward
            #(ord('d'), ): 2, ## turn right
            #(ord('a'), ): 3, ## turn left
            (ord('1'), ): 0,
            (ord('2'), ): 1, 
            (ord('3'), ): 2, 
            (ord('4'), ): 3, 
            (ord('5'), ): 4, 
            (ord('6'), ): 5, 
            (ord('7'), ): 6, 
            (ord('8'), ): 7, 
            (ord('q'), ): 8, 
            (ord('w'), ): 9, 
            (ord('e'), ): 10, 
            (ord('r'), ): 11, 
            (ord('t'), ): 12, 
            (ord('y'), ): 13, 
            (ord('u'), ): 14, 
            (ord('i'), ): 15, 
            (): 4
        }


class AntClimber(Ant):
    def __init__(self, initial_pos, initial_orn, is_discrete=True, target_pos=[1, 0, 0], resolution="NORMAL", mode="RGBD"):
        Ant.__init__(self, initial_pos, initial_orn, 
            is_discrete=is_discrete, 
            target_pos=target_pos, 
            resolution=resolution, 
            mode=mode, power=5)
        self.eye_offset_orn = euler2quat(np.pi/4, 0, np.pi/2, axes='sxyz')  ## looking 45 degs down
        
    def robot_specific_reset(self):
        Ant.robot_specific_reset(self)
        amplify = 1
        for j in self.jdict.keys():
            self.jdict[j].power_coef *= amplify
        '''
        self.jdict["ankle_1"].power_coef = amplify * self.jdict["ankle_1"].power_coef
        self.jdict["ankle_2"].power_coef = amplify * self.jdict["ankle_2"].power_coef
        self.jdict["ankle_3"].power_coef = amplify * self.jdict["ankle_3"].power_coef
        self.jdict["ankle_4"].power_coef = amplify * self.jdict["ankle_4"].power_coef
        '''
        debugmode=0
        if debugmode:
            for k in self.jdict.keys():
                print("Power coef", self.jdict[k].power_coef)


    def calc_potential(self):
        #base_potential = Ant.calc_potential(self)
        #height_coeff   = 3
        #height_potential = - height_coeff * self.walk_height_diff / self.scene.dt
        debugmode = 0
        if debugmode:
            print("Ant xyz potential", self.walk_target_dist_xyz)
        return - self.walk_target_dist_xyz / self.scene.dt
        

    def alive_bonus(self, roll, pitch):
        """Alive requires the ant's head to not touch the ground, it's roll
        and pitch cannot be too large"""
        #return +1 if z > 0.26 else -1  # 0.25 is central sphere rad, die if it scrapes the ground
        alive = roll < np.pi/2 and roll > -np.pi/2 and pitch > -np.pi/2 and pitch < np.pi/2
        debugmode = 0
        if debugmode:
            print("roll, pitch")
            print(roll, pitch)
            print("alive")
            print(alive)
        return +1 if alive else -1

    def is_close_to_goal(self):
        body_pose = self.robot_body.pose()
        parts_xyz = np.array([p.pose().xyz() for p in self.parts.values()]).flatten()
        self.body_xyz = (parts_xyz[0::3].mean(), parts_xyz[1::3].mean(), body_pose.xyz()[2])  # torso z is more informative than mean z
        dist_to_goal = np.linalg.norm([self.body_xyz[0] - self.walk_target_x, self.body_xyz[1] - self.walk_target_y, self.body_xyz[2] - self.walk_target_z])
        debugmode = 0
        if debugmode:
            print(np.linalg.norm([self.body_xyz[0] - self.walk_target_x, self.body_xyz[1] - self.walk_target_y, self.body_xyz[2] - self.walk_target_z]), [self.body_xyz[0], self.body_xyz[1], self.body_xyz[2]], [self.walk_target_x, self.walk_target_y, self.walk_target_z])
        return dist_to_goal < 0.5


    def reset_base_position(self, enabled, delta_orn = np.pi/9, delta_pos = 0.2):
        #print("Reset base enabled", enabled)
        if not enabled:
            return
        pos = self.robot_body.current_position()
        orn = self.robot_body.current_orientation()

        #print("collision", len(p.getContactPoints(self.robot_body.bodyIndex)))

        #while True:
        new_pos = [ pos[0] + self.np_random.uniform(low=configs.RANDOM_INITIAL_RANGE_X[0], high=configs.RANDOM_INITIAL_RANGE_X[1]),
                    pos[1] + self.np_random.uniform(low=configs.RANDOM_INITIAL_RANGE_Y[0], high=configs.RANDOM_INITIAL_RANGE_Y[1]),
                    pos[2] + self.np_random.uniform(low=configs.RANDOM_INITIAL_RANGE_Z[0], high=configs.RANDOM_INITIAL_RANGE_Z[1])]
        new_orn = quat.qmult(quat.axangle2quat([1, 0, 0], self.np_random.uniform(low=configs.RANDOM_INITIAL_RANGE_DEG[0], high=configs.RANDOM_INITIAL_RANGE_DEG[1])), orn)
        self.robot_body.reset_orientation(new_orn)
        self.robot_body.reset_position(new_pos)





class Humanoid(WalkerBase):
    self_collision = True
    foot_list = ["right_foot", "left_foot"]  # "left_hand", "right_hand"


    def __init__(self, is_discrete, initial_pos, initial_orn, target_pos=[1, 0, 0], resolution="NORMAL"):
        self.model_type = "MJCF"
        self.mjcf_scaling = 0.6
        self.is_discrete = is_discrete
        WalkerBase.__init__(self, 'humanoid.xml', 'torso', action_dim=17, sensor_dim=44, power=0.41, scale = self.mjcf_scaling, target_pos=target_pos, resolution=resolution)
        self.initial_pos = initial_pos
        self.initial_orn = initial_orn
        #self.eye_offset_orn = euler2quat(0, 0, 0, axes='sxyz')
        self.glass_id = None
        if self.is_discrete:
            self.action_space = gym.spaces.Discrete(5)
            ## specific offset for husky.urdf
            #self.eye_offset_orn = euler2quat(np.pi/2, 0, np.pi/2, axes='sxyz')
            self.torque = 0.1
            self.action_list = np.concatenate((np.ones((1, 17)), np.zeros((1, 17)))).tolist()

            self.setup_keys_to_action()

    def robot_specific_reset(self):
        WalkerBase.robot_specific_reset(self)
        
        humanoidId = -1
        numBodies = p.getNumBodies()
        for i in range (numBodies):
            bodyInfo = p.getBodyInfo(i)
            if bodyInfo[1].decode("ascii") == 'humanoid':
                humanoidId = i
        ## Spherical radiance/glass shield to protect the robot's camera
        if self.glass_id is None:
            glass_id = p.loadMJCF(os.path.join(self.physics_model_dir, "glass.xml"))[0]
            #print("setting up glass", glass_id, humanoidId)
            p.changeVisualShape(glass_id, -1, rgbaColor=[0, 0, 0, 0])
            cid = p.createConstraint(humanoidId, -1, glass_id,-1,p.JOINT_FIXED,[0,0,0],[0,0,1.4],[0,0,1])

        self.motor_names  = ["abdomen_z", "abdomen_y", "abdomen_x"]
        self.motor_power  = [100, 100, 100]
        self.motor_names += ["right_hip_x", "right_hip_z", "right_hip_y", "right_knee"]
        self.motor_power += [100, 100, 300, 200]
        self.motor_names += ["left_hip_x", "left_hip_z", "left_hip_y", "left_knee"]
        self.motor_power += [100, 100, 300, 200]
        self.motor_names += ["right_shoulder1", "right_shoulder2", "right_elbow"]
        self.motor_power += [75, 75, 75]
        self.motor_names += ["left_shoulder1", "left_shoulder2", "left_elbow"]
        self.motor_power += [75, 75, 75]
        self.motors = [self.jdict[n] for n in self.motor_names]
        if self.random_yaw:
            position = [0,0,0]
            orientation = [0,0,0]
            yaw = self.np_random.uniform(low=-3.14, high=3.14)
            if self.random_lean and self.np_random.randint(2)==0:
                cpose.set_xyz(0, 0, 1.4)
                if self.np_random.randint(2)==0:
                    pitch = np.pi/2
                    position = [0, 0, 0.45]
                else:
                    pitch = np.pi*3/2
                    position = [0, 0, 0.25]
                roll = 0
                orientation = [roll, pitch, yaw]
            else:
                position = [0, 0, 1.4]
            self.robot_body.reset_position(position)
            # just face random direction, but stay straight otherwise
            self.robot_body.reset_orientation(quatWXYZ2quatXYZW(euler2quat(0, 0, yaw)))
        self.initial_z = 0.8

        roll  = self.initial_orn[0]
        pitch = self.initial_orn[1]
        yaw   = self.initial_orn[2]
        self.robot_body.reset_orientation(quatWXYZ2quatXYZW(euler2quat(roll, pitch, yaw)))
        self.robot_body.reset_position(self.initial_pos)

        self.reset_base_position(configs.RANDOM_INITIAL_POSE)



    random_yaw = False
    random_lean = False

    def apply_action(self, a):
        if self.is_discrete:
            realaction = self.action_list[a]
        else:
            force_gain = 1
            for i, m, power in zip(range(17), self.motors, self.motor_power):
                m.set_motor_torque( float(force_gain * power*self.power*a[i]) )
            #m.set_motor_torque(float(force_gain * power * self.power * np.clip(a[i], -1, +1)))

    def alive_bonus(self, z, pitch):
        return +2 if z > 0.78 else -1   # 2 here because 17 joints produce a lot of electricity cost just from policy noise, living must be better than dying
    
    def setup_keys_to_action(self):
        self.keys_to_action = {
            (ord('w'), ): 0,
            (): 1
        }


class Husky(WalkerBase):
    foot_list = ['front_left_wheel_link', 'front_right_wheel_link', 'rear_left_wheel_link', 'rear_right_wheel_link']


    def __init__(self, is_discrete, initial_pos, initial_orn, target_pos=[1, 0, 0], resolution="NORMAL", mode="RGB"):
        self.model_type = "URDF"
        self.is_discrete = is_discrete
        self.mjcf_scaling = 1
        WalkerBase.__init__(self, "husky.urdf", "base_link", action_dim=4, sensor_dim=20, power=2.5, scale = 0.6, target_pos=target_pos, resolution=resolution, mode=mode)
        self.initial_pos = initial_pos
        self.initial_orn = initial_orn
        self.eye_offset_orn = euler2quat(np.pi / 2, 0, np.pi / 2, axes='sxyz')
        if self.is_discrete:
            self.action_space = gym.spaces.Discrete(5)
        ## specific offset for husky.urdf
        #self.eye_offset_orn = euler2quat(np.pi/2, 0, np.pi/2, axes='sxyz')
            self.torque = 0.1
            self.action_list = [[self.torque/2, self.torque/2, self.torque/2, self.torque/2],
                                #[-self.torque * 2, -self.torque * 2, -self.torque * 2, -self.torque * 2],
                                [-self.torque * 0.9, -self.torque * 0.9, -self.torque * 0.9, -self.torque * 0.9],
                                [self.torque, -self.torque, self.torque, -self.torque],
                                [-self.torque, self.torque, -self.torque, self.torque],
                                [0, 0, 0, 0]]

            self.setup_keys_to_action()
        else:
            action_high = 0.02 * np.ones([4])
            self.action_space = gym.spaces.Box(-action_high, action_high)


        ## specific offset for husky.urdf
        #self.eye_offset_orn = euler2quat(np.pi/2, 0, np.pi/2, axes='sxyz')
    
        
    def apply_action(self, action):
        if self.is_discrete:
            realaction = self.action_list[action]
        else:
            realaction = action
        WalkerBase.apply_action(self, realaction)

    def steering_cost(self, action):
        if not self.is_discrete:
            return 0
        if action == 2 or action == 3:
            return -0.1
        else:
            return 0

    def angle_cost(self):
        angle_const = 0.2
        diff_to_half = np.abs(self.angle_to_target - 1.57)
        is_forward = self.angle_to_target > 1.57
        diff_angle = np.abs(1.57 - diff_to_half) if is_forward else 3.14 - np.abs(1.57 - diff_to_half)
        debugmode = 0
        if debugmode:
            print("is forward", is_forward)
            print("diff to half", diff_to_half)
            print("angle to target", self.angle_to_target)
            print("diff angle", diff_angle)
        return -angle_const* diff_angle


    def robot_specific_reset(self):
        WalkerBase.robot_specific_reset(self)
        roll  = self.initial_orn[0]
        pitch = self.initial_orn[1]
        yaw   = self.initial_orn[2]
        self.robot_body.reset_orientation(quatWXYZ2quatXYZW(euler2quat(roll, pitch, yaw)))
        self.robot_body.reset_position(self.initial_pos)

        self.reset_base_position(configs.RANDOM_INITIAL_POSE)


    def alive_bonus(self, z, pitch):
        return +1 if z > 0.26 else -1  # 0.25 is central sphere rad, die if it scrapes the ground

    def setup_keys_to_action(self):
        self.keys_to_action = {
            (ord('s'), ): 0, ## backward
            (ord('w'), ): 1, ## forward
            (ord('d'), ): 2, ## turn right
            (ord('a'), ): 3, ## turn left
            (): 4
        }


class HuskyClimber(Husky):
    def calc_potential(self):
        base_potential = Husky.calc_potential(self)
        height_potential = - 4 * self.walk_height_diff / self.scene.dt
        print("Husky climber", base_potential, height_potential)
        return base_potential + height_potential

    def robot_specific_reset(self):
        Ant.robot_specific_reset(self)
        for j in self.jdict.keys():
            self.jdict[j].power_coef = 1.5 * self.jdict[j].power_coef
        
        debugmode=0
        if debugmode:
            for k in self.jdict.keys():
                print("Power coef", self.jdict[k].power_coef)

class HuskyHighCamera(Husky):
    def __init__(self, is_discrete, initial_pos, initial_orn, target_pos=[1, 0, 0], resolution="NORMAL", mode="RGB"):
        self.model_type = "URDF"
        self.is_discrete = is_discrete
        self.mjcf_scaling = 1
        WalkerBase.__init__(self, "husky_high.urdf", "base_link", action_dim=4, sensor_dim=20, power=2.5, scale = 0.6, target_pos=target_pos, resolution=resolution, mode=mode)
        self.initial_pos = initial_pos
        self.initial_orn = initial_orn
        self.eye_offset_orn = euler2quat(np.pi / 2, 0, np.pi / 2, axes='sxyz')
        if self.is_discrete:
            self.action_space = gym.spaces.Discrete(5)
        ## specific offset for husky.urdf
        #self.eye_offset_orn = euler2quat(np.pi/2, 0, np.pi/2, axes='sxyz')
            self.torque = 0.1
            self.action_list = [[self.torque/4, self.torque/4, self.torque/4, self.torque/4],
                                #[-self.torque * 2, -self.torque * 2, -self.torque * 2, -self.torque * 2],
                                [-self.torque * 0.9, -self.torque * 0.9, -self.torque * 0.9, -self.torque * 0.9],
                                [self.torque, -self.torque, self.torque, -self.torque],
                                [-self.torque, self.torque, -self.torque, self.torque],
                                [0, 0, 0, 0]]

            self.setup_keys_to_action()
        else:
            action_high = 0.02 * np.ones([4])
            self.action_space = gym.spaces.Box(-action_high, action_high)

        ## specific offset for husky.urdf
        #self.eye_offset_orn = euler2quat(np.pi/2, 0, np.pi/2, axes='sxyz')
    