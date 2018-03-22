from gibson.core.physics.robot_bases import BaseRobot
import numpy as np
import pybullet as p
import os
import gym, gym.spaces
from transforms3d.euler import euler2quat, euler2mat
import transforms3d.quaternions as quat
import sys

OBSERVATION_EPS = 0.01


class WalkerBase(BaseRobot):
    """ Built on top of BaseRobot
    Handles action_dim, sensor_dim, scene
    base_position, apply_action, calc_state
    reward
    """
    eye_offset_orn = euler2quat(0, 0, 0)
        
    def __init__(self, 
        filename,           # robot file name 
        robot_name,         # robot name
        action_dim,         # action dimension
        power,
        initial_pos,
        target_pos,
        sensor_dim=None,
        scale = 1, 
        resolution=512,
        env = None
    ):
        BaseRobot.__init__(self, filename, robot_name, scale, env)

        self.resolution = resolution
        self.obs_dim = None
        self.obs_dim = [self.resolution, self.resolution, 0]

        if "rgb_filled" in self.env.config["output"]:
            self.obs_dim[2] += 3
        if "depth" in self.env.config["output"]:
            self.obs_dim[2] += 1

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
        self.target_pos = target_pos
        self.initial_pos = initial_pos
        self.body_xyz=[0, 0, 0]
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
        '''Get current robot position
        '''
        return self.robot_body.current_position()

    def get_orientation(self):
        '''Get current orientation
        '''
        return self.robot_body.current_orientation()

    def set_position(self, pos):
        self.robot_body.reset_position(pos)

    def move_by(self, delta):
        new_pos = np.array(delta) + self.robot_body.current_position()
        self.robot_body.reset_position(new_pos)

    def move_forward(self, forward=1.0):
        orn = self.robot_body.current_orientation()
        print(euler2mat(orn))
        self.move_by(euler2mat(orn).dot(np.array(forward, 0, 0)))

    def move_backward

    def get_rpy(self):
        return self.robot_body.bp_pose.rpy()

    def apply_action(self, a):
        for n, j in enumerate(self.ordered_joints):
            j.set_motor_torque(self.power * j.power_coef * float(np.clip(a[n], -1, +1)))

    def get_target_position(self):
        return self.target_pos

    def set_target_position(self, pos):
        self.target_pos = pos

    def calc_state(self):
        j = np.array([j.current_relative_position() for j in self.ordered_joints], dtype=np.float32).flatten()
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
        self.walk_target_theta = np.arctan2(self.target_pos[1] - self.body_xyz[1],
                                            self.target_pos[0] - self.body_xyz[0])
        self.walk_target_dist = np.linalg.norm(
            [self.target_pos[1] - self.body_xyz[1], self.target_pos[0] - self.body_xyz[0]])
        self.walk_target_dist_xyz = np.linalg.norm(
            [self.target_pos[2] - self.body_xyz[2], self.target_pos[0] - self.body_xyz[1], self.target_pos[0] - self.body_xyz[0]])
        angle_to_target = self.walk_target_theta - yaw
        self.angle_to_target = angle_to_target

        self.walk_height_diff = np.abs(self.target_pos[2] - self.body_xyz[2])

        self.dist_to_start = np.linalg.norm(np.array(self.body_xyz) - np.array(self.initial_pos))

        debugmode= 0
        if debugmode:
            print("Robot dsebug mode: walk_height_diff", self.walk_height_diff)
            print("Robot dsebug mode: walk_target_z", self.target_pos[2])
            print("Robot dsebug mode: body_xyz", self.body_xyz[2])

        rot_speed = np.array(
            [[np.cos(-yaw), -np.sin(-yaw), 0],
             [np.sin(-yaw), np.cos(-yaw), 0],
             [        0,             0, 1]]
        )
        vx, vy, vz = np.dot(rot_speed, self.robot_body.speed())  # rotate speed back to body point of view

        debugmode=0
        if debugmode:
            print("Robot state", self.target_pos[1] - self.body_xyz[1], self.target_pos[0] - self.body_xyz[0])

        more = np.array([ z-self.initial_z,
            np.sin(angle_to_target), np.cos(angle_to_target),
            0.3* vx , 0.3* vy , 0.3* vz ,  # 0.3 is just scaling typical speed into -1..+1, no physical sense here
            r, p], dtype=np.float32)

        if debugmode:
            print("Robot more", more)


        if not 'nonvis_sensor' in self.env.config["output"]:
            j.fill(0)
            more.fill(0)

        return np.clip( np.concatenate([more] + [j] + [self.feet_contact]), -5, +5)

    def calc_potential(self):
        # progress in potential field is speed*dt, typical speed is about 2-3 meter per second, this potential will change 2-3 per frame (not per second),
        # all rewards have rew/frame units and close to 1.0 (hzyjerry) ==> make rewards similar scale
        debugmode=0
        if (debugmode):
            print("calc_potential: self.walk_target_dist x y", self.walk_target_dist)
            print("robot position", self.body_xyz, "target position", [self.target_pos[0], self.target_pos[1], self.target_pos[2]])
#            print("self.scene.dt")
#            print(self.scene.dt)
#            print("self.scene.frame_skip")
#            print(self.scene.frame_skip)
            #print("self.scene.timestep", self.scene.timestep)
        return - self.walk_target_dist / self.scene.dt


    def calc_goalless_potential(self):
        return self.dist_to_start / self.scene.dt

    def dist_to_target(self):
        return np.linalg.norm(self.get_position() - self.get_target_position())

    def _is_close_to_goal(self):
        body_pose = self.robot_body.pose()
        parts_xyz = np.array([p.pose().xyz() for p in self.parts.values()]).flatten()
        self.body_xyz = (
        parts_xyz[0::3].mean(), parts_xyz[1::3].mean(), body_pose.xyz()[2])  # torso z is more informative than mean z
        dist_to_goal = np.linalg.norm([self.body_xyz[0] - self.target_pos[0], self.body_xyz[1] - self.target_pos[1]])
        #print("dist to goal", dist_to_goal)
        #print(self.body_xyz[0], self.walk_target_x, self.body_xyz[1], self.walk_target_y)
        #print(self.body_xyz)
        return dist_to_goal < 2

    def _get_scaled_position(self):
        '''Private method, please don't use this method outside
        Used for downscaling MJCF models
        '''
        return self.robot_body.current_position() / self.mjcf_scaling


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
    eye_offset_orn = euler2quat(np.pi/2, 0, np.pi/2, axes='sxyz')
        
    def __init__(self, config, env=None):
        self.config = config
        self.model_type = "MJCF"
        self.mjcf_scaling = 0.25
        WalkerBase.__init__(self, "ant.xml", "torso", action_dim=8, 
                            sensor_dim=28, power=2.5, scale = self.mjcf_scaling, 
                            initial_pos=config['initial_pos'],
                            target_pos=config["target_pos"], 
                            resolution=config["resolution"], 
                            env = env)
        self.r_f = 0.1
        if config["is_discrete"]:
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
        if self.config["is_discrete"]:
            realaction = self.action_list[action]
        else:
            realaction = action
        WalkerBase.apply_action(self, realaction)

    def robot_specific_reset(self):
        WalkerBase.robot_specific_reset(self)

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
    eye_offset_orn = euler2quat(np.pi/4, 0, np.pi/2, axes='sxyz')  ## looking 45 degs down
    def __init__(self, config, env=None):
        Ant.__init__(self, config, env=env)
        
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

    def _is_close_to_goal(self):
        body_pose = self.robot_body.pose()
        parts_xyz = np.array([p.pose().xyz() for p in self.parts.values()]).flatten()
        self.body_xyz = (parts_xyz[0::3].mean(), parts_xyz[1::3].mean(), body_pose.xyz()[2])  # torso z is more informative than mean z
        dist_to_goal = np.linalg.norm([self.body_xyz[0] - self.target_pos[0], self.body_xyz[1] - self.target_pos[1], self.body_xyz[2] - self.target_pos[2]])
        debugmode = 0
        if debugmode:
            print(np.linalg.norm([self.body_xyz[0] - self.target_pos[0], self.body_xyz[1] - self.target_pos[1], self.body_xyz[2] - self.target_pos[2]]), [self.body_xyz[0], self.body_xyz[1], self.body_xyz[2]], [self.target_pos[0], self.target_pos[1], self.target_pos[2]])
        return dist_to_goal < 0.5


class Humanoid(WalkerBase):
    self_collision = True
    foot_list = ["right_foot", "left_foot"]  # "left_hand", "right_hand"

    #eye_offset_orn = euler2quat(np.pi/2, 0, np.pi/2, axes='sxyz')
    def __init__(self, config, env=None):
        self.config = config
        self.model_type = "MJCF"
        self.mjcf_scaling = 0.6
        WalkerBase.__init__(self, "humanoid.xml", "torso", action_dim=17, 
                            sensor_dim=44, power=2.5, scale = self.mjcf_scaling, 
                            initial_pos=config['initial_pos'],
                            target_pos=config["target_pos"], 
                            resolution=config["resolution"], 
                            env = env)
        self.glass_id = None
        self.is_discrete = config["is_discrete"]
        if self.is_discrete:
            self.action_space = gym.spaces.Discrete(5)
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
        '''
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
        '''

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
    mjcf_scaling = 1
    model_type = "URDF"
    eye_offset_orn = euler2quat(np.pi / 2, 0, np.pi / 2, axes='sxyz')

    def __init__(self, config, env=None):
        self.config = config
        WalkerBase.__init__(self, "husky.urdf", "base_link", action_dim=4, 
                            sensor_dim=20, power=2.5, scale = 0.6, 
                            initial_pos=config['initial_pos'],
                            target_pos=config["target_pos"], 
                            resolution=config["resolution"], 
                            env = env)
        self.is_discrete = config["is_discrete"]

        #self.eye_offset_orn = euler2quat(np.pi/2, 0, np.pi/2, axes='sxyz')
        if self.is_discrete:
            self.action_space = gym.spaces.Discrete(5)        
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


class Quadrotor(WalkerBase):
    eye_offset_orn = euler2quat(np.pi / 2, 0, np.pi / 2, axes='sxyz')
    def __init__(self, config, env=None):
        self.model_type = "URDF"
        self.mjcf_scaling = 1
        self.config = config
        self.is_discrete = config["is_discrete"]
        WalkerBase.__init__(self, "quadrotor.urdf", "base_link", action_dim=4, 
                            sensor_dim=20, power=2.5, scale = self.mjcf_scaling, 
                            initial_pos=config['initial_pos'],
                            target_pos=config["target_pos"], 
                            resolution=config["resolution"], 
                            env = env)
        if self.is_discrete:
            self.action_space = gym.spaces.Discrete(7)

            self.action_list = [[1,0,0,0,0,0],
                                [-1,0,0,0,0,0],
                                [0,1,0,0,0,0],
                                [0,-1,0,0,0,0],
                                [0,0,1,0,0,0],
                                [0,0,-1,0,0,0],
                                [0,0,0,0,0,0]
                                ]
            self.setup_keys_to_action()
        else:
            action_high = 0.02 * np.ones([6])
            self.action_space = gym.spaces.Box(-action_high, action_high)

        self.foot_list = []
    def apply_action(self, action):
        if self.is_discrete:
            realaction = self.action_list[action]
        else:
            realaction = action

        p.setGravity(0, 0, 0)
        p.resetBaseVelocity(self.robot_ids[0], realaction[:3], realaction[3:])

    def robot_specific_reset(self):
        WalkerBase.robot_specific_reset(self)

    def setup_keys_to_action(self):
        self.keys_to_action = {
            (ord('s'),): 0,  ## backward
            (ord('w'),): 1,  ## forward
            (ord('d'),): 2,  ## turn right
            (ord('a'),): 3,  ## turn left
            (ord('z'),): 4,  ## turn left
            (ord('x'),): 5,  ## turn left
            (): 6
        }