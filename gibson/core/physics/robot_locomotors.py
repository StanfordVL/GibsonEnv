from gibson.core.physics.robot_bases import BaseRobot
import numpy as np
import pybullet as p
import os
import gym, gym.spaces
from transforms3d.euler import euler2quat, euler2mat
from transforms3d.quaternions import quat2mat, qmult
import transforms3d.quaternions as quat
import sys

OBSERVATION_EPS = 0.01


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
        initial_pos,
        target_pos,
        scale,
        sensor_dim=None,
        resolution=512,
        control = 'torque',
        env = None
    ):
        BaseRobot.__init__(self, filename, robot_name, scale, env)
        self.control = control
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
            j.reset_joint_state(self.np_random.uniform(low=-0.1, high=0.1), 0)

        self.feet = [self.parts[f] for f in self.foot_list]
        self.feet_contact = np.array([0.0 for f in self.foot_list], dtype=np.float32)

        self.scene.actor_introduce(self)
        self.initial_z = None

    def get_position(self):
        '''Get current robot position
        '''
        return self.robot_body.get_position()

    def get_orientation(self):
        '''Return robot orientation
        '''
        return self.robot_body.get_orientation()

    def set_position(self, pos):
        self.robot_body.reset_position(pos)

    def move_by(self, delta):
        new_pos = np.array(delta) + self.get_position()
        self.robot_body.reset_position(new_pos)

    def move_forward(self, forward=0.05):
        x, y, z, w = self.robot_body.get_orientation()
        self.move_by(quat2mat([w, x, y, z]).dot(np.array([forward, 0, 0])))
        
    def move_backward(self, backward=0.05):
        x, y, z, w = self.robot_body.get_orientation()
        self.move_by(quat2mat([w, x, y, z]).dot(np.array([-backward, 0, 0])))

    def turn_left(self, delta=0.03):
        orn = self.robot_body.get_orientation()
        new_orn = qmult((euler2quat(-delta, 0, 0)), orn)
        self.robot_body.set_orientation(new_orn)

    def turn_right(self, delta=0.03):
        orn = self.robot_body.get_orientation()
        new_orn = qmult((euler2quat(delta, 0, 0)), orn)
        self.robot_body.set_orientation(new_orn)

    def set_orientation(self, x, y=np.pi, z=0):
        new_orn = euler2quat(x, y, z)
        self.robot_body.set_orientation(new_orn)
        
    def get_rpy(self):
        return self.robot_body.bp_pose.rpy()

    def apply_action(self, a):
        #print(self.ordered_joints)
        if self.control == 'torque':
            for n, j in enumerate(self.ordered_joints):
                j.set_motor_torque(self.power * j.power_coef * float(np.clip(a[n], -1, +1)))
        elif self.control == 'velocity':
            for n, j in enumerate(self.ordered_joints):
                j.set_motor_velocity(self.power * j.power_coef * float(np.clip(a[n], -1, +1)))
        elif self.control == 'position':
            for n, j in enumerate(self.ordered_joints):
                j.set_motor_position(a[n])
        elif type(self.control) is list or type(self.control) is tuple: #if control is a tuple, set different control
        # type for each joint
            for n, j in enumerate(self.ordered_joints):
                if self.control[n] == 'torque':
                    j.set_motor_torque(self.power * j.power_coef * float(np.clip(a[n], -1, +1)))
                elif self.control[n] == 'velocity':
                    j.set_motor_velocity(self.power * j.power_coef * float(np.clip(a[n], -1, +1)))
                elif self.control[n] == 'position':
                    j.set_motor_position(a[n])
        else:
            pass

    def get_target_position(self):
        return self.target_pos

    def set_target_position(self, pos):
        self.target_pos = pos

    def calc_state(self):
        j = np.array([j.get_joint_relative_state() for j in self.ordered_joints], dtype=np.float32).flatten()
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
        robot_orn = self.get_rpy()

        self.walk_target_theta = np.arctan2(self.target_pos[1] - self.body_xyz[1],
                                            self.target_pos[0] - self.body_xyz[0])
        self.walk_target_dist = np.linalg.norm(
            [self.target_pos[1] - self.body_xyz[1], self.target_pos[0] - self.body_xyz[0]])
        self.walk_target_dist_xyz = np.linalg.norm(
            [self.target_pos[2] - self.body_xyz[2], self.target_pos[0] - self.body_xyz[1], self.target_pos[0] - self.body_xyz[0]])
        
        self.angle_to_target = self.walk_target_theta - yaw
        if self.angle_to_target > np.pi:
            self.angle_to_target -= 2 * np.pi
        elif self.angle_to_target < -np.pi:
            self.angle_to_target += 2 * np.pi

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
            np.sin(self.angle_to_target), np.cos(self.angle_to_target),
            0.3* vx , 0.3* vy , 0.3* vz ,  # 0.3 is just scaling typical speed into -1..+1, no physical sense here
            r, p], dtype=np.float32)

        if debugmode:
            print("Robot more", more)


        if not 'nonviz_sensor' in self.env.config["output"]:
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
        return - self.walk_target_dist / self.scene.dt


    def calc_goalless_potential(self):
        return self.dist_to_start / self.scene.dt

    def dist_to_target(self):
        return np.linalg.norm(np.array(self.body_xyz) - np.array(self.get_target_position()))


    def angle_cost(self):
        angle_const = 0.2
        is_forward = np.abs(self.angle_to_target) < 1.57
        diff_angle = np.abs(self.angle_to_target)
        debugmode = 0
        if debugmode:
            print("is forward", is_forward)
            print("angle to target", self.angle_to_target)
            print("diff angle", diff_angle)
        return -angle_const* diff_angle

    def _is_close_to_goal(self):
        body_pose = self.robot_body.pose()
        parts_xyz = np.array([p.pose().xyz() for p in self.parts.values()]).flatten()
        self.body_xyz = (
        parts_xyz[0::3].mean(), parts_xyz[1::3].mean(), body_pose.xyz()[2])  # torso z is more informative than mean z
        dist_to_goal = np.linalg.norm([self.body_xyz[0] - self.target_pos[0], self.body_xyz[1] - self.target_pos[1]])
        return dist_to_goal < 2

    def _get_scaled_position(self):
        '''Private method, please don't use this method outside
        Used for downscaling MJCF models
        '''
        return self.robot_body.get_position() / self.mjcf_scaling



class Ant(WalkerBase):
    foot_list = ['front_left_foot', 'front_right_foot', 'left_back_foot', 'right_back_foot']
    model_type = "MJCF"
    default_scale = 0.25

    def __init__(self, config, env=None):
        self.config = config
        scale = config["robot_scale"] if "robot_scale" in config.keys() else self.default_scale
        self.mjcf_scaling = scale
        WalkerBase.__init__(self, "ant.xml", "torso", action_dim=8, 
                            sensor_dim=28, power=2.5, scale=scale, 
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
            (ord('1'), ): 0,
            (ord('2'), ): 1, 
            (ord('3'), ): 2, 
            (ord('4'), ): 3, 
            (ord('5'), ): 4, 
            (ord('6'), ): 5, 
            (ord('7'), ): 6, 
            (ord('8'), ): 7, 
            (ord('9'), ): 8, 
            (ord('0'), ): 9, 
            (ord('q'), ): 10, 
            (ord('w'), ): 11, 
            (ord('e'), ): 12, 
            (ord('r'), ): 13, 
            (ord('t'), ): 14, 
            (ord('y'), ): 15, 
            (): 4
        }


class AntClimber(Ant):
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
    model_type = "MJCF"
    default_scale = 0.6
    glass_offset = 0.3

    def __init__(self, config, env=None):
        self.config = config
        scale = config["robot_scale"] if "robot_scale" in config.keys() else self.default_scale
        self.mjcf_scaling = scale
        WalkerBase.__init__(self, "humanoid.xml", "torso", action_dim=17, 
                            sensor_dim=44, power=0.41, scale=scale, 
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
        humanoidId = -1
        numBodies = p.getNumBodies()
        for i in range (numBodies):
            bodyInfo = p.getBodyInfo(i)
            if bodyInfo[1].decode("ascii") == 'humanoid':
                humanoidId = i
        ## Spherical radiance/glass shield to protect the robot's camera
        
        WalkerBase.robot_specific_reset(self)


        if self.glass_id is None:
            glass_path = os.path.join(self.physics_model_dir, "glass.xml")
            glass_id = p.loadMJCF(glass_path)[0]
            self.glass_id = glass_id
            p.changeVisualShape(self.glass_id, -1, rgbaColor=[0, 0, 0, 0])
            p.createMultiBody(baseVisualShapeIndex=glass_id, baseCollisionShapeIndex=-1)
            cid = p.createConstraint(humanoidId, -1, self.glass_id,-1,p.JOINT_FIXED, 
                jointAxis=[0,0,0], parentFramePosition=[0, 0, self.glass_offset],
                childFramePosition=[0,0,0])
        
        robot_pos = list(self._get_scaled_position())
        robot_pos[2] += self.glass_offset
        robot_orn = self.get_orientation()
        p.resetBasePositionAndOrientation(self.glass_id, robot_pos, robot_orn)
                

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
    default_scale = 0.6
    
    def __init__(self, config, env=None):
        self.config = config
        scale = config["robot_scale"] if "robot_scale" in config.keys() else self.default_scale
        
        WalkerBase.__init__(self, "husky.urdf", "base_link", action_dim=4, 
                            sensor_dim=23, power=2.5, scale=scale,
                            initial_pos=config['initial_pos'],
                            target_pos=config["target_pos"], 
                            resolution=config["resolution"], 
                            env = env)
        self.is_discrete = config["is_discrete"]

        if self.is_discrete:
            self.action_space = gym.spaces.Discrete(5)        
            self.torque = 0.03
            self.action_list = [[self.torque, self.torque, self.torque, self.torque],
                                [-self.torque, -self.torque, -self.torque, -self.torque],
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


    def robot_specific_reset(self):
        WalkerBase.robot_specific_reset(self)

    def alive_bonus(self, z, pitch):
        top_xyz = self.parts["top_bumper_link"].pose().xyz()
        bottom_xyz = self.parts["base_link"].pose().xyz()
        alive = top_xyz[2] > bottom_xyz[2]
        return +1 if alive else -100  # 0.25 is central sphere rad, die if it scrapes the ground

    def setup_keys_to_action(self):
        self.keys_to_action = {
            (ord('w'), ): 0, ## forward
            (ord('s'), ): 1, ## backward
            (ord('d'), ): 2, ## turn right
            (ord('a'), ): 3, ## turn left
            (): 4
        }

    def calc_state(self):
        base_state = WalkerBase.calc_state(self)

        angular_speed = self.robot_body.angular_speed()
        return np.concatenate((base_state, np.array(angular_speed)))




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
    model_type = "URDF"
    default_scale=1
    mjcf_scaling=1

    def __init__(self, config, env=None):
        self.config = config
        scale = config["robot_scale"] if "robot_scale" in config.keys() else self.default_scale
        self.is_discrete = config["is_discrete"]
        WalkerBase.__init__(self, "quadrotor.urdf", "base_link", action_dim=4, 
                            sensor_dim=20, power=2.5, scale = scale, 
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
            (ord('w'),): 0,  ## +x
            (ord('s'),): 1,  ## -x
            (ord('d'),): 2,  ## +y
            (ord('a'),): 3,  ## -y
            (ord('z'),): 4,  ## +z
            (ord('x'),): 5,  ## -z
            (): 6
        }


class Turtlebot(WalkerBase):
    foot_list = []
    mjcf_scaling = 1
    model_type = "URDF"
    default_scale = 1
    
    def __init__(self, config, env=None):
        self.config = config
        scale = config["robot_scale"] if "robot_scale" in config.keys() else self.default_scale
        WalkerBase.__init__(self, "turtlebot/turtlebot.urdf", "base_link", action_dim=4,
                            sensor_dim=20, power=2.5, scale=scale,
                            initial_pos=config['initial_pos'],
                            target_pos=config["target_pos"],
                            resolution=config["resolution"],
                            control = 'velocity',
                            env=env)
        self.is_discrete = config["is_discrete"]

        if self.is_discrete:
            self.action_space = gym.spaces.Discrete(5)
            self.vel = 0.1
            self.action_list = [[self.vel, self.vel],
                                [-self.vel, -self.vel],
                                [self.vel, -self.vel],
                                [-self.vel, self.vel],
                                [0, 0]]

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

    def robot_specific_reset(self):
        WalkerBase.robot_specific_reset(self)

    def alive_bonus(self, z, pitch):
        return +1 if z > 0.26 else -1  # 0.25 is central sphere rad, die if it scrapes the ground

    def setup_keys_to_action(self):
        self.keys_to_action = {
            (ord('w'),): 0,  ## forward
            (ord('s'),): 1,  ## backward
            (ord('d'),): 2,  ## turn right
            (ord('a'),): 3,  ## turn left
            (): 4
        }

    def calc_state(self):
        base_state = WalkerBase.calc_state(self)

        angular_speed = self.robot_body.angular_speed()
        return np.concatenate((base_state, np.array(angular_speed)))



class JR(WalkerBase):
    foot_list = []
    mjcf_scaling = 1
    model_type = "URDF"
    default_scale = 0.6
    
    def __init__(self, config, env=None):
        self.config = config
        scale = config["robot_scale"] if "robot_scale" in config.keys() else self.default_scale
        WalkerBase.__init__(self, "jr1_urdf/jr1_gibson.urdf", "base_link", action_dim=4,
                            sensor_dim=20, power=2.5, scale=scale,
                            initial_pos=config['initial_pos'],
                            target_pos=config["target_pos"],
                            resolution=config["resolution"],
                            control = 'velocity',
                            env=env)
        self.is_discrete = config["is_discrete"]

        if self.is_discrete:
            self.action_space = gym.spaces.Discrete(5)
            self.vel = 0.1
            self.action_list = [[self.vel, self.vel],
                                [-self.vel, -self.vel],
                                [self.vel, -self.vel],
                                [-self.vel, self.vel],
                                [0, 0]]

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

    def robot_specific_reset(self):
        WalkerBase.robot_specific_reset(self)

    def alive_bonus(self, z, pitch):
        return +1 if z > 0.26 else -1  # 0.25 is central sphere rad, die if it scrapes the ground

    def setup_keys_to_action(self):
        self.keys_to_action = {
            (ord('w'),): 0,  ## forward
            (ord('s'),): 1,  ## backward
            (ord('d'),): 2,  ## turn right
            (ord('a'),): 3,  ## turn left
            (): 4
        }

    def calc_state(self):
        base_state = WalkerBase.calc_state(self)

        angular_speed = self.robot_body.angular_speed()
        return np.concatenate((base_state, np.array(angular_speed)))


class JR2(WalkerBase):
    foot_list = []
    mjcf_scaling = 1
    model_type = "URDF"
    default_scale = 1

    def __init__(self, config, env=None):
        self.config = config
        scale = config["robot_scale"] if "robot_scale" in config.keys() else self.default_scale
        WalkerBase.__init__(self, "jr2_urdf/jr2.urdf", "base_link", action_dim=4,
                            sensor_dim=20, power=2.5, scale=scale,
                            initial_pos=config['initial_pos'],
                            target_pos=config["target_pos"],
                            resolution=config["resolution"],
                            control=['velocity', 'velocity', 'position', 'position'],
                            env=env)
        self.is_discrete = config["is_discrete"]

        if self.is_discrete:
            self.action_space = gym.spaces.Discrete(5)
            self.vel = 0.01
            self.action_list = [[self.vel, self.vel,0,0.2],
                                [-self.vel, -self.vel,0,-0.2],
                                [self.vel, -self.vel,-0.5,0],
                                [-self.vel, self.vel,0.5,0],
                                [0, 0,0,0]]

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

    def robot_specific_reset(self):
        WalkerBase.robot_specific_reset(self)

    def alive_bonus(self, z, pitch):
        return +1 if z > 0.26 else -1  # 0.25 is central sphere rad, die if it scrapes the ground

    def setup_keys_to_action(self):
        self.keys_to_action = {
            (ord('w'),): 0,  ## forward
            (ord('s'),): 1,  ## backward
            (ord('d'),): 2,  ## turn right
            (ord('a'),): 3,  ## turn left
            (): 4
        }

    def calc_state(self):
        base_state = WalkerBase.calc_state(self)

        angular_speed = self.robot_body.angular_speed()
        return np.concatenate((base_state, np.array(angular_speed)))
