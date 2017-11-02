from realenv.envs.env_modalities import CameraRobotEnv, SensorRobotEnv
from realenv.core.physics.robot_locomotors import Husky
from transforms3d import quaternions
import numpy as np
import pybullet as p

HUMANOID_TIMESTEP  = 1.0/(4 * 22)
HUMANOID_FRAMESKIP = 4

class HuskyEnv:
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }
    def __init__(self, is_discrete=False, mode="SENSOR"):
        self.physicsClientId=-1
        self.robot = Husky(is_discrete, mode)

    def get_keys_to_action(self):
        return self.robot.keys_to_action
        

class HuskyCameraEnv(HuskyEnv, CameraRobotEnv):
    def __init__(self, human=True, timestep=HUMANOID_TIMESTEP, 
        frame_skip=HUMANOID_FRAMESKIP, enable_sensors=False,
        is_discrete=False, mode="RGBD", use_filler=True):
        self.human = human
        self.timestep = timestep
        self.frame_skip = frame_skip
        self.enable_sensors = enable_sensors
        HuskyEnv.__init__(self, is_discrete, mode)
        CameraRobotEnv.__init__(self, use_filler)

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

class HuskySensorEnv(HuskyEnv, SensorRobotEnv):
    def __init__(self, human=True, timestep=HUMANOID_TIMESTEP, 
        frame_skip=HUMANOID_FRAMESKIP, enable_sensors=False,
        is_discrete=False):
        self.human = human
        self.timestep = timestep
        self.frame_skip = frame_skip
        HuskyEnv.__init__(self, is_discrete)
        SensorRobotEnv.__init__(self)
        self.nframe = 0



    def  _reset(self):
        obs = SensorRobotEnv._reset(self)
        self.nframe = 0
        return obs
    def _step(self, a=None):
        self.nframe += 1

        # dummy state if a is None
        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            if not a is None:
                self.robot.apply_action(a)
            self.scene.global_step()

        state = self.robot.calc_state()  # also calculates self.joints_at_limit

        alive = float(self.robot.alive_bonus(state[0] + self.robot.initial_z, self.robot.body_rpy[
            1]))  # state[0] is body height above ground, body_rpy[1] is pitch

        done = self.nframe > 100
        print(self.nframe)
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
            print("electricity_cost")
            print(electricity_cost)
            print("joints_at_limit_cost")
            print(joints_at_limit_cost)
            print("feet_collision_cost")
            print(feet_collision_cost)

        self.rewards = [
            #alive,
            progress,
            #electricity_cost,
            #joints_at_limit_cost,
            #feet_collision_cost
        ]
        if (debugmode):
            print("rewards=")
            print(self.rewards)
            print("sum rewards")
            print(sum(self.rewards))
        if not a is None:
            self.HUD(state, a, done)
        self.reward += sum(self.rewards)

        if self.human:
            humanPos, humanOrn = p.getBasePositionAndOrientation(self.robot_tracking_id)
            humanPos = (humanPos[0], humanPos[1], humanPos[2] + self.tracking_camera['z_offset'])

            p.resetDebugVisualizerCamera(self.tracking_camera['distance'], self.tracking_camera['yaw'],
                                         self.tracking_camera['pitch'], humanPos)  ## demo: kitchen, living room
            # p.resetDebugVisualizerCamera(distance,yaw,-42,humanPos);        ## demo: stairs

        eye_pos = self.robot.eyes.current_position()
        x, y, z, w = self.robot.eyes.current_orientation()
        eye_quat = quaternions.qmult([w, x, y, z], self.robot.eye_offset_orn)
        print(sum(self.rewards))
        return state, sum(self.rewards), bool(done), {"eye_pos": eye_pos, "eye_quat": eye_quat}

        #self.tracking_camera['pitch'] = -45 ## stairs
        yaw = 90     ## demo: living room
        #yaw = 30    ## demo: kitchen
        offset = 0.5
        distance = 0.7 ## living room
        #self.tracking_camera['yaw'] = 90     ## demo: stairs

        
        self.tracking_camera['yaw'] = yaw   ## living roon
        self.tracking_camera['pitch'] = -10
        
        self.tracking_camera['distance'] = distance
        self.tracking_camera['z_offset'] = offset