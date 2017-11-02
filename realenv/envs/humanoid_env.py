from realenv.envs.env_modalities import CameraRobotEnv, SensorRobotEnv
from realenv.core.physics.robot_locomotors import Humanoid
import gym

HUMANOID_TIMESTEP  = 1.0/(4 * 22)
HUMANOID_FRAMESKIP = 4

class HumanoidEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }
    timestep   = 1/(20 * 4)
    frame_skip = 20
    def __init__(self, mode="SENSOR"):
        self.robot = Humanoid(mode)
        self.physicsClientId=-1
        self.electricity_cost  = 4.25*SensorRobotEnv.electricity_cost
        self.stall_torque_cost = 4.25*SensorRobotEnv.stall_torque_cost

    def calc_rewards(self, a, state):
        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            self.robot.apply_action(a)
            self.scene.global_step()

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

        return [
            alive,
            progress,
            electricity_cost,
            joints_at_limit_cost,
            feet_collision_cost
            ], done


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