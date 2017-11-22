from realenv.envs.env_modalities import CameraRobotEnv, SensorRobotEnv
from realenv.envs.env_bases import *
from realenv.core.physics.robot_locomotors import Ant, AntClimber
from transforms3d import quaternions
from realenv import configs
import numpy as np
import sys
import pybullet as p
from realenv.core.physics.scene_stadium import SinglePlayerStadiumScene
import pybullet_data

ANT_TIMESTEP  = 1.0/(4 * 22)
ANT_FRAMESKIP = 4

"""Task specific classes for Ant Environment
Each class specifies: 
    (1) Target position
    (2) Reward function
    (3) Done condition
    (4) (Optional) Curriculum learning condition
"""

tracking_camera = {
    'pitch': -20,
    # 'pitch': -24  # demo: stairs
    #self.tracking_camera['pitch'] = -45 ## stairs
    'yaw': 46,     ## demo: living room
    #yaw = 30    ## demo: kitchen
    'z_offset': 0.5,
    'distance': 5 ## living room
    #self.tracking_camera['yaw'] = 90     ## demo: stairs
}

class AntNavigateEnv(CameraRobotEnv):
    def __init__(
            self, 
            human=True, 
            timestep=ANT_TIMESTEP, 
            frame_skip=ANT_FRAMESKIP, 
            is_discrete=False, 
            mode="RGBD", 
            use_filler=True, 
            gpu_count=0, 
            resolution="NORMAL"):
        self.human = human
        self.model_id = configs.NAVIGATE_MODEL_ID
        self.timestep = timestep
        self.frame_skip = frame_skip
        self.resolution = resolution
        self.tracking_camera = tracking_camera
        target_orn, target_pos   = configs.TASK_POSE[configs.NAVIGATE_MODEL_ID]["navigate"][-1]
        initial_orn, initial_pos = configs.TASK_POSE[configs.NAVIGATE_MODEL_ID]["navigate"][0]
        
        self.robot = Ant(initial_pos, initial_orn, 
            is_discrete=is_discrete, 
            target_pos=target_pos,
            resolution=resolution)
        CameraRobotEnv.__init__(
            self, 
            mode, 
            gpu_count, 
            scene_type="building", 
            use_filler=use_filler)

        self.total_reward = 0
        self.total_frame = 0
        
        
    def calc_rewards_and_done(self, a, state):
        ### TODO (hzyjerry): this is directly taken from husky_env, needs to be tuned 

        # dummy state if a is None
        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            self.robot.apply_action(a)
            self.scene.global_step()
       
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

    def flag_reposition(self):
        walk_target_x = self.robot.walk_target_x / self.robot.mjcf_scaling
        walk_target_y = self.robot.walk_target_y / self.robot.mjcf_scaling

        self.flag = None
        if self.human and not configs.DISPLAY_UI:
            self.visual_flagId = p.createVisualShape(p.GEOM_MESH, fileName=os.path.join(pybullet_data.getDataPath(), 'cube.obj'), meshScale=[0.5, 0.5, 0.5], rgbaColor=[1, 0, 0, 0.7])
            self.last_flagId = p.createMultiBody(baseVisualShapeIndex=self.visual_flagId, baseCollisionShapeIndex=-1, basePosition=[walk_target_x, walk_target_y, 0.5])
        
    def  _reset(self):
        self.total_frame = 0
        self.total_reward = 0
        obs = CameraRobotEnv._reset(self)
        self.flag_reposition()
        return obs


class AntClimbEnv(CameraRobotEnv):
    delta_target = [configs.RANDOM_TARGET_RANGE, configs.RANDOM_TARGET_RANGE]

    def __init__(
            self, 
            human=True, 
            timestep=ANT_TIMESTEP, 
            frame_skip=ANT_FRAMESKIP,           ## increase frame skip, as the motion is too flashy 
            is_discrete=False, 
            mode="RGBD", 
            use_filler=True, 
            gpu_count=0, 
            resolution="NORMAL"):
        self.human = human
        self.model_id = configs.CLIMB_MODEL_ID
        self.timestep = timestep
        self.frame_skip = frame_skip
        self.resolution = resolution
        self.tracking_camera = tracking_camera
        target_orn, target_pos   = configs.TASK_POSE[configs.CLIMB_MODEL_ID]["climb"][-1]
        initial_orn, initial_pos = configs.TASK_POSE[configs.CLIMB_MODEL_ID]["climb"][0]

        self.target_pos_gt = target_pos
        self.robot = AntClimber(initial_pos, initial_orn, 
            is_discrete=is_discrete, 
            target_pos=target_pos,
            resolution=resolution,
            mode=mode)
        CameraRobotEnv.__init__(
            self, 
            mode, 
            gpu_count, 
            scene_type="building", 
            use_filler=use_filler)

        self.total_reward = 0
        self.total_frame = 0
        self.visual_flagId = None
        
        
    def calc_rewards_and_done(self, a, state):
        #time.sleep(0.1)
        ### TODO (hzyjerry): this is directly taken from husky_env, needs to be tuned 

        # dummy state if a is None
        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            self.robot.apply_action(a)
            self.scene.global_step()
       
        alive = float(self.robot.alive_bonus(self.robot.body_rpy[0], self.robot.body_rpy[1]))  # state[0] is body height above ground (z - z initial), body_rpy[1] is pitch

        done = self.nframe > 700 or alive < 0 or self.robot.body_xyz[2] < 0

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
        
        close_to_goal = 0
        if self.robot.is_close_to_goal():
            close_to_goal = 1

        done = self.nframe > 600 or alive < 0 or close_to_goal

        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True


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

        reward = [
            #alive,
            progress,
            #close_to_goal,
            #electricity_cost,
            #joints_at_limit_cost,
            #feet_collision_cost
         ]

        debugmode = 0
        if (debugmode):
            print("reward")
            print(reward)
        return reward, done

    def randomize_target(self):
        if configs.RANDOM_TARGET_POSE:
            delta_x = self.np_random.uniform(low=-self.delta_target[0],
                                             high=+self.delta_target[0])
            delta_y = self.np_random.uniform(low=-self.delta_target[1],
                                             high=+self.delta_target[1])
        else:
            delta_x = 0
            delta_y = 0
        self.temp_target_x = (self.target_pos_gt[0] + delta_x)
        self.temp_target_y = (self.target_pos_gt[1] + delta_y)

    def flag_reposition(self):
        walk_target_x = self.temp_target_x
        walk_target_y = self.temp_target_y
        walk_target_z = self.robot.walk_target_z

        self.robot.walk_target_x = walk_target_x    # Change robot target accordingly
        self.robot.walk_target_y = walk_target_y    # Important for stair climbing
        self.robot.walk_target_z = walk_target_z

        self.flag = None
        if not self.human:
            return

        if self.visual_flagId is None:
            if configs.DISPLAY_UI:
                self.visual_flagId = -1
            else:
                self.visual_flagId = p.createVisualShape(p.GEOM_MESH, fileName=os.path.join(pybullet_data.getDataPath(), 'cube.obj'), meshScale=[0.5, 0.5, 0.5], rgbaColor=[1, 0, 0, 0.7])
            self.last_flagId = p.createMultiBody(baseVisualShapeIndex=self.visual_flagId, baseCollisionShapeIndex=-1, basePosition=[walk_target_x / self.robot.mjcf_scaling, walk_target_y / self.robot.mjcf_scaling, walk_target_z / self.robot.mjcf_scaling])
        else:
            last_flagPos, last_flagOrn = p.getBasePositionAndOrientation(self.last_flagId)
            p.resetBasePositionAndOrientation(self.last_flagId, [walk_target_x  / self.robot.mjcf_scaling, walk_target_y / self.robot.mjcf_scaling, walk_target_z / self.robot.mjcf_scaling], last_flagOrn)
        
    def  _reset(self):
        self.total_frame = 0
        self.total_reward = 0
        self.randomize_target()
        self.flag_reposition()
        obs = CameraRobotEnv._reset(self)       ## Important: must come after flat_reposition
        return obs




class AntFlagRunEnv(CameraRobotEnv):
    """Specfy flagrun reward
    """
    def __init__(self, human=True, timestep=ANT_TIMESTEP,
                 frame_skip=ANT_FRAMESKIP, is_discrete=False, 
                 gpu_count=0):
        self.human = human
        self.timestep = timestep
        self.frame_skip = frame_skip
        ## Mode initialized with mode=SENSOR
        self.flag_timeout = 1
        self.tracking_camera = tracking_camera
        initial_pos, initial_orn = [0, 0, 1], [0, 0, 0, 1]
        self.robot = Ant(initial_pos, initial_orn, 
            is_discrete=is_discrete)
        CameraRobotEnv.__init__(
            self, 
            "SENSOR", 
            gpu_count, 
            scene_type="stadium", 
            use_filler=False)

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
        self.walk_target_x *= more_compact / self.robot.mjcf_scaling
        self.walk_target_y *= more_compact / self.robot.mjcf_scaling

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

        #alive = len(self.robot.parts['top_bumper_link'].contact_list())
        head_touch_ground = 0
        if head_touch_ground == 0:
            alive_score = 0.1
        else:
            alive_score = -0.1


        done = head_touch_ground > 0 or self.nframe > 500

        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
        debugmode = 0
        if (debugmode):
            print("alive=")
            print(head_touch_ground)
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


class AntFetchEnv(CameraRobotEnv):
    """Specfy flagrun reward
    """
    def __init__(self, human=True, timestep=ANT_TIMESTEP,
                 frame_skip=ANT_FRAMESKIP, is_discrete=False,
                 gpu_count=0, scene_type="building", mode = 'SENSOR'):

        target_orn, target_pos = INITIAL_POSE["ant"][configs.FETCH_MODEL_ID][-1]
        initial_orn, initial_pos = configs.INITIAL_POSE["ant"][configs.FETCH_MODEL_ID][0]

        self.human = human
        self.timestep = timestep
        self.frame_skip = frame_skip
        self.model_id = configs.FETCH_MODEL_ID
        ## Mode initialized with mode=SENSOR
        self.tracking_camera = tracking_camera

        self.robot = Ant(
            is_discrete=is_discrete,
            initial_pos=initial_pos,
            initial_orn=initial_orn)

        CameraRobotEnv.__init__(
            self,
            mode,
            gpu_count,
            scene_type="building")
        self.flag_timeout = 1


        self.visualid = -1

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

        #alive = len(self.robot.parts['top_bumper_link'].contact_list())
        head_touch_ground = 1
        if head_touch_ground == 0:
            alive_score = 0.1
        else:
            alive_score = -0.1


        done = head_touch_ground > 0 or self.nframe > 500

        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
        debugmode = 0
        if (debugmode):
            print("head_touch_ground=")
            print(head_touch_ground)
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


class AntFetchKernelizedRewardEnv(CameraRobotEnv):
    """Specfy flagrun reward
    """
    def __init__(self, human=True, timestep=ANT_TIMESTEP,
                 frame_skip=ANT_FRAMESKIP, is_discrete=False,
                 gpu_count=0, scene_type="building"):
        self.human = human
        self.timestep = timestep
        self.frame_skip = frame_skip
        ## Mode initialized with mode=SENSOR
        self.model_id = configs.FETCH_MODEL_ID

        self.flag_timeout = 1
        self.tracking_camera = tracking_camera

        initial_pos, initial_orn = configs.INITIAL_POSE["ant"][configs.FETCH_MODEL_ID][0]
        self.robot = Ant(initial_pos, initial_orn, 
            is_discrete=is_discrete)
        CameraRobotEnv.__init__(
            self, 
            "SENSOR", 
            gpu_count, 
            scene_type="building", 
            use_filler=False)


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

        #alive = len(self.robot.parts['top_bumper_link'].contact_list())
        head_touch_ground = 1
        if head_touch_ground == 0:
            alive_score = 0.1
        else:
            alive_score = -0.1


        done = head_touch_ground > 0 or self.nframe > 500

        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
        debugmode = 0
        if (debugmode):
            print("head_touch_ground=")
            print(head_touch_ground)
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
