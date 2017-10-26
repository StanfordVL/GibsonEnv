from realenv.core.physics.scene_building import SinglePlayerBuildingScene
from realenv.data.datasets import ViewDataSet3D, get_model_path
from realenv.core.render.show_3d2 import PCRenderer
from realenv.core.physics.physics_env import PhysicsEnv
from realenv.envs.env_bases import MJCFBaseEnv
import realenv
from gym import error
from gym.utils import seeding
import pybullet as p
import progressbar
import subprocess, os, signal
import numpy as np
import sys
import zmq
import socket
import shlex
import gym
import cv2


DEFAULT_TIMESTEP  = 1.0/(4 * 9)
DEFAULT_FRAMESKIP = 4


class SensorRobotEnv(MJCFBaseEnv):
    def __init__(self, human, timestep=DEFAULT_TIMESTEP, frame_skip=DEFAULT_FRAMESKIP):
        MJCFBaseEnv.__init__(self, human)
        self.camera_x = 0
        self.walk_target_x = 1e3  # kilometer away
        self.walk_target_y = 0

        self.k = 5
        self.timestep=timestep
        self.frame_skip=frame_skip
        self.robot_tracking_id = -1

        self.model_path, self.model_id = get_model_path()
        self.scale_up  = 1
        self.dataset  = ViewDataSet3D(
            transform = np.array, 
            mist_transform = np.array, 
            seqlen = 2, 
            off_3d = False, 
            train = False, 
            overwrite_fofn=True)
        self.ground_ids = None
        
    def _reset(self):
        MJCFBaseEnv._reset(self)
        if not self.ground_ids:
            self.parts, self.jdict, self.ordered_joints, self.robot_body = self.robot.addToScene(
                self.building_scene.building_obj)
            #print(self.parts)
            #self.ground_ids = set([(self.parts[f].bodies[self.parts[f].bodyIndex], self.parts[f].bodyPartIndex) for f in self.foot_ground_object_names])
            self.ground_ids = set([(self.building_scene.building_obj, 0)])
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
        for i in range (p.getNumBodies()):
            if (p.getBodyInfo(i)[0].decode() == self.robot_body.get_name()):
               self.robot_tracking_id=i
        i = 0

        ## TODO (hzyjerry), the original reset() in gym interface returns an env, 
        #return r

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

        if self.isRender:
            distance=2.5 ## demo: living room ,kitchen
            #distance=1.7   ## demo: stairs
            #yaw = 0     ## demo: living room
            yaw = 30    ## demo: kitchen
            #yaw = 90     ## demo: stairs
            humanPos, humanOrn = p.getBasePositionAndOrientation(self.robot_tracking_id)
            p.resetDebugVisualizerCamera(distance,yaw,-35,humanPos);       ## demo: kitchen, living room
            #p.resetDebugVisualizerCamera(distance,yaw,-42,humanPos);        ## demo: stairs

        eye_pos = self.robot.eyes.current_position()
        x, y, z ,w = self.robot.eyes.current_orientation()
        eye_quat = [w, x, y, z]

        return state, sum(self.rewards), bool(done), {"eye_pos":eye_pos, "eye_quat":eye_quat}


    def move_robot(self, init_x, init_y, init_z):
        "Used by multiplayer building to move sideways, to another running lane."
        self.cpp_robot.query_position()
        pose = self.cpp_robot.root_part.pose()
        pose.move_xyz(init_x, init_y, init_z)  # Works because robot loads around (0,0,0), and some robots have z != 0 that is left intact
        self.cpp_robot.set_pose(pose)

    def camera_adjust(self):
        x, y, z = self.body_xyz
        self.camera_x = 0.98*self.camera_x + (1-0.98)*x
        self.camera.move_and_look_at(self.camera_x, y-2.0, 1.4, x, y, 1.0)

    def find_best_k_views(self, eye_pos, all_dist, all_pos):
        least_order = (np.argsort(all_dist))
        #print(eye_pos, all_pos)
        if len(all_pos) <= p.MAX_RAY_INTERSECTION_BATCH_SIZE:
            collisions = list(p.rayTestBatch([eye_pos] * len(all_pos), all_pos))
        else:
            collisions = []
            curr_i = 0
            while (curr_i < len(all_pos)):
                curr_n = min(len(all_pos), curr_i + p.MAX_RAY_INTERSECTION_BATCH_SIZE - 1)
                collisions = collisions + list(p.rayTestBatch([eye_pos] * (curr_n - curr_i), all_pos[curr_i: curr_n]))
                curr_i = curr_n
        collisions  = [c[0] for c in collisions]
        top_k = []
        for i in range(len(least_order)):
            if len(top_k) >= self.k:
                break
            if collisions[least_order[i]] < 0:
                top_k.append(least_order[i])
        if len(top_k) < self.k:
            for o in least_order:
                if o not in top_k:
                    top_k.append(o)
                if len(top_k) >= self.k:
                    break 
        print("Found %d views" % len(top_k), top_k)
        return top_k


    def create_single_player_scene(self):
        self.building_scene = SinglePlayerBuildingScene(gravity=9.8, timestep=self.timestep, frame_skip=self.frame_skip)
        return self.building_scene


    def getExtendedObservation(self):
        pass

    

class CameraRobotEnv(SensorRobotEnv):
    def __init__(self, human, timestep=DEFAULT_TIMESTEP, frame_skip=DEFAULT_FRAMESKIP, enable_sensors=False):
        SensorRobotEnv.__init__(self, human, timestep, frame_skip)
        self.r_camera_rgb = None     ## Rendering engine
        self.r_camera_mul = None     ## Multi channel rendering engine
        self.enable_sensors = enable_sensors
        
    def _reset(self):
        SensorRobotEnv._reset(self)
        if not self.r_camera_rgb or not self.r_camera_mul:
            self.check_port_available()
            self.setup_camera_multi()
            self.setup_camera_rgb()

    def _step(self, a):
        sensor_state, sensor_reward, done, sensor_meta = SensorRobotEnv._step(self, a)
        pose = [sensor_meta['eye_pos'], sensor_meta['eye_quat']]
        
        ## Select the nearest points
        all_dist, all_pos = self.r_camera_rgb.rankPosesByDistance(pose)
        top_k = self.find_best_k_views(sensor_meta['eye_pos'], all_dist, all_pos)
        
        sensor_meta.pop("eye_pos", None)
        sensor_meta.pop("eye_quat", None)
        
        #with Profiler("Render to screen"):
        #if not self.debug_mode:
        #    visuals = self.r_camera_rgb.renderOffScreen(pose, top_k)
        #else:
        visuals = self.r_camera_rgb.renderToScreen(pose, top_k)

        if self.enable_sensors:
            sensor_meta["rgb"] = visuals
            return sensor_state, sensor_reward , done, sensor_meta
        else:
            return visuals, sensor_reward, done, sensor_meta
        #return sensor_state, sensor_reward, done, sensor_meta
        

    def _close(self):
        self.r_camera_mul.terminate()

    def setup_camera_rgb(self):
        scene_dict = dict(zip(self.dataset.scenes, range(len(self.dataset.scenes))))
        ## Todo: (hzyjerry) more error handling
        self.scale_up = 2
        if not self.model_id in scene_dict.keys():
             raise error.Error("Dataset not found: model {} cannot be loaded".format(self.model_id))
        else:
            scene_id = scene_dict[self.model_id]
        uuids, rts = self.dataset.get_scene_info(scene_id)

        targets, sources, source_depths, poses = [], [], [], []
        pbar  = progressbar.ProgressBar(widgets=[
            ' [ Initializing Environment ] ', progressbar.Bar(), ' (', progressbar.ETA(), ') ',])
        
        for k,v in pbar(uuids):
            data = self.dataset[v]
            target, target_depth = data[1], data[3]
            if self.scale_up !=1:
                target =  cv2.resize(
                    target,None,
                    fx=1.0/self.scale_up, 
                    fy=1.0/self.scale_up, 
                    interpolation = cv2.INTER_CUBIC)
                target_depth =  cv2.resize(
                    target_depth,None,
                    fx=1.0/self.scale_up, 
                    fy=1.0/self.scale_up, 
                    interpolation = cv2.INTER_CUBIC)
            pose = data[-1][0].numpy()
            targets.append(target)
            poses.append(pose)
            sources.append(target)
            source_depths.append(target_depth)
        context_mist = zmq.Context()
        socket_mist = context_mist.socket(zmq.REQ)
        socket_mist.connect("tcp://localhost:5555")
        ## TODO (hzyjerry): make sure 5555&5556 are not occupied, or use configurable ports

        PCRenderer.sync_coords()
        renderer = PCRenderer(5556, sources, source_depths, target, rts, self.scale_up)
        self.r_camera_rgb = renderer


    def setup_camera_multi(self):
        def camera_multi_excepthook(exctype, value, tb):
            print("killing", self.r_camera_mul)
            self.r_camera_mul.terminate()
            while tb:
                filename = tb.tb_frame.f_code.co_filename
                name = tb.tb_frame.f_code.co_name
                lineno = tb.tb_lineno
                print '   File "%.500s", line %d, in %.500s' %(filename, lineno, name)
                tb = tb.tb_next
            print ' %s: %s' %(exctype.__name__, value)
        sys.excepthook = camera_multi_excepthook

        dr_path = os.path.join(os.path.dirname(os.path.abspath(realenv.__file__)), 'core', 'channels', 'depth_render')
        cur_path = os.getcwd()
        os.chdir(dr_path)
        cmd = "./depth_render --modelpath {}".format(self.model_path)
        self.r_camera_mul = subprocess.Popen(shlex.split(cmd), shell=False)
        os.chdir(cur_path)


    def check_port_available(self):
        # TODO (hzyjerry) not working
        """
        s = socket.socket()
        try:
            s.connect(("127.0.0.1", 5555))
        except socket.error as e:
            raise e
            raise error.Error("Realenv starting error: port {} is in use".format(5555))
        try:
            s.connect(("127.0.0.1", 5556))
        except socket.error as e:
            raise error.Error("Realenv starting error: port {} is in use".format(5556))
        """
        return



        