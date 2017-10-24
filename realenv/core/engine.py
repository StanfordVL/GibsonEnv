from realenv.data.datasets import ViewDataSet3D
from realenv.core.render.show_3d2 import PCRenderer, sync_coords
from realenv.core.channels.depth_render import run_depth_render
from realenv.core.physics.physics_env import PhysicsEnv
from realenv import error

import progressbar
import subprocess, os, signal
import numpy as np
import sys
import zmq
import socket
import shlex
import gym
from realenv.data.datasets import get_model_path


class Engine(object):
    def __init__(self, model_id, human, debug, physics_env):
        self.dataset  = ViewDataSet3D(transform = np.array, mist_transform = np.array, seqlen = 2, off_3d = False, train = False)
        self.model_id  = model_id
        self.scale_up  = 1
        self.human = human
        self.debug = debug
        self.physics_env = physics_env

        self.r_visuals = None
        self.r_physics = None
        self.p_channel = None

    def setup_all(self):
        def channel_excepthook(exctype, value, tb):
            print("killing", self.p_channel)
            self.p_channel.terminate()
            while tb:
                filename = tb.tb_frame.f_code.co_filename
                name = tb.tb_frame.f_code.co_name
                lineno = tb.tb_lineno
                print '   File "%.500s", line %d, in %.500s' %(filename, lineno, name)
                tb = tb.tb_next
            print ' %s: %s' %(exctype.__name__, value)
        sys.excepthook = channel_excepthook
        
        self._checkPortClear()
        self._setupChannel()
        self._setupVisuals()
        
        ## Sync initial poses
        pose_init = self.r_visuals.renderOffScreenInitialPose()
        self._setupPhysics(self.human, pose_init)

        if self.debug:
            self.r_visuals.renderToScreenSetup()
            #self.r_displayer = RewardDisplayer() #MPRewardDisplayer()

        return self.r_visuals, self.r_physics, self.p_channel

    def _checkPortClear(self):
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

    def _setupChannel(self):
        model_path, model_id = get_model_path()
        dr_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'channels', 'depth_render')
        cur_path = os.getcwd()
        os.chdir(dr_path)
        cmd = "./depth_render --modelpath {}".format(model_path)
        self.p_channel = subprocess.Popen(shlex.split(cmd), shell=False)
        os.chdir(cur_path)
      

    def _setupVisuals(self):
        scene_dict = dict(zip(self.dataset.scenes, range(len(self.dataset.scenes))))
        ## Todo: (hzyjerry) more error handling
        if not self.model_id in scene_dict.keys():
             raise error.Error("Dataset not found: model {} cannot be loaded".format(self.model_id))
        else:
            scene_id = scene_dict[self.model_id]
        uuids, rts = self.dataset.get_scene_info(scene_id)
        targets = []
        sources = []
        source_depths = []
        poses = []
        pbar  = progressbar.ProgressBar(widgets=[
                            ' [ Initializing Environment ] ',
                            progressbar.Bar(),
                            ' (', progressbar.ETA(), ') ',
                            ])
        for k,v in pbar(uuids):
            data = self.dataset[v]
            target = data[1]
            target_depth = data[3]
            
            if self.scale_up !=1:
                target =  cv2.resize(target,None,fx=1.0/self.scale_up, fy=1.0/self.scale_up, interpolation = cv2.INTER_CUBIC)
                target_depth =  cv2.resize(target_depth,None,fx=1.0/self.scale_up, fy=1.0/self.scale_up, interpolation = cv2.INTER_CUBIC)
            
            pose = data[-1][0].numpy()
            targets.append(target)
            poses.append(pose)
            sources.append(target)
            source_depths.append(target_depth)
        context_mist = zmq.Context()
        socket_mist = context_mist.socket(zmq.REQ)
        socket_mist.connect("tcp://localhost:5555")

        ## TODO (hzyjerry): make sure 5555&5556 are not occupied, or use configurable ports
        sync_coords()

        renderer = PCRenderer(5556, sources, source_depths, target, rts, self.scale_up)
        self.r_visuals = renderer

    def _setupPhysics(self, human, pose_init):
        """
        framePerSec = 13
        renderer = PhysicsEnv(self.dataset.get_model_obj(), render_mode="human_play",fps=framePerSec, pose=pose_init)
        self.r_physics = renderer
        """
        env = gym.make(self.physics_env)
        env.render(mode="human")
        env.reset()
        self.r_physics = env

    def cleanUp(self):
        self.p_channel.terminate()
        