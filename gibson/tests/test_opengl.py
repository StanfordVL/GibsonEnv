from __future__ import print_function
import time
import numpy as np
import sys
import gym
import os.path as osp
import os, zmq
import subprocess, shlex
import gibson
import gibson.core.render.utils as utils


WINDOW_SZ = 512
FOV = 120
PORT_DEPTH = 5555
ROTATION_CONST = np.array([[0,1,0,0],[0,0,1,0],[-1,0,0,0],[0,0,0,1]])
POSE_DEFAULT = np.array([[0,1,0,0],[0,0,1,0],[-1,0,0,0],[0,0,0,1]])
POSE_TARGET = np.array([[0,1,0,0],[0,0,1,0],[-1,0,0,0],[0,0,0,1]])

def testOpengl(model_id):
    render_path = osp.join(osp.dirname(osp.abspath(gibson.__file__)), "core", "channels", "depth_render", "depth_render")
    model_path = osp.join(osp.dirname(osp.abspath(gibson.__file__)), "assets", "dataset", model_id)

    render_main  = "./depth_render" + " --modelpath {} --GPU {} -w {} -h {} -f {} -p {}".format(model_path, 0, WINDOW_SZ, WINDOW_SZ, FOV, PORT_DEPTH)


    dr_path = osp.join(osp.dirname(osp.abspath(gibson.__file__)), 'core', 'channels', 'depth_render')
    cur_path = os.getcwd()
    os.chdir(dr_path)
    #subprocess.Popen(shlex.split(render_main))
    os.chdir(cur_path)


    p = (POSE_DEFAULT).dot(np.linalg.inv(POSE_TARGET))
    p = p.dot(np.linalg.inv(ROTATION_CONST))
    s = utils.mat_to_str(p)

    #with Profiler("Render: depth request round-trip"):
    ## Speed bottleneck: 100fps for mist + depth
    
    _context = zmq.Context()
    _client = _context.socket(zmq.REQ)
    _client.connect("tcp://localhost:{}".format(PORT_DEPTH))
    _client.send_string(s)
    _msg = _client.recv()
    print("Received", _msg)

    #render_norm  = render_path + " --modelpath {} -n 1 -w {} -h {} -f {} -p {}".format(self.model_path, WINDOW_SZ, WINDOW_SZ, FOV, self.port_normal)
    #render_semt  = render_path + " --modelpath {} -t 1 -r {} -c {} -w {} -h {} -f {} -p {}".format(self.model_path, self._semantic_source, self._semantic_color, WINDOW_SZ, WINDOW_SZ, FOV, self.port_sem)



if __name__ == '__main__':
    testOpengl("2Q9V7ETLJ2x")