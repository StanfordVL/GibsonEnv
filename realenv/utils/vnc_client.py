from __future__ import print_function

from realenv.client import constants
import go_vncdriver
import time
import yaml
import os
# from realenv.client.client_actions import *
from realenv.client.client_actions import client_newloc

DEFAULT_ADDRESS = 'capri19.stanford.edu:5901'
CONNECTION_NAME = 'Conn'


def main():
    client = VNCClient()
    client.connect()
    client.step()

def keycode(key):
    if key in constants.KEYMAP:
        return constants.KEYMAP.get(key)
    elif len(key) == 1:
        return ord(key)
    else:
        raise error.Error('Not sure how to translate to keycode: {!r}'.format(key))

class VNCClient:
    def __init__(self):
        self.vnc_session = go_vncdriver.VNCSession()
        self._configure()
    
    def _configure(self):
        remote_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'remote.yml')
        # remote_file = './remote.yml'
        with open(remote_file) as f:
            spec = yaml.load(f)
        conn_addr = '{}:{}'.format(spec['capri']['addr'], spec['capri']['port'])
        conn_pswd = spec['capri']['password'] 
        self.kwargs = {
            'address': conn_addr,
            'password': conn_pswd,
            'name': CONNECTION_NAME,
        }
        vnc_kwargs  = {}
        vnc_kwargs.setdefault('start_timeout', 1000)
        vnc_kwargs.setdefault('encoding', 'zrle')
        vnc_kwargs = {k: v for k, v in vnc_kwargs.items() if v is not None}
        self.kwargs.update(vnc_kwargs)

    def connect(self):
        print(self.kwargs)
        self.vnc_session.connect(**self.kwargs)


    def step(self, action):
        observations, infos, errors = self.vnc_session.step({CONNECTION_NAME: [("KeyEvent", keycode(action), 1)]})
        # TODO: botton release causes bug here
        # self.vnc_session.step({CONNECTION_NAME: [("KeyEvent", keycode(action), 0)]})
        # observations, infos, errors = self.vnc_session.step({CONNECTION_NAME: [("KeyEvent", keycode(action), 0)]})

        return observations, infos, errors


    def reset(self):
        refresh_action = client_newloc[0]
        ## Hardcoded, pressing 'n' to refresh
        observations, infos, errors = self.vnc_session.step({CONNECTION_NAME: [("KeyEvent", keycode(refresh_action), 1)]})
        observations, infos, errors = self.vnc_session.step({CONNECTION_NAME: [("KeyEvent", keycode(refresh_action), 0)]})
        return observations, infos, errors



if __name__ == '__main__':
    main()
