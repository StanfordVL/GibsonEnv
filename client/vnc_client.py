from __future__ import print_function
import go_vncdriver
import time
from universe.vncdriver import constants

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
		self.count = 0
		
		self.vnc_kwargs  = {}
		self.vnc_kwargs.setdefault('start_timeout', 1000)
		self.vnc_kwargs.setdefault('encoding', 'zrle')
		self.vnc_kwargs = {k: v for k, v in self.vnc_kwargs.items() if v is not None}
		

	def connect(self):
		kwargs = {
			'address': DEFAULT_ADDRESS,
			'password': 'qwertyui',
			'name': CONNECTION_NAME,
		}
		kwargs.update(self.vnc_kwargs)
		h = self.vnc_session.connect(**kwargs)

	def step(self):
		keydown = 0 if (self.count % 2 == 0) else 1
		# print(self.vnc_session.step({CONNECTION_NAME: [("KeyEvent", keycode('a'), keydown)]}))
		observations, infos, errors = self.vnc_session.step({CONNECTION_NAME: [("KeyEvent", keycode('a'), keydown)]})
		self.count += 1
		# observations, infos, errors = self.vnc_session.flip()
		# print(observations, infos, errors)
		return observations, infos, errors

	def stepN(self):
		observations, infos, errors = self.vnc_session.step({CONNECTION_NAME: [("KeyEvent", keycode('n'), 1)]})
		observations, infos, errors = self.vnc_session.step({CONNECTION_NAME: [("KeyEvent", keycode('n'), 0)]})
		return observations, infos, errors



if __name__ == '__main__':
	main()
