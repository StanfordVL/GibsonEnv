import pybullet as p
import time
import random
import zmq
import argparse
import os
import json
import numpy as np
from transforms3d import euler, quaternions, taitbryan

from numpy import sin, cos

PHYSICS_FIRST = True

def camera_init_orientation(quat):
    to_z_facing = euler.euler2quat(np.pi/2, np.pi, 0)
    return quaternions.qmult(to_x_facing, quat_wxyz)

def setPosViewOrientation(objectUid, pos, rot):
	return

def getUpdateFromKeyboard(object):
	# Controls: p.B3G_RIGHT_ARROW, p.B3G_LEFT_ARROW, p.B3G_DOWN_ARROW
	# 		p.B3G_UP_ARROW, 
	keys = p.getKeyboardEvents()
	print(keys)
	#print(p.getBasePositionAndOrientation(objectUid))
	#print(p.getContactPoints(boundaryUid, objectUid))
	## Down
	action = {
		'up'	  : False,
		'down'	  : False,
		'left'	  : False,
		'right'	  : False,
		'forward' : False,
		'backward': False,
		'alpha'   : 0,
		'beta'    : 0,
		'gamma'   : 0
	}
	if (ord('d') in keys):
		action['right'] = True
	if (ord('a') in keys):
		action['left'] = True
	if (ord('s') in keys):
		action['backward'] = True
	if (ord('w') in keys):
		action['forward'] = True
	if (ord('z') in keys):
		action['up'] = True
	if (ord('c') in keys):
		action['down'] = True

	if (ord('u') in keys):
		action['alpha'] = 1
	if (ord('j') in keys):
		action['alpha'] = -1
	if (ord('i') in keys):
		action['beta'] = 1
	if (ord('k') in keys):
		action['beta'] = -1
	if (ord('o') in keys):
		action['gamma'] = 1
	if (ord('l') in keys):
		action['gamma'] = -1
	object.parseAction(action)
	object.updatePositionOrientation()

def getCollisionFromUpdate():
	message = socket.recv().decode("utf-8")
	
	x, y, z, r_w, r_x, r_y, r_z = map(float, message.split())
	
	p.resetBasePositionAndOrientation(objectUid, [x, y, z], [r_w, r_x, r_y, r_z])
	p.stepSimulation()
	print("step simulation done")
	collisions = p.getContactPoints(boundaryUid, objectUid)
	if len(collisions) == 0:
		print("No collisions")
	else:
		print("Collisions!")
	print("collision length", len(collisions))
	socket.send_string(str(len(collisions)))
	return


def synchronizeWithViewPort():
	#step
	view_pose = json.loads(socket.recv().decode("utf-8"))
	changed = view_pose['changed']
	## Always send pose from last frame
	pos, rot = p.getBasePositionAndOrientation(objectUid)

	print("receiving changed ?", changed)
	print("receiving from view", view_pose['pos'])
	print("original view posit", pos)
	print("receiving from view", view_pose['quat'])
	print("original view posit", rot)
	if changed:
		## Apply the changes
		new_pos = view_pose['pos']
		new_quat = view_pose['quat']
		#new_quat = [0, 0, 0, 1]
		p.resetBasePositionAndOrientation(objectUid, new_pos, new_quat)
	p.stepSimulation()
	pos, rot = p.getBasePositionAndOrientation(objectUid)
	print("after applying pose", pos)
	print("")
	#print(changed, pos, rot)
	socket.send_string(json.dumps([pos, rot]))


## wxyz: numpy array format
def quatWxyzTOXyzw(wxyz):
    return np.concatenate((wxyz[1:], wxyz[:1]))

## xyzw: numpy array format
def quatXyzwToWxyz(xyzw):
    return np.concatenate((xyzw[-1:], xyzw[:-1]))

def rotate_quat_by_euler(xyzw, e_x, e_y, e_z):
	wxyz = quatXyzwToWxyz(xyzw)
	rot_mat = euler.euler2mat(e_x, e_y, e_z)
	wxyz = quaternions.qmult(rot_mat, wxyz)
	return quatWxyzTOXyzw(wxyz)


class PhysicsObject():
	## By OpenGL convention, object is default: +x facing,
	#  which is incompatible with camera view matrix (-z facing).
	#  The inconsistency is handled by _camera_calibrate function.
	def __init__(self, uid):
		self.uid = uid
		## Initial position
		self.pos_init, self.quat_init = p.getBasePositionAndOrientation(uid)
		self.pos_init = np.array([self.pos_init[0], self.pos_init[1], 1])

		## Relative rotation of object to world
		self.alpha, self.beta, self.gamma = 0, 0, 0
		## Relative position inside object's world view
		self.xyz = np.array([0, 0, 0])

		## DEPRECATED: roll, pitch, yaw
		self.roll, self.pitch, self.yaw = 0, np.pi/6, 0
		
		
	## Convert object's head from +x facing to -z facing
	#  Note that this is not needed when you're computing view_matrix,
	#  only use this function for adjusting object head
	#  To get object rotation at current object pose:
	#   	rotation = self.camera_calibrate(self._rotate_intrinsic(
	#				   self.quat_init))
	#  To get view rotation at current object pose:
	#  		rotation = self._rotate_intrinsic(self.quat_init)
	def _camera_calibrate(self, quat_xyzw):
		z_facing_wxyz = euler.euler2quat(-np.pi/2, np.pi/2, 0)
		quat_wxyz = quatXyzwToWxyz(quat_xyzw)
		return quatWxyzTOXyzw(quaternions.qmult(quat_wxyz, z_facing_wxyz))

	## Update physics simulation (object position, object rotation)
	def updatePositionOrientation(self):
		quat_xyzw = self._rotate_intrinsic(self.quat_init)
		quat_xyzw = self._camera_calibrate(quat_xyzw)
		pos_xyz   = self._translate_intrinsic()
		p.resetBasePositionAndOrientation(self.uid, pos_xyz, quat_xyzw)

	## Convert delta_relative (movement in object's world) to 
	#  delta_absolute (movement in actual world)
	def _positionDeltaToAbsolute(self, delta_relative):
		quat_xyzw = self._rotate_intrinsic(self.quat_init)
		quat_wxyz = quatXyzwToWxyz(quat_xyzw)
		delta_abs = quaternions.quat2mat(quat_wxyz).dot(delta_relative)
		return delta_abs
		
	## Convert intrinsic (x, y, z) translation to extrinsic 	
	def _translate_intrinsic(self):
		return self.pos_init + self.xyz

	## Convert intrinsic (alpha, beta, gamma) rotation to extrinsic 
	def _rotate_intrinsic(self, quat_xyzw):
		quat_wxyz = quatXyzwToWxyz(quat_xyzw)
		intrinsic = euler.euler2quat(self.alpha, self.beta, self.gamma, 'rxyz')
		return quatWxyzTOXyzw(quaternions.qmult(intrinsic, quat_wxyz))
		
	## DEPRECATED: roll, pitch, yaw
	def principle_to_mat(self):
		alpha = self.yaw
		beta  = self.pitch
		gamma = self.roll
		mat   = np.eye(4)
		mat[0, 0] = cos(alpha)*cos(beta)
		mat[1, 0] = sin(alpha)*cos(beta)
		mat[2, 0] = -sin(beta)

		mat[0, 1] = cos(alpha)*sin(beta)*sin(gamma) - sin(alpha)*cos(gamma)
		mat[1, 1] = sin(alpha)*sin(beta)*sin(gamma) + cos(alpha)*cos(gamma)
		mat[2, 1] = cos(beta)*sin(gamma)

		mat[0, 2] = cos(alpha)*sin(beta)*cos(gamma) + sin(alpha)*sin(gamma)
		mat[1, 2] = sin(alpha)*sin(beta)*cos(gamma) - cos(alpha)*sin(gamma)
		mat[2, 2] = cos(beta)*cos(gamma)
		return mat

	def getPosAndOrientation(self):
		pos  = self.pos_init
		quat = self.quat_init
		quat = self.principle_to_mat() * quatXyzwToWxyz(quat)

	def parseAction(self, action):
		## Update position: because the object's rotation
		#  changes every time, the position needs to be updated
		#  by delta
		delta_xyz = np.array([0, 0, 0], dtype=float)
		if action['up']:
			delta_xyz[1] =  0.1
		if action['down']:
			delta_xyz[1] = -0.1
		if action['left']:
			delta_xyz[0] = -0.1
		if action['right']:
			delta_xyz[0] = 0.1
		if action['forward']:
			delta_xyz[2] = -0.1
		if action['backward']:
			delta_xyz[2] = 0.1
		self.xyz = self.xyz + self._positionDeltaToAbsolute(delta_xyz)

		## Update rotation: reset the rotation every time
		if action['alpha'] > 0:
			self.alpha = self.alpha + np.pi/16
		if action['alpha'] < 0:
			self.alpha = self.alpha - np.pi/16
		if action['beta'] > 0:
			self.beta = self.beta + np.pi/16
		if action['beta'] < 0:
			self.beta = self.beta - np.pi/16
		if action['gamma'] > 0:
			self.gamma = self.gamma + np.pi/16
		if action['gamma'] < 0:
			self.gamma = self.gamma - np.pi/16


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--datapath'  , required = True, help='dataset path')
	parser.add_argument('--model'  , type = str, default = '', help='path of model')

	opt = parser.parse_args()


	## Turn on p.GUI for visualization
	## Turn on p.GUI for headless mode
	p.connect(p.GUI)
	#p.connect(p.DIRECT)


	obj_path = os.path.join(opt.datapath, opt.model, "modeldata", 'out_z_up.obj')

	p.setRealTimeSimulation(0)
	#boundaryUid = p.createCollisionShape(p.GEOM_MESH, fileName=obj_path, meshScale=[1, 1, 1], flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
	#print("Exterior boundary", boundaryUid)
	p.createMultiBody(0,0)

	sphereRadius = 0.05
	colSphereId = p.createCollisionShape(p.GEOM_SPHERE,radius=sphereRadius)
	colBoxId = p.createCollisionShape(p.GEOM_BOX,halfExtents=[sphereRadius,sphereRadius,sphereRadius])

	mass = 1
	visualShapeId = -1

		
		
	link_Masses=[1]
	linkCollisionShapeIndices=[colBoxId]
	linkVisualShapeIndices=[-1]
	linkPositions=[[0,0,0.11]]
	linkOrientations=[[0,0,0,1]]
	linkInertialFramePositions=[[0,0,0]]
	linkInertialFrameOrientations=[[0,0,0,1]]
	indices=[0]
	jointTypes=[p.JOINT_REVOLUTE]
	axis=[[0,0,1]]

	allSpheres = []


	#objectUid = p.loadURDF("models/quadrotor.urdf", globalScaling = 0.8)
	objectUid = p.loadURDF("models/husky.urdf", globalScaling = 0.8)
	cart = PhysicsObject(objectUid)
	cart.updatePositionOrientation()
	#pos, rot = p.getBasePositionAndOrientation(objectUid)
	#newpos = (pos[0], pos[1], 1)
	#p.resetBasePositionAndOrientation(objectUid, newpos, rotate_quat_by_euler(rot, np.pi/6, 0, 0))


	print("Generated cart", objectUid)

	#p.setGravity(0,0,-10)
	#p.setRealTimeSimulation(1)
	
	## same as cv.waitKey(5) in viewPort
	#p.setTimeStep(0.01)


	context = zmq.Context()
	socket = context.socket(zmq.REP)
	socket.bind("tcp://*:5556")
	while (1):
		getUpdateFromKeyboard(cart)
		#p.stepSimulation()
		time.sleep(0.05)
		#if PHYSICS_FIRST:
			## Physics-first simulation
			#synchronizeWithViewPort()
			#p.stepSimulation()
			#time.sleep(0.05)
		
		#else:
			## Visual-first simulation
			#getCollisionFromUpdate()
