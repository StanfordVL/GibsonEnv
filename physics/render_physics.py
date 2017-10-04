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
	#print(keys)
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
	object.parseActionAndUpdate(action)
	
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

##  Physics Object stands for every controllable object in world
#     This class mostly handles action parsing
# 	  Note: only stores realtime delta pose, absolute pose is not 
# 	  stored
class PhysicsObject():
	## By OpenGL convention, object is default: +x facing,
	#  	 which is incompatible with camera view matrix (-z facing).
	#  	 The inconsistency is handled by _cameraCalibrate function.
	## By default, internal quaternion variables are [x, y, z, w]
	def __init__(self, uid):
		self.uid = uid
		## Relative delta rotation of object to world
		self.d_alpha, self.d_beta, self.d_gamma = 0, 0, 0
		## Relative delta position inside object's world view
		self.d_xyz = np.array([0, 0, 0], dtype=float)

		## DEPRECATED: roll, pitch, yaw
		self.roll, self.pitch, self.yaw = 0, np.pi/6, 0
		self.updateInitialPositionOrientation()

	## Convert object's head from +x facing to -z facing
	#  Note that this is not needed when you're computing view_matrix,
	#  only use this function for adjusting object head
	#  To get object rotation at current object pose:
	#   	rotation = self._cameraCalibrate(self._rotateIntrinsic(
	#				   self.quat_world))
	#  To get view rotation at current object pose:
	#  		rotation = self._rotateIntrinsic(self.quat_world)
	def _cameraCalibrate(self, org_quat_xyzw):
		z_facing_wxyz = euler.euler2quat(-np.pi/2, np.pi/2, 0)
		org_quat_wxyz = quatXyzwToWxyz(org_quat_xyzw)
		new_quat_xyzw = quatWxyzTOXyzw(quaternions.qmult(org_quat_wxyz, z_facing_wxyz))
		return new_quat_xyzw

	def _cameraUncalibrate(self, new_quat_xyzw):
		x_facing_wxyz = euler.euler2quat(np.pi/2, 0, -np.pi/2)
		new_quat_wxyz = quatXyzwToWxyz(new_quat_xyzw)
		org_quat_wxyz = quatWxyzTOXyzw(quaternions.qmult(new_quat_wxyz, x_facing_wxyz))
		return org_quat_wxyz

	## Update physics simulation (object position, object rotation)
	#  This is the function where obj communicates with the world,
	#  in manual step mode, you need to call updatePositionOrientation()
	#  periodically. 
	def updateInitialPositionOrientation(self):
		pos_world_xyz, quat_world_xyzw = p.getBasePositionAndOrientation(self.uid)
		new_pos_world_xyz = np.array([pos_world_xyz[0], pos_world_xyz[1], 1])

		## parse xyz, alpha, beta, gamma from world
		## apply local delta xyz, delta alpha, delta beta, delta gamma
		## update to world
		new_quat_world_xyzw = self._rotateIntrinsic(quat_world_xyzw)
		## Calibrate
		new_quat_world_xyzw = self._cameraCalibrate(new_quat_world_xyzw)
		#pos_xyz   = self._translateIntrinsic()
		p.resetBasePositionAndOrientation(self.uid, new_pos_world_xyz, new_quat_world_xyzw)


	def updatePositionOrientation(self):
		pos_world_xyz, quat_world_xyzw = p.getBasePositionAndOrientation(self.uid)
		quat_world_xyzw 	= self._cameraUncalibrate(quat_world_xyzw)

		new_pos_world_xyz   = self._translateIntrinsic(pos_world_xyz, quat_world_xyzw)
		new_quat_world_xyzw = self._rotateIntrinsic(quat_world_xyzw)
		## Calibrate

		print("last world pos", quat_world_xyzw)
		print()
		print("new  world pos", new_quat_world_xyzw)
		new_quat_world_xyzw = self._cameraCalibrate(new_quat_world_xyzw)
		
		p.resetBasePositionAndOrientation(self.uid, new_pos_world_xyz, new_quat_world_xyzw)


	## Convert delta_relative (movement in object's world) to 
	#  delta_absolute (movement in actual world)
	'''
	def _positionDeltaToAbsolute(self, delta_relative):
		quat_xyzw = self._rotateIntrinsic()
		quat_wxyz = quatXyzwToWxyz(quat_xyzw)
		delta_abs = quaternions.quat2mat(quat_wxyz).dot(delta_relative)
		return delta_abs
	'''
	## Convert intrinsic (x, y, z) translation to extrinsic 	
	def _translateIntrinsic(self, pos_world_xyz, quat_world_xyzw):
		delta_objec_xyz = self.d_xyz
		quat_world_wxyz = quatXyzwToWxyz(quat_world_xyzw)
		delta_world_xyz = quaternions.quat2mat(quat_world_wxyz).dot(delta_objec_xyz)
		return pos_world_xyz + delta_world_xyz

	## Add intrinsic (d_alpha, d_beta, d_gamma) rotation to extrinsic 
	def _rotateIntrinsic(self, quat_world_xyzw):
		quat_world_wxyz = quatXyzwToWxyz(quat_world_xyzw)
		euler_world 	= euler.quat2euler(quat_world_wxyz)
		
		alpha = euler_world[0] + self.d_alpha
		beta  = euler_world[1] + self.d_beta
		gamma = euler_world[2] + self.d_gamma

		#print("delta rotationss", self.d_alpha, self.d_beta, self.d_gamma)
		#print('alpha beta gamma', alpha, beta, gamma)
		new_quat_world_wxyz  = euler.euler2quat(alpha, beta, gamma, 'sxyz')
		new_quat_world_xyzw  = quatWxyzTOXyzw(new_quat_world_wxyz)
		return new_quat_world_xyzw
		
	## DEPRECATED: roll, pitch, yaw
	def _principle_to_mat(self):
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
		pos  = self.pos_world
		quat = self.quat_world
		quat = self._principle_to_mat() * quatXyzwToWxyz(quat)

	def parseActionAndUpdate(self, action):
		## Update position: because the object's rotation
		#  changes every time, the position needs to be updated
		#  by delta
		#print(action)
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
		self.d_xyz = delta_xyz

		## Update rotation: reset the rotation every time
		if action['alpha'] > 0:
			self.d_alpha = np.pi/16
		if action['alpha'] < 0:
			self.d_alpha = - np.pi/16
		if action['beta'] > 0:
			self.d_beta = np.pi/16
		if action['beta'] < 0:
			self.d_beta = - np.pi/16
		if action['gamma'] > 0:
			self.d_gamma = np.pi/16
		if action['gamma'] < 0:
			self.d_gamma = - np.pi/16

		self.updatePositionOrientation()
		self._clearUpDelta()	

	def _clearUpDelta(self):
		self.d_xyz = np.array([0, 0, 0], dtype=float)
		self.d_alpha, self.d_beta, self.d_gamma = 0, 0, 0
	


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
	boundaryUid = p.createCollisionShape(p.GEOM_MESH, fileName=obj_path, meshScale=[1, 1, 1], flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
	print("Exterior boundary", boundaryUid)
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
	#pos, rot = p.getBasePositionAndOrientation(objectUid)
	#newpos = (pos[0], pos[1], 1)
	#p.resetBasePositionAndOrientation(objectUid, newpos, rotate_quat_by_euler(rot, np.pi/6, 0, 0))


	print("Generated cart", objectUid)

	#p.setGravity(0,0,-10)
	p.setRealTimeSimulation(0)
	
	## same as cv.waitKey(5) in viewPort
	#p.setTimeStep(0.01)


	context = zmq.Context()
	socket = context.socket(zmq.REP)
	socket.bind("tcp://*:5556")
	while (1):
		getUpdateFromKeyboard(cart)
		p.stepSimulation()
		time.sleep(0.05)
		#if PHYSICS_FIRST:
			## Physics-first simulation
			#synchronizeWithViewPort()
			#p.stepSimulation()
			#time.sleep(0.05)
		
		#else:
			## Visual-first simulation
			#getCollisionFromUpdate()
