## Issue related to time resolution/smoothness
#  http://bulletphysics.org/mediawiki-1.5.8/index.php/Stepping_The_World

import pybullet as p
import time
import random
import zmq
import argparse
import os
import json
import numpy as np
import settings
from transforms3d import euler, quaternions
from PhysicsObject import PhysicsObject
from numpy import sin, cos

PHYSICS_FIRST = True

def camera_init_orientation(quat):
    to_z_facing = euler.euler2quat(np.pi/2, np.pi, 0)
    return quaternions.qmult(to_x_facing, quat_wxyz)

def setPosViewOrientation(objectUid, pos, rot):
	return

	
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

def sendPoseToViewPort(pose):
	socket.send_string(json.dumps(pose))
	socket.recv()

def getInitialPositionOrientation():
	print("waiting to receive initial")
	socket.send_string("Initial")
	pos, quat = json.loads(socket.recv().decode("utf-8"))
	print("received initial", pos, quat)
	return pos, quat

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
	
def stepNsteps(N, object):
	for _ in range(N):
		p.stepSimulation()
		object.parseActionAndUpdate()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--datapath'  , required = True, help='dataset path')
	parser.add_argument('--model'  , type = str, default = '', help='path of model')

	opt = parser.parse_args()


	context = zmq.Context()
	socket = context.socket(zmq.REQ)
	socket.bind("tcp://*:5556")

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


	framePerSec = 5


	#objectUid = p.loadURDF("models/quadrotor.urdf", globalScaling = 0.8)
	objectUid = p.loadURDF("models/husky.urdf", globalScaling = 0.8)

	pos, quat_xyzw = getInitialPositionOrientation()

	v_t = 1 			# 1m/s max speed
	v_r = np.pi/5 		# 36 degrees/s
	#pos  = [0, 0, 3]
	#quat_xyzw = [0, 0, 0, 3]
	cart = PhysicsObject(objectUid, p, pos, quat_xyzw, v_t, v_r, framePerSec)
	
	#pos, quat = p.getViewPosAndOrientation(cart)
	#p.resetBasePositionAndOrientation(cart, [pos[0], pos[1], 1], quat)

	print("Generated cart", objectUid)

	p.setGravity(0,0,-10)
	p.setRealTimeSimulation(0)
	
	## same as cv.waitKey(5) in viewPort


	#p.setTimeStep(1.0/framePerSec)
	p.setTimeStep(1.0/settings.STEPS_PER_SEC)

	lasttime = time.time()
	while (1):
		## Execute one frame
		cart.getUpdateFromKeyboard()
		sendPoseToViewPort(cart.getViewPosAndOrientation())
		#time.sleep(1.0/framePerSec)
		#p.stepSimulation()
		simutime = time.time()
		print('time step', 1.0/settings.STEPS_PER_SEC, 'stepping', settings.STEPS_PER_SEC/framePerSec)
		stepNsteps(int(settings.STEPS_PER_SEC/framePerSec), cart)
		#for _ in range(int(settings.STEPS_PER_SEC/framePerSec)):
		#	p.stepSimulation()

		print("passed time", time.time() - lasttime, "simulation time", time.time() - simutime)
		lasttime = time.time()
		
		#if PHYSICS_FIRST:
			## Physics-first simulation
			#synchronizeWithViewPort()
			#p.stepSimulation()
			#time.sleep(0.05)
		
		#else:
			## Visual-first simulation
			#getCollisionFromUpdate()
