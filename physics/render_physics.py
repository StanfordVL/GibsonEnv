import pybullet as p
import time
import random
import zmq
import argparse
import os
import json


def getUpdateFromKeyboard():
	keys = p.getKeyboardEvents()
	#print(p.getBasePositionAndOrientation(objectUid))
	#print(p.getContactPoints(boundaryUid, objectUid))
	## Down
	if (65298 in keys):
		pos, rot = p.getBasePositionAndOrientation(objectUid)
		#print(pos)
		newpos = (pos[0] + 0.1, pos[1], pos[2])
		p.resetBasePositionAndOrientation(objectUid, newpos, rot)
		#p.setVRCameraState(rootPosition = newpos)
	## Up
	if (65297 in keys):
		pos, rot = p.getBasePositionAndOrientation(objectUid)
		#print(pos)
		newpos = (pos[0] - 0.1, pos[1], pos[2])
		p.resetBasePositionAndOrientation(objectUid, newpos, rot)
		#p.setVRCameraState(rootPosition = newpos)
	if (65295 in keys):
		pos, rot = p.getBasePositionAndOrientation(objectUid)
		#print(pos)
		newpos = (pos[0], pos[1] - 0.1, pos[2])
		p.resetBasePositionAndOrientation(objectUid, newpos, rot)
		#p.setVRCameraState(rootPosition = newpos)
	if (65296 in keys):
		pos, rot = p.getBasePositionAndOrientation(objectUid)
		#print(pos)
		newpos = (pos[0], pos[1] + 0.1, pos[2])
		p.resetBasePositionAndOrientation(objectUid, newpos, rot)
		#p.setVRCameraState(rootPosition = newpos)
	#for uid in allSpheres:
	#	p.resetBaseVelocity(uid, [4-random.randint(0, 8), 4-random.randint(0, 8), 1])


def getCollisionFromUpdate():
	message = socket.recv().decode("utf-8")
	#print(message)
	x, y, z, r_w, r_x, r_y, r_z = map(float, message.split())
	#print("received", x, y, z, r_w, r_x, r_y, r_z)
	p.resetBasePositionAndOrientation(objectUid, [x, y, z], [r_w, r_x, r_y, r_z])
	#p.resetBasePositionAndOrientation(objectUid, [0, 0, 0], [r_w, r_x, r_y, r_z])
	p.stepSimulation()
	print("step simulation done")
	collisions = p.getContactPoints(boundaryUid, objectUid)
	if len(collisions) == 0:
	#if True:
		print("No collisions")
	else:
		print("Collisions!")
	print("collision length", len(collisions))
	socket.send_string(str(len(collisions)))
	#socket.send_string(str(0))
	return


def synchronizeWithViewPort():
	#step
	view_pose = json.loads(socket.recv().decode("utf-8"))
	changed = view_pose['changed']
	pos, rot = p.getBasePositionAndOrientation(objectUid)
	print(changed, pos, rot)
	if changed:
		## Apply the changes
		new_pos = view_pose['pos']
		new_quat = view_pose['quat']
		p.resetBasePositionAndOrientation(objectUid, new_pos, new_quat)
	socket.send_string(json.dumps([pos, rot]))
	


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


	objectUid = p.loadURDF("models/quadrotor.urdf", globalScaling = 0.8)
	print("Generated cart", objectUid)

	#p.setGravity(0,0,-10)
	p.setRealTimeSimulation(1)
	
	## same as cv.waitKey(5) in viewPort
	#p.setTimeStep(1)


	context = zmq.Context()
	socket = context.socket(zmq.REP)
	socket.bind("tcp://*:5556")
	while (1):
		#getUpdateFromKeyboard()
		
		## Visual-first simulation
		#getCollisionFromUpdate()

		## Physics-first simulation
		synchronizeWithViewPort()
		#p.stepSimulation()
		#time.sleep(0.05)
		