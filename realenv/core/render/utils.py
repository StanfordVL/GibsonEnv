
# coding: utf-8

# In[1]:

import numpy as np
import math
import PIL
import transforms3d
from numpy import cos,sin

# In[36]:

def qmul(q1, q2):
    a1, b1, c1, d1 = q1
    a2, b2, c2, d2 = q2
    r1 = a1*a2 - b1*b2 - c1*c2 - d1*d2
    r2 = a1*b2 + a2*b1 + c1*d2 - c2*d1
    r3 = a1*c2 + a2*c1 - b1*d2 + b2*d1
    r4 = a1*d2 + a2*d1 + b1*c2 - b2*c1
    return np.array([r1, r2, r3, r4]).astype(np.float32)


# In[73]:

def qinv(q):
    a, b, c, d = q
    norm2 = float(a**2 + b**2 + c**2 + d**2)
    return np.array([a, -b, -c, -d]).astype(np.float32)/(norm2 + 1e-10)


# In[74]:

def qtrans(q1, q2):
    #q1 / q2 = q1 * q2^-1
    return qmul(qinv(q2), q1)


# In[75]:

def ptrans(p1, p2):
    return p1 - p2


# In[76]:

def trans(z1, z2):
    p = ptrans(z1[:3], z2[:3])
    q = qtrans(z1[3:], z2[3:])
    return np.concatenate([p,q],0)


def to_r(q):
    a,b,c,d = q
    R = np.array([[a**2 + b**2 - c**2 - d**2, 2*b*c - 2*a*d, 2*b*d + 2*a*c],
                 [2*b*c + 2*a*d, a**2-b**2+c**2-d**2, 2*c*d - 2*a*b],
                 [2*b*d - 2*a*c, 2*c*d + 2*a*b, a**2 - b**2 - c**2 + d**2]], dtype=np.float32)
    return R



def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-2


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :

    assert(isRotationMatrix(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

def trans2(z1, z2):
    R1 = to_r(z1[3:])
    T1 = -np.dot(R1, np.array(z1[:3]))
    R2 = to_r(z2[3:])
    T2 = -np.dot(R2, np.array(z2[:3]))

    RT1 = np.concatenate([R1, np.expand_dims(T1,1)], 1);
    RT2 = np.concatenate([R2, np.expand_dims(T2,1)], 1);

    e =  np.array([[0,0,0,1]])

    M1 = np.concatenate([RT1, e], 0)
    M2 = np.concatenate([RT2, e], 0)
    #print(M1, M2)

    dM = np.dot(M1, np.linalg.inv(M2))
    #dR = np.dot(R2.transpose(), R1)
    dR = rotationMatrixToEulerAngles(dM[:3, :3])
    #dT = dM[:3, -1]
    #dT = np.dot(R2.transpose(), T1 - T2)
    dT = dM[:3, -1]
    return (dT, dR)

def transfromM(dM):
    dR = rotationMatrixToEulerAngles(dM[:3, :3])
    dT = dM[:3, -1]
    return (dT, dR)




def rotateImage(img, angle):
    img = np.asarray(img)
    width = img.shape[1]
    angle = angle/np.pi

    if angle < 0:
        angle += 2

    if angle >= 0:
        shift = int(angle * width / 2)
        img = np.concatenate([img[:,shift:,:], img[:, :shift, :]], 1)

    return PIL.Image.fromarray(img)


def generate_transformation_matrix(x,y,z,yaw,pitch,roll):
    current_t = np.eye(4)
    current_t[0,-1] = x
    current_t[1,-1] = y
    current_t[2,-1] = z       
    alpha = yaw
    beta = pitch
    gamma = roll
    cpose = np.zeros(16) 
    cpose[0] = cos(alpha) * cos(beta);
    cpose[1] = cos(alpha) * sin(beta) * sin(gamma) - sin(alpha) * cos(gamma);
    cpose[2] = cos(alpha) * sin(beta) * cos(gamma) + sin(alpha) * sin(gamma);
    cpose[3] = 0
    cpose[4] = sin(alpha) * cos(beta);
    cpose[5] = sin(alpha) * sin(beta) * sin(gamma) + cos(alpha) * cos(gamma);
    cpose[6] = sin(alpha) * sin(beta) * cos(gamma) - cos(alpha) * sin(gamma);
    cpose[7] = 0
    cpose[8] = -sin(beta);
    cpose[9] = cos(beta) * sin(gamma);
    cpose[10] = cos(beta) * cos(gamma);
    cpose[11] = 0
    cpose[12:16] = 0
    cpose[15] = 1           
    cpose = cpose.reshape((4,4))      
    cpose = np.dot(cpose, current_t)
    current_rt = cpose
    rotation = np.array([[0,-1,0,0],[-1,0,0,0],[0,0,1,0],[0,0,0,1]])
    current_rt = np.dot(rotation, current_rt)
    return current_rt




def mat_to_str(matrix):
    s = ""
    for row in range(4):
        for col in range(4):
            s = s + " " + str(matrix[row][col])
    return s.strip()


def mat_to_posi_xyz(cpose):
    return cpose[:3, -1]

def mat_to_quat_xyzw(cpose):
    rot = cpose[:3, :3]
    ## Return: [r_x, r_y, r_z, r_w]
    wxyz = transforms3d.quaternions.mat2quat(rot)
    return quat_wxyz_to_xyzw(wxyz)



## Blender generated poses and OpenGL default poses adhere to
## different conventions. To better understand the nitty-gritty
## transformations inside this file, check out this:
## https://en.wikibooks.org/wiki/OpenGL_Programming/Modern_OpenGL_Tutorial_Navigation
##  Blender: z-is-up
##      same with: csv, rt_camera_matrix
##  OpenGL: y-is-up
##      same with: obj
##  Default camera: y is up-direction, -z facing


## Quat(wxyz)
def quat_pos_to_mat(pos, quat):
    r_w, r_x, r_y, r_z = quat
    #print("quat", r_w, r_x, r_y, r_z)
    mat = np.eye(4)
    mat[:3, :3] = transforms3d.quaternions.quat2mat([r_w, r_x, r_y, r_z])
    mat[:3, -1] = pos
    # Return: roll, pitch, yaw
    return mat

## Used for URDF models that are default -x facing
##  Rotate the model around its internal x axis for 90 degrees
##  so that it is at "normal" pose when applied camera_rt_matrix 
## Format: wxyz for input & return
def z_up_to_y_up(quat_wxyz):
    ## Operations (1) rotate around y for pi/2, 
    ##            (2) rotate around z for pi/2
    to_y_up = transforms3d.euler.euler2quat(0, np.pi/2, np.pi/2)
    return transforms3d.quaternions.qmult(quat_wxyz, to_y_up)

## Models coming out of opengl are negative x facing
##  Transform the default to 
def y_up_to_z_up(quat_wxyz):
    to_z_up = transforms3d.euler.euler2quat(np.pi/2, 0, np.pi/2)
    return transforms3d.quaternions.qmult(to_z_up, quat_wxyz)


def quat_wxyz_to_euler(wxyz):
    q0, q1, q2, q3 = wxyz
    sinr = 2 * (q0 * q1 + q2 * q3)
    cosr = 1 - 2 * (q1 * q1 + q2 * q2)
    sinp = 2 * (q0 * q2 - q3 * q1)
    siny = 2 * (q0 * q3 + q1 * q2)
    cosy = 1 - 2 * (q2 * q2 + q3 * q3)

    roll  = np.arctan2(sinr, cosr)
    pitch = np.arcsin(sinp)
    yaw   = np.arctan2(siny, cosy)
    return [roll, pitch, yaw]


## wxyz: numpy array format
def quat_wxyz_to_xyzw(wxyz):
    return np.concatenate((wxyz[1:], wxyz[:1]))

## xyzw: numpy array format
def quat_xyzw_to_wxyz(xyzw):
    return np.concatenate((xyzw[-1:], xyzw[:-1]))