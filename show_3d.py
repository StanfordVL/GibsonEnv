import numpy as np
import ctypes as ct
import cv2
import sys
import argparse
from datasets import ViewDataSet3D
showsz = 256
mousex,mousey=0.5,0.5
changed=True

def onmouse(*args):
    global mousex,mousey,changed
    y=args[1]
    x=args[2]
    mousex=x/float(showsz)
    mousey=y/float(showsz * 2)
    changed=True

cv2.namedWindow('show3d')
cv2.moveWindow('show3d',0,0)
cv2.setMouseCallback('show3d',onmouse)

dll=np.ctypeslib.load_library('render','.')

def showpoints(img, depth):
    global mousex,mousey,changed
    show=np.zeros((showsz,showsz * 2,3),dtype='uint8')
    
    
    show = img
    
    def render():
        pass
    
    while True:
        cv2.imshow('show3d',show)
        cmd=cv2.waitKey(10)%256
    
        if cmd==ord('q'):
            break
    

if __name__=='__main__':
    
    idx = 0
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug'  , action='store_true', help='debug mode')
    parser.add_argument('--dataroot'  , required = True, help='dataset path')
    opt = parser.parse_args()
    d = ViewDataSet3D(root=opt.dataroot, transform = np.array, mist_transform = np.array, seqlen = 2)
    
    source = d[idx][0][0]
    source_depth = d[idx][2][0]
    
    print(source.shape, source_depth.shape)
    
    showpoints(source, source_depth)
    
    
    