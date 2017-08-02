import numpy as np
import ctypes as ct
import cv2
import sys
mousex,mousey=0.5,0.5
changed=True


def onmouse(*args):
    global mousex,mousey,changed
    y=args[1]
    x=args[2]
    mousex=x/float(showsz)
    mousey=y/float(showsz)
    changed=True

cv2.namedWindow('show3d')
cv2.moveWindow('show3d',0,0)
cv2.setMouseCallback('show3d',onmouse)

dll=np.ctypeslib.load_library('render','.')

def showpoints():
    global mousex,mousey,changed
    
    

if __name__=='__main__':
    np.random.seed(100)
    showpoints()

