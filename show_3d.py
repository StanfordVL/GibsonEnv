import numpy as np
import ctypes as ct
import cv2
import sys
import argparse
from datasets import ViewDataSet3D
showsz = 256
mousex,mousey=0.5,0.5
changed=True
pitch,yaw,x,y,z = 0,0,0,0,0
org_pitch, org_yaw, org_x, org_y, org_z = 0,0,0,0,0
mousedown = False
clickstart = (0,0)

def onmouse(*args):
    global mousex,mousey,changed
    global pitch,yaw,x,y,z
    global org_pitch, org_yaw, org_x, org_y, org_z
    global clickstart
    
    if args[0] == cv2.EVENT_LBUTTONDOWN:
        org_pitch, org_yaw, org_x, org_y, org_z =\
        pitch,yaw,x,y,z
        clickstart = (mousex, mousey)

        
    if (args[3] & cv2.EVENT_FLAG_LBUTTON):
        pitch = org_pitch + (mousex - clickstart[0])/10
        yaw = org_yaw + (mousey - clickstart[1])
        changed=True
        
    my=args[1]
    mx=args[2]
    mousex=mx/float(showsz)
    mousey=my/float(showsz * 2)
    

cv2.namedWindow('show3d')
cv2.moveWindow('show3d',0,0)
cv2.setMouseCallback('show3d',onmouse)

dll=np.ctypeslib.load_library('render','.')

def showpoints(img, depth):
    global mousex,mousey,changed
    global pitch,yaw,x,y,z
    show=np.zeros((showsz,showsz * 2,3),dtype='uint8')
    
    def render(img, depth, pose):
        dll.render(ct.c_int(img.shape[0]),
                   ct.c_int(img.shape[1]),
                   img.ctypes.data_as(ct.c_void_p),
                   depth.ctypes.data_as(ct.c_void_p),
                   pose.ctypes.data_as(ct.c_void_p),
                   show.ctypes.data_as(ct.c_void_p)
                  )
        
    
    while True:
        
        if changed:
            render(img, depth, np.array([x,y,z,pitch,yaw]).astype(np.float32))
            changed = False
                
        cv2.putText(show,'pitch %.2f yaw %.2f x %.2f y %.2f z %.2f'%(pitch, yaw, x, y, z),(15,showsz-15),0,0.5,cv2.cv.CV_RGB(255,0,0))
        
        cv2.imshow('show3d',show)
        cmd=cv2.waitKey(10)%256
    
        if cmd==ord('q'):
            break
            
        elif cmd == ord('w'):
            y += 0.01
            changed = True
        elif cmd == ord('s'):
            y -= 0.01
            changed = True
        elif cmd == ord('a'):
            x -= 0.01
            changed = True
        elif cmd == ord('d'):
            x += 0.01    
            changed = True
        elif cmd == ord('r'):
            pitch,yaw,x,y,z = 0,0,0,0,0
            changed = True
            
    

if __name__=='__main__':
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug'  , action='store_true', help='debug mode')
    parser.add_argument('--dataroot'  , required = True, help='dataset path')
    parser.add_argument('--idx'  , type = int, default = 0, help='index of data')
    opt = parser.parse_args()
    d = ViewDataSet3D(root=opt.dataroot, transform = np.array, mist_transform = np.array, seqlen = 2)
    idx = opt.idx
    source = d[idx][0][0]
    source_depth = d[idx][2][0]
    
    #print(source_depth)
    print(source.shape, source_depth.shape)
    
    showpoints(source, source_depth)
    
    
    