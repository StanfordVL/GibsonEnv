#cython: boundscheck=False, wraparound=False, nonecheck=False
import numpy as np

def transfer2(float [:,:,:,:] in_img, int [:,:,:]coords, int h, int w, float [:,:] out_img):
    
    # cdef double [:,:,:] out_img = np.zeros((h,w,3))
    
    cdef int xcoord, ycoord
    cdef int ind, corrx, corry
    cdef int img_h
    cdef int c
    
    img_h = in_img.shape[1]
    for ycoord in range(0, h):
        for xcoord in range(0, w):
            ind = coords[ycoord, xcoord, 0]   
            corrx = coords[ycoord, xcoord, 1]
            corry = coords[ycoord, xcoord, 2]  
            out_img[ycoord, xcoord] =  in_img[ind, img_h-corry, corrx, 0]
    # return np.array(out_img)
