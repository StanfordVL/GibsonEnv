import numpy as np

def transfer2(unsigned char [:,:,:,:] in_img, int [:,:,:]coords, int h, int w):
    
    cdef unsigned char [:,:,:] out_img = np.zeros((h,w,3)).astype(np.uint8)
    
    cdef int xcoord, ycoord
    cdef int ind, corrx, corry
    
    cdef int c
    
    for ycoord in range(0, h):
        for xcoord in range(0, w):
            ind = coords[ycoord, xcoord, 0]   
            corrx = coords[ycoord, xcoord, 1]
            corry = coords[ycoord, xcoord, 2]  
            for c in range(3):
                out_img[ycoord, xcoord, c] =  in_img[ind, corry, corrx, c]
    return np.array(out_img)