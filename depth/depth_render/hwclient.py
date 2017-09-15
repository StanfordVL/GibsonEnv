#
#   Hello World client in Python
#   Connects REQ socket to tcp://localhost:5555
#   Sends "Hello" to server, expects "World" back
#

import zmq
import numpy as np
import PIL
from PIL import Image
import scipy.misc
import os
from cube2equi import find_corresponding_pixel

from transfer import transfer2


context = zmq.Context()

img_path  = "./"

#  Socket to talk to server
print("Connecting to hello world server...")
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

#  Do 10 requests, waiting each time for a response
for request in range(6):
    print("Sending request %s ..." % request)
    socket.send(b"Hello")

    #  Get the reply.
    message = socket.recv()
    data = np.array(np.frombuffer(message, dtype=np.uint16)).reshape((512, 512, 3))
    print(data.shape)

    #data = np.log(data / 256 * (256/np.log(256))).astype(np.uint8) 
    #data = (data % 256).astype(np.uint8) 
    data = (data / 256.0).astype(np.uint8) * 12
    print(np.max(data), np.min(data))

    # todo: debug
    data = data[:][::-1][:]
    #img = Image.fromarray(data[0])
    #img.save(img_path + str(request) + ".tiff")
    scipy.misc.imsave(img_path + str(request) + ".tiff", data)
    print("Received reply %s [ %s ]" % (request, data))


def transfer(in_img, coords, h, w):
    out_img = np.zeros((h,w,3)).astype(np.uint8)
    
    for ycoord in range(0, h):
        for xcoord in range(0, w):
            ind, corrx, corry = coords[ycoord, xcoord, :]   
            out_img[ycoord, xcoord, :] =  in_img[ind, corrx, corry, :]
    return out_img
    
def convert_img():
    #inimg = Image.open(infile)

    inimg = InImg()
    
    in_imgs =  np.array([np.array(Image.open("0.tiff")),
    np.array(Image.open("1.tiff")),
    np.array(Image.open("2.tiff")),
    np.array(Image.open("3.tiff")),
    np.array(Image.open("4.tiff")),
    np.array(Image.open("5.tiff"))]).astype(np.uint8)
    
    print(in_imgs.shape)
    
    wo, ho = inimg.grid * 4, inimg.grid * 3

    # Calculate height and width of output image, and size of each square face
    h = wo/3
    w = 2*h
    n = ho/3

    # Create new image with width w, and height h
    outimg = np.zeros((h,w,3)).astype(np.uint8)

    if not os.path.isfile('coord.npy'):
        coords = np.zeros((h,w,3)).astype(np.int32)
        
        for ycoord in range(0, h):
            for xcoord in range(0, w):
                corrx, corry = find_corresponding_pixel(xcoord, ycoord, w, h, n)
                coords[ycoord, xcoord, :] = inimg.getpixel((corrx, corry))
        
        np.save('coord.npy', coords)
    else:
        coords = np.load('coord.npy')
    
    
    # For each pixel in output image find colour value from input image
    outimg = transfer2(in_imgs, coords, h, w)
    
    

    outimg = Image.fromarray(outimg)
    outimg.save('output', 'PNG')


class InImg(object):
    def __init__(self):
        self.grid = 512

    def getpixel(self, key):
        corrx, corry = key[0], key[1]
        
        indx = int(corrx / self.grid)
        indy = int(corry / self.grid)

        remx = int(corrx % self.grid)
        remy = int(corry % self.grid)

        if (indy == 0):
            return (0, remx, remy)
        elif (indy == 2):
            return (5, remx, remy)
        else:
            return (indx + 1, remx, remy)

convert_img()
