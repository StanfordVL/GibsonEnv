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
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from transfer import transfer2



img_path  = "./"
blender_path = "./point_29b9558f6a244ca493d2bf52709684e2_view_equirectangular_domain_mist.png"



def transfer(in_img, coords, h, w):
    out_img = np.zeros((h,w,3)).astype(np.uint16)
    
    for ycoord in range(0, h):
        for xcoord in range(0, w):
            ind, corrx, corry = coords[ycoord, xcoord, :]   
            out_img[ycoord, xcoord, :] =  in_img[ind, corrx, corry, :]
    return out_img
    
def convert_img():
    inimg = InImg()
    
    wo, ho = inimg.grid * 4, inimg.grid * 3

    # Calculate height and width of output image, and size of each square face
    h = wo/3
    w = 2*h
    n = ho/3

    # Create new image with width w, and height h
    outimg = np.zeros((h,w,3)) #.astype(np.uint8)

    in_imgs = np.array([np.array(Image.open("0.tiff")),
        np.array(Image.open("1.tiff")),
        np.array(Image.open("2.tiff")),
        np.array(Image.open("3.tiff")),
        np.array(Image.open("4.tiff")),
        np.array(Image.open("5.tiff"))]).astype(np.uint8)
        
    print("Received image array", len(in_imgs))

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
    print(outimg.shape)

    # todo: for some reason the image is flipped 180 degrees
    outimg = transfer2(in_imgs, coords, h, w)[::-1, ::-1, ::]

    outimg = Image.fromarray(outimg)
    outimg.save('output', 'PNG')

    return outimg


def convert_array(img_array):
    inimg = InImg()
    
    wo, ho = inimg.grid * 4, inimg.grid * 3

    # Calculate height and width of output image, and size of each square face
    h = wo/3
    w = 2*h
    n = ho/3

    # Create new image with width w, and height h
    outimg = np.zeros((h,w,3)) #.astype(np.uint8)

    in_imgs = None 
    print("converting images", len(img_array))

    print("Passed in image array", len(img_array), np.max(img_array[0]))
    in_imgs = img_array

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
    print(outimg.shape)

    # todo: for some reason the image is flipped 180 degrees
    outimg = transfer(in_imgs, coords, h, w)[::-1, ::-1, ::]
    
    return outimg




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



def plot_histogram(opengl_arr):
    blender_img = Image.open(blender_path)
    blender_arr = np.asarray(blender_img)

    print("blender", blender_arr.shape, "opengl", opengl_arr.shape)


    blender_arr = blender_arr.reshape((-1, 1))
    ## TODO: blender array has 65535 elements
    blender_arr = blender_arr[blender_arr < 65535]

    opengl_arr = opengl_arr[:, :, 0].reshape((-1, 1))
    ## TODO: opengl array has 0 elements
    opengl_arr = opengl_arr[opengl_arr > 0]

    print(blender_arr.shape, opengl_arr.shape, np.mean(blender_arr), np.min(blender_arr), np.max(blender_arr))

    plt.subplot(2, 1, 1)
    n, bins, patches = plt.hist(opengl_arr, 50, normed=1, label='opengl', alpha=0.75)
    plt.legend(loc='upper right')
    plt.subplot(2, 1, 2)
    n, bins, patches = plt.hist(blender_arr, 50, normed=1, label='blender', alpha=0.75)
    plt.legend(loc='upper right')
    plt.show()



if __name__ == '__main__':
    context = zmq.Context()
    #  Socket to talk to server
    print("Connecting to hello world server...")
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")

    img_array = []

    #  Do 10 requests, waiting each time for a response
    for request in range(10):
        print("Sending request %s ..." % request)
        socket.send(b"Hello")

        #  Get the reply.
        message = socket.recv()
        data = np.array(np.frombuffer(message, dtype=np.uint16)).reshape((512, 512, 3))

        img_array.append(data)

        #data = np.log(data / 256 * (256/np.log(256))).astype(np.uint8) 
        #data = (data % 256).astype(np.uint8) 
        #data = (data / 256.0).astype(np.uint8) * 12
        print(np.max(data), np.min(data))

        # todo: debug
        data = data[:][::-1][:]
        #img = Image.fromarray(data)
        #img.save(img_path + str(request) + ".tiff")
        #scipy.misc.imsave(img_path + str(request) + ".png", data, 'L', bits=16)
        #print("Received reply %s [ %s ]" % (request, data))

    print("max value", np.max(data[0]))
    opengl_arr = convert_array(np.array(img_array))
    #plot_histogram(opengl_arr)
