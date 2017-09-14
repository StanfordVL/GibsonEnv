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

from cube2equi import find_corresponding_pixel


context = zmq.Context()

img_path  = "./"

#  Socket to talk to server
print("Connecting to hello world server...")
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

#  Do 10 requests, waiting each time for a response
for request in range(10):
    print("Sending request %s ..." % request)
    socket.send(b"Hello")

    #  Get the reply.
    message = socket.recv()
    data = np.array(np.frombuffer(message, dtype=np.uint16)).reshape((512, 512, 3))
    print(data.shape)

    #data = np.log(data / 256 * (256/np.log(256))).astype(np.uint8) 
    data = (data % 256).astype(np.uint8) 


    # todo: debug
    data = data[:][::-1][:]
    #img = Image.fromarray(data[0])
    #img.save(img_path + str(request) + ".tiff")
    scipy.misc.imsave(img_path + str(request) + ".tiff", data)
    print("Received reply %s [ %s ]" % (request, data))



def convert_img():
    #inimg = Image.open(infile)

    inimg = InImg()
    wo, ho = inimg.grid * 4, inimg.grid * 3

    # Calculate height and width of output image, and size of each square face
    h = wo/3
    w = 2*h
    n = ho/3

    # Create new image with width w, and height h
    outimg = Image.new('RGB', (w, h))

    # For each pixel in output image find colour value from input image
    for ycoord in range(0, h):
        for xcoord in range(0, w):
            corrx, corry = find_corresponding_pixel(xcoord, ycoord, w, h, n)

            outimg.putpixel((xcoord, ycoord), inimg.getpixel((corrx, corry)))
        # Print progress percentage
        print str(round((float(ycoord)/float(h))*100, 2)) + '%'


    outimg.save('output', 'PNG')


class InImg(object):
    def __init__(self):
        self.grid = 512
        self.img0 = Image.open("0.tiff")
        self.img1 = Image.open("1.tiff")
        self.img2 = Image.open("2.tiff")
        self.img3 = Image.open("3.tiff")
        self.img4 = Image.open("4.tiff")
        self.img5 = Image.open("5.tiff")

    def getpixel(self, key):
        corrx, corry = key[0], key[1]
        imgs = [[self.img0], [self.img1, self.img2, self.img3, self.img4], [self.img5]]
        
        indx = int(corrx / self.grid)
        indy = int(corry / self.grid)

        remx = int(corrx % self.grid)
        remy = int(corry % self.grid)

        if (indy == 0  or indy == 2):
            return imgs[indy][0].getpixel((remx, remy))
        else:
            return imgs[1][indx].getpixel((remx, remy))

convert_img()