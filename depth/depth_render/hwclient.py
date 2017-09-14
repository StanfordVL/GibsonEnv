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

context = zmq.Context()

img_path  = "/home/jerry/Pictures/hwclient"

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
    data = data[:][::-1][:]
    #img = Image.fromarray(data[0])
    #img.save(img_path + str(request) + ".tiff")
    scipy.misc.imsave(img_path + str(request) + ".tiff", data)
    print("Received reply %s [ %s ]" % (request, data))