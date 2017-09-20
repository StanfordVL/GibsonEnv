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
from matplotlib import cm as CM
import matplotlib.pyplot as plt
import argparse
import json
from shutil import copy2
#import transform3d

from transfer import transfer2



img_path  = "./"
blender_path = "./point_29b9558f6a244ca493d2bf52709684e2_view_equirectangular_domain_mist.png"



def transfer(in_img, coords, h, w):
    ## For computing cosine (float)
    #if(in_img.dtype == np.dtype('float64')):
    #    in_img = (in_img * 256).astype(np.uint16)
    
    out_img = np.zeros((h,w,3)).astype(in_img.dtype)

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
        coords = np.zeros((h,w,1)).astype(np.int32)

        for ycoord in range(0, h):
            for xcoord in range(0, w):
                corrx, corry = find_corresponding_pixel(xcoord, ycoord, w, h, n)
                coords[ycoord, xcoord, :] = inimg.getpixel((corrx, corry))

        np.save('coord.npy', coords)
    else:
        coords = np.load('coord.npy')


    # For each pixel in output image find colour value from input image
    # todo: for some reason the image is flipped 180 degrees
    #outimg = transfer2(in_imgs, coords, h, w)[::-1, ::-1, ::]
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
    outimg = np.zeros((h,w,1)) #.astype(np.uint8)

    in_imgs = None
    #print("converting images", len(img_array))

    print("Passed in image array", len(img_array), np.max(img_array[0]), np.mean(img_array[0]), np.min(img_array[0]))
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
    #print(outimg.shape)

    # todo: for some reason the image is flipped 180 degrees
    outimg = transfer2(in_imgs, coords, h, w)[:, ::-1, :]

    return outimg




class InImg(object):
    def __init__(self):
        self.grid = 768

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

def plot_2_histogram(opengl_arr, blender_arr):
    blender_arr_tmp = blender_arr.reshape((-1, 1))
    ## TODO: blender array has 65535 elements
    blender_arr_tmp = blender_arr_tmp[blender_arr_tmp < 65535]

    opengl_arr_tmp = opengl_arr[:, :, 0].reshape((-1, 1))
    ## TODO: opengl array has 0 elements
    opengl_arr_tmp = opengl_arr_tmp[opengl_arr_tmp > 0]

    plt.subplot(2, 1, 1)
    n, bins, patches = plt.hist(opengl_arr_tmp, 50, normed=1, label='opengl', alpha=0.75)
    plt.legend(loc='upper right')
    plt.subplot(2, 1, 2)
    n, bins, patches = plt.hist(blender_arr_tmp, 50, normed=1, label='blender', alpha=0.75)
    plt.legend(loc='upper right')
    plt.show()


def read_pose_from_csv(root, model_id, idx):
    posefile = os.path.join(root, model_id, 'sweep_locations.csv')
    line_i = 0
    l_list  = []
    l_str   = ""
    with open(posefile) as f:
        for line in f:
            l_list = line.strip().split(',')
            l_str  = ' '.join(l_list[1:])
            if line_i == idx:
                break
            line_i += 1
    return l_list, l_str

def read_pose_from_json(root, model_id, idx):
    posedir = os.path.join(root, model_id, 'pano', 'points')
    pose_i  = os.listdir(posedir)[idx]
    item    = os.path.join(root, model_id, 'pano', 'points', pose_i)
    #print(pose_i)
    
    f = open(item)
    pose_dict = json.load(f)
    p = np.concatenate(np.array(pose_dict[0][u'camera_rt_matrix'] + [[0,0,0,1]])).astype(np.float32).reshape((4,4))
    
    trans = -np.dot(p[:3, :3].T, p[:3, -1])
    #rotation = np.array([[0,0,-1],[0,-1,0],[1,0,0]])
    #rot = np.dot(np.dot(rotation, p[:3, :3]), rotation)
    
    rot = np.dot(np.array([[-1,0,0],[0,-1,0],[0,0,1]]),  np.linalg.inv(p[:3, :3]))
    
    p2 = np.eye(4)
    
    p2[:3, :3] = rot
    p2[:3, -1] = trans
    
    #print(p2)
    
    return mat_to_str(p2), pose_i[:pose_i.index(".")]

def mat_to_str(matrix):
    s = ""
    for row in range(4):
        for col in range(4): 
            s = s + " " + str(matrix[row][col])
    return s.strip()

def find_blender_output(root, model_id, name):
    mistdir = os.path.join(root, model_id, 'pano', 'mist')
    mist_pano = ""
    #print(name)
    for f in os.listdir(mistdir):
        if name in f:
            mist_pano = f
    mist_path = os.path.join(mistdir,  mist_pano)
    mist_img_arr = np.array(Image.open(mist_path), dtype=np.uint16)
    #print('max blender', np.max(mist_img_arr))
    return mist_img_arr

## Return 1024 * 2048
def find_cosine_output(root, model_id, name):
    cosine_dir = "."
    cosine_pano = ""
    #print(name)
    for f in os.listdir(cosine_dir):
        if name in f and "opengl" in f:
            cosine_pano = f
    cosine_path = os.path.join(cosine_dir,  cosine_pano)
    cosine_img_arr = np.array(Image.open(cosine_path), dtype = float) / 255
    print('max cosine', np.max(cosine_img_arr))
    return cosine_img_arr[:, :, 0]


def compare_err_neighbors(opengl_arr, blender_arr, width, height):
    gap = 5
    width_small = width - 2 * gap
    height_small = height - 2 * gap
    error_map_small = np.ones((height_small, width_small)) * 50
    error_str = ""
    for dx in range(2 * gap):
        for dy in range(2 * gap):
            diff = np.abs(opengl_arr[dy:dy+height_small, dx:dx + width_small] - blender_arr[gap:gap + height_small, gap:gap + width_small])
            error_map_small = np.minimum(error_map_small, diff)
            error_str = error_str + " " + str(np.sum(error_map_small))
    error_map = np.zeros((height, width))
    print("neighbor error", error_str)
    error_map[gap:gap +height_small, gap:gap + width_small] = error_map_small
    return error_map

def plot_heat_map(error_map):
    error_map_heat = np.copy(error_map)
    error_map_heat[np.abs(error_map_heat) > 10] = 0

    plt.imshow(error_map_heat, extent=(0, 2048, 0, 1024),
           interpolation='nearest', cmap=CM.PiYG)
    cb = plt.colorbar()
    cb.set_label('opengl - blender')
    plt.show()

def save_concatenated_images(im_size, gap, all_opengl_images, filename, num_samples, num_diffs):
    gap = 5
    new_im = Image.new('RGB', (num_samples * im_size[0] + (num_samples - 1) * gap, num_diffs * im_size[1] + (num_diffs - 1) * gap))

    x_offset = 0
    y_offset = 0

    for idx_i in range(num_samples):
        x_offset = idx_i * (im_size[0] + gap)
        ims_i = map(Image.open, all_opengl_images[idx_i])
        for diff_i in range(num_diffs):
            y_offset = diff_i * (im_size[1] + gap)
            new_im.paste(ims_i[diff_i], (x_offset, y_offset))

    new_im.save(filename)


def plot_error_vs_alpha(all_cosine_values, all_error_values):
    print(np.max(all_cosine_values), np.min(all_cosine_values), len(all_error_values))
    plt.plot(180 * np.arccos(all_cosine_values) / np.pi,all_error_values, 'ro',  markersize=0.1)
    plt.ylabel("Error (m)")
    plt.xlabel("Alpha (degree)")
    plt.show()

def plot_error_vs_mist(all_mist_values, all_error_values):
    print(np.max(all_mist_values), np.min(all_mist_values), len(all_error_values))
    plt.plot(all_mist_values,all_error_values, 'ro',  markersize=0.1)
    plt.ylabel("Error (m)")
    plt.xlabel("Mist (m)")
    plt.show()

def plot_error_histogram(error_values, cap):
    plt.subplot(2, 1, 1)
    error_abs = np.abs(error_values)
    print("(Plotting error values)", np.max(error_abs), np.std(error_abs), np.mean(error_abs))
    n, bins, patches = plt.hist(error_abs[error_abs < cap], 80, normed=1, facecolor='red', alpha=0.75)
    # add a 'best fit' line
    print("Below", 0.02, float(np.sum(error_abs < 0.02)) / len(error_abs))
    plt.ylabel("Overall Error")
    plt.xlabel("Error (m)")
    #plt.show()
    plt.subplot(2, 1, 2)
    error_abs = np.abs(error_values)
    error_abs = error_abs[error_abs > 0.02]
    print("(Plotting error values)", np.max(error_abs), np.std(error_abs), np.mean(error_abs))
    print("Below", 0.08, float(np.sum(error_abs < 0.08)) / len(error_abs))
    n, bins, patches = plt.hist(error_abs[error_abs < cap * 3], 200, normed=1, facecolor='red', alpha=0.75)
    # add a 'best fit' line
    plt.ylabel("Error above 2cm")
    plt.xlabel("Error (m)")
    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot'  , required = True, help='dataset path')
    parser.add_argument('--idx'  , type = int, default = 0, help='index of data')
    parser.add_argument('--model'  , type = str, default = '', help='path of model')

    opt = parser.parse_args()

    root     = opt.dataroot
    model_id = opt.model

    pose_idx = opt.idx

    all_opengl_images = []

    num_samples = 5
    diff_caps = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12]

    all_cosine_values = []
    all_error_values = []
    all_mist_values = []


    for pose_idx in range(6, 6 + num_samples):
        #pose_list, pose_str = read_pose_from_csv(root, model_id, 0)
        #print('parsed pose str', pose_str)

        mat_str, pose_i = read_pose_from_json(root, model_id, pose_idx)
        img_array = []


        #  Socket to talk to server
        context = zmq.Context()
        print("Connecting to hello world server...", mat_str)
        socket = context.socket(zmq.REQ)
        socket.connect("tcp://localhost:5555")
        print("Sending request ..." )
        socket.send(mat_str)

        #  Get the reply.
        message = socket.recv()
        #data = np.array(np.frombuffer(message, dtype=np.uint16)).reshape((6, 768, 768, 1))

        ## 32bit float seems precise enough for receiver
        data = np.array(np.frombuffer(message, dtype=np.float32)).reshape((6, 768, 768, 1))
        
        data = data[:, ::-1,::-1,:]
        
        for i in range(6):
            img_array.append(data[i])

        img_array2 = [img_array[0], img_array[3], img_array[2], img_array[1], img_array[4], img_array[5]]
        print("(Incoming array) opengl max", np.max(np.array(img_array2)), "opengl mean", np.mean(np.array(img_array2)))
        
        #opengl_arr = convert_array(np.array(255 * np.array(img_array2), dtype = np.uint16))
        #opengl_arr = convert_array(np.array(img_array2, dtype=np.uint16))
        opengl_arr = convert_array(np.array(img_array2))
        
        #plot_histogram(opengl_arr)
        blender_arr = find_blender_output(root, model_id, pose_i)

        #plot_2_histogram(opengl_arr, blender_arr)

        blender_arr[blender_arr == 65535] = 0 
        #print("(Incoming array) opengl max", np.max(opengl_arr), "opengl mean", np.mean(opengl_arr))
        opengl_arr_float = np.array(opengl_arr[:, :, 0]).astype(float)
        blender_arr_float = np.array(blender_arr).astype(float) * 128 / 65535
        #print("(Incoming array) blender max", np.max(blender_arr_float), "blender mean", np.mean(blender_arr_float))
        #error_map = np.array(opengl_arr_float - blender_arr_float)
        ## Compare error with neighboring pixels
        error_map = compare_err_neighbors(opengl_arr_float, blender_arr_float, 2048, 1024)

        cosine_arr = find_cosine_output(root, model_id, pose_i)
        #print(error_map.shape, cosine_arr.shape)
        print("(Incoming array) cosine max", np.max(cosine_arr), "cosine mean", np.mean(cosine_arr))
        

        #if (len(all_error_values) == 0):
        all_error_values  = all_error_values  + error_map.reshape((1, -1))[0].tolist()
        all_cosine_values = all_cosine_values + cosine_arr.reshape((1, -1))[0].tolist()
        all_mist_values  = all_mist_values  + opengl_arr_float.reshape((1, -1))[0].tolist()
        
        

        #print("Error map max min", np.max(error_map), np.min(error_map))
        #print("(Incoming array) Blender max min", np.max(blender_arr), np.min(blender_arr))
        #print("(Incoming array) Opengl max min", np.max(opengl_arr[:, :, 0]), np.min(opengl_arr[:, :, 0]))
        #error_map = (error_map / 65535) * 12800
        print("(Error array) max", np.max(np.abs(error_map)), "std", np.std(error_map), "mean", np.mean(error_map))

        #plot_heat_map(error_map)
        

        curr_opengl_images = []


        for diff_cap in diff_caps:
            #opengl_arr_save = (opengl_arr / 10).astype(np.uint8)
            #opengl_arr_save = (opengl_arr * 255).astype(np.uint8)
            #print("(Before Saving image array) opengl max", opengl_arr.dtype, np.max(opengl_arr), "opengl mean", np.mean(opengl_arr))
            
            opengl_arr_save = (opengl_arr * 100).astype(np.uint8)
            print("(Saving image array) opengl max", np.max(opengl_arr_save), "opengl mean", np.mean(opengl_arr_save[:, :, 0]))
            opengl_arr_save[np.abs(error_map) > diff_cap] = 255
            outimg = Image.fromarray(opengl_arr_save)
            opengl_name = 'output_opengl_' + str(pose_i) + '_' + str(diff_cap) + '.png'
            outimg.save(opengl_name, 'PNG')
            
            target_opengl = os.path.join('/home', 'jerry', 'Dropbox', opengl_name)
            #print(target_opengl)
            #copy2(opengl_name, target_opengl)

            curr_opengl_images.append(opengl_name)

            blender_arr_save = (blender_arr * 100).astype(np.uint8)
            #blender_arr_save = (blender_arr * 256).astype(np.uint8)
            #print("mean opengl cosine", np.mean(opengl_arr_save), "max blender cosine", np.max(blender_arr_save))
            blender_arr_save[np.abs(error_map) > diff_cap] = 255
            blender_img = Image.fromarray(blender_arr_save)
            blender_name = 'output_blender_' + str(pose_i) + '_' + str(diff_cap) + '.png'
            blender_img.save(blender_name, 'PNG')
            target_blender = os.path.join('/home', 'jerry', 'Dropbox', blender_name)
            #copy2(blender_name, target_blender)
        
        all_opengl_images.append(curr_opengl_images)

    #plot_error_vs_alpha(all_cosine_values, all_error_values)
    #plot_error_vs_mist(all_mist_values, all_error_values)
    plot_error_histogram(all_error_values, 0.05)
    save_concatenated_images([2048, 1024], 5, all_opengl_images, "cosine.png", num_samples, len(diff_caps))