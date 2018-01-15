from PIL import Image
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

#opengl_path  = "/home/jerry/Pictures/point_0_view_2_domain_fixatedpose.png"

opengl_path  = "/home/jerry/Pictures/point_0_view_2_domain_fixatedpose_mist.png"
#blender_path = "/home/jerry/Desktop/Data/1CzjpjNF8qk/depth/point_0_view_2_domain_depth.png"
blender_path = "/home/jerry/Desktop/Data/1CzjpjNF8qk/mist/point_0_view_2_domain_mist.png"

outline_path  = "/home/jerry/Pictures/point_0_view_2_domain_outline.png"

opengl_viz  = "/home/jerry/Pictures/point_0_view_2_domain_viz.png"

opengl_img  = Image.open(opengl_path)
blender_img = Image.open(blender_path)


opengl_arr  = np.asarray(opengl_img)
blender_arr = np.asarray(blender_img)

## Opengl:  opengl_arr[:, :, 0]
## Blender: blender_arr

#opengl_arr  = opengl_arr[:, :, 0].reshape((1, -1))[0]  ## fpa version


# quick fix for visualization
## TODO: cpp png saving upside down
#opengl_arr 
opengl_arr = opengl_arr[::-1][:]
#print(opengl_arr.shape, blender_arr.shape)


outline_arr = opengl_arr.copy()

for row in range(opengl_arr.shape[0]):
    for col in range(opengl_arr.shape[1]):
        if np.abs(opengl_arr[row][col] - blender_arr[row][col]) > 3:
            print(opengl_arr[row][col], blender_arr[row][col])
            outline_arr[row][col] = 65535


im = Image.new('L', (512, 512))
im.putdata(outline_arr.flatten().tolist())
im.save(outline_path)

viz_arr = opengl_arr.copy()
viz_arr = viz_arr * 128.0 / 65535
viz_arr = np.power(viz_arr, 5)
print(viz_arr)


im = Image.new('L', (512, 512))
im.putdata(viz_arr.flatten().tolist())
im.save(opengl_viz)

opengl_arr  = opengl_arr.reshape((1, -1))[0]  ## png version
blender_arr = blender_arr.reshape((1, -1))[0]
print("before clamping, max blender: ", max(blender_arr)) 
print(opengl_arr)
print(np.min(opengl_arr), np.max(opengl_arr), len(opengl_arr))
print(np.min(blender_arr),np.max(blender_arr), len(blender_arr))

diff_count = np.sum((opengl_arr != blender_arr))
diff_sqsum = np.sum(np.square(opengl_arr - blender_arr))

total_count = len(opengl_arr)
total_sqsum = np.sum(np.square(opengl_arr))
print('How many different', diff_count, float(diff_count) / total_count)
print('Total square diff', diff_sqsum, float(diff_sqsum) / total_sqsum)

blender_arr = blender_arr[blender_arr < 65535]

plt.subplot(2, 1, 1)
n, bins, patches = plt.hist(opengl_arr, 50, normed=1, label='opengl', alpha=0.75)
plt.legend(loc='upper right')
plt.subplot(2, 1, 2)
n, bins, patches = plt.hist(blender_arr, 50, normed=1, label='blender', alpha=0.75)
plt.legend(loc='upper right')
plt.show()

