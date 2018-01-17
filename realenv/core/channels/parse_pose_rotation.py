import json
import os

pose_path = '/home/jerry/Desktop/Data/1CzjpjNF8qk/points'

pose_output = '/home/jerry/Desktop/view3d/realenv/depth/posefile'

fout = open(pose_output, 'w')

all_paths = []
for posefile in os.listdir(pose_path):
    all_paths.append(posefile)

all_paths = sorted(all_paths)

for posefile in all_paths:
    with open(os.path.join(pose_path, posefile)) as pf:
        js = json.load(pf)
        cam_loc = js['camera_location']
        cam_rot = js['camera_rotation_final']
        newname = posefile[:posefile.index('.')]
        fout.write("%f %f %f %f %f %f %s\n" %(cam_loc[0], cam_loc[1], cam_loc[2], cam_rot[0], cam_rot[1], cam_rot[2], newname))

fout.close()
