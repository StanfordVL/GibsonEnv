## Data generation entry point for real environment
## Output:
## /dataroot/model_id
## 	  /pano
##	    /points:  json formatted view point data
## 		/mist  :  mist panorama
## 		/rgb   :  rgb panorama
## 		/normal:  surface normal panorama

## How to run
## 		cd /dataroot
## 		source activate xxx (blender virtual env)
## 		python start.py

## Requirements:
## (1) Have blender installed (v78 or v79)
##	   Have python3 environment
##	   Packages:
##			npm install optimist bytebuffer long
## 
## (2) Have the following initial model files
## /dataroot/model_id
##  	modeldata/sweep_locations.csv
##  	modeldata/out_res.obj       (if use obj)
##  	modeldata/out_res.ply 		(if use ply)
## 		modeldata/img/high
##		modeldata/img/low

from __future__ import print_function
from datatasks import DataTasks
import argparse
import os


def model_finished(model_root):
	#check pano/points, pano/rgb, pano/mist, pano/normal
	#check file counts
	#check point.json
	return False

def model_process(model_root, model_dest, model_id):
	pano_dir = os.path.join(model_root, model_id, "pano")
	dt = DataTasks(model_root, model_dest, model_id)
	#dt.generate_points(2, 1, 1)
	dt.create_obj_file()
	#dt.create_rgb_images()
	#dt.create_mist_images()
	#dt.create_normal_images()
	#dt.move_point_folder()
	dt.move_model_to_dest()
	return

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--dataroot", type = str, default = "/cvgl/group/taskonomy/raw/", help = "Root of model folders (absolute).")
	parser.add_argument("--datadest", type = str, default = os.getcwd(), help = "Destination of processed model folders (absolute).")

	## Assert blender version
	## Assert node js installed

	opt = parser.parse_args()
	models = []

	for root_file in os.listdir(opt.dataroot):
		file_path = os.path.join(opt.dataroot, root_file)
		print(os.path.isdir(file_path))
		if (os.path.isdir(file_path) and "__" not in file_path and "decode" not in file_path and "node" not in file_path):
			models.append(root_file)
	
	print("Loaded %d models" % len(models))

	data_root = opt.dataroot
	data_dest = opt.datadest

	for model in models:
		print("Processing model: %s" % model)
		model_root = os.path.join(data_root, model)
		model_dest = os.path.join(data_dest, model)
		if model_finished(model_root):
			print("\tModel %s finished" %model)
		else:
			model_process(model_root, model_dest, model)



