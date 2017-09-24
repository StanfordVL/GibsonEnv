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
##			npm install optimist
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

def model_finished(model_path, model_id):
	#check pano/points, pano/rgb, pano/mist, pano/normal
	#check file counts
	#check point.json
	return False

def model_process(model_root, model_id):
	pano_dir = os.path.join(model_root, model_id, "pano")
	dt = DataTasks(".", model_root, model_id)
	dt.generate_points(2, 1, 1)
	dt.create_obj_file()
	dt.create_rgb_images()
	dt.create_mist_images()
	dt.create_normal_images()
	dt.move_point_folder()
	return

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--dataroot", type = str,default = ".", help = "root of model folders")

	## Assert blender version
	## Assert node js installed

	opt = parser.parse_args()
	models = []

	for root_file in os.listdir(opt.dataroot):
		if (os.path.isdir(root_file) and "__" not in root_file and "decode" not in root_file and "node" not in root_file):
			models.append(root_file)
	
	for model in models:
		print("Processing model: %s" % model)
		model_path = os.path.join(os.getcwd(), model)
		if model_finished(model_path, model):
			print("\tModel %s finished" %model)
		else:
			model_process(opt.dataroot, model)



