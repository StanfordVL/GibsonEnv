from __future__ import print_function
import argparse
import os
import csv
from shutil import copyfile


def restore_all_identities(model_dirs):
	for model in model_dirs:
		sweep_origin  = os.path.join(".", model, "sweep_locations_origin.csv")
		sweep_restore = os.path.join(".", model, "sweep_locations.csv")
		sweep_modeldata = os.path.join(".", model, "modeldata", "sweep_locations.csv")

		if os.path.isfile(sweep_restore):
			os.remove(sweep_restore)
		if os.path.isfile(sweep_origin):
			os.rename(sweep_origin, sweep_restore)

		copyfile(sweep_restore, sweep_modeldata)


def create_identities(model_dirs):
	for model in model_dirs:
		sweep_origin  = os.path.join(".", model, "sweep_locations_origin.csv")
		sweep_default = os.path.join(".", model, "sweep_locations.csv")
		sweep_uuid    = os.path.join(".", model, "camera_uuids.csv")

		sweep_modeldata = os.path.join(".", model, "modeldata", "sweep_locations.csv")
		print(sweep_origin)
		if os.path.isfile(sweep_default):
			os.rename(sweep_default, sweep_origin)
			write_to_identity(sweep_default, sweep_uuid, sweep_origin)
		copyfile(sweep_default, sweep_modeldata)


def write_to_identity(dest_path, uuid_path, pose_path):
	dest_file   = open(dest_path, 'w')
	uuid_file   = open(uuid_path, 'r')
	pose_file   = open(pose_path, 'r')
	uuid_reader = csv.reader(uuid_file, delimiter=",")
	pose_reader = csv.reader(pose_file, delimiter=",")
	pose_writer = csv.writer(dest_file, delimiter=",")

	uuid_line = uuid_reader.next()
	while uuid_line:
		pose_line = pose_reader.next()
		uuid = uuid_line[0]
		pose_position = pose_line[1:4]
		pose_identity = [uuid] + pose_position + [0.44443511962890625, 0.3106224536895752, -0.7182869911193848, 0.4359528422355652] + [0, 0]
		pose_writer.writerow(pose_identity)

		try:
			uuid_line = uuid_reader.next()
		except:
			break

	dest_file.close()
	uuid_file.close()
	pose_file.close()



if __name__ == "__main__":
	all_models = []
	for model in os.listdir('.'):
		if os.path.isfile(os.path.join(".", model, "sweep_locations.csv")) or os.path.isfile(os.path.join(".", model, "sweep_locations_origin.csv")):
			all_models.append(model)

	parser = argparse.ArgumentParser()
	parser.add_argument('--restore', type=bool, default=False, help="Delete created identity file. Default false, set true to clean up")
	opt = parser.parse_args()

	if opt.restore:
		restore_all_identities(all_models)
	else:
		create_identities(all_models)
