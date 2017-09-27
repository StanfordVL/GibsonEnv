from subprocess import call, Popen
import shutil
import os

class DataTasks:
	code_dir = ""
	blender_py = "blender -b -noaudio --enable-autoexec --python"

	def __init__(self, model_root, model_dest, model_id):
		self.code_dir   = os.path.dirname(os.path.abspath(__file__))
		self.model_root = model_root
		self.model_dest = model_dest
		self.model_id   = model_id

	def generate_points(self, num_needed, min_view, max_view):
		code_point   = os.path.join(self.code_dir, "generate_points.py")
		command_str  = "%s %s --" % (self.blender_py, code_point)
		print(command_str)
		command_list = command_str.split()
		try:
			proc = Popen(command_list + ["--NUM_POINTS_NEEDED", str(num_needed), "--MIN_VIEWS", str(min_view), "--MAX_VIEWS", str(max_view), "--BASEPATH", self.model_root])
			print("Finished: %s point generation" % self.model_id)
			proc.wait()
		except KeyboardInterrupt:
			proc.kill()
		return True

	def create_obj_file(self):
		code_obj     = os.path.join(self.code_dir, 'decode', 'decode.js')
		command_str  = "node %s --rootdir=%s --model=%s" % (code_obj, self.model_root, self.model_id)
		command_list = command_str.split()
		try:
			proc = Popen(command_list)
			print("Finished: %s object creation" % self.model_id)
			proc.wait()
		except KeyboardInterrupt:
			proc.kill()
		return True

	def create_rgb_images(self):
		code_rgb     = os.path.join(self.code_dir, "create_rgb_images.py")
		command_str  = "%s %s --" % (self.blender_py, code_rgb)
		command_list = command_str.split()
		try:
			proc = Popen(command_list + ["--BASEPATH", self.model_root])
			print("Finished: %s create rgb images" % self.model_id)
			proc.wait()
		except KeyboardInterrupt:
			proc.kill()
		return True

	def create_mist_images(self):
		code_mist    = os.path.join(self.code_dir, "create_mist_images.py")
		command_str  = "%s %s --" % (self.blender_py, code_mist)
		command_list = command_str.split()
		try:
			proc = Popen(command_list + ["--BASEPATH", self.model_root])
			print("Finished: %s create mist images" % self.model_id)
			proc.wait()
		except KeyboardInterrupt:
			proc.kill()
		return True


	def create_normal_images(self):
		code_normal  = os.path.join(self.code_dir, "create_normal_images.py")
		command_str  = "%s %s --" % (self.blender_py, code_normal)
		command_list = command_str.split()
		proc = Popen(command_list + ["--BASEPATH", self.model_root])
		print("Finished: %s create normal images" % self.model_id)
		proc.wait()
		return True


	def move_point_folder(self):
		old_folder = os.path.join(self.model_root, "points")
		new_folder = os.path.join(self.model_root, "pano", "points")
		if os.path.isdir(new_folder):
			shutil.rmtree(new_folder)
		shutil.copytree(old_folder, new_folder)


	def move_model_to_dest(self):
		## Move pano folder, sweep_locations.csv to destination
		pano_folder = os.path.join(self.model_root, "pano")
		new_folder  = os.path.join(self.model_dest, "pano")
		if not os.path.isdir(self.model_dest):
			os.mkdir(self.model_dest)
		if os.path.isdir(new_folder):
			shutil.rmtree(new_folder)
		shutil.copytree(pano_folder, new_folder)

		csv_file = os.path.join(self.model_root, "modeldata", "sweep_locations.csv")
		shutil.copy2(csv_file, self.model_dest)
		
		obj_file = os.path.join(self.model_root, "modeldata", "out_res.obj")
		shutil.copy2(obj_file, self.model_dest)
