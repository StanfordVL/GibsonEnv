from subprocess import call
import shutil
import os

class DataTasks:
	code_dir = ""
	blender_py = "blender -b -noaudio --enable-autoexec --python"

	def __init__(self, code_dir, model_root, model_id):
		self.code_dir   = os.path.abspath(code_dir)
		self.model_root = os.path.abspath(model_root)
		self.model_path = os.path.join(model_root, model_id)
		self.model_id   = model_id

	def generate_points(self, num_needed, min_view, max_view):
		os.chdir(self.model_path)
		code_point   = os.path.abspath(os.path.join(self.code_dir, "generate_points.py"))
		command_str  = "%s %s --" % (self.blender_py, code_point)
		command_list = command_str.split()
		call(command_list + ["--NUM_POINTS_NEEDED", str(num_needed), "--MIN_VIEWS", str(min_view), "--MAX_VIEWS", str(max_view)])
		os.chdir(self.code_dir)
		print("Finished: %s point generation" % self.model_id)
		return True

	def create_obj_file(self):
		os.chdir(self.model_path)
		code_obj     = os.path.abspath(os.path.join(self.code_dir, 'decode', 'decode.js'))
		command_str  = "node %s --rootdir=%s --model=%s" % (code_obj, self.model_root, self.model_id)
		command_list = command_str.split()
		call(command_list)
		os.chdir(self.code_dir)
		print("Finished: %s object creation" % self.model_id)
		return True

	def create_rgb_images(self):
		os.chdir(self.model_path)
		code_rgb     = os.path.abspath(os.path.join(self.code_dir, "create_rgb_images.py"))
		command_str  = "%s %s --" % (self.blender_py, code_rgb)
		command_list = command_str.split()
		call(command_list)
		os.chdir(self.code_dir)
		print("Finished: %s create rgb images" % self.model_id)
		return True

	def create_mist_images(self):
		os.chdir(self.model_path)
		code_mist    = os.path.abspath(os.path.join(self.code_dir, "create_mist_images.py"))
		command_str  = "%s %s --" % (self.blender_py, code_mist)
		command_list = command_str.split()
		call(command_list)
		os.chdir(self.code_dir)
		print("Finished: %s create mist images" % self.model_id)
		return True

	def create_normal_images(self):
		os.chdir(self.model_path)
		code_normal  = os.path.abspath(os.path.join(self.code_dir, "create_normal_images.py"))
		command_str  = "%s %s --" % (self.blender_py, code_normal)
		command_list = command_str.split()
		call(command_list)
		os.chdir(self.code_dir)
		print("Finished: %s create normal images" % self.model_id)
		return True

	def move_point_folder(self):
		old_folder = os.path.join(self.model_path, "points")
		new_folder = os.path.join(self.model_path, "pano", "points")
		if os.isdir(new_folder):
			shutil.rmtree(new_folder)
		shutil.move(old_folder, new_folder)