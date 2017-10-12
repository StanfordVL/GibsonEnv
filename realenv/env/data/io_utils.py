"""
  Name: io_utils.py
  Author: Sasha Sax, CVGL
  Modified by: Zhiyang He
  Desc: Contains utilities for saving and loading information

  Usage: for import only
"""

import sys
import os
sys.path.append( os.path.dirname( os.path.realpath(__file__) ) )
from activate_env import add_on_path
sys.path.append(add_on_path)

from load_settings import settings

try:
  import bpy
  from mathutils import Vector, Matrix, Quaternion, Euler
  import utils
  from utils import create_camera, axis_and_positive_to_cube_face, cube_face_idx_to_skybox_img_idx
except:
  if settings.VERBOSITY >= settings.VERBOSITY_LEVELS[ 'WARNING' ]:
    print( "Can't import Blender-dependent libraries in io_utils.py. Proceeding, and assuming this is kosher...")

import ast
import csv
import glob
import json
import logging
import math
# import numpy as np
import os
import time

axis_and_positive_to_skybox_idx = {  
    ( "X", True ): 1,
    ( "X", False ): 3,
    ( "Y", True ): 0,
    ( "Y", False ): 5,
    ( "Z", True ): 2,
    ( "Z", False ): 4
  }

skybox_number_to_axis_and_rotation = { 5: ('X', -math.pi / 2),
                                       0: ('X', math.pi / 2), 
                                       4: ('Y', 0.0), 
                                       3: ('Y', math.pi / 2), 
                                       2: ('Y', math.pi), 
                                       1: ('Y', -math.pi / 2) }

img_format_to_ext = { "png": 'png', "jpeg": "jpg", "jpg": "jpg" }

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel( logging.INFO ) 


def collect_camera_poses_from_csvfile( infile ):
  """
    Reads the camera uuids and locations from the given file

    Returns:
      points: A Dict of the camera locations from uuid -> position, rotation, and quaterion.
              Quaterions are wxyz ordered
  """
  points = {}
  
  with open(infile) as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
      uuid = row[0]
      position = (float(row[1]), float(row[2]), float(row[3]))
      quaternion_wxyz = (float(row[7]), float(row[4]), float(row[5]), float(row[6]))
      if settings.VERBOSITY >= settings.VERBOSITY_LEVELS[ 'DEBUG' ]:
        print( "Camera: {0}, rotation: {1}".format( uuid, quaternion_wxyz ) )
      # quaternion_xyzw = (float(row[4]), float(row[5]), float(row[6]), float(row[7]))
      rotation = convert_quaternion_to_euler(quaternion_wxyz)
      points[uuid] = (position, rotation, quaternion_wxyz)
      points[uuid] = { 'position': position, 'rotation': rotation, 'quaternion': quaternion_wxyz }
  csvfile.close()
  return points


def convert_quaternion_to_euler(quaternion):
    
    blender_quat = Quaternion( quaternion )
    result = blender_quat.to_euler( settings.EULER_ROTATION_ORDER ) 

    # levels the quaternion onto the plane images were taken at
    result.rotate_axis( 'X', math.pi/2 )
    # result[0] = result[0] + (math.pi / 2)

    return result

def create_logger( logger_name ):
    logging.basicConfig()
    logger = logging.getLogger(logger_name)
    logger.setLevel( settings.LOGGING_LEVEL ) 
    return logger

def delete_materials():
    ''' Deletes all materials in the scene. This can be useful for stanardizing meshes. '''
    # https://blender.stackexchange.com/questions/27190/quick-way-to-remove-thousands-of-materials-from-an-object
    C = bpy.context

    for i in range(1,len(C.object.material_slots)):
        C.object.active_material_index = 1
        bpy.ops.object.material_slot_remove()

    bpy.ops.object.mode_set(mode = 'EDIT') 
    bpy.ops.mesh.select_all(action = 'SELECT')
    bpy.ops.object.material_slot_assign()
    bpy.ops.object.mode_set(mode = 'OBJECT') 

def filter_camera_sweeps_by_enabled_in_file( camera_sweeps, infile ):
  """
    Keeps the points which are enabled in jsonfile
  """
  with open(infile) as jsonfile:
    data = json.load(jsonfile)

  return {k: v for k, v in camera_sweeps.items() if data['sweeps'][k]['enabled']}    


def get_2d_point_from_3d_point(three_d_point, K, RT):
    '''  By Farhan  '''
    P = K * RT
    product = P*Vector(three_d_point)
    two_d_point = (product[0] / product[2], product[1] / product[2])
    return two_d_point


def get_2d_point_and_decision_vector_from_3d_point(camera_data, location, rotation, target):
    '''  By Farhan  '''
    K = get_calibration_matrix_K_from_blender(camera_data)
    RT = get_3x4_RT_matrix_from_blender(Vector(location), rotation)
    P = K*RT
    decision_vector = P*Vector(target)
    x, y = get_2d_point_from_3d_point(target, K, RT)
    return (x, y, decision_vector)


def get_3x4_RT_matrix_from_blender(location, rotation):
    '''  By Farhan  '''
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0,  0),
         (0, -1, 0),
         (0, 0, -1)))
    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    # R_world2bcam = rotation.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints
    R_world2bcam = rotation.to_matrix()
    #R_world2bcam.invert()
    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:     
    T_world2bcam = location
    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = R_world2bcam #R_bcam2cv*R_world2bcam
    T_world2cv = T_world2bcam #R_bcam2cv*T_world2bcam
    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],), 
        (0, 0, 0, 1)
         ))
    return RT


def get_calibration_matrix_K_from_blender(camd):  
    '''  By Farhan  '''
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    # scale = scene.render.resolution_percentage / 100
    scale = 1
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    f_in_px = (f_in_mm * resolution_x_in_px) / sensor_width_in_mm

    if (camd.sensor_fit == 'VERTICAL'):
        # the sensor height is fixed (sensor fit is horizontal), 
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio 
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else: # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal), 
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm
        
    # Parameters of intrinsic calibration matrix K
    u_0 = resolution_x_in_px*scale / 2
    v_0 = resolution_y_in_px*scale / 2
    skew = 0 # only use rectangular pixels

    K = Matrix(
        ((f_in_px, skew,    u_0),
        (    0  ,  f_in_px, v_0),
        (    0  ,    0,      1 )))

    return K


def get_camera_is_enabled_file( dir ):
  return os.path.join( dir, settings.CAMERA_IS_ENABLED_FILE )


def get_camera_pose_file( dir ):
  return os.path.join( dir, "modeldata", settings.CAMERA_POSE_FILE )


def get_file_name_for( dir, point_uuid, view_number, camera_uuid, task, ext ):
  """ 
    Returns the filename for the given point, view, and task 

    Args:
      dir: The parent directory for the model
      task: A string definint the task name 
      point_uuid: The point identifier
      view_number: This is the nth view of the point 
      camera_uuid: An identifier for the camera
      ext: The file extension to use
  """
  view_specifier = view_number
  #   if settings.FILE_NAMING_CONVENTION == 'DEBUG':
  #     view_specifier = camera_uuid
  filename = "point_{0}_view_{1}_domain_{2}.{3}".format( point_uuid, view_specifier, task, ext )
  return os.path.join( dir, filename )


def get_model_file( dir, typ='RAW' ):
  if typ == 'RAW':
      model_file =  settings.MODEL_FILE
  elif typ == 'SEMANTIC':
      model_file = settings.SEMANTIC_MODEL_FILE
  elif typ == 'SEMANTIC_PRETTY':
      model_file = settings.SEMANTIC_PRETTY_MODEL_FILE
  elif typ == 'LEGO':
      model_file =  settings.LEGO_MODEL_FILE
  else:
      raise ValueError( 'Unknown type of model file: {0}'.format( typ ) )
  return os.path.join( dir, "modeldata", model_file )

def get_point_loc_in_model_from_view_dict( view_info ):
  """ Returns the location (3-tuple) of the point in the model given the loaded view_dict """
  return ( view_info['model_x'], view_info['model_y'], view_info['model_z'] )


def get_pitch_of_point( camera, point ):
  """
    Args: 
      camera: A Blender camera
      point: A 3-tuple of coordinates of the target point
    
    Returns: 
      pitch: A float
  """
  # Just check whether the direction of the target point is within pi / 12 of the plane of rotation
  point_in_local_coords = camera.matrix_world.inverted() * Vector( point )
  angle_to_normal = Vector( (0,1,0) ).angle( point_in_local_coords )
  angle_to_plane = math.pi / 2. - angle_to_normal
  return angle_to_plane


def get_pixel_in_skybox_for_point_from_view_dict( view_info ):
  """ Returns the pixel location (pair) of the point in the skybox image given the loaded view_dict """
  return ( view_info['skybox_pixel_x'], view_info['skybox_pixel_y'] )


def get_save_info_for_correspondence( empty, point, point_uuid, point_normal, camera_uuid, cameras, obliqueness_angle, resolution ):
  """ 
    Creates info for a point and camera that allows easy loading of a camera in Blender

    Args:
      empty: An Empty located at the point
      point: The xyz coordinates of the point to create the save info for
      point_uuid: The uuid pertaining to this point
      point_normal: The normal of the face the point lies on
      camera_uuid: The uuid of the camera for which we will be creating info for
      cameras: This a dict of many cameras for which camera_uuid is a key
      obliqueness_angle: Angle formed between the point_normal and camera->point_location, in rads
      resolution: Skybox camera resolution

    Returns:
      save_dict: A Dict of useful information. Currently it's keys are
        camera_distance: The distance from the camera to the point in meters
        camera_location: The location of the camera in the 3d model
        camera_original_rotation: The rotation_euler of the camera in the 3d model
        img_path: The path to the unique image for this uuid that has line-of-sight on the point 
        model_x: The x coordinate of the point in the model
        model_y: The y coordinate of the point in the model
        model_z: The z coordinate of the point in the model
        nonfixated_pixel_x:
        nonfixated_pixel_y:
        obliqueness_angle: Angle formed between the point_normal and camera->point_location, in rads
        point_normal: The normal of the face the point lies on        
        rotation_of_skybox: The Euler rotation that, when the camera is set to inside the cube, will provide the skybox image
        rotation_from_original_to_point: Apply to camera_original_rotation to aim camera at target
        skybox_img: The unique skybox image number that has line-of-sight on the point
        skybox_pixel_x: The exact x pixel in the skybox image where the point will be
        skybox_pixel_y: The exact y pixel in the skybox image where the point will be
        uuid: The uuid of this camera
  """
  # TODO(sasha): The arguments are ugly
  point_data = {}

  # Save basic info
  point_data[ 'model_x' ] = point[0]
  point_data[ 'model_y' ] = point[1]
  point_data[ 'model_z' ] = point[2]
  point_data[ 'camera_uuid' ] = camera_uuid
  point_data[ 'point_uuid' ] = point_uuid
  point_data[ 'field_of_view_rads' ] = settings.FIELD_OF_VIEW_RADS
  
  # Unpack the camera extrinsics
  camera_extrinsics = cameras[ camera_uuid ]
  location = camera_extrinsics['position']
  rotation_euler = camera_extrinsics['rotation'] 
  point_data[ 'camera_distance' ] = ( Vector( location ) - Vector( point ) ).magnitude
  point_data[ 'camera_location' ] = location
  point_data[ 'obliqueness_angle' ] = obliqueness_angle
  point_data[ 'point_normal' ] = point_normal

  # rotation_euler = Euler( rotation_euler ) # This is for debugging camera movement
  # rotation_euler.rotate_axis( 'Z', math.pi / 4 )
  
  quaternion = Quaternion( camera_extrinsics['quaternion'] )
  
  ## SKYBOX
  # Find and save skybox number
  camera, camera_data, scene = create_camera( location, rotation_euler, 
              field_of_view=settings.FIELD_OF_VIEW_MATTERPORT_RADS, 
              camera_name="Camera_save_point_1" )                
  skybox_number = get_skybox_img_number_containing_point( location, rotation_euler, empty )
  point_data[ 'camera_original_rotation' ] = tuple( rotation_euler )
  point_data[ 'skybox_img' ] = skybox_number 
  point_data[ 'img_path' ] = os.path.join("./img/high", "{0}_skybox{1}.jpg".format( camera_uuid, skybox_number ) )
  point_data[ 'point_pitch' ] = get_pitch_of_point( camera, point )

  # Save the rotation_euler for the camera to point at the skybox image in the model
  new_camera, new_camera_data, scene = create_camera( location, rotation_euler, 
          resolution=settings.MATTERPORT_SKYBOX_RESOLUTION,
          field_of_view=settings.FIELD_OF_VIEW_MATTERPORT_RADS, 
          camera_name="Camera_save_point_2" )
  axis_of_rotation, rotation_from_start = skybox_number_to_axis_and_rotation[ skybox_number ]
  new_camera.rotation_euler.rotate_axis( axis_of_rotation, rotation_from_start )
  if skybox_number == 0:  # Need to rotate top and bottom images
      new_camera.rotation_euler.rotate_axis( 'Z', math.pi / 2 )
  if skybox_number == 5:  # Need to rotate top and bottom images
      new_camera.rotation_euler.rotate_axis( 'Z', -math.pi / 2 ) # Not sure if this is correct, but we should never have this img
  scene.update()

  # And save the x, y pixel coordinates for the skybox image
  x, y, _ = get_2d_point_and_decision_vector_from_3d_point( 
          new_camera_data, location, new_camera.rotation_euler, point )
  point_data[ 'rotation_of_skybox' ] = tuple( new_camera.rotation_euler )
  point_data[ 'skybox_pixel_x' ] = int( round( x ) )
  point_data[ 'skybox_pixel_y' ] = int( round( y ) )

  ## FIXATED
  # Now save the rotation needed to point at the target
  new_camera, new_camera_data, scene = create_camera( location, rotation_euler, 
              resolution=settings.RESOLUTION,
              field_of_view=settings.FIELD_OF_VIEW_RADS, 
              camera_name="Camera_save_point_3" )
  utils.point_camera_at_target( new_camera, empty )
  point_data[ 'rotation_from_original_to_point' ] = tuple( 
          utils.get_euler_rotation_between( 
                  camera.rotation_euler, 
                  new_camera.rotation_euler ) )
 
  #   other_calculated_normal = camera.matrix_world.to_quaternion() * Vector( (0,1,0) )
  #   centered_camera_dir = new_camera.matrix_world.to_quaternion() * Vector( (0,0,-1) )
  #   other_calculated_pitch = math.pi/2 - centered_camera_dir.angle( other_calculated_normal )
    
  # Local coords method
  #   point_in_local_coords = ( camera.matrix_world.inverted() * point ).normalized()
  #   angle_to_normal_local_method = Vector( (0,1,0) ).angle( point_in_local_coords )
  #   print( "-----------pitch (manual method):", math.degrees( other_calculated_pitch ) )
  #   print("\tnormal_dir", other_calculated_normal)
  #   print("\tcamera_dir:", centered_camera_dir)
  #   print("\tpoint_dir:", (Vector( point ) - camera.location ).normalized() )
  #   print("\tpoint_dir unnormalized:", (Vector( point ) - camera.location ) )
  #   print("\tangle_to_normal: {0} degrees".format( math.degrees( centered_camera_dir.angle( other_calculated_normal ) ) ) )
  #   print( "-----------pitch (local method):", math.degrees( math.pi / 2 - angle_to_normal_local_method ) )
  #   print("\tpoint_dir_in_local_coords:", ( camera.matrix_world.inverted() * point ).normalized() )

        ## NONFIXATED
  #   # Generate nonfixated image
  #   x_jitter = np.random.uniform( -settings.FIELD_OF_VIEW_RADS / 2., settings.FIELD_OF_VIEW_RADS / 2. )
  #   new_camera.rotation_euler.rotate_axis( axis_of_rotation, x_jitter )
    
  #   new_camera.rotation_euler.rotate_axis( 'X', -point_data[ 'point_pitch' ]  ) # Back into the plane
  # And save the x, y pixel coordinates the nonfixated image
  x, y, _ = get_2d_point_and_decision_vector_from_3d_point( 
          new_camera_data, location, new_camera.rotation_euler, point )
  point_data[ 'rotation_from_original_to_nonfixated' ] = tuple( 
          utils.get_euler_rotation_between( 
                  camera.rotation_euler, 
                  new_camera.rotation_euler ) )
  point_data[ 'nonfixated_pixel_x' ] = int( round( x ) )
  point_data[ 'nonfixated_pixel_y' ] = int( round( y ) )


  utils.delete_objects_starting_with( "Camera_save_point_1" ) # Clean up
  utils.delete_objects_starting_with( "Camera_save_point_2" ) # Clean up
  utils.delete_objects_starting_with( "Camera_save_point_3" ) # Clean up
  # utils.delete_objects_starting_with( "Camera" ) # Clean up
  return point_data






def get_save_info_for_sweep( fov, pitch, yaw, point_uuid, camera_uuid, cameras, resolution ):
  """ 
    Creates info for a point and camera that allows easy loading of a camera in Blender

    Args:
      fov: The field of view of the camera
      pitch: The pitch of the camera relative to its plane of rotation
      yaw: The yaw of the camera compared to its initial Euler coords
      point_uuid: The uuid pertaining to this point
      camera_uuid: The uuid of the camera for which we will be creating info for
      cameras: This a dict of many cameras for which camera_uuid is a key
      resolution: Skybox camera resolution

    Returns:
      save_dict: A Dict of useful information. Currently it's keys are
        {
        "camera_k_matrix":  # The 3x3 camera K matrix. Stored as a list-of-lists, 
        "field_of_view_rads": #  The Camera's field of view, in radians, 
        "camera_original_rotation": #  The camera's initial XYZ-Euler rotation in the .obj, 
        "rotation_from_original_to_point": 
        #  Apply this to the original rotation in order to orient the camera for the corresponding picture, 
        "point_uuid": #  alias for camera_uuid, 
        "camera_location": #  XYZ location of the camera, 
        "frame_num": #  The frame_num in the filename, 
        "camera_rt_matrix": #  The 4x3 camera RT matrix, stored as a list-of-lists, 
        "final_camera_rotation": #  The camera Euler in the corresponding picture, 
        "camera_uuid": #  The globally unique identifier for the camera location, 
        "room": #  The room that this camera is in. Stored as roomType_roomNum_areaNum 
        }
  """
  # TODO(sasha): The arguments are ugly
  point_data = {}

  # Save basic info
  point_data[ 'camera_uuid' ] = camera_uuid
  point_data[ 'point_uuid' ] = point_uuid
  
  # Unpack the camera extrinsics
  camera_extrinsics = cameras[ camera_uuid ]
  location = camera_extrinsics['position']
  rotation_euler = camera_extrinsics['rotation']
  point_data[ 'camera_original_rotation' ] = tuple( rotation_euler ) 
  point_data[ 'camera_location' ] = location
  
  # Save initial camera locatoin
  camera, camera_data, scene = create_camera( location, rotation_euler, 
              field_of_view=settings.FIELD_OF_VIEW_MATTERPORT_RADS, 
              camera_name="Camera_save_point_1" )

  # Save the rotation_euler for the camera to point at the skybox image in the model
  new_camera, new_camera_data, scene = create_camera( location, rotation_euler, 
          resolution=settings.RESOLUTION,
          field_of_view=fov, 
          camera_name="Camera_save_point_2" )
  new_camera.rotation_euler.rotate_axis( 'Y', yaw )
  new_camera.rotation_euler.rotate_axis( 'X', pitch )                
  point_data[ 'rotation_from_original_to_point' ] = tuple( 
          utils.get_euler_rotation_between( 
                  camera.rotation_euler, 
                  new_camera.rotation_euler ) )
  point_data[ 'final_camera_rotation' ] = tuple( new_camera.rotation_euler )
  point_data[ 'field_of_view_rads' ] = fov
  def matrix_to_list_of_lists( mat ):
      lst_of_lists = list( mat )
      lst_of_lists = [ list( vec ) for vec in lst_of_lists ]
      return lst_of_lists

  point_data[ 'camera_rt_matrix' ] = matrix_to_list_of_lists( 
            get_3x4_RT_matrix_from_blender( Vector( location ), new_camera.rotation_euler ) )
  point_data[ 'camera_k_matrix' ] = matrix_to_list_of_lists( 
            get_calibration_matrix_K_from_blender( new_camera_data ) )
  
  utils.delete_objects_starting_with( "Camera_save_point_1" ) # Clean up
  utils.delete_objects_starting_with( "Camera_save_point_2" ) # Clean up
  # utils.delete_objects_starting_with( "Camera" ) # Clean up
  return point_data





def get_skybox_img_number_containing_point( camera_location, camera_rotation_euler, empty_at_target ):
  """
    This gets the image index of the skybox image. 

    It works by finding the direction of the empty from the camera and then by rotating that vector into a 
    canonical orientation. Then we can use the dimension with the greatest magnitude, and the sign of that 
    coordinate in order to determine the face of the cube that the empty projects onto. 
  """
  empty_direction = ( empty_at_target.location - Vector( camera_location ) ).normalized()
  empty_direction.normalize()
  empty_direction.rotate( camera_rotation_euler.to_matrix().inverted() )

  # The trick to finding the cube face here is that we can convert the direction
  max_axis, coord_val = max( enumerate( empty_direction ), key=lambda x: abs( x[1] ) ) # Find the dim with 
  sign = ( coord_val >= 0.0 ) 
  max_axis = ["X", "Y", "Z"][ max_axis ] # Just make it more readable

  return axis_and_positive_to_skybox_idx[ ( max_axis, sign ) ]

def get_task_image_fpath( directory, point_data, view_num, task ):
    ''' Builds and returnes a standardized filepath for an point/image '''
    view_dict = point_data[ view_num ]
    if task == 'skybox':
        return os.path.join( directory, view_dict[ 'img_path' ] )
    elif 'depth' in task:
        directory = os.path.join( directory, 'depth')
    elif 'normals' in task:
        directory = os.path.join( directory, 'normals')
    elif 'rgb':
        directory = os.path.join( directory, 'rgb')
    preferred_ext = img_format_to_ext[ settings.PREFERRED_IMG_EXT.lower() ]
    fname = get_file_name_for( directory,
                        point_uuid=view_dict[ 'point_uuid' ], 
                        view_number=view_num, 
                        camera_uuid=view_dict[ 'camera_uuid' ], 
                        task=task, 
                        ext=preferred_ext )
    return os.path.join( directory, fname )



def import_mesh( dir, typ='RAW' ):
  ''' Imports a mesh with the appropriate processing beforehand.
  Args: 
    dir: The dir from which to import the mesh. The actual filename is given from settings.
    typ: The type of mesh to import. Must be one of ['RAW', 'SEMANTIC', 'SEMANTIC_PRETTY', 'LEGO']
      Importing a raw model will remove all materials and textures. 
  Returns:
    mesh: The imported mesh.
  '''
  model_fpath = get_model_file( dir, typ=typ )
  if '.obj' in model_fpath:
    bpy.ops.import_scene.obj( filepath=model_fpath ) 
    model = join_meshes() # OBJs often come in many many peices
    for img in bpy.data.images: # remove all images
        bpy.data.images.remove(img, do_unlink=True)
    bpy.context.scene.objects.active = model
    if typ == 'SEMANTIC' or typ == 'SEMANTIC_PRETTY':
        return
    #model.matrix_world *= Matrix.Rotation(-math.pi/2., 4, 'Z')
    if typ == 'LEGO':
        return
    delete_materials()
  elif '.ply' in model_fpath:
    bpy.ops.import_mesh.ply( filepath=model_fpath )
  model = bpy.context.object
  return model

def join_meshes():
    ''' Takes all meshes in the scene and joins them into a single mesh.
    Args:
        None
    Returns:
        mesh: The single, combined, mesh 
    '''
    # https://blender.stackexchange.com/questions/13986/how-to-join-objects-with-python
    scene = bpy.context.scene
    obs = []
    for ob in scene.objects:
        # whatever objects you want to join...
        if ob.type == 'MESH':
            obs.append(ob)
    ctx = bpy.context.copy()
    # one of the objects to join

    #ctx['active_object'] = obs[0]
    #ctx['selected_objects'] = obs
    # we need the scene bases as well for joining
    #ctx['selected_editable_bases'] = [scene.object_bases[ob.name] for ob in obs]
    #print('finished join mesh selection')
    #bpy.ops.object.join(ctx)
    #for ob in scene.objects:
    #    print(ob)
        # whatever objects you want to join...
    #    if ob.type == 'MESH':
    #        return ob
    scene.objects.active = obs[0]
    bpy.ops.object.join()
    return obs[0]

def load_camera_poses( dir, enabled_only ):
  """ 
    Loads the cameras from disk.

    Args:
      dir: Parent directory of the model. E.g. '/path/to/model/u8isYTAK3yP'
      enabled_only: Whether to load only enabled cameras
    
    Returns:
      camera_poses: A dict of camera_uuid -> { position:, quaternion:, rotation: }
  """
  camera_locations = collect_camera_poses_from_csvfile( get_camera_pose_file( dir ) )
  if enabled_only:
      camera_locations = filter_camera_sweeps_by_enabled_in_file( camera_locations, get_camera_is_enabled_file( dir ) )
  logger.info("Loaded {0} cameras.".format( len( camera_locations ) ) )
  return camera_locations   


def load_saved_points_of_interest( dir ):
  """
    Loads all the generated points that have multiple views. 

    Args:
      dir: Parent directory of the model. E.g. '/path/to/model/u8isYTAK3yP'
    
    Returns:
      point_infos: A list where each element is the parsed json file for a point
  """
  point_files = glob.glob( os.path.join( dir, "points", "point_*.json" ) )
  point_files.sort()
  point_infos = []
  for point_file in point_files:
    with open( point_file ) as f:
      point_infos.append( json.load( f ) )
  logger.info( "Loaded {0} points of interest.".format( len( point_infos ) ) )
  return point_infos

def load_model_and_points( basepath, typ='RAW' ):
    ''' Loads the model and points
    Args:
        basepath: The model path
    Returns:
        A Dict:
            'camera_poses': 
            'point_infos':
            'model: The blender mesh
    '''
    utils.delete_all_objects_in_context()
    camera_poses = load_camera_poses( basepath, settings.USE_ONLY_ENABLED_CAMERAS )   
    point_infos = load_saved_points_of_interest( basepath )
    model = import_mesh( basepath, typ=typ )
    return { 'camera_poses': camera_poses, 'point_infos': point_infos, 'model': model }

def parse_semantic_label( label ):
    ''' Pareses a semantic label string into 
      semantic_class, instance_num, roomtype, roomnum, area_num
    Args: 
        label: A string to be parsed
    Returns:
        semantic_class, instance_num, roomtype, roomnum, area_num
    '''
    toks = label.split('_')
    clazz, instance_num, roomtype, roomnum, area_num = toks[0], toks[1], toks[2], toks[3], toks[4]
    return clazz, instance_num, roomtype, roomnum, area_num

def track_empty_with_axis_lock(cam, lock_axis, track_axis, empty):
  ''' Turns the camera along its axis of rotation in order to points (as much as possible) at the 
    target empty.

  Args:
    cam: A Blender camera 
    lock_axis: The axis to rotate about
    track_axis: The axis of the camera which should point (as much as possible) at the empty: Use 'NEGATIVE_Z'
    empty: The empty to point 'track_axis' at

  Returns:
    None (points the camera at the empty)
  '''
  constraint = cam.constraints.new(type='LOCKED_TRACK')
  constraint.lock_axis = lock_axis
  constraint.track_axis = track_axis
  constraint.target = empty
  bpy.ops.object.select_all(action='DESELECT')
  cam.select = True
  bpy.ops.object.visual_transform_apply()
  cam.constraints.remove(constraint)


def try_get_data_dict( point_datas, point_uuid ):
    point_data = [ p for p in point_datas if p[0][ 'point_uuid' ] == point_uuid ]
    if not point_data:
        raise KeyError( "Point uuid {0} not found".format( point_uuid ) )
    return point_data[0]

def try_get_task_image_fpaths( directory, point_uuid, point_datas=None, task='markedskybox' ):
    if task == 'rgb_nonfixated':
        if not point_datas:
            raise ValueError( "If using rgb_nonfixated then point_datas must be specified" )
        return sorted( [ os.path.join( directory, view_dict[ 'img_path' ] ) 
                            for view_dict in try_get_data_dict( point_datas, point_uuid ) ] ) 
    else: 
        bash_regex = "point_{0}__view_*__{1}.png".format( point_uuid, task )
        return sorted( glob.glob( os.path.join( directory, bash_regex ) ) )

if __name__=='__main__':
    import argparse
    args = argparse.Namespace()
    # import settings
    # settings.__dict__['DEPTH_BITS_PER_CHANNEL'] = 1000000000000
    # print( settings.DEPTH_BITS_PER_CHANNEL )
    load_settings( args )