"""
  Name: generate_points.py
  Author: Sasha Sax, CVGL
  Modified by: Zhiyang He
  Desc: Selects points that have at least a given number of views and saves information useful for loading them.
    Will also mark the point as a pixel on point_{uuid}__view_{num}__markedskybox.{ext} if MARK_POINT_WITH_X is
    enabled in settings.py

  Usage:
    blender -b -noaudio --enable-autoexec --python generate_points.py -- NUM_POINTS MIN_VIEWS MAX_VIEWS
"""

# Import these two first so that we can import other packages
import os
import sys
sys.path.append( os.path.dirname( os.path.realpath(__file__) ) )
from activate_env import add_on_path
sys.path.append(add_on_path)

from load_settings import settings
import io_utils

# Import remaining packages
import argparse
import bpy
import bpy_extras.mesh_utils
import glob
import json
import math
from mathutils import Vector, Euler
import numpy as np
import random
import time
import utils
from utils import Profiler, create_empty
import uuid

utils.set_random_seed()


parser = argparse.ArgumentParser()

parser.add_argument('--NUM_POINTS_NEEDED', type=int, required=True,
                    help='The number of points to generate')
parser.add_argument('--MIN_VIEWS', type=int, required=True,
                    help='The minimum number of views per point')
parser.add_argument('--MAX_VIEWS', type=int, required=True,
                    help='The maximum number of views per point (-1 to disable)')
parser.add_argument('--BASEPATH', type=str, required=True,
                    help='The (absolute) base path of the current model')

TASK_NAME = 'points'

def parse_local_args( args ):
  local_args = args[ args.index( '--' ) + 1: ]
  return parser.parse_known_args( local_args )

def main():
  global args, logger 
  args, remaining_args = parse_local_args( sys.argv )
  logger = io_utils.create_logger( __name__ )  
#   io_utils.load_settings( remaining_args )
#   utils.validate_blender_settings( settings )

  assert(args.BASEPATH)
  basepath = args.BASEPATH
  
  assert(args.NUM_POINTS_NEEDED > 1)
  utils.delete_all_objects_in_context()
  if settings.VERBOSITY >= settings.VERBOSITY_LEVELS[ 'INFO' ]:
    print( "Num points: {0} | Min views: {1} | Max views: {2}".format( args.NUM_POINTS_NEEDED, args.MIN_VIEWS, args.MAX_VIEWS ) )
 
  # Get camera locations and optionally filter by enabled
  camera_poses = io_utils.collect_camera_poses_from_csvfile( io_utils.get_camera_pose_file( basepath ) )
  if settings.USE_ONLY_ENABLED_CAMERAS:
      camera_poses = io_utils.filter_camera_sweeps_by_enabled_in_file( camera_poses, io_utils.get_camera_is_enabled_file( basepath ) )
#   valid_cameras = [ 'fab20a57533646ce8da7ced527766b93', '1d12cda3bb31406ab49646bf27376d6a' ] # 'e1071efa828c432087a60ecb7b498453', 
#   valid_cameras = [ '1d12cda3bb31406ab49646bf27376d6a' ] # 'e1071efa828c432087a60ecb7b498453', 
#   camera_poses = { k:cp for k, cp in camera_poses.items() if k in valid_cameras }

  # Load the model
  model = io_utils.import_mesh( basepath )
  if not os.path.isdir( os.path.join( basepath, TASK_NAME ) ):
    os.mkdir( os.path.join( basepath, TASK_NAME ) )
  
  # Generate the points
  if settings.POINT_TYPE == 'SWEEP':
    generate_points_from_each_sweep( camera_poses, basepath )
  elif settings.POINT_TYPE == 'CORRESPONDENCES':
    generate_point_correspondences( model, camera_poses, basepath )
  else:
    raise NotImplementedError( 'Unknown settings.POINT_TYPE: ' + settings.POINT_TYPE )

def generate_points_from_each_sweep( camera_poses, basepath ):
  ''' Generates and saves points into basepath. Each point file corresponds to one cameara and 
    contains an array of different view_dicts for that camera. These view_dicts are distinct from
    the ones created by generate_point_correspondences since these views to not share a target point.
  Args:
    camera_poses: A Dict of camera_uuids -> camera extrinsics
    basepath: The directory in which to save points
  Returns:
    None (saves points)
  '''
  def sample( sample_i ):
    if settings.CREATE_PANOS: 
      if sample_i == 0: # Top
        return math.pi, math.pi / 2, settings.FIELD_OF_VIEW_MATTERPORT_RADS
      elif sample_i == 1: # Front
        return 0.0, 0.0, settings.FIELD_OF_VIEW_MATTERPORT_RADS
      elif sample_i == 2: # Right
        return math.pi / 2, 0.0, settings.FIELD_OF_VIEW_MATTERPORT_RADS
      elif sample_i == 3: # Back
        return math.pi , 0.0, settings.FIELD_OF_VIEW_MATTERPORT_RADS
      elif sample_i == 4: # Left
        return -math.pi / 2., 0.0, settings.FIELD_OF_VIEW_MATTERPORT_RADS
      elif sample_i == 5: # Bottom
        return math.pi, -math.pi/2, settings.FIELD_OF_VIEW_MATTERPORT_RADS
      else:
       raise ValueError( 'Too many samples for a panorama! (Max 6)')
    else:
      # How to generate samples from a camera sweep
      yaw = np.random.uniform( low=-math.pi, high=math.pi )
      pitch = settings.MAX_ANGLE_OF_CAMERA_BASE_FROM_PLANE_OF_ROTATION + 1
      while abs( pitch ) > settings.MAX_ANGLE_OF_CAMERA_BASE_FROM_PLANE_OF_ROTATION:
        pitch = np.random.normal( loc=0.0, scale=math.radians( 15. ) )
      # FOV
      z_val = 2
      while z_val > 1:
        z_val = np.random.normal( loc=0.0, scale=1. )
        z_val = np.abs( z_val )
        fov = settings.FIELD_OF_VIEW_MAX_RADS - z_val * ( settings.FIELD_OF_VIEW_MAX_RADS - settings.FIELD_OF_VIEW_MIN_RADS )
      return yaw, pitch, fov

  # Generate random points for each camera:
  for camera_uuid in camera_poses.keys():
    save_point_from_camera_sweep( sample, camera_uuid, camera_poses, basepath )

def generate_point_correspondences( model, camera_poses, basepath ):
  ''' Generates and saves points into basepath. These points are generated as correspondences
    where each point_uuid.json is an array of view_dicts, or information about a camera which
    has line-of-sight to the desired point. Each view_dict includes information about the
    target point, too. 
  Args:
    model: A Blender mesh that will be used to propose points
    camera_poses: A Dict of camera_uuids -> camera extrinsics
    basepath: The directory in which to save points
  Returns:
    None (saves points)
  '''
  n_generated = 0
  while n_generated < args.NUM_POINTS_NEEDED:
    utils.delete_objects_starting_with( "Camera" )
    with Profiler( "Generate point", logger ):
      point_uuid = str( uuid.uuid4() ) # Can also use hardcoded "TEST"
      if settings.FILE_NAMING_CONVENTION == 'DEBUG':
        point_uuid = str( n_generated ) 
      next_point, visible_cameras, obliquenesses_dict = get_viable_point_and_corresponding_cameras( 
                                                                                model, 
                                                                                camera_poses, 
                                                                                min_views=args.MIN_VIEWS,
                                                                                point_num=n_generated)
      save_point_from_correspondence( visible_cameras, next_point, point_uuid, obliquenesses_dict, basepath )
      n_generated += 1

def save_point_from_camera_sweep( sampling_fn, camera_uuid, camera_poses, basepath ):
    '''
    Args:
        sampling_fn: A function which taskes in (sample_number) and returns (yaw, pitch, fov)
        camera_uuid: The key of this camera inside camera_poses
        camera_poses: All of the camera extrinsics for all cameras
        basepath: The directory to save this point in
    Returns:
        None (samples point and saves it in basepath)
    '''
    with Profiler( "Save point", logger ):
        point_data = []
        point_uuid = str( uuid.uuid4() ) # Can also use hardcoded "TEST"  
        if settings.FILE_NAMING_CONVENTION == 'DEBUG':
            point_uuid = str( camera_uuid ) 
        
        # Save each sampled camera position into point_data
        for sample_i in range( args.NUM_POINTS_NEEDED ):
            yaw, pitch, fov = sampling_fn( sample_i )
            print("Get yaw, pitch, fov", yaw, pitch, fov, sample_i)
            view_dict = io_utils.get_save_info_for_sweep( 
                    fov, pitch, yaw, point_uuid, camera_uuid, camera_poses, settings.RESOLUTION  )     
            point_data.append( view_dict )
        
        # Save result out
        outfile_path = os.path.join( basepath, TASK_NAME, "point_" + point_uuid + ".json" )    
        with open( outfile_path, 'w' ) as outfile:
            json.dump( point_data, outfile )

def save_point_from_correspondence( visible_cameras, next_point, point_uuid, obliquenesses_dict, basepath ):
    ''' Saves out a CORRESPONDENCE-type point to a file in basepath. 
    Each point_uuid.json is an array of view_dicts, or information about a camera which
    has line-of-sight to the desired point. Each view_dict includes information about the
    target point, too. 

    Args: 
        visible_cameras: A list of all camera_poses which have line-of-sight to next_point
        next_point: A 3-tuple of the XYZ coordinates of the target_point
        point_uuid: A uuid to call this point. Defines the filename. 
        obliquenesses_dict: A dict of camera_uuid -> obliqueness of the face relative to camera
        basepath: Directory under which to save point information
    Returns:
        None (Save a point file under basepath)
    '''
    with Profiler( "Save point" ):
      empty = utils.create_empty( "Empty", next_point )
      point_data = []
 
      # So that we're not just using the same camera for each point
      shuffled_views = list( visible_cameras )
      random.shuffle( shuffled_views )
      for view_number, camera_uuid in enumerate( shuffled_views ):
        point_normal, obliqueness_angle = obliquenesses_dict[ camera_uuid ]
        next_point_data = io_utils.get_save_info_for_correspondence( empty, 
                                                  point=next_point, 
                                                  point_uuid=point_uuid,
                                                  point_normal=tuple( point_normal ),
                                                  camera_uuid=camera_uuid, 
                                                  cameras=visible_cameras, 
                                                  obliqueness_angle=obliqueness_angle,
                                                  resolution=settings.RESOLUTION ) 
        point_data.append( next_point_data )
        if view_number == int( args.MAX_VIEWS ):
          break

        if view_number == settings.STOP_VIEW_NUMBER: 
          break
 
      outfile_path = os.path.join(basepath, TASK_NAME, "point_" + point_uuid + ".json")
      with open( outfile_path, 'w' ) as outfile:
        json.dump( point_data, outfile )

def get_random_point_from_mesh( num_points, model ):
  """
    Generates a given number of random points from the mesh
  """
  # return [ Vector( ( -1, 0, 0 ) ) ] # Sink
  me = model.data

  me.calc_tessface() # recalculate tessfaces
  tessfaces_select = [f for f in me.tessfaces if f.select]
  random.shuffle( tessfaces_select )
  multiplier = 1 if len(tessfaces_select) >= num_points else num_points // len(tessfaces_select)
  return bpy_extras.mesh_utils.face_random_points(multiplier, tessfaces_select[:num_points])


def get_viable_point_and_corresponding_cameras( model, camera_locations, min_views=3, point_num=None ):
  """
    Keeps randomly sampling points from the mesh until it gets one that is viewable from at least
      'min_views' camera locations.

    Args:
      model: A Blender mesh object
      min_views: The minimum viable number of views
      camera_locations: A list of dicts which have information about the camera location
      point_num: The index of the point in test_assets/points_to_generate.json - needs to be 
          specified iff settings.MODE == 'TEST'

    Returns:
      point: A point that has at least 'min_views' cameras with line-of-sight on point
      visible: A Dict of visible cameras---camera_uuid -> extrinsics
      obliquness: A Dict of camera_uuid->( point_normal, obliqueness_angle )
  """
  count = 0
  while True:
    # Generate point and test
    if settings.MODE == 'TEST':
      with open( "test_assets/points_to_generate.json", 'r' ) as fp:
        candidate_point_tuple = io_utils.get_point_loc_in_model_from_view_dict( json.load( fp )[ point_num ] )        
        candidate_point = Vector( candidate_point_tuple )
    else:  
      candidate_point = get_random_point_from_mesh( 1, model )[0]
    #   candidate_point = Vector( (-1.8580, -0.9115, 3.6539) )

    cameras_with_view_of_candidate = {}
    obliquenesses_dict = {}
    n_views_with_los = 0
    for camera_uuid, camera_extrinsics in camera_locations.items():
      camera_rotation_euler = Euler( camera_extrinsics[ 'rotation' ], settings.EULER_ROTATION_ORDER )
      camera_location = camera_extrinsics[ 'position' ]
      camera, _, scene = utils.create_camera( location=camera_location, rotation=camera_rotation_euler, 
        field_of_view=settings.FIELD_OF_VIEW_MATTERPORT_RADS, 
        camera_name="viable_point_camera" )  
      
      # Compute whether to use this view
      los_normal_and_obliquness = try_get_line_of_sight_obliqueness( camera.location, candidate_point  )
      contains_base = is_view_of_point_containing_camera_base( camera, candidate_point )
      
      # Debug logging
      if settings.VERBOSITY >= settings.VERBOSITY_LEVELS[ 'DEBUG' ]:
        print( "\nCamera name:", camera_uuid )
        print( "\tCamera position:", tuple( camera.location ) )
        print( "\tCamera initial rotation:", tuple( camera.rotation_euler ) )
        print( "\tPitch:", math.degrees( io_utils.get_pitch_of_point( camera, candidate_point ) ) )
        print( "\tPoint:", tuple( candidate_point ) )
        print( "\tLine of Sight: {0} | Contains Base: {1}".format( los_normal_and_obliquness, contains_base ) )
      
      # Count the number of cameras with a view
      if los_normal_and_obliquness:
        n_views_with_los += 1

      # if use viable view, save it for this point
      if los_normal_and_obliquness and not contains_base:
        point_normal, obliquness = los_normal_and_obliquness
        cameras_with_view_of_candidate[ camera_uuid ] = camera_extrinsics
        obliquenesses_dict[ camera_uuid ] = ( point_normal, obliquness )
    
    # Decide whether to continue looking for points
    count += 1
    if settings.VERBOSITY >= settings.VERBOSITY_LEVELS[ 'INFO' ]:
      print( "N views: {0} | line of sight: {1} ".format( len( cameras_with_view_of_candidate ), n_views_with_los ) )
    if len( cameras_with_view_of_candidate ) >= min_views:
      break
    if count % 100 == 0:  # Don't look more than 100 times
      if settings.VERBOSITY >= settings.VERBOSITY_LEVELS[ 'INFO' ]:
        print( "Finding a point taking a long time... {0} iters".format( count ) )
        break
  utils.delete_objects_starting_with( "viable_point_camera" ) # Clean up
  return candidate_point, cameras_with_view_of_candidate, obliquenesses_dict


def is_view_of_point_containing_camera_base( camera, point ):
  """
    Checks whether the given camera has a valid view of the point. 
    This currently just checks for line of sight, and that the blurry camera
    base is not visible in the picture. 

    Args: 
      camera: A Blender camera
      point: A 3-tuple of coordinates of the target point
    
    Returns: 
      bool
  """
  angle_to_plane = io_utils.get_pitch_of_point( camera=camera, point=point )
  return abs( angle_to_plane ) > settings.MAX_ANGLE_OF_CAMERA_BASE_FROM_PLANE_OF_ROTATION


def try_get_line_of_sight_obliqueness( start, end, scene=bpy.context.scene ):
  """
    Casts a ray in the direction of start to end and returns the surface 
    normal of the face containing 'end', and also the angle between the 
    normal and the cast ray. If the cast ray does not hit 'end' before 
    hitting anything else, it returns None.
    
    Args:
      start: A Vector
      end: A Vector
      scene: A Blender scene

    Returns:
      ( normal_of_end, obliqueness_angle )
      normal_of_end: A Vector normal to the face containing end
      obliqueness_angle: A scalar in rads
  """
  scene = bpy.context.scene
  if ( bpy.app.version[1] >= 75 ):
    direction = end - Vector(start)
    (ray_hit, location, normal, index, obj, matrix) = scene.ray_cast( start, direction )
  else:
    direction = end - Vector(start) # We need to double the distance since otherwise 
    farther_end = end + direction   # The ray might stop short of the target
    (ray_hit, obj, matrix, location, normal) = scene.ray_cast( start, farther_end )
  
  if not ray_hit or (location - end).length > settings.LINE_OF_SITE_HIT_TOLERANCE:
    return None
  obliqueness_angle = min( direction.angle( normal ), direction.angle( -normal ) )
  return normal, obliqueness_angle


if __name__=='__main__':
  with Profiler( "generate_points.py" ):
    main()
