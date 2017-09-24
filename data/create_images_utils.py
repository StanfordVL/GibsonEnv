"""
  Name: create_images_utils.py
  Author: Sasha Sax, CVGL
  Desc: Contains utilities which can be used to run 
  Usage:
    blender -b -noaudio -enable-autoexec --python create_normal_images.py --
"""

# Import these two first so that we can import other packages
import os
import sys
sys.path.append( os.path.dirname( os.path.realpath(__file__) ) )
from activate_env import add_on_path
sys.path.append(add_on_path)

from load_settings import settings


# Import remaining packages
import bpy
import bpy_extras.mesh_utils
import glob
import io_utils
import json
import logging
import math
from   mathutils import Vector, Euler
import numpy as np
import random
import shutil # Temporary dir
import time
import tempfile # Temporary dir
import utils
import uuid as uu
from   utils import Profiler

def start_logging():
  ''' '''
#   global logger
  logger = io_utils.create_logger( __name__ )  
  utils.set_random_seed()
  basepath = os.getcwd()
  return logger, basepath

def setup_rendering( setup_scene_fn, setup_nodetree_fn, logger, save_dir, apply_texture=None ):
    ''' Sets up everything required to render a scene 
    Args:
    Returns:
        render_save_path: A path where rendered images will be saved (single file)
    '''
    scene=bpy.context.scene 
    if apply_texture: 
        apply_texture( scene=bpy.context.scene )
    setup_scene_fn( scene )
    render_save_path = setup_nodetree_fn( scene, save_dir )
    return render_save_path



def setup_and_render_image( task_name, basepath, view_number, view_dict, camera_poses, execute_render_fn, logger=None, clean_up=True ):
    ''' Mutates the given camera and uses it to render the image called for in 
        'view_dict'
    Args:
        task_name: task name + subdirectory to save images
        basepath: model directory
        view_number: The index of the current view
        view_dict: A 'view_dict' for a point/view
        camera_poses: A dict of camera_uuid -> camera_pose
        execute_render_fn: A function which renders the desired image
        logger: A logger to write information out to
        clean_up: Whether to delete cameras after use
    Returns:
        None (Renders image)
    '''
    scene = bpy.context.scene
    camera_uuid = view_dict[ "camera_uuid" ]
    point_uuid = view_dict[ "point_uuid" ]
    camera, camera_data, scene = utils.create_camera( 
            location=camera_poses[ camera_uuid ][ "position" ],
            rotation=view_dict[ "camera_original_rotation" ],
            field_of_view=view_dict[ "field_of_view_rads" ],
            scene=scene, 
            camera_name='RENDER_CAMERA' )

    if settings.CREATE_PANOS:
        utils.make_camera_data_pano( camera_data )
        save_path = io_utils.get_file_name_for( 
                dir=get_save_dir( basepath, task_name ), 
                point_uuid=camera_uuid, 
                view_number=settings.PANO_VIEW_NAME, 
                camera_uuid=camera_uuid, 
                task=task_name, 
                ext=io_utils.img_format_to_ext[ settings.PREFERRED_IMG_EXT.lower() ] )
        #camera.rotation_euler = Euler( view_dict["camera_original_rotation"], settings.EULER_ROTATION_ORDER )
        camera.rotation_euler = Euler( view_dict["camera_original_rotation"] )
        camera.rotation_euler.rotate( Euler( view_dict[ "rotation_from_original_to_point" ] ) )
        execute_render_fn( scene, save_path )
    else:
        if settings.CREATE_NONFIXATED:
            save_path = io_utils.get_file_name_for( 
                    dir=get_save_dir( basepath, task_name ), 
                    point_uuid=point_uuid, 
                    view_number=view_number, 
                    camera_uuid=camera_uuid, 
                    task=task_name + "_nonfixated", 
                    ext=io_utils.img_format_to_ext[ settings.PREFERRED_IMG_EXT.lower() ] )
            camera.rotation_euler = Euler( view_dict["camera_original_rotation"], settings.EULER_ROTATION_ORDER )
            camera.rotation_euler.rotate( Euler( view_dict[ "rotation_from_original_to_nonfixated" ] , settings.EULER_ROTATION_ORDER ) )
            execute_render_fn( scene, save_path )

        if settings.CREATE_FIXATED:
            save_path = io_utils.get_file_name_for( 
                    dir=get_save_dir( basepath, task_name ), 
                    point_uuid=point_uuid, 
                    view_number=view_number, 
                    camera_uuid=camera_uuid, 
                    task=task_name + "_fixated", 
                    ext=io_utils.img_format_to_ext[ settings.PREFERRED_IMG_EXT.lower() ] )
            # Aim camera at target by rotating a known amount
            camera.rotation_euler = Euler( view_dict["camera_original_rotation"] )
            camera.rotation_euler.rotate( Euler( view_dict[ "rotation_from_original_to_point" ] ) )
            execute_render_fn( scene, save_path )
    
    if clean_up:
        utils.delete_objects_starting_with( "RENDER_CAMERA" ) # Clean up

def get_save_dir( basepath, task_name ):
    if settings.CREATE_PANOS:
        return os.path.join( basepath, 'pano', task_name )
    else: 
        return os.path.join( basepath, task_name )

def get_number_imgs( point_infos ):
    if settings.CREATE_PANOS:
        return len( point_infos )
    else: 
        n_imgs = 0
        if settings.CREATE_FIXATED:
            n_imgs += sum( [len( pi ) for pi in point_infos] )
        if settings.CREATE_NONFIXATED:
            n_imgs += sum( [len( pi ) for pi in point_infos] )
        return n_imgs

def run( setup_scene_fn, setup_nodetree_fn, model_dir, task_name, apply_texture_fn=None ):
    ''' Runs image generation given some render helper functions 
    Args:
        stop_at: A 2-Tuple of (pt_idx, view_idx). If specified, running will cease (not cleaned up) at the given point/view'''
    utils.set_random_seed()
    logger = io_utils.create_logger( __name__ )  

    with Profiler( "Setup", logger ) as prf:
        save_dir = os.path.join( model_dir, 'pano', task_name )
        model_info = io_utils.load_model_and_points( model_dir, typ='LEGO' )
        scene = bpy.context.scene
        if apply_texture_fn: 
            apply_texture_fn( scene=bpy.context.scene )
        execute_render = utils.make_render_fn( setup_scene_fn, setup_nodetree_fn, logger=logger) # takes (scene, save_dir)
        debug_at = ( settings.DEBUG_AT_POINT, settings.DEBUG_AT_VIEW )
        n_imgs = get_number_imgs( model_info[ 'point_infos' ] )
    
    with Profiler( 'Render', logger ) as pflr:
        img_number = 0
        for point_number, point_info in enumerate( model_info[ 'point_infos' ] ):
            for view_number, view_dict in enumerate( point_info ):
                if settings.CREATE_PANOS and view_number != 1:
                    continue  # we only want to create 1 pano per camera
                img_number += 1
                if debug_at[0] is not None:
                    if debug_at != ( point_number, view_number ):
                        continue
                setup_and_render_image( task_name, model_dir, 
                    camera_poses=model_info[ 'camera_poses' ],
                    clean_up=debug_at == (None, None),
                    execute_render_fn=execute_render,
                    logger=logger,
                    view_dict=view_dict, 
                    view_number=view_number )
                pflr.step( 'finished img {}/{}'.format( img_number, n_imgs ) )

                if debug_at == ( point_number, view_number ): 
                    return
    return