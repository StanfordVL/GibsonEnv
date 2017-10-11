"""
  Name: utils.py
  Author: Sasha Sax, CVGL
  Modified by: Zhiyang He
  Desc: Contains Blender and Matterport utility functions

  Usage: for import only
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from load_settings import settings
import os

import sys
sys.path.append( os.path.dirname( os.path.realpath(__file__) ) )
from activate_env import add_on_path
sys.path.append(add_on_path)

import bpy
import logging
import math
from   mathutils import Euler
import numpy as np
import os
import random
import shutil # Temporary dir
import time
import uuid as uu

cube_face_idx_to_skybox_img_idx = { 0: 4, 1: 1, 2: 2, 3: 3, 4: 5, 5: 0 }
img_format_to_ext = { "png": 'png', "jpeg": "jpg", "jpg": "jpg" }
skybox_img_idx_to_cube_face_idx = { v: k for k, v in cube_face_idx_to_skybox_img_idx.items() }
# order of axes is XYZ
axis_and_positive_to_cube_face = { (0, False): 4,
                                   (0, True): 2,
                                   (1, False): 3,
                                   (1, True): 5,
                                   (2, False): 0,
                                   (2, True): 1}

class Profiler(object):
  def __init__(self, name, logger=None, level=logging.INFO):
    self.name = name
    self.logger = logger
    self.level = level

  def step( self, name ):
    """ Returns the duration and stepname since last step/start """
    self.summarize_step( start=self.step_start, step_name=name, level=self.level )
    now = time.time()
    self.step_start = now

  def __enter__( self ):
    self.start = time.time()
    self.step_start = time.time()
    return self
 
  def __exit__( self, exception_type, exception_value, traceback ):
    self.summarize_step( self.start )

  def summarize_step( self, start, step_name="", level=None ):
    duration = time.time() - start
    step_semicolon = ':' if step_name else ""
    if self.logger:
        level = level or self.level
        self.logger.log( self.level, "{name}{step}: {secs} seconds".format( name=self.name, step=step_semicolon + step_name, secs=duration) )
    return duration

def change_shading( color, factor, lighter=True ):
    old_color = np.array( color )
    if lighter: 
        return tuple( old_color + (1. - old_color) * factor )
    else: 
        return tuple( old_color * (1. - factor) )

def create_camera( location, rotation, 
                   field_of_view=settings.FIELD_OF_VIEW_RADS,
                   sensor_dim=settings.SENSOR_DIM, 
                   resolution=settings.RESOLUTION, 
                   scene=bpy.context.scene, 
                   camera_name="Camera" ):   
  """
    Creates a camera in the context scene 

    Args:
      location: The origin of the create_camera
      rotation: Rotation to start the camera in, relative to global (?)
      focal_length: 
      sensor_dim: Size of aperture
      resolution: Sets render x and y sizes
    
    Returns:
      (camera, camera_data, scene): 
        camera: The camera from bpy.data.objects.new
        camera_data: The camera data from bpy.data.cameras.new
        scene: The scene passed in (?)
  """ 
  scene.render.resolution_x = resolution
  scene.render.resolution_y = resolution
  camera_data = bpy.data.cameras.new( camera_name )
  camera = bpy.data.objects.new( name=camera_name, object_data=camera_data )
  scene.objects.link(camera)
  scene.camera = camera
  scene.camera.location = location
  scene.camera.rotation_mode = settings.EULER_ROTATION_ORDER
  scene.camera.rotation_euler = rotation
  # From https://en.wikibooks.org/wiki/Blender_3D:_Noob_to_Pro/Understanding_the_Camera#Specifying_the_Field_of_View
  camera_data.lens_unit = 'FOV'
  focal_length = sensor_dim / ( 2 * math.tan( field_of_view / 2. ) ) 
  camera_data.lens = focal_length
  camera_data.sensor_width = sensor_dim
  camera_data.sensor_height = sensor_dim
  scene.update()
  return (camera, camera_data, scene)


def create_empty(name, location):
  """
    Creates an empty at the given location.

    Args:
      name: Name for the Empty in Blender.
      location: Location of the empty in the Blender model.
    
    Returns: 
      A reference to the created Empty
  """
  scene = bpy.context.scene    
  empty = bpy.data.objects.new(name, None)
  scene.objects.link(empty)
  empty.location = location
  return empty


def create_material_with_texture(texture, name="material"):
  """ 
    Creates a new material in blender and applies the given texture to it using UV mapping

    Args:
      texture: A Blender texture

    Returns:
      material: A Blender material with the texture applied
  """
  material = bpy.data.materials.new( name )
  material.use_shadeless = True
  m_texture = material.texture_slots.add()
  m_texture.texture = texture
  m_texture.texture_coords = 'UV'
  m_texture.use_map_color_diffuse = True 
  m_texture.use_map_color_emission = True 
  m_texture.emission_color_factor = 0.5
  m_texture.use_map_density = True 
  m_texture.mapping = 'FLAT' 

  return material

def create_material_with_color( rgb, name="material", engine='BI' ):
  """ 
    Creates a new material in blender and applies the given texture to it using UV mapping

    Args:
      rgb: A 3-tuple of the RGB values that the material will have

    Returns:
      material: A Blender material with the texture applied
  """
  material = bpy.data.materials.new( name )
  if engine=='BI':
    material.use_shadeless = True
    #   material.use_shadows = False
    #   material.use_cast_shadows = False
    #   material.use_mist = False
    #   material.use_raytrace = False
    material.diffuse_color = rgb
  elif engine=='CYCLES':   
    # Create material
    material.use_nodes = True
    tree = material.node_tree
    links = tree.links

    # Make sure there are no existing nodes
    for node in tree.nodes:
        tree.nodes.remove( node )

    nodes = tree.nodes
    # Use bump map to get normsls
    color_input = nodes.new("ShaderNodeRGB")
    color_input.outputs[0].default_value = list(rgb) + [1.0]

    # Make the material emit that color (so it's visible in render)
    emit_node = nodes.new("ShaderNodeEmission")
    links.new( color_input.outputs[ 0 ], emit_node.inputs[ 0 ] )

    # Now output that color
    out_node = nodes.new("ShaderNodeOutputMaterial")
    links.new( emit_node.outputs[ 0 ], out_node.inputs[ 0 ] )

    material.use_shadeless = True
  return material

def create_render_nodetree( scene ):
    ''' Clears and creates a render nodetree for a scene 
        Args:
            scene: The Blender scene to render
        Returns:
            tree, links
    '''
    scene.use_nodes = True
    tree = scene.node_tree
    links = tree.links

    # Make sure there are no existing nodes
    for node in tree.nodes:
        tree.nodes.remove( node )
    return tree, links

def create_output_node( tree, output_data, tmpdir=None, color_mode='RGB', color_depth=settings.COLOR_BITS_PER_CHANNEL, file_format=settings.PREFERRED_IMG_EXT.upper() ):
  ''' Creates an output node for a scene render nodetree 
    Args: 
        tree: The scene's nodetree
        output_data: This will be fed into the input slot of the output node
        tmpdir: Dir to save the file. If is None, then the fn will create a compositor node.
        color_mode: In ['RGB', 'BW' ]
        file_format: The format to save the image as. 
    Returns:
        save_path: The path where the image will be saved. 
  '''
  links = tree.links
  if tmpdir:
    out = tree.nodes.new('CompositorNodeOutputFile')
    ident = str(uu.uuid4())
    out.file_slots[0].path = ident 
    out.base_path = tmpdir
    out.format.color_mode = color_mode
    out.format.color_depth = color_depth
    out.format.file_format = file_format.upper()
    links.new( output_data, out.inputs[ 0 ] )
    
    # Blender pecululiarly names its files with 0001 (frame #) at the end 
    ext = img_format_to_ext[ file_format.lower() ]
    temp_filename = "{0}0001.{1}".format( ident, ext )

    return os.path.join( tmpdir, temp_filename )
  else:
    out = tree.nodes.new('CompositorNodeComposite')
    links.new( output_data, out.inputs[ 0 ] )
    return None

def delete_all_objects_in_context():
  """ Selects all objects in context scene and deletest them. """
  for obj in bpy.context.scene.objects:
    obj.select = True
  bpy.ops.object.delete() 


def delete_objects_starting_with( prefix ):
  """ Removes all objects whose Blender name begins with the prefix """
  old_mode  = bpy.context.mode
  bpy.ops.object.mode_set(mode='OBJECT')

  for obj in bpy.data.objects:
      if obj.name.startswith( prefix ):
          obj.select = True
          bpy.context.scene.objects.unlink( obj )
          bpy.data.objects.remove( obj )
      else:
          obj.select = False

  bpy.ops.object.delete()
  bpy.ops.object.mode_set( mode=old_mode )


def delete_cameras_and_empties():
  # oldMode = bpy.context.mode
  # bpy.ops.object.mode_set(mode='OBJECT')

  for obj in bpy.data.objects:
      if obj.name.startswith("Camera") or obj.name.startswith("Empty"):
          obj.select = True
          bpy.context.scene.objects.unlink(obj)
          bpy.data.objects.remove(obj)
      else:
          obj.select = False

  bpy.ops.object.delete()
  # bpy.ops.object.mode_set(mode=oldMode)


def get_euler_rotation_between( start, end ):
  """ 
    Returns the Euler rotation so that start.rotate( get_euler_rotation_between( start, end ) ) == end

    Args: 
      start: An Euler
      end: An Euler with the same ordering
    
    Returns: 
      An Euler
  """
  # Gets the rotation by converting Euler angles to rotation matrices and composing
  # return end.to_quaternion().rotation_difference( start.to_quaternion() ).to_euler()
  return ( end.to_matrix() * start.to_matrix().inverted() ).to_euler()

def get_mesh():
  scene = bpy.context.scene
  for ob in scene.objects:
    # whatever objects you want to join...
    if ob.type == 'MESH':
      return ob

def make_camera_data_pano( camera_data ):
  render = bpy.context.scene.render
  render.engine = 'CYCLES'
  camera_data.type = 'PANO'
  camera_data.cycles.panorama_type = 'EQUIRECTANGULAR'
  render.resolution_x, render.resolution_y = settings.PANO_RESOLUTION

def point_camera_at_target_OLD( camera, target ):
  """ 
    Points the given camera at the target. If the target moves, so will the camera. 
    This will leave only the camera selected. 
  """
  constraint = camera.constraints.new(type="TRACK_TO") # Works via adding a constraint to camera
  constraint.target = target
  constraint.track_axis = 'TRACK_NEGATIVE_Z' # Points the local negative z axis (lens) at target
  constraint.up_axis = 'UP_Y' # Keeps the local y axis pointing in the global positive z direction. for orientation
  bpy.ops.object.select_all(action='DESELECT') # Make sure that only the camera is transformd
  camera.select = True
  bpy.ops.object.visual_transform_apply()
  camera.constraints.remove( constraint )

def point_camera_at_target( camera, target, align_with_global_up_axis=False, lock_pitch=False ):
  """ 
    Points the given camera at the target. If the target moves, so will the camera. 
    This will leave only the camera selected. 
  """
  # constraint = camera.constraints.new(type="TRACK_TO") # Works via adding a constraint to camera
  # constraint.target = target
  # constraint.track_axis = 'TRACK_NEGATIVE_Z' # Points the local negative z axis (lens) at target
  # constraint.up_axis = 'UP_Y' # Keeps the local y axis pointing in the global positive z direction. for orientation
  # bpy.ops.object.select_all(action='DESELECT') # Make sure that only the camera is transformd
  # camera.select = True
  # bpy.ops.object.visual_transform_apply()
  # camera.constraints.remove( constraint )
  target_old_rotation_euler = target.rotation_euler.copy()
  target.rotation_euler = camera.rotation_euler 
  target.rotation_euler.rotate_axis( "X", -math.pi / 2) # Since the empty's axes are ordered differently than the camera's
  
  if not lock_pitch: # Use unlocked track
    constraint = camera.constraints.new(type="TRACK_TO") # Works via adding a constraint to camera
    constraint.target = target
    constraint.track_axis = 'TRACK_NEGATIVE_Z' # Points the local negative z axis (lens) at target
    constraint.up_axis = 'UP_Y' # Keeps the local y axis pointing in the global positive z direction. for orientation
    
    if not align_with_global_up_axis:
        # target.rotation_euler = camera.rotation_euler
        constraint.use_target_z = True # Keeps the local y axis pointing in the global positive z direction. for orientation
  else:   
    constraint = camera.constraints.new(type="LOCKED_TRACK") # Works via adding a constraint to camera
    constraint.target = target
    constraint.track_axis = 'TRACK_NEGATIVE_Z' # Points the local negative z axis (lens) at target
    constraint.lock_axis = 'LOCK_Y' # Keeps the local y axis pointing in the global positive z direction. for orientation

  bpy.ops.object.select_all(action='DESELECT') # Make sure that only the camera is transformd
  camera.select = True
  bpy.ops.object.visual_transform_apply()
  camera.constraints.remove( constraint ) 
  # camera.rotation_euler = camera.rotation_euler
  # target.rotation_euler = target_old_rotation_euler

def make_render_fn( setup_scene_fn, setup_nodetree_fn, logger=None ):
  """ Renders a scene 

    Args:
        setup_scene_fn: A function which accepts (scene) 
        setup_nodetree_fn: A function which accepts (scene, output_dir)
    Returns:
        A function which accepts( scene, save_path ) and renders the scene 
            to that save path, using the given nodetree function. 
  """
  def render_fn( scene, save_path ):
    """
        Renders an image from the POV of the camera and save it out

        Args:
        scene: A Blender scene that the camera will render
        save_path: Where to save the image
    """
    outdir, _ = os.path.split(save_path)
    setup_scene_fn( scene )
    render_save_path = setup_nodetree_fn( scene, outdir )
    quiet_render()
    shutil.move( render_save_path, save_path )

    return render_fn
    with Profiler( "Render", logger ) as prf:
        setup_scene_fn( scene )
        render_save_path = setup_nodetree_fn( scene, outdir )
        prf.step( "Setup" )
    
        # bpy.ops.render.render()
        quiet_render()
        prf.step( "Render" )

    with Profiler( "Saving", logger ) as prf:
        shutil.move( render_save_path, save_path )
  return render_fn

def quiet_render():
    ''' sends the noisy blender render info to the bitbucket '''
    # redirect output to log file
    logfile = 'blender_render.log'
    open(logfile, 'a').close()
    old = os.dup(1)
    sys.stdout.flush()
    os.close(1)
    os.open(logfile, os.O_WRONLY)

    # do the rendering
    bpy.ops.render.render(write_still=True)

    # disable output redirection
    os.close(1)
    os.dup(old)
    os.close(old)
    os.remove(logfile)


def set_use_of_gpu_to_render_scene( use=True, compute_device='CUDA_0' ):
  """
    Enables or disables GPU when using cycles

    Args:
      use: Whether to use GPU
      compute_device: Which device to use for cycles. Usually one of ['CUDA_MULTI_0', 'CUDA_0', 'CUDA_1', ...]
  """
  bpy.context.scene.cycles.device = 'GPU' if use else 'CPU'
  bpy.context.user_preferences.system.compute_device_type = 'CUDA'
  bpy.context.user_preferences.system.compute_device = 'CUDA_0'
  print("Default CUDA device: " + bpy.context.user_preferences.system.compute_device)
  print("Default cycles device: " + bpy.context.scene.cycles.device)

  if settings.VERBOSITY >= settings.VERBOSITY_LEVELS[ 'DEBUG' ]:
    print("Default CUDA device: " + bpy.context.user_preferences.system.compute_device)
    print("Render engine: " + scene.render.engine )


def set_preset_render_settings( scene, presets=[ 'BASE' ] ):
  """ Sets Blender render settings to common preset.
    Many of the tasks don't require sampling in cycles, and don't
    require antialiasing. This function disables such features. 

    Args:
        scene: The scene for which to set settings.
        preset: The types of preset to use. Allowable types:
            [ 'BASE', 'RAW' ]
  """
  if 'BASE' in presets:
    # If using cycles, don't sample. 
    scene.cycles.samples = 1
    scene.cycles.max_bounces = 1
    scene.cycles.min_bounces = 1

    # Quality settings
    scene.render.resolution_percentage = 100
    scene.render.tile_x = settings.TILE_SIZE
    scene.render.tile_y = settings.TILE_SIZE

    # Turn off all but the first renderlayer
    for i, layer in enumerate( scene.layers ):
        layer = ( i == 0 )
    render_layer = scene.render.layers["RenderLayer"]
    bpy.types.WorldLighting.indirect_bounces = 1
    scene.render.layers[0].use_all_z = True

    # We don't need raytracing or shadows
    render_layer.use_edge_enhance = False
    scene.render.use_sss = False
    scene.render.use_envmaps = False
    scene.render.use_raytrace = False
    scene.render.use_shadows = False
    scene.render.use_simplify = True

    # Antialiasing leads to incorrect values
    scene.render.use_antialiasing = False

  if 'NON-COLOR' in presets: # Save as non-color data
    scene.view_settings.view_transform = 'Raw'

def set_random_seed():
  # Set seeds
  if settings.RANDOM_SEED:
    np.random.seed( settings.RANDOM_SEED )
    random.seed( settings.RANDOM_SEED )
  
