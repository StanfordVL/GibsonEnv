"""
  Name: create_depth_images.py
  Author: Sasha Sax, CVGL
  Modified by: Zhiyang He
  Desc: Creates mist versions of standard RGB images by using the matterport models.
    This reads in all the point#.json files and rendering the corresponding images in depth,
    where depth is defined relative to the center of the camera sensor. 

  Usage:
    blender -b -noaudio -enable-autoexec --python create_mist_images.py --
"""
import bpy
import os
import sys

sys.path.append( os.path.dirname( os.path.realpath(__file__) ) )
from activate_env import add_on_path
sys.path.append(add_on_path)

from   load_settings import settings
import create_images_utils
import utils

TASK_NAME = 'mist'

def main():
    apply_texture_fn = None
    create_images_utils.run( 
        set_render_settings, 
        setup_nodetree_for_render, 
        model_dir=os.getcwd(),
        task_name=TASK_NAME, 
        apply_texture_fn=apply_texture_fn )

def set_render_settings( scene ):
  """
    Sets the render settings for speed.

    Args:
      scene: The scene to be rendered 
  """
  if settings.CREATE_PANOS:
    scene.render.engine = 'CYCLES'
  else:
    scene.render.engine = 'BLENDER_RENDER'
  utils.set_preset_render_settings( scene, presets=[ 'BASE', 'NON-COLOR' ] )

  # Simplifying assummptions for depth
  scene.render.layers[ "RenderLayer" ].use_pass_combined = False
  scene.render.layers[ "RenderLayer" ].use_pass_z = False
  scene.render.layers[ "RenderLayer" ].use_pass_mist = True

  # Use mist to simulate depth
  world = bpy.data.worlds["World"] 
  world.horizon_color = (1.,1.,1.)
  world.ambient_color = (0,0,0)
  world.mist_settings.use_mist = True
  world.mist_settings.start = 0. # min range
  world.mist_settings.depth = settings.MIST_MAX_DISTANCE_METERS # max range
  world.mist_settings.intensity = 0. # minimum mist level
  world.mist_settings.height = 0. # mist is prevalent at all z-values
  world.mist_settings.falloff = 'LINEAR'


def setup_nodetree_for_render( scene, outdir ):
  """
    Creates the scene so that a depth image will be saved.

    Args:
      scene: The scene that will be rendered
      outdir: The directory to save raw renders to

    Returns:
      save_path: The path to which the image will be saved
  """
  # Use node rendering for python control
  scene.use_nodes = True
  tree = scene.node_tree
  links = tree.links

  # Make sure there are no existing nodes
  for node in tree.nodes:
      tree.nodes.remove( node )

  #  Set up a renderlayer and plug it into our remapping layer
  inp = tree.nodes.new('CompositorNodeRLayers')
  mist_output = 16 # index 16 is the mist pass
  if scene.render.engine == 'CYCLES':
    image_data = inp.outputs[ mist_output ]
  elif scene.render.engine == 'BLENDER_RENDER':
    inv = tree.nodes.new('CompositorNodeInvert')
    links.new( inp.outputs[ mist_output ], inv.inputs[ 1 ] )
    image_data = inv.outputs[ 0 ]

  save_path = utils.create_output_node( tree, image_data, outdir, 
        color_mode='BW',
        file_format=settings.PREFERRED_IMG_EXT,
        color_depth=settings.DEPTH_BITS_PER_CHANNEL )
  return save_path



if __name__=="__main__":
  with utils.Profiler( "create_mist_images.py" ):
    main()