"""
  Name: create_normals_images.py
  Author: Sasha Sax, CVGL
  Modified by: Zhiyang He
  Desc: Creates surface normals versions of standard RGB images by using the matterport models.
    This reads in all the point#.json files and rendering the corresponding images with surface normals. 

  Usage:
    blender -b -noaudio -enable-autoexec --python create_normal_images.py --
"""

# Import these two first so that we can import other packages
import os
import sys
import bpy

# Import remaining packages
sys.path.append( os.path.dirname( os.path.realpath(__file__) ) )
from activate_env import add_on_path
sys.path.append(add_on_path)

from   load_settings import settings
import create_images_utils
import utils

TASK_NAME = 'normal'

def main():
    apply_texture_fn = None
    if settings.CREATE_PANOS:
        apply_texture_fn = apply_normals_texture
    create_images_utils.run( 
        set_scene_render_settings, 
        setup_nodetree_for_render, 
        model_dir=os.getcwd(),
        task_name=TASK_NAME, 
        apply_texture_fn=apply_texture_fn )

def apply_normals_texture( scene ):
  if not settings.CREATE_PANOS:
      raise EnvironmentError( 'Only panoramic normal images need a texture, but settings.CREATE_PANOS is True' )

  render = bpy.context.scene.render
  render.engine = 'CYCLES'
  
  # Create material
  mat = bpy.data.materials.new( 'normals' )
  mat.use_nodes = True
  tree = mat.node_tree
  links = tree.links

  # Make sure there are no existing nodes
  for node in tree.nodes:
      tree.nodes.remove( node )

  nodes = tree.nodes
  # Use bump map to get normsls
  bump = nodes.new( 'ShaderNodeBump' )

  # Map to new color
  map_node = nodes.new("ShaderNodeMapping")
  map_node.translation[0] = 0.5
  map_node.translation[1] = 0.5 
  map_node.translation[2] = 0.5 
  map_node.scale[0] = 0.5
  map_node.scale[1] = 0.5 
  map_node.scale[2] = -0.5
  links.new( bump.outputs[ 0 ], map_node.inputs[ 0 ] )

  split_node = nodes.new("ShaderNodeSeparateRGB")
  links.new( map_node.outputs[ 0 ], split_node.inputs[ 0 ] )

  combine_node = nodes.new("ShaderNodeCombineRGB")
  links.new( split_node.outputs[ 0 ], combine_node.inputs[ 0 ] ) # R
  links.new( split_node.outputs[ 1 ], combine_node.inputs[ 2 ] ) # G
  links.new( split_node.outputs[ 2 ], combine_node.inputs[ 1 ] ) # B 
  
  # Make the material emit that color (so it's visible in render)
  emit_node = nodes.new("ShaderNodeEmission")
  links.new( combine_node.outputs[ 0 ], emit_node.inputs[ 0 ] )


  # Now output that color
  out_node = nodes.new("ShaderNodeOutputMaterial")
  links.new( emit_node.outputs[ 0 ], out_node.inputs[ 0 ] )

  mat.use_shadeless = True

  # Now apply this material to the mesh
  mesh = utils.get_mesh()
  bpy.context.scene.objects.active = mesh
  bpy.ops.object.material_slot_add()
  mesh.material_slots[ 0 ].material = mat


def set_scene_render_settings( scene ):
  """
    Sets the render settings for speed.

    Args:
      scene: The scene to be rendered 
  """
  utils.set_preset_render_settings( scene, presets=['BASE', 'NON-COLOR'] )

  # Set passes
  scene.render.layers[ "RenderLayer" ].use_pass_combined = True
  scene.render.layers[ "RenderLayer" ].use_pass_z = False
  scene.render.layers[ "RenderLayer" ].use_pass_normal = True


def setup_nodetree_for_render( scene, tmpdir ):
  """
    Creates the scene so that a surface normals image will be saved.
    Note that this method works, but not for Blender 2.69 which is
    the version that exists on Napoli. Therefore, prefer the other 
    method 'setup_scene_for_normals_render_using_matcap'
    
    Args:
      scene: The scene that will be rendered
      tmpdir: The directory to save raw renders to

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

  # We want to use the normals pass in blender, but Blender uses its own
  #   special colors for normals. So we need to map from Blender's colors
  #   to the standard ones. 
  #  Set up a renderlayer and plug it into our remapping layer
  inp = tree.nodes.new('CompositorNodeRLayers')

  if settings.CREATE_PANOS:  # Panos get the normals from texture
    image_data = inp.outputs[ 0 ]
    bpy.data.worlds["World"].horizon_color = (0.5, 0.5, 0.5)
  else:  # Other images get the normals from the scene
    # Remap Blender colors to std
    grey = ( 0.5, 0.5, 0.5, 1 ) #BCBCBC

    mix1 = tree.nodes.new('CompositorNodeMixRGB')
    mix1.blend_type = 'MULTIPLY'
    mix1.inputs[ 2 ].default_value = grey
    links.new( inp.outputs[ 3 ], mix1.inputs[ 1 ] ) # inp.outputs[ 3 ] is the normals socket

    mix2 = tree.nodes.new('CompositorNodeMixRGB')
    mix2.blend_type = 'ADD'
    mix2.inputs[ 2 ].default_value = grey
    links.new( mix1.outputs[ 0 ], mix2.inputs[ 1 ] )

    split = tree.nodes.new('CompositorNodeSepRGBA')
    links.new( mix2.outputs[ 0 ], split.inputs[ 0 ] )

    inv = tree.nodes.new('CompositorNodeInvert')
    links.new( split.outputs[ 0 ], inv.inputs[ 1 ] )

    combine = tree.nodes.new('CompositorNodeCombRGBA')
    links.new( inv.outputs[ 0 ], combine.inputs[ 0 ] ) # R
    links.new( split.outputs[ 1 ], combine.inputs[ 1 ] ) # G
    links.new( split.outputs[ 2 ], combine.inputs[ 2 ] ) # B 
    links.new( split.outputs[ 3 ], combine.inputs[ 3 ] ) # A
    image_data = combine.outputs[ 0 ]

  # Now save out the normals image and return the path
  save_path = utils.create_output_node( tree, image_data, tmpdir, 
        color_mode='RGB',
        file_format=settings.PREFERRED_IMG_EXT )
  return save_path 


if __name__=="__main__":
  with utils.Profiler( "create_normal_images.py" ):
    main()
